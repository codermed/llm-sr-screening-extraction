"""Offline OpenRouter integration/concurrency checks; never contact a live API."""
import argparse
import contextlib
import copy
import io
import json
import os
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

import revision_experiment as exp
from test_revision_experiment import fake_result

BASE=Path(__file__).parent


def model():
    return copy.deepcopy(exp.read_json(BASE/'models.example.json')['models'][1])


class Response:
    def __init__(self,data,status=200,headers=None):
        self.data=data;self.status=status;self.headers=headers or {}
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self): return json.dumps(self.data).encode()


def answer(**extra):
    result={'provider':'AtlasCloud','model':'deepseek/deepseek-v3.2-20251201','id':'gen-test',
        'choices':[{'message':{'content':'Final categorization: OK'},'finish_reason':'stop','native_finish_reason':'stop'}],
        'usage':{'prompt_tokens':100,'completion_tokens':10,'cost':0.0123}}
    result.update(extra)
    return result


def call_with(responses,retries=0):
    opener=Mock();opener.open.side_effect=responses
    with patch.dict(os.environ,{'OPENROUTER_API_KEY':'TEST_KEY_NOT_REAL'}), \
        patch.object(exp.urllib.request,'build_opener',return_value=opener),patch.object(exp.time,'sleep') as sleep:
        result=exp.api_call(model(),'system','full paper text','DS',retries,1)
    return result,opener,sleep


class OpenRouterTests(unittest.TestCase):
    def test_fixed_host_and_no_cached_or_fallback_generation(self):
        result,opener,_=call_with([Response(answer())])
        request=opener.open.call_args.args[0]
        payload=json.loads(request.data)
        self.assertEqual(request.full_url,exp.OPENROUTER_ENDPOINT)
        self.assertEqual(request.get_header('Authorization'),'Bearer TEST_KEY_NOT_REAL')
        self.assertEqual(dict((k.lower(),v) for k,v in request.header_items())['x-openrouter-cache'],'false')
        self.assertEqual(payload['provider'],{'only':['atlas-cloud/fp8'],'order':['atlas-cloud/fp8'],
                                            'allow_fallbacks':False,'require_parameters':True})
        self.assertEqual(payload['transforms'],[])
        self.assertFalse(payload['stream'])
        self.assertFalse(payload['reasoning']['enabled'])
        self.assertNotIn('models',payload)
        self.assertEqual(result['returned_provider'],'AtlasCloud')
        self.assertEqual(result['reported_cost_usd'],0.0123)
        self.assertEqual(result['attempts_without_reported_cost'],0)
        self.assertNotIn('TEST_KEY_NOT_REAL',json.dumps(result))

    def test_http_200_error_is_technical_and_retry_uses_same_provider(self):
        error={'error':{'code':429,'message':'limited','metadata':{'provider_name':'AtlasCloud'}}}
        result,opener,sleep=call_with([Response(error,headers={'Retry-After':'7'}),Response(answer())],retries=1)
        self.assertEqual(result['automated_status'],'usable')
        self.assertEqual(result['attempts_without_reported_cost'],1)
        self.assertEqual(result['attempts'][0]['returned_provider'],'AtlasCloud')
        sleep.assert_called_once_with(7)
        self.assertEqual(opener.open.call_args_list[0].args[0].data,opener.open.call_args_list[1].args[0].data)
        result,_,_=call_with([Response(error)])
        self.assertEqual(result['automated_status'],'technical_failure')
        self.assertIsNone(result['reported_cost_usd'])

    def test_temporary_in_flight_budget_402_is_retried(self):
        error={'error':{'code':402,'message':'temporary in-flight budget limit','metadata':{
            'reason':'in_flight_budget_exhausted','headers':{'Retry-After':'120'}}}}
        result,opener,sleep=call_with([Response(error,status=402),Response(answer())],retries=1)
        self.assertEqual(result['automated_status'],'usable')
        self.assertEqual(opener.open.call_count,2)
        sleep.assert_called_once_with(120)
        self.assertEqual(result['attempts'][0]['retry_after'],'120')

    def test_saved_temporary_402_is_archived_and_only_that_call_is_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'run';calls=out/'calls';calls.mkdir(parents=True)
            frozen=root/'frozen';frozen.mkdir();text='paper text '*150
            (frozen/'p.txt').write_text(text)
            row=dict(task='DS',paper_id='p1',text_file='p.txt',text_sha256=exp.digest(text))
            job=(row,model(),1,'DS__p1__deepseek__r1','fingerprint')
            previous=dict(fingerprint='fingerprint',automated_status='technical_failure',http_status=402,
                error={'code':402,'metadata':{'reason':'in_flight_budget_exhausted'}},attempts=[])
            exp.save_json(calls/'DS__p1__deepseek__r1.json',previous)
            config={'transport_retries':2,'timeout_seconds':180}
            prompts={'DS':{'system':'s','user':'{full_text}'}}
            with patch.object(exp,'api_call',return_value=fake_result('DS')) as api:
                call_id,status=exp.execute_job(job,out,config,prompts,frozen)
            self.assertEqual((call_id,status),('DS__p1__deepseek__r1','usable'))
            api.assert_called_once()
            archived=list((out/'transient_failure_history').glob('*.json'))
            self.assertEqual(len(archived),1)
            current=exp.read_json(calls/'DS__p1__deepseek__r1.json')
            self.assertEqual(current['prior_transient_result_files'],
                [str(archived[0].relative_to(out))])

    def test_saved_insufficient_credit_402_waits_for_account_top_up(self):
        result={'automated_status':'technical_failure','http_status':402,
            'error':{'code':402,'message':
                'This request requires more credits, or fewer max_tokens. You requested up to 4096 tokens.'}}
        self.assertTrue(exp.saved_result_is_retryable(result))
        result['error']['message']='Unrelated payment error'
        self.assertFalse(exp.saved_result_is_retryable(result))

    def test_protocol_identity_ignores_only_execution_concurrency(self):
        first={'configuration':{'max_parallel_total':8,'max_parallel_per_model':2,
            'models':[{'name':'m','model':'fixed','max_parallel_requests':2}]},
            'runner_sha256':'old','inputs_sha256':'same','prompts':{'DS':'same'}}
        second=copy.deepcopy(first)
        second['configuration']['max_parallel_total']=4
        second['configuration']['max_parallel_per_model']=1
        second['configuration']['models'][0]['max_parallel_requests']=1
        second['runner_sha256']='approved-new'
        second['compatible_runner_sha256']=['approved-new']
        self.assertEqual(exp.protocol_scientific_identity(first),exp.protocol_scientific_identity(second))
        second['configuration']['models'][0]['model']='different'
        self.assertNotEqual(exp.protocol_scientific_identity(first),exp.protocol_scientific_identity(second))

    def test_policy_error_not_retried_even_with_server_error_code(self):
        data={'error':{'code':502,'message':'content_policy block','metadata':{'provider_name':'AtlasCloud'}}}
        result,opener,sleep=call_with([Response(data)],retries=2)
        self.assertEqual(result['automated_status'],'content_block_candidate')
        self.assertEqual(opener.open.call_count,1)
        sleep.assert_not_called()

    def test_cache_hit_not_usable_repetition(self):
        result,_,_=call_with([Response(answer(),headers={'X-OpenRouter-Cache-Status':'HIT'})])
        self.assertEqual(result['automated_status'],'response_cache_hit')

    def test_native_refusal_signal_retained(self):
        data=answer()
        data['choices'][0]['native_finish_reason']='refusal'
        result,_,_=call_with([Response(data)])
        self.assertEqual(result['automated_status'],'explicit_refusal_signal')

    def test_partial_error_costs_are_included(self):
        a=answer(error={'code':502,'message':'upstream dropped connection'})
        a['usage']['cost']=0.02
        result,_,_=call_with([Response(a),Response(answer())],retries=1)
        self.assertAlmostEqual(result['reported_cost_usd'],0.0323)
        self.assertEqual(result['attempts_with_reported_cost'],2)

    def test_metadata_check_is_free_and_rejects_ambiguous_or_unsupported_host(self):
        m=model()
        ep={'tag':'atlas-cloud/fp8','provider_name':'AtlasCloud','pricing':{'prompt':'0.00000026'},
            'supported_parameters':['temperature','max_tokens','reasoning']}
        opener=Mock();opener.open.return_value=Response({'data':{'endpoints':[ep]}})
        with patch.object(exp.urllib.request,'build_opener',return_value=opener):
            report=exp.inspect_openrouter_models([m])
        self.assertTrue(report['all_available'])
        request=opener.open.call_args.args[0]
        self.assertEqual(request.get_method(),'GET')
        self.assertIsNone(request.data)
        self.assertIsNone(request.get_header('Authorization'))
        m['provider_slug']='atlas-cloud'
        opener.open.return_value=Response({'data':{'endpoints':[ep,dict(ep,tag='atlas-cloud/fp4')]}})
        with patch.object(exp.urllib.request,'build_opener',return_value=opener):
            report=exp.inspect_openrouter_models([m])
        self.assertFalse(report['all_available'])
        m['provider_slug']='atlas-cloud/fp8';ep['supported_parameters']=['max_tokens']
        opener.open.return_value=Response({'data':{'endpoints':[ep]}})
        with patch.object(exp.urllib.request,'build_opener',return_value=opener):
            report=exp.inspect_openrouter_models([m])
        self.assertFalse(report['all_available'])
        self.assertIn('parameters',report['models'][0]['message'])

    def test_configuration_cannot_enable_model_fallbacks_or_router(self):
        for params in ({'models':['another/model']},{'provider':{'allow_fallbacks':True}},{'preset':'cached'}):
            m=model();m['parameters'].update(params)
            with self.assertRaises(ValueError): exp.validate_model(m)
        m=model();m['model']='openrouter/auto'
        with self.assertRaises(ValueError): exp.validate_model(m)

    def test_two_per_model_and_eight_total_can_actually_run_together(self):
        jobs=[({}, {'name':name},r,'id','hash') for name in 'abcd' for r in range(5)]
        active=Counter();peak=Counter();lock=threading.Lock();barrier=threading.Barrier(8)
        started=0;max_total=0
        def worker(job):
            nonlocal started,max_total
            name=job[1]['name']
            with lock:
                active[name]+=1;started+=1;position=started
                peak[name]=max(peak[name],active[name]);max_total=max(max_total,sum(active.values()))
            if position<=8: barrier.wait(timeout=3)
            time.sleep(.005)
            with lock: active[name]-=1
            return name,'ok'
        result=list(exp.run_scheduled(jobs,worker,8,dict.fromkeys('abcd',2)))
        self.assertEqual(len(result),20)
        self.assertEqual(max_total,8)
        self.assertEqual(dict(peak),dict.fromkeys('abcd',2))

    def test_global_limit_and_unequal_model_limits(self):
        jobs=[({}, {'name':name},r,'id','hash') for name in 'ab' for r in range(8)]
        active=Counter();peak=Counter();lock=threading.Lock();max_total=0
        def worker(job):
            nonlocal max_total
            name=job[1]['name']
            with lock:
                active[name]+=1;peak[name]=max(peak[name],active[name]);max_total=max(max_total,sum(active.values()))
            time.sleep(.005)
            with lock: active[name]-=1
            return name,'ok'
        list(exp.run_scheduled(jobs,worker,3,{'a':1,'b':2}))
        self.assertLessEqual(max_total,3);self.assertLessEqual(peak['a'],1);self.assertLessEqual(peak['b'],2)

    def test_four_model_formal_comparison_and_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frozen=root/'inputs';frozen.mkdir()
            text='paper text '*150;(frozen/'p.txt').write_text(text)
            rows=[dict(task='DS',paper_id='p'+str(i),title='Paper '+str(i),doi='',human_label=label,
                groups='DS_random_120',in_random_sample='1',sampling_weight='1',text_file='p.txt',text_sha256=exp.digest(text))
                for i,label in enumerate(exp.LABELS)]
            rows.append(dict(rows[0],paper_id='ie',task='IE',in_random_sample='0',groups='IE_comparison'))
            exp.save_json(frozen/'inputs.json',rows)
            conf=exp.read_json(BASE/'models.example.json');exp.save_json(root/'models.json',conf)
            prompts={t:{'system':'s','user':'{full_text}'} for t in ['DS','IE']};exp.save_json(root/'prompts.json',prompts)
            args=argparse.Namespace(frozen=str(frozen),config=str(root/'models.json'),prompts=str(root/'prompts.json'),
                out=str(root/'formal'),execute=True,model=None,pilot=0,workers=8)
            def fake(m,s,u,task,*a):
                result=fake_result(task)
                result.update(returned_provider=m['provider_slug'],requested_provider=m['provider_slug'],reported_cost_usd=.01,
                    attempts_with_reported_cost=1,attempts_without_reported_cost=0)
                return result
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'TEST_KEY_NOT_REAL'}),patch.object(exp,'api_call',side_effect=fake) as api,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(exp.run(args),0);self.assertEqual(api.call_count,60)
                self.assertEqual(exp.run(args),0);self.assertEqual(api.call_count,60)
                exp.summarize(argparse.Namespace(run=str(root/'formal'),out=str(root/'summary'),review=None))
            costs=exp.read_csv(root/'summary/cost_summary.csv')
            self.assertEqual(len(costs),4)
            self.assertTrue(all(abs(float(x['reported_cost_usd'])-.15)<1e-10 for x in costs))
            comparisons=exp.read_csv(root/'summary/paired_model_comparisons.csv')
            self.assertEqual(len(comparisons),6)
            self.assertTrue(all(x['complete_paired_papers']=='4' and float(x['mean_difference_correct_and_usable'])==0 for x in comparisons))
            self.assertEqual(len(exp.read_csv(root/'summary/deployment_audit.csv')),4)


if __name__=='__main__': unittest.main()
