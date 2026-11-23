"""Offline regression tests. No provider calls and no Zotero changes.
Run: python -m unittest discover -s revision_experiment -p 'test_*.py' -v
"""
import argparse
import contextlib
import io
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import revision_experiment as exp
import build_revision_sample as sample


def fake_result(task):
    text = 'Final categorization: OK (Justification: eligible)'
    if task=='IE':
        text='\n'.join(k+': Not specified' for k in exp.IE_FIELDS)
    return dict(http_status=200,returned_model='mock-fixed',response_id='mock',usage={},
        finish_reason='stop',text=text,automated_status='usable',parsed_label='OK' if task=='DS' else '',
        extracted_fields=exp.parse_ie(text) if task=='IE' else {},attempts=[{'attempt':1}],total_seconds=0.1)


class OutcomeTests(unittest.TestCase):
    def test_http_retry_payload_and_secret_redaction(self):
        response=type('Response',(),{
            'status':200,'headers':{},
            '__enter__':lambda self:self,'__exit__':lambda self,*a:False,
            'read':lambda self:json.dumps({'model':'fixed','choices':[{
                'message':{'content':'Final categorization: OK'},'finish_reason':'stop'}]}).encode()})()
        opener=unittest.mock.Mock()
        opener.open.side_effect=[urllib.error.HTTPError('https://example.invalid',429,'rate limit',{},
            io.BytesIO(b'unit-test-not-real rate limit')),response]
        model=dict(name='test',api='openai_chat',model='fixed',api_key_env='REVISION_TEST_KEY',
            endpoint='https://example.invalid/chat/completions',parameters={'temperature':0.6})
        with patch.dict(os.environ,{'REVISION_TEST_KEY':'unit-test-not-real'}),patch.object(exp.urllib.request,'build_opener',return_value=opener),patch.object(exp.time,'sleep'):
            result=exp.api_call(model,'system','paper full text','DS',retries=1,timeout=1)
        self.assertEqual(len(result['attempts']),2)
        self.assertEqual(result['automated_status'],'usable')
        self.assertNotIn('unit-test-not-real',result['attempts'][0]['raw_body'])
        payload=json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(payload['messages'],[{'role':'system','content':'system'},{'role':'user','content':'paper full text'}])
        self.assertNotIn('human_label',payload)

    def test_anthropic_refusal_signal(self):
        text,reason,refusal=exp.response_parts({'content':[{'type':'text','text':'I cannot do that.'}],'stop_reason':'refusal'},'anthropic')
        self.assertTrue(refusal)
        self.assertEqual(exp.classify(text,reason,refusal,200,'{}','DS'),'explicit_refusal_signal')

    def test_ds_requires_unambiguous_explicit_label(self):
        self.assertEqual(exp.parse_ds('**Final categorization: OK** (Justification: text)'),'OK')
        self.assertEqual(exp.parse_ds('Maybe OK, maybe EXCLUDED'),'')
        self.assertEqual(exp.parse_ds('Final categorization: OK\nFinal categorization: EXCLUDED'),'')

    def test_ie_multiline_and_missing_fields(self):
        text='\n'.join('**'+k+':** Not specified' for k in exp.IE_FIELDS)
        self.assertEqual(len(exp.parse_ie(text)),21)
        self.assertEqual(exp.classify(text,'stop',False,200,'{}','IE'),'usable')
        self.assertEqual(exp.classify('objective: something','stop',False,200,'{}','IE'),'invalid_output')

    def test_failures_do_not_become_confirmed_refusals(self):
        self.assertEqual(exp.classify('','',False,429,'rate limit','DS'),'technical_failure')
        self.assertEqual(exp.classify('','',False,400,'DataInspectionFailed','DS'),'content_block_candidate')
        self.assertEqual(exp.classify('I cannot process this content.','stop',False,200,'{}','DS'),'possible_refusal_text')
        self.assertEqual(exp.classify('','stop',False,200,'{}','DS'),'empty_output')
        self.assertEqual(exp.classify('Final categorization: OK','length',False,200,'{}','DS'),'truncated')

    def test_category_mapping_retains_unknowns(self):
        self.assertEqual(sample.human_label('REVIEW (EXCLUDED)'),'REVIEW')
        self.assertEqual(sample.human_label('POSSIBLE (REFERENCE)'),'POSSIBLE')
        self.assertEqual(sample.human_label('REFERENCE (PAPER)?'),'EXCLUDED')
        self.assertEqual(sample.human_label('??? TWN'),'')


class WorkflowTests(unittest.TestCase):
    def test_local_zotero_match_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);db=root/'zotero.sqlite';storage=root/'storage'
            (storage/'ATTKEY').mkdir(parents=True)
            (storage/'ATTKEY'/'paper.pdf').write_bytes(b'test')
            con=sqlite3.connect(db)
            con.executescript('''CREATE TABLE items(itemID INTEGER,key TEXT);
              CREATE TABLE fields(fieldID INTEGER,fieldName TEXT);
              CREATE TABLE itemData(itemID INTEGER,fieldID INTEGER,valueID INTEGER);
              CREATE TABLE itemDataValues(valueID INTEGER,value TEXT);
              CREATE TABLE itemAttachments(itemID INTEGER,parentItemID INTEGER,path TEXT,contentType TEXT);
              CREATE TABLE deletedItems(itemID INTEGER);
              INSERT INTO items VALUES(1,'PAPERKEY'),(2,'ATTKEY');
              INSERT INTO fields VALUES(1,'title'),(2,'DOI');
              INSERT INTO itemDataValues VALUES(1,'Paper A'),(2,'10.1/example');
              INSERT INTO itemData VALUES(1,1,1),(1,2,2);
              INSERT INTO itemAttachments VALUES(2,1,'storage:paper.pdf','application/pdf');''')
            con.commit();con.close()
            before=db.read_bytes()
            items=exp.zotero_index(db)
            row={'title':'Paper A','doi':'10.1/example'}
            path,key,_=exp.locate_pdf(row,items,storage)
            self.assertEqual(key,'PAPERKEY');self.assertTrue(Path(path).exists())
            self.assertEqual(exp.locate_pdf(row,items+items,storage)[0],'')
            self.assertEqual(before,db.read_bytes())

    def test_dry_run_resume_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frozen=root/'frozen';frozen.mkdir()
            text='This is frozen full text. '*100
            (frozen/'a.txt').write_text(text)
            rows=[dict(task='DS',paper_id='paper1',title='Title',doi='10.1/x',human_label='OK',
                groups='DS_random_120',in_random_sample='1',sampling_weight='1',
                text_file='a.txt',text_sha256=exp.digest(text))]
            exp.save_json(frozen/'inputs.json',rows)
            config=dict(replicates=3,order_seed=1,transport_retries=0,timeout_seconds=30,
                models=[dict(name='test',model='test-model',api='openai_chat',enabled=True,
                   endpoint='https://example.invalid/chat/completions',api_key_env='REVISION_TEST_KEY',parameters={})])
            prompts={t:dict(system='s',user='{full_text}') for t in ('DS','IE')}
            exp.save_json(root/'config.json',config);exp.save_json(root/'prompts.json',prompts)
            args=argparse.Namespace(frozen=str(frozen),config=str(root/'config.json'),prompts=str(root/'prompts.json'),
                out=str(root/'run'),model=None,pilot=0,execute=False,workers=2)
            with patch.object(exp,'api_call') as api,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(exp.run(args),0);api.assert_not_called()
            args.execute=True
            with patch.dict(os.environ,{'REVISION_TEST_KEY':'unit-test-not-real'}),patch.object(exp,'api_call',side_effect=lambda *a:fake_result(a[3])) as api,contextlib.redirect_stdout(io.StringIO()):
                exp.run(args);self.assertEqual(api.call_count,3)
                exp.run(args);self.assertEqual(api.call_count,3)
                exp.summarize(argparse.Namespace(run=str(root/'run'),out=str(root/'summary'),review=None))
            recurrence=exp.read_csv(root/'summary'/'paper_recurrence.csv')
            self.assertEqual(len(recurrence),1)
            self.assertEqual(recurrence[0]['observed_runs'],'3')
            self.assertEqual(recurrence[0]['all_three_same_DS_label'],'1')
            metrics=exp.read_csv(root/'summary'/'screening_metrics.csv')
            self.assertTrue(all(x['weighted_correct_and_usable_per_planned']=='1.0' for x in metrics))
            (frozen/'a.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                exp.run(args)

    def test_prepare_blocks_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            exp.write_csv(root/'manifest.csv',[dict(task='DS',paper_id='p1',title='T',doi='',text_path='',pdf_path='')],
                          ['task','paper_id','title','doi','text_path','pdf_path'])
            args=argparse.Namespace(manifest=str(root/'manifest.csv'),out=str(root/'frozen'),repo_root='.',
                zotero_db=None,zotero_storage=None,max_words=10000,min_words=100)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(exp.prepare(args),2)
            self.assertFalse((root/'frozen'/'inputs.json').exists())

    def test_provided_text_is_frozen_once_for_both_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'input.txt').write_text('word '*150)
            rows=[dict(task=t,paper_id='p1',title='T',doi='',text_path='input.txt',pdf_path='') for t in ('DS','IE')]
            exp.write_csv(root/'manifest.csv',rows,list(rows[0]))
            args=argparse.Namespace(manifest=str(root/'manifest.csv'),out=str(root/'frozen'),repo_root='.',
                zotero_db=None,zotero_storage=None,max_words=120,min_words=100)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(exp.prepare(args),0)
            inputs=exp.read_json(root/'frozen'/'inputs.json')
            self.assertEqual(inputs[0]['text_sha256'],inputs[1]['text_sha256'])
            self.assertEqual(inputs[0]['word_count'],120)

    def test_pilot_is_one_call_per_selected_task(self):
        rows=[dict(task=t,paper_id='p'+str(i),text_sha256='x') for t in ('DS','IE') for i in range(5)]
        config=dict(order_seed=1,replicates=3,transport_retries=0,timeout_seconds=1)
        jobs=exp.make_jobs(rows,[{'name':'a'},{'name':'b'}],config,'hash',pilot=5)
        self.assertEqual(len(jobs),10)
        self.assertEqual({j[2] for j in jobs},{1})
        self.assertEqual({j[0]['task'] for j in jobs},{'DS','IE'})


if __name__=='__main__':
    unittest.main()
