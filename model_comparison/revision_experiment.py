#!/usr/bin/env python3
"""Fixed-input, three-replicate DS/IE experiment. No Zotero or repository writes.

Commands: prepare (offline), run (dry-run unless --execute), summarize (offline).
Uses Python's standard library; PDF extraction reuses the existing repository's
AcademicPDFProcessor. All provider parameters are explicit in models.json.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import socket
import sqlite3
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

LABELS = ['EXCLUDED', 'OK', 'POSSIBLE', 'REVIEW']
IE_FIELDS = ['objective','country','land_use_classes','baseline_years','target_future_years',
 'scenarios','analysis_methods','simulation_method_advantages','predictors','population_selected',
 'data_sources','tools_software_used','validation_performed','validation_methods',
 'validation_data_sources','key_findings','implications','limitations','future_directions','scalability','novelty']
FINAL_STATUSES = {'usable','refusal','content_block','invalid_output','empty_output','truncated','technical_failure'}
OPENROUTER_ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'


def now():
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds):
    """Compact human-readable duration for notebook progress output."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return '%dh %02dm %02ds' % (hours, minutes, seconds)
    return '%dm %02ds' % (minutes, seconds)


def digest(data):
    if not isinstance(data, bytes):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def norm_title(text):
    return re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', text).casefold())


def norm_doi(text):
    return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi\s*:?\s*)', '', text.strip(), flags=re.I).lower().rstrip('.')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_manifest(path):
    rows = read_csv(path)
    keys = [(r['task'], r['paper_id']) for r in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError('Empty manifest or duplicate paper/task rows')
    for r in rows:
        if r['task'] not in ('DS','IE') or not re.fullmatch(r'[A-Za-z0-9_-]+',r['paper_id']):
            raise ValueError('Unsafe paper identifier or unknown task')
    return rows


def zotero_index(db):
    """Read-only SQLite lookup. Match DOI/title, never trust an export ID alone."""
    con = sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True)
    con.execute('PRAGMA query_only = ON')
    items = {}
    query = '''SELECT i.itemID,i.key,f.fieldName,v.value FROM items i
      JOIN itemData d ON d.itemID=i.itemID JOIN fields f ON f.fieldID=d.fieldID
      JOIN itemDataValues v ON v.valueID=d.valueID
      WHERE f.fieldName IN ('title','DOI') AND i.itemID NOT IN
      (SELECT itemID FROM deletedItems)'''
    for item_id,key,field,value in con.execute(query):
        items.setdefault(item_id, {'key':key, 'title':'', 'DOI':'', 'pdfs':[]})[field] = value or ''
    for parent,key,path,ctype in con.execute('''SELECT a.parentItemID,i.key,a.path,a.contentType
       FROM itemAttachments a JOIN items i ON i.itemID=a.itemID
       WHERE a.parentItemID IS NOT NULL AND i.itemID NOT IN (SELECT itemID FROM deletedItems)'''):
        if parent in items and (ctype == 'application/pdf' or (path or '').lower().endswith('.pdf')):
            items[parent]['pdfs'].append((key,path or ''))
    con.close()
    return list(items.values())


def locate_pdf(row, dbitems, storage):
    if not dbitems:
        return '', '', 'No database supplied; fill pdf_path or text_path in the manifest.'
    exact = [x for x in dbitems if norm_title(x['title']) == norm_title(row['title'])]
    doi = [x for x in dbitems if row['doi'] and norm_doi(x['DOI']) == norm_doi(row['doi'])]
    both = [x for x in exact if x in doi]
    found = both or doi or exact
    if len(found) != 1:
        return '', '', 'No unique DOI/title match (%d matches).' % len(found)
    item = found[0]
    paths = []
    for key, path in item['pdfs']:
        if path.startswith('storage:'):
            if not storage:
                continue
            candidate = Path(storage)/key/path[len('storage:'):]
        elif path.startswith('attachments:'):
            continue  # Relative linked attachments need an explicit manifest path.
        else:
            candidate = Path(path)
        if candidate.is_file():
            paths.append(str(candidate.resolve()))
    paths = sorted(set(paths))
    if len(paths) != 1:
        return '', item['key'], 'Expected one accessible PDF, found %d; set pdf_path explicitly.' % len(paths)
    return paths[0], item['key'], 'Unique DOI/title match and one accessible PDF.'


def prepare(args):
    if args.min_words < 1 or args.max_words < 0 or (args.max_words and args.max_words < args.min_words):
        raise ValueError('Invalid word limits')
    manifest = Path(args.manifest).resolve()
    rows = checked_manifest(manifest)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new/empty frozen-input directory.')
    out.mkdir(parents=True,exist_ok=True)
    dbitems = zotero_index(args.zotero_db) if args.zotero_db else []
    processor = None
    processor_hash = ''
    text_cache = {}
    prepared,report = [],[]
    for row in rows:
        pid = row['paper_id']
        item = dict(row)
        status,reason = 'ready',''
        try:
            if pid not in text_cache:
                source, mode = row.get('text_path','').strip(), 'provided_text'
                key = row.get('zotero_key','')
                if source:
                    p = Path(source).expanduser()
                    p = p if p.is_absolute() else manifest.parent/p
                    text = p.read_text(encoding='utf-8-sig')
                else:
                    pdf = row.get('pdf_path','').strip()
                    if not pdf:
                        pdf,key,reason = locate_pdf(row,dbitems,args.zotero_storage)
                        if not pdf:
                            raise ValueError(reason)
                    p = Path(pdf).expanduser()
                    p = p if p.is_absolute() else manifest.parent/p
                    if processor is None:
                        module_path = Path(args.repo_root).resolve()/'Libs/Zotero/AcademicPDFProcessor.py'
                        processor_hash = digest(module_path.read_bytes())
                        spec = importlib.util.spec_from_file_location('revision_pdf_processor',module_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        processor = module.AcademicPDFProcessor
                    obj = processor(p)
                    obj.extract_text(method='markdown',chunk_size=None)
                    obj.clean_text(remove_refs=True,remove_urls=True,remove_empty_lines=True)
                    text = obj.get_processed_text('cleaned')
                    backend = getattr(obj,'extraction_backend','') or 'unknown'
                    mode = 'repository_cleaned_pdf:'+backend
                original_words = len(text.split())
                if original_words < args.min_words:
                    raise ValueError('Only %d extracted words; inspect PDF/OCR.' % original_words)
                # Apply the cap AFTER cleaning. The old processor's clean_text
                # reads raw text and can ignore truncate_text called beforehand.
                if args.max_words and original_words > args.max_words:
                    text = ' '.join(text.split()[:args.max_words])
                frozen = out/'texts'/(pid+'.txt')
                frozen.parent.mkdir(exist_ok=True)
                frozen.write_text(text,encoding='utf-8')
                text_cache[pid] = dict(text_file=str(frozen.relative_to(out)),text_sha256=digest(text),
                    word_count=len(text.split()),original_word_count=original_words,
                    source_file=str(p.resolve()),source_sha256=digest(p.read_bytes()),
                    text_source=mode,zotero_key=key,cap_applied=original_words>len(text.split()))
            item.update(text_cache[pid])
            prepared.append(item)
        except Exception as exc:
            status,reason = 'missing_or_invalid_input',str(exc)
        report.append(dict(paper_id=pid,task=row['task'],title=row['title'],status=status,reason=reason,
                           text_file=item.get('text_file',''),word_count=item.get('word_count','')))
    write_csv(out/'input_report.csv',report,list(report[0]))
    versions = {}
    for name in ('pymupdf','pymupdf4llm','pypdf','fonttools'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    save_json(out/'preparation.json',dict(created_utc=now(),manifest_sha256=digest(manifest.read_bytes()),
        python=platform.python_version(),packages=versions,processor_sha256=processor_hash,
        text_sources=sorted({x['text_source'] for x in text_cache.values()}),
        max_words=args.max_words,min_words=args.min_words,
        protocol='Freeze one cleaned input per paper; cap after cleaning; reuse for every model and repetition.',
        expected_tasks=len(rows),ready_tasks=len(prepared)))
    if len(prepared) != len(rows):
        print('Inputs incomplete: %d/%d. Inspect %s; fix paths and prepare into a NEW directory.' % (len(prepared),len(rows),out/'input_report.csv'))
        return 2
    save_json(out/'inputs.json',prepared)
    print('Prepared %d tasks from %d unique papers in %s' % (len(prepared),len(text_cache),out))
    return 0


def parse_ds(text):
    matches = re.findall(r'Final\s+categor(?:ization|isation)\s*:\s*\**\s*(EXCLUDED|OK|POSSIBLE|REVIEW)\b',text,re.I)
    return matches[0].upper() if matches and len(set(x.upper() for x in matches))==1 else ''


def parse_ie(text):
    fields = {}
    current = None
    pattern = re.compile(r'^\s*(?:\d+[.)]\s*)?\**('+'|'.join(IE_FIELDS)+r')\**\s*:\s*\**\s*(.*)$',re.I)
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            current = m.group(1).lower()
            fields[current] = m.group(2).strip()
        elif current and line.strip():
            fields[current] += '\n'+line.strip()
    return fields


def response_parts(data,api):
    if api == 'anthropic':
        blocks = data.get('content') or []
        text = '\n'.join(x.get('text','') for x in blocks if x.get('type')=='text')
        reason = data.get('stop_reason','')
        refusal = reason == 'refusal' or any(x.get('type')=='refusal' for x in blocks)
        return text,reason,refusal
    choices = data.get('choices') or []
    if not choices:
        return '','',False
    msg = choices[0].get('message') or {}
    content = msg.get('content') or ''
    if isinstance(content,list):
        content = '\n'.join(x.get('text','') for x in content if isinstance(x,dict))
    return str(content),choices[0].get('finish_reason',''),bool(msg.get('refusal'))


def classify(text,reason,refusal,http_status,raw,task):
    if http_status != 200:
        if re.search(r'DataInspectionFailed|content[_ -]?filter|content[_ -]?policy|inappropriate.content|sensitive.content',raw,re.I):
            return 'content_block_candidate'
        return 'technical_failure'
    if refusal or reason in ('content_filter','refusal'):
        return 'explicit_refusal_signal'
    # Text heuristics only flag a review; they never establish a refusal.
    if re.search(r'^\s*(?:I(?:\s+am|[’\x27]m)?\s+sorry[,.:]?\s*)?(?:I\s+(?:cannot|can[’\x27]t|am unable|will not)|Sorry[,.:]?\s+I)|无法(?:提供|处理|回答)|拒绝(?:回答|提供)',text,re.I):
        return 'possible_refusal_text'
    if reason in ('length','max_tokens'):
        return 'truncated'
    if not text.strip():
        return 'empty_output'
    if task=='DS':
        return 'usable' if parse_ds(text) else 'invalid_output'
    extracted = parse_ie(text)
    return 'usable' if all(extracted.get(k,'').strip() for k in IE_FIELDS) else 'invalid_output'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    # Do not forward API credentials to an unexpected redirect target.
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None


def validate_model(model):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',model['name']) or model['api'] not in ('openai_chat','anthropic','openrouter'):
        raise ValueError('Invalid model name or adapter')
    if not model['endpoint'].startswith('https://'):
        raise ValueError('API endpoint must use HTTPS')
    reserved = {'messages','system','model','models','stream','route','provider','preset','plugins','transforms'}
    if set(model.get('parameters',{})) & reserved:
        raise ValueError('Generation parameters may not override messages, routing, plugins or model identity')
    if model['api']=='openrouter':
        if model['endpoint'] != OPENROUTER_ENDPOINT:
            raise ValueError('OpenRouter adapter requires the official chat-completions endpoint')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',model['model']) or model['model'].startswith('openrouter/'):
            raise ValueError('Use one explicit OpenRouter model ID, without an auto-router or routing suffix')
        if not re.fullmatch(r'[A-Za-z0-9_./-]+',model.get('provider_slug','')):
            raise ValueError('Set one provider_slug for '+model['name'])


def inspect_openrouter_models(models,timeout=30):
    """Free public metadata only: no key, prompts, paper text or inference calls."""
    report = dict(checked_utc=now(),models=[],all_available=True)
    opener = urllib.request.build_opener(NoRedirect)
    for model in models:
        validate_model(model)
        if model['api']!='openrouter':
            continue
        slug=model['provider_slug']
        url='https://openrouter.ai/api/v1/models/'+model['model']+'/endpoints'
        entry=dict(name=model['name'],model=model['model'],requested_provider=slug,url=url,status='unavailable')
        try:
            with opener.open(urllib.request.Request(url,headers={'Accept':'application/json'}),timeout=timeout) as response:
                data=json.loads(response.read().decode('utf-8'))['data']
            endpoints=data.get('endpoints',[])
            # A base provider slug also matches its region/quantization variants.
            matches=[e for e in endpoints if e.get('tag')==slug or ('/' not in slug
                and e.get('tag','').startswith(slug+'/') and e.get('tag','').split('/')[-1] not in ('fast','flex'))]
            entry['available_provider_slugs']=[e.get('tag','') for e in endpoints]
            entry['matching_endpoints']=matches
            if len(matches)!=1:
                raise ValueError('Expected one endpoint; found %d. Choose an exact tag from available_provider_slugs.'%len(matches))
            ep=matches[0]
            supported=set(ep.get('supported_parameters',[]))
            unsupported=set(model.get('parameters',{}))-supported
            if unsupported:
                raise ValueError('Endpoint does not advertise these parameters: '+', '.join(sorted(unsupported)))
            entry.update(status='available',provider_name=ep.get('provider_name',''),
                exact_provider_slug=ep.get('tag',''),pricing=ep.get('pricing',{}),context_length=ep.get('context_length'))
        except (urllib.error.URLError,TimeoutError,OSError,ValueError,KeyError,TypeError) as exc:
            entry['message']=str(exc)
            report['all_available']=False
        report['models'].append(entry)
    return report


def numeric(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def response_error(data):
    error=data.get('error')
    if not error:
        choices=data.get('choices') or []
        error=choices[0].get('error') if choices else None
    return error if isinstance(error,dict) else ({'message':str(error)} if error else {})


def in_flight_budget_error(status,error):
    """Identify OpenRouter's temporary credit-reservation response."""
    metadata=error.get('metadata') if isinstance(error,dict) else {}
    metadata=metadata if isinstance(metadata,dict) else {}
    try:
        code=int(error.get('code',status)) if isinstance(error,dict) else int(status)
    except (ValueError,TypeError):
        code=0
    return code==402 and metadata.get('reason')=='in_flight_budget_exhausted'


def saved_result_is_retryable(result):
    """Retry saved router credit failures after waiting or adding credit."""
    error=result.get('error') or {}
    try:
        code=int(error.get('code',result.get('http_status',0)))
    except (ValueError,TypeError):
        code=0
    message=str(error.get('message','')).lower()
    insufficient_credit=(code==402 and
        'requires more credits, or fewer max_tokens' in message)
    return (result.get('automated_status')=='technical_failure' and
            (in_flight_budget_error(result.get('http_status',0),error) or insufficient_credit))


def error_retry_after(response_headers,error_metadata):
    """Read Retry-After from either HTTP headers or OpenRouter metadata."""
    value=response_headers.get('retry-after','')
    headers=error_metadata.get('headers') if isinstance(error_metadata,dict) else {}
    if not value and isinstance(headers,dict):
        value=headers.get('Retry-After',headers.get('retry-after',''))
    return value


def retry_delay(header,attempt):
    """Honor Retry-After (seconds or HTTP date); waiting occurs inside one slot."""
    delay=min(2**attempt,30)
    if header:
        try:
            parsed=float(header)
        except ValueError:
            try:
                parsed=(parsedate_to_datetime(header)-datetime.now(timezone.utc)).total_seconds()
            except (ValueError,TypeError,OverflowError):
                parsed=0
        if math.isfinite(parsed):
            delay=max(delay,parsed)
    return max(0,delay)


def api_call(model,system,user,task,retries,timeout):
    validate_model(model)
    api = model['api']
    key = os.environ[model['api_key_env']]
    params = dict(model.get('parameters',{}))
    if api=='anthropic':
        payload = dict(model=model['model'],system=system,messages=[dict(role='user',content=user)],**params)
        headers = {'x-api-key':key,'anthropic-version':'2023-06-01','Content-Type':'application/json'}
    else:
        payload = dict(model=model['model'],messages=[dict(role='system',content=system),dict(role='user',content=user)],stream=False,**params)
        headers = {'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    if api=='openrouter':
        payload['provider']=dict(only=[model['provider_slug']],order=[model['provider_slug']],
            allow_fallbacks=False,require_parameters=True)
        payload['transforms']=[]
        headers['X-OpenRouter-Cache']='false'  # Each repetition must generate a fresh response.
    opener = urllib.request.build_opener(NoRedirect)
    attempts = []
    start = time.monotonic()
    for attempt in range(1,retries+2):
        request = urllib.request.Request(model['endpoint'],data=canonical(payload).encode(),headers=headers,method='POST')
        t = time.monotonic()
        stamp = now()
        status,body,request_id = 0,'',''
        response_headers={}
        try:
            with opener.open(request,timeout=timeout) as response:
                status = response.status
                body = response.read().decode('utf-8',errors='replace')
                request_id = response.headers.get('x-request-id',response.headers.get('request-id',''))
                response_headers={k.lower():v for k,v in response.headers.items()}
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = exc.read().decode('utf-8',errors='replace')
            request_id = exc.headers.get('x-request-id',exc.headers.get('request-id',''))
            response_headers={k.lower():v for k,v in exc.headers.items()}
        except (urllib.error.URLError,socket.timeout,TimeoutError,OSError) as exc:
            body = str(exc)
        body = body.replace(key,'[REDACTED]')
        try:
            data = json.loads(body)
            if not isinstance(data,dict):
                data = {}
        except json.JSONDecodeError:
            data = {}
        error=response_error(data)
        error_metadata=error.get('metadata') or {}
        if not isinstance(error_metadata,dict):
            error_metadata={}
        retry_after=error_retry_after(response_headers,error_metadata)
        provider=data.get('provider') or error_metadata.get('provider_name','')
        usage=data.get('usage') or {}
        if not isinstance(usage,dict):
            usage={}
        cache_status=response_headers.get('x-openrouter-cache-status','')
        attempts.append(dict(attempt=attempt,utc=stamp,http_status=status,request_id=request_id,
            latency_seconds=round(time.monotonic()-t,3),raw_body=body,response_id=data.get('id',''),
            returned_model=data.get('model',''),returned_provider=provider,usage=usage,error=error,
            response_cache_status=cache_status,retry_after=retry_after))
        # Retry transport/rate/server errors only. A retry is part of this
        # planned replicate, not an extra independent observation.
        effective_status=status
        if error:
            try:
                effective_status=int(error.get('code',502))
            except (ValueError,TypeError):
                effective_status=502
            if effective_status<400:
                effective_status=502
        blocked=classify('','',False,effective_status,body,task)=='content_block_candidate'
        temporary_budget=in_flight_budget_error(effective_status,error)
        if (effective_status==0 or effective_status in (408,429,500,502,503,504,529)
                or temporary_budget) and not blocked and attempt <= retries:
            time.sleep(retry_delay(retry_after,attempt))
            continue
        text,finish,refusal = response_parts(data,api)
        choice=(data.get('choices') or [{}])[0]
        native_finish=choice.get('native_finish_reason','')
        refusal=refusal or native_finish in ('refusal','content_filter')
        automated=classify(text,finish,refusal,effective_status,body,task)
        if error and not blocked:
            automated='technical_failure'
        elif finish=='error':
            automated='technical_failure'
        if cache_status.upper()=='HIT':
            automated='response_cache_hit'  # Never score a cached answer as a new replicate.
        reported_costs=[a['usage']['cost'] for a in attempts if numeric(a['usage'].get('cost'))]
        result = dict(http_status=status,returned_model=data.get('model',''),response_id=data.get('id',''),
            returned_provider=provider,requested_provider=model.get('provider_slug',''),
            error=error,router_metadata=data.get('router_metadata',{}),response_cache_status=cache_status,
            reported_cost_usd=sum(reported_costs) if reported_costs else None,
            attempts_with_reported_cost=len(reported_costs),
            attempts_without_reported_cost=len(attempts)-len(reported_costs),
            usage=usage,finish_reason=finish,native_finish_reason=native_finish,text=text,
            automated_status=automated,
            parsed_label=parse_ds(text) if task=='DS' else '',
            extracted_fields=parse_ie(text) if task=='IE' else {},attempts=attempts,
            total_seconds=round(time.monotonic()-start,3))
        return result


def execute_job(job,out,config,prompts,frozen):
    row,model,rep,call_id,fingerprint = job
    target = out/'calls'/(call_id+'.json')
    prior_transient_result_files=[]
    if target.exists():
        previous = read_json(target)
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Existing result has incompatible inputs: '+call_id)
        if not saved_result_is_retryable(previous):
            return call_id,'already_saved'
        prior_transient_result_files=list(previous.get('prior_transient_result_files',[]))
        history=out/'transient_failure_history'
        history.mkdir(exist_ok=True)
        number=1
        archived=history/(call_id+'__failed_attempt_%d.json'%number)
        while archived.exists():
            number+=1
            archived=history/(call_id+'__failed_attempt_%d.json'%number)
        target.replace(archived)
        prior_transient_result_files.append(str(archived.relative_to(out)))
    text = (frozen/row['text_file']).read_text(encoding='utf-8')
    if digest(text) != row['text_sha256']:
        raise ValueError('Frozen text changed: '+row['paper_id'])
    task_prompt = prompts[row['task']]
    user = task_prompt['user'].replace('{full_text}',text)
    # No reference decisions, group membership, geography labels or old model
    # outputs are ever sent in the request.
    result = api_call(model,task_prompt['system'],user,row['task'],config['transport_retries'],config['timeout_seconds'])
    result.update(call_id=call_id,fingerprint=fingerprint,paper_id=row['paper_id'],task=row['task'],
        model_name=model['name'],requested_model=model['model'],replicate=rep,completed_utc=now(),
        input_sha256=row['text_sha256'],prompt_sha256=digest(canonical(task_prompt)),
        settings={k:v for k,v in model.items() if k!='api_key_env'},
        prior_transient_result_files=prior_transient_result_files)
    save_json(target,result)
    return call_id,result['automated_status']


def make_jobs(rows,models,config,prompt_hash,pilot=0):
    if pilot:
        # Same deterministic pilot tasks for every model; pilot output is kept
        # in its own directory and must not be pooled with formal repetitions.
        candidates = sorted(rows,key=lambda r:(r['task'],r['paper_id']))
        rng = random.Random(config['order_seed'])
        # Include both task prompts when available, then fill randomly.
        selected = []
        for task in ('DS','IE'):
            options = [r for r in candidates if r['task']==task and r['paper_id'] not in {x['paper_id'] for x in selected}]
            if options and len(selected)<pilot:
                selected.append(rng.choice(options))
        rest = [r for r in candidates if r['paper_id'] not in {x['paper_id'] for x in selected}]
        rng.shuffle(rest)
        for row in rest:
            if len(selected)>=pilot: break
            if row['paper_id'] not in {x['paper_id'] for x in selected}: selected.append(row)
        rows = selected
    jobs = []
    reps = 1 if pilot else config['replicates']
    for rep in range(1,reps+1):
        round_jobs = []
        for row in rows:
            for m in models:
                call_id = '%s__%s__%s__r%d' % (row['task'],row['paper_id'],m['name'],rep)
                fingerprint = digest(canonical(dict(row=row,model=m,prompt_hash=prompt_hash,replicate=rep,
                    transport_retries=config['transport_retries'],timeout_seconds=config['timeout_seconds'],pilot=bool(pilot))))
                round_jobs.append((row,m,rep,call_id,fingerprint))
        random.Random(config['order_seed']+rep).shuffle(round_jobs)
        jobs.extend(round_jobs)
    return jobs


def run_scheduled(jobs,worker,max_workers,model_limits):
    """Fair bounded submission: no waiting model occupies another model's slot."""
    queues=defaultdict(deque)
    for job in jobs:
        queues[job[1]['name']].append(job)
    order=deque(queues)
    active=Counter()
    futures={}
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        while any(queues.values()) or futures:
            while len(futures)<max_workers:
                selected=None
                for _ in range(len(order)):
                    name=order[0];order.rotate(-1)
                    if queues[name] and active[name]<model_limits[name]:
                        selected=name
                        break
                if selected is None:
                    break
                future=pool.submit(worker,queues[selected].popleft())
                futures[future]=selected
                active[selected]+=1
            if not futures:
                break
            done,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                active[futures.pop(future)]-=1
                yield future.result()
    finally:
        # Only in-flight work can finish after an interrupt; thousands of queued
        # paid calls are never submitted in advance.
        pool.shutdown(wait=True,cancel_futures=True)


def saved_job_is_final(path,fingerprint):
    """Validate a saved call and distinguish final from retryable results."""
    if not path.exists():
        return False
    result=read_json(path)
    if result['fingerprint']!=fingerprint:
        raise ValueError('Existing call fingerprint mismatch')
    return not saved_result_is_retryable(result)


def protocol_scientific_identity(protocol):
    """Exclude execution-only concurrency and approved runner hashes."""
    value=json.loads(canonical(protocol))
    value.pop('runner_sha256',None)
    value.pop('compatible_runner_sha256',None)
    config=value.get('configuration',{})
    config.pop('max_parallel_total',None)
    config.pop('max_parallel_per_model',None)
    for model in config.get('models',[]):
        model.pop('max_parallel_requests',None)
    return value


def run(args):
    frozen = Path(args.frozen).resolve()
    rows = read_json(frozen/'inputs.json')
    config = read_json(args.config)
    prompts = read_json(args.prompts)
    models = [m for m in config['models'] if m.get('enabled',True) and (not args.model or m['name']==args.model)]
    if not models or len({m['name'] for m in models}) != len(models):
        raise ValueError('No selected models or duplicate model names')
    workers=getattr(args,'workers',None)
    if workers is None:
        workers=config.get('max_parallel_total',8)
    limits={m['name']:m.get('max_parallel_requests',config.get('max_parallel_per_model',2)) for m in models}
    if type(workers) is not int or workers<1 or any(type(n) is not int or n<1 for n in limits.values()):
        raise ValueError('Concurrency limits must be positive integers')
    for m in models:
        validate_model(m)
    for row in rows:
        if digest((frozen/row['text_file']).read_bytes()) != row['text_sha256']:
            raise ValueError('Frozen input checksum mismatch')
    if config['replicates']!=3:
        raise ValueError('This protocol specifies three replicates; set replicates=3.')
    for task in ('DS','IE'):
        if prompts[task]['user'].count('{full_text}') != 1:
            raise ValueError('Task prompt must contain exactly one {full_text} placeholder')
    jobs = make_jobs(rows,models,config,digest(canonical(prompts)),args.pilot)
    out = Path(args.out).resolve()
    existing = sum(saved_job_is_final(out/'calls'/(j[3]+'.json'),j[4]) for j in jobs)
    print('%d planned requests; %d already saved; %d pending.' % (len(jobs),existing,len(jobs)-existing))
    print('Concurrency: total <= %d; per model %s.' % (workers,', '.join('%s <= %d'%x for x in limits.items())))
    if not args.execute:
        print('Dry run only. No requests sent. Add --execute after the input and model configuration checks.')
        return 0
    for m in models:
        if not m['model'] or 'CHOOSE_' in m['model'] or not os.environ.get(m['api_key_env']):
            raise ValueError('Set the exact model ID and environment variable for '+m['name'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'calls').mkdir(exist_ok=True)
    lock = out/'run.lock'
    fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.close(fd)
    try:
        protocol = dict(configuration=config,prompts=prompts,
            inputs_sha256=digest((frozen/'inputs.json').read_bytes()),
            runner_sha256=digest(Path(__file__).read_bytes()),pilot_tasks=args.pilot)
        if (out/'protocol.json').exists():
            saved_protocol=read_json(out/'protocol.json')
            if protocol_scientific_identity(saved_protocol)!=protocol_scientific_identity(protocol):
                raise ValueError('Configuration changed. Use a new run directory; do not mix experiments.')
            allowed_runners={saved_protocol.get('runner_sha256',''),
                *saved_protocol.get('compatible_runner_sha256',[])}
            if protocol['runner_sha256'] not in allowed_runners:
                raise ValueError('Runner code changed without an audited compatible hash. Use a new run directory.')
        else:
            save_json(out/'protocol.json',protocol)
        save_json(out/'input_manifest.json',rows)
        with (out/'execution_sessions.jsonl').open('a',encoding='utf-8') as log:
            log.write(canonical(dict(started_utc=now(),max_parallel_total=workers,model_limits=limits,
                runner_sha256=protocol['runner_sha256']))+'\n')
        pending = []
        for j in jobs:
            p = out/'calls'/(j[3]+'.json')
            if not saved_job_is_final(p,j[4]):
                pending.append(j)
        session_started = time.monotonic()
        session_completed = 0
        pending_at_start = len(pending)
        for call_id,status in run_scheduled(pending,lambda j:execute_job(j,out,config,prompts,frozen),workers,limits):
            session_completed += 1
            completed = existing + session_completed
            elapsed = time.monotonic() - session_started
            remaining = pending_at_start - session_completed
            eta = elapsed * remaining / session_completed
            print('[%d/%d] %.2f%% | elapsed %s | ETA %s | %s %s' % (
                completed,len(jobs),100*completed/len(jobs),format_duration(elapsed),
                format_duration(eta),call_id,status),flush=True)
        if not pending:
            print('All planned requests already have saved final results.',flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return 0


def summarize(args):
    run_dir = Path(args.run).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    protocol = read_json(run_dir/'protocol.json')
    config = protocol['configuration']
    inputs = read_json(run_dir/'input_manifest.json')
    meta = {(r['task'],r['paper_id']):r for r in inputs}
    results = [read_json(p) for p in sorted((run_dir/'calls').glob('*.json'))]
    decisions = {}
    if args.review:
        for r in read_csv(args.review):
            if r.get('final_status',''):
                if r['final_status'] not in FINAL_STATUSES:
                    raise ValueError('Unknown reviewed status: '+r['final_status'])
                if r['call_id'] in decisions:
                    raise ValueError('Duplicate adjudication ID')
                decisions[r['call_id']] = r
    reviews,all_rows = [],[]
    for r in results:
        m = meta[(r['task'],r['paper_id'])]
        auto = r['automated_status']
        review = decisions.get(r['call_id'],{})
        final = review.get('final_status') or (auto if auto in FINAL_STATUSES else 'unreviewed')
        if auto=='response_cache_hit' and final=='usable':
            raise ValueError('A response-cache hit is not a fresh repetition; do not mark it usable.')
        label = review.get('corrected_label') or r['parsed_label']
        if label and label not in LABELS:
            raise ValueError('Invalid reviewed label')
        if final=='usable' and r['task']=='DS' and not label:
            raise ValueError('Usable DS result needs a label')
        entry = dict(call_id=r['call_id'],task=r['task'],paper_id=r['paper_id'],title=m['title'],doi=m['doi'],
            groups=m['groups'],model_name=r['model_name'],replicate=r['replicate'],
            random_subset_ids=m.get('random_subset_ids',''),forced_inclusion=m.get('forced_inclusion','0'),
            sampling_stratum=m.get('sampling_stratum',m['human_label']),
            human_label=m['human_label'],predicted_label=label,automated_status=auto,final_status=final,
            in_random_sample=int(m['in_random_sample']),sampling_weight=m['sampling_weight'],
            requested_model=r.get('requested_model',''),returned_model=r['returned_model'],
            requested_provider=r.get('requested_provider',''),returned_provider=r.get('returned_provider',''),
            response_id=r.get('response_id',''),http_status=r.get('http_status',''),
            finish_reason=r.get('finish_reason',''),native_finish_reason=r.get('native_finish_reason',''),
            response_cache_status=r.get('response_cache_status',''),error_json=canonical(r.get('error',{})),
            prompt_tokens=r['usage'].get('prompt_tokens',r['usage'].get('input_tokens','')),
            completion_tokens=r['usage'].get('completion_tokens',r['usage'].get('output_tokens','')),
            reasoning_tokens=(r['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens',''),
            reported_cost_usd=r.get('reported_cost_usd'),
            attempts_with_reported_cost=r.get('attempts_with_reported_cost',0),
            attempts_without_reported_cost=r.get('attempts_without_reported_cost',len(r['attempts'])),
            total_seconds=r['total_seconds'],attempt_count=len(r['attempts']),
            usage_json=canonical(r['usage']),raw_response_file=str(Path('calls')/(r['call_id']+'.json')),
            **{'ie_'+k:r['extracted_fields'].get(k,'') for k in IE_FIELDS})
        all_rows.append(entry)
        # Review all unusable/flagged outputs, including technically successful
        # but malformed outputs, so a refusal without a keyword is not missed.
        if auto!='usable' or review:
            reviews.append(dict(call_id=r['call_id'],task=r['task'],paper_id=r['paper_id'],title=m['title'],
                model_name=r['model_name'],replicate=r['replicate'],automated_status=auto,
                final_status=review.get('final_status',''),corrected_label=review.get('corrected_label',''),
                evidence_note=review.get('evidence_note',''),raw_response_file=entry['raw_response_file'],
                response_text=r['text']))
    if all_rows:
        write_csv(out/'all_results.csv',all_rows,list(all_rows[0]))
        costs=[]
        deployment=[]
        for model_name in sorted({r['model_name'] for r in all_rows}):
            rr=[r for r in all_rows if r['model_name']==model_name]
            known=[r['reported_cost_usd'] for r in rr if numeric(r['reported_cost_usd'])]
            costs.append(dict(model_name=model_name,completed_calls=len(rr),
                http_attempts=sum(r['attempt_count'] for r in rr),
                reported_cost_usd=sum(known) if known else '',
                attempts_with_reported_cost=sum(r['attempts_with_reported_cost'] for r in rr),
                attempts_without_reported_cost=sum(r['attempts_without_reported_cost'] for r in rr),
                note='Reported API usage only; missing costs are unknown, not zero. Excludes top-up fees/taxes.'))
            groups=Counter((r['requested_model'],r['returned_model'],r['requested_provider'],r['returned_provider'],
                            r['response_cache_status']) for r in rr)
            for (requested,returned,pin,provider,cache),count in sorted(groups.items()):
                deployment.append(dict(model_name=model_name,requested_model=requested,returned_model=returned,
                    requested_provider=pin,returned_provider=provider or 'NOT_REPORTED',
                    response_cache_status=cache,count=count))
        write_csv(out/'cost_summary.csv',costs,list(costs[0]))
        write_csv(out/'deployment_audit.csv',deployment,list(deployment[0]))
    review_fields=['call_id','task','paper_id','title','model_name','replicate','automated_status','final_status',
                   'corrected_label','evidence_note','raw_response_file','response_text']
    if args.review and Path(args.review).resolve()==(out/'outcome_review.csv').resolve():
        raise ValueError('Use a new summary directory; do not overwrite your review file.')
    write_csv(out/'outcome_review.csv',reviews,review_fields)
    metrics,confusions,per_class = [],[],[]
    model_names = sorted({r['model_name'] for r in results})
    expected_random = sum(int(r['in_random_sample']) for r in inputs if r['task']=='DS')
    for model in model_names:
        for rep in range(1,config['replicates']+1):
            rows = [r for r in all_rows if r['task']=='DS' and r['in_random_sample'] and r['model_name']==model and r['replicate']==rep]
            if not rows:
                continue
            usable = [r for r in rows if r['final_status']=='usable']
            correct = sum(r['predicted_label']==r['human_label'] for r in usable)
            full_weight = sum(float(r['sampling_weight']) for r in inputs if r['task']=='DS' and int(r['in_random_sample']))
            usable_weight = sum(float(r['sampling_weight']) for r in usable)
            correct_weight = sum(float(r['sampling_weight']) for r in usable if r['predicted_label']==r['human_label'])
            complete = len(rows)==expected_random
            metrics.append(dict(model_name=model,replicate=rep,expected_random=expected_random,attempted=len(rows),
                usable=len(usable),pending=expected_random-len(rows),
                usable_output_rate=len(usable)/len(rows),
                accuracy_among_usable=correct/len(usable) if usable else '',
                correct_and_usable_per_attempt=correct/len(rows),
                weighted_usable_rate=usable_weight/full_weight if complete else '',
                weighted_accuracy_among_usable=correct_weight/usable_weight if complete and usable_weight else '',
                weighted_correct_and_usable_per_planned=correct_weight/full_weight if complete else '',
                unreviewed=sum(r['final_status']=='unreviewed' for r in rows)))
            for human in LABELS:
                stratum = [r for r in rows if r['human_label']==human]
                truepos = sum(r['final_status']=='usable' and r['predicted_label']==human for r in stratum)
                usable_h = sum(r['final_status']=='usable' for r in stratum)
                per_class.append(dict(model_name=model,replicate=rep,human_label=human,attempted=len(stratum),
                    usable=usable_h,correct=truepos,
                    recall_among_usable=truepos/usable_h if usable_h else '',
                    correct_and_usable_per_attempt=truepos/len(stratum) if stratum else ''))
                for predicted in LABELS+['NO_USABLE_LABEL']:
                    n = sum((r['predicted_label'] if r['final_status']=='usable' else 'NO_USABLE_LABEL')==predicted for r in stratum)
                    confusions.append(dict(model_name=model,replicate=rep,human_label=human,predicted_label=predicted,count=n))
    for name,rows in [('screening_metrics',metrics),('screening_per_class',per_class),('screening_confusion',confusions)]:
        if rows:
            write_csv(out/(name+'.csv'),rows,list(rows[0]))
    recurrence=[]
    grouped=defaultdict(list)
    for r in all_rows:
        grouped[(r['task'],r['paper_id'],r['model_name'])].append(r)
    for (task,pid,model),rows in sorted(grouped.items()):
        m=meta[(task,pid)]
        labels=[r['predicted_label'] for r in rows if r['final_status']=='usable']
        recurrence.append(dict(task=task,paper_id=pid,title=m['title'],groups=m['groups'],model_name=model,
            expected_runs=1 if protocol['pilot_tasks'] else config['replicates'],observed_runs=len(rows),
            confirmed_refusal_runs=sum(r['final_status']=='refusal' for r in rows),
            provider_content_block_runs=sum(r['final_status']=='content_block' for r in rows),
            technical_failure_runs=sum(r['final_status']=='technical_failure' for r in rows),
            unreviewed_runs=sum(r['final_status']=='unreviewed' for r in rows),
            usable_runs=sum(r['final_status']=='usable' for r in rows),
            all_three_same_DS_label=(int(len(labels)==3 and len(set(labels))==1) if task=='DS' else ''),
            run_outcomes=';'.join('%s:%s:%s'%(r['replicate'],r['final_status'],r['predicted_label']) for r in sorted(rows,key=lambda x:x['replicate']))))
    if recurrence:
        write_csv(out/'paper_recurrence.csv',recurrence,list(recurrence[0]))
    pairs=[]
    for i,a in enumerate(model_names):
        for b in model_names[i+1:]:
            paired=[]
            for (task,pid),m in meta.items():
                if task!='DS' or not int(m['in_random_sample']):
                    continue
                aa=grouped.get((task,pid,a),[]);bb=grouped.get((task,pid,b),[])
                if len(aa)!=3 or len(bb)!=3 or any(r['final_status']=='unreviewed' for r in aa+bb):
                    continue
                score=lambda rr:sum(r['final_status']=='usable' and r['predicted_label']==r['human_label'] for r in rr)/3
                paired.append((m.get('sampling_stratum',m['human_label']),score(aa)-score(bb)))
            complete=len(paired)==expected_random and expected_random>0
            # Pair at paper level; stratified resampling preserves the
            # observed sampling-stratum allocation. Replicate calls are not independent observations.
            lo=hi=delta=''
            if complete:
                delta=sum(d for _,d in paired)/len(paired)
                strata=defaultdict(list)
                for h,d in paired:
                    strata[h].append(d)
                rng=random.Random(20260916)
                boot=[]
                for _ in range(2000):
                    ds=[rng.choice(v) for v in strata.values() for _ in v]
                    boot.append(sum(ds)/len(ds))
                boot.sort();lo=boot[49];hi=boot[1949]
            pairs.append(dict(model_a=a,model_b=b,complete_paired_papers=len(paired),expected_random=expected_random,
                mean_difference_correct_and_usable=delta,bootstrap_95_low=lo,bootstrap_95_high=hi,
                estimand='unweighted union of random papers; mean of 3 calls per paper; failures score zero; descriptive comparison'))
    if pairs:
        write_csv(out/'paired_model_comparisons.csv',pairs,list(pairs[0]))
    write_subset_summaries(all_rows,inputs,config,out)
    frame_description=next((r.get('sampling_frame_description') for r in inputs if r.get('sampling_frame_description')),
        'reconstructed 2088-record RDF DS frame')
    (out/'INTERPRETATION.txt').write_text(
        'Use screening_metrics only for the random sample. Controls/candidates are selected deliberately.\n'
        'Weighted rates target: '+frame_description+'. They do not describe the original search pool.\n'
        'Refusal and content-block counts require reviewing outcome_review.csv and rerunning summarize.\n'
        'Zero confirmed refusals before adjudication does not establish absence of refusals.\n'
        'Technical failures are not refusals. A missing label is not necessarily a content block.\n'
        'IE is a targeted failure check, not validation of general extraction accuracy.\n'
        'One paper remains one independent paper even when called three times per model.\n'
        'Deployment differences and recurrence do not by themselves establish geopolitical bias.\n',encoding='utf-8')
    with (out/'INTERPRETATION.txt').open('a',encoding='utf-8') as notes:
        notes.write('Inspect deployment_audit.csv for the actual reported models/hosts; missing provider identity stays unknown.\n'
            'Response-cache hits are not fresh repetitions and must not be scored as usable replicates.\n'
            'cost_summary.csv sums reported costs across recorded HTTP attempts; absent costs are unknown.\n'
            'Costs exclude credit-purchase fees/taxes and possibly requests billed after a client timeout.\n')
    print('Summarized %d saved calls. Review flagged/unusable responses in %s.' % (len(results),out/'outcome_review.csv'))
    return 0


def write_subset_summaries(all_rows,inputs,config,out):
    """Same saved calls reused for each membership; never duplicate paper evidence."""
    subsets=sorted({s for r in inputs for s in r.get('random_subset_ids','').split(';') if s})
    if not subsets: return
    models=sorted({r['model_name'] for r in all_rows})
    grouped=defaultdict(list)
    for r in all_rows: grouped[(r['task'],r['paper_id'],r['model_name'])].append(r)
    metrics=[];pairs=[]
    for subset in subsets:
        sample={r['paper_id']:r for r in inputs if r['task']=='DS' and subset in r.get('random_subset_ids','').split(';')}
        for model in models:
            for rep in range(1,config['replicates']+1):
                rr=[r for r in all_rows if r['task']=='DS' and r['paper_id'] in sample and r['model_name']==model and r['replicate']==rep]
                if not rr: continue
                usable=[r for r in rr if r['final_status']=='usable']
                correct=sum(r['predicted_label']==r['human_label'] for r in usable)
                weights={pid:float(r['stratum_N'])/float(r['subset_per_class']) for pid,r in sample.items()}
                complete=len(rr)==len(sample)
                metrics.append(dict(subset_id=subset,model_name=model,replicate=rep,expected_papers=len(sample),
                    attempted=len(rr),pending=len(sample)-len(rr),usable=len(usable),
                    accuracy_among_usable=correct/len(usable) if usable else '',
                    correct_and_usable_per_attempt=correct/len(rr),
                    weighted_correct_and_usable=sum(weights[r['paper_id']] for r in usable if r['predicted_label']==r['human_label'])/sum(weights.values()) if complete else '',
                    unreviewed=sum(r['final_status']=='unreviewed' for r in rr)))
        for i,a in enumerate(models):
            for b in models[i+1:]:
                differences=[]
                for pid in sample:
                    aa=grouped[('DS',pid,a)];bb=grouped[('DS',pid,b)]
                    if len(aa)!=3 or len(bb)!=3 or any(r['final_status']=='unreviewed' for r in aa+bb): continue
                    score=lambda rows:sum(r['final_status']=='usable' and r['predicted_label']==r['human_label'] for r in rows)/3
                    differences.append(score(aa)-score(bb))
                pairs.append(dict(subset_id=subset,model_a=a,model_b=b,expected_papers=len(sample),complete_paired_papers=len(differences),
                    mean_difference_correct_and_usable=sum(differences)/len(differences) if differences and len(differences)==len(sample) else '',
                    note='Descriptive subset sensitivity; repeated draws may overlap. Main inference uses unique papers.'))
    if metrics: write_csv(out/'subset_screening_metrics.csv',metrics,list(metrics[0]))
    if pairs: write_csv(out/'subset_model_comparisons.csv',pairs,list(pairs[0]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare',help='Resolve local PDFs/text and freeze identical inputs; offline.')
    s.add_argument('--manifest',required=True);s.add_argument('--out',required=True)
    s.add_argument('--repo-root',default='.');s.add_argument('--zotero-db');s.add_argument('--zotero-storage')
    s.add_argument('--max-words',type=int,default=10000,help='Cap after cleaning; 0 retains all text.')
    s.add_argument('--min-words',type=int,default=100)
    s.set_defaults(func=prepare)
    s=sub.add_parser('run',help='Plan or execute the fixed cross-model experiment.')
    s.add_argument('--frozen',required=True);s.add_argument('--config',required=True)
    s.add_argument('--prompts',required=True);s.add_argument('--out',required=True)
    s.add_argument('--model',help='One configured model name; omitted means all enabled models.')
    s.add_argument('--workers',type=int,default=None,help='Total concurrency; defaults to config max_parallel_total (8).')
    s.add_argument('--pilot',type=int,default=0,metavar='TASKS')
    s.add_argument('--execute',action='store_true');s.set_defaults(func=run)
    s=sub.add_parser('summarize',help='Summarize saved results and create the outcome-review sheet.')
    s.add_argument('--run',required=True);s.add_argument('--out',required=True);s.add_argument('--review')
    s.set_defaults(func=summarize)
    args=p.parse_args()
    try:
        return args.func(args)
    except (ValueError,FileNotFoundError,FileExistsError,ImportError,sqlite3.Error) as exc:
        print('STOP:',exc,file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
