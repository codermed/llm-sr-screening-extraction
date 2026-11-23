"""Read-only Zotero collection sampling with explicit availability and provenance.

No API calls, no Zotero writes. Used to draw the balanced random sample of the
supplementary model comparison from a local Zotero database (see README.md:
per_class=13, subsets=3, seed=20260917, mode="disjoint", human labels in "rights").
"""
import csv
import hashlib
import json
import platform
import random
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from build_revision_sample import human_label

LABELS=['EXCLUDED','OK','POSSIBLE','REVIEW']
FIELDS=['title','DOI','date','rights','archive','callNumber','archiveLocation','libraryCatalog']


def canonical(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def sha(x): return hashlib.sha256(x.encode()).hexdigest()
def norm_title(x): return re.sub(r'[^\w]+','',unicodedata.normalize('NFKC',str(x)).casefold())
def norm_doi(x): return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi\s*:?\s*)','',str(x).strip(),flags=re.I).lower().rstrip('.')


def connection(db):
    con=sqlite3.connect(Path(db).resolve().as_uri()+'?mode=ro',uri=True)
    con.row_factory=sqlite3.Row
    con.execute('PRAGMA query_only=ON')
    con.execute('BEGIN')
    return con


def _collections(con):
    tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    deleted={r[0] for r in con.execute('SELECT collectionID FROM deletedCollections')} if 'deletedCollections' in tables else set()
    rows={r['collectionID']:dict(r) for r in con.execute('SELECT collectionID,collectionName,parentCollectionID,libraryID,key FROM collections')}
    def path(cid,seen=None):
        seen=set() if seen is None else seen
        if cid in seen: raise ValueError('Cycle in Zotero collection hierarchy')
        seen.add(cid);r=rows[cid];parent=r['parentCollectionID']
        if cid in deleted or parent in deleted: return None
        if parent in rows:
            prefix=path(parent,seen)
            return None if prefix is None else prefix+' / '+r['collectionName']
        return r['collectionName']
    result=[]
    for cid,r in rows.items():
        p=path(cid)
        if p is not None:
            result.append(dict(collection_id=cid,library_id=r['libraryID'],collection_key=r['key'],
                collection_path=p,parent_id=r['parentCollectionID']))
    return sorted(result,key=lambda x:(x['library_id'],x['collection_path'],x['collection_key']))


def list_collections(db):
    con=connection(db)
    try:
        rows=_collections(con)
        counts=dict(con.execute('SELECT collectionID,COUNT(*) FROM collectionItems WHERE itemID NOT IN (SELECT itemID FROM deletedItems) GROUP BY collectionID'))
        for r in rows: r['direct_member_count']=counts.get(r['collection_id'],0)
        return rows
    finally: con.close()


def read_collection(db,collection_key,library_id=None,include_children=True,item_types=('journalArticle',)):
    """Load article metadata/attachment paths only inside the selected collection."""
    if not collection_key: raise ValueError('Choose COLLECTION_KEY from the collection list first.')
    con=connection(db)
    try:
        collections=_collections(con)
        selected=[r for r in collections if r['collection_key']==collection_key and (library_id is None or r['library_id']==library_id)]
        if len(selected)!=1:
            raise ValueError('Collection key must identify one collection; set LIBRARY_ID if the key is ambiguous.')
        chosen=selected[0];ids={chosen['collection_id']}
        if include_children:
            while True:
                expanded=ids|{r['collection_id'] for r in collections if r['parent_id'] in ids and r['library_id']==chosen['library_id']}
                if expanded==ids: break
                ids=expanded
        slots=','.join('?' for _ in ids)
        types=tuple(item_types)
        if not types: raise ValueError('Set at least one ITEM_TYPE.')
        query=f'''SELECT DISTINCT i.itemID,i.key,i.libraryID,t.typeName FROM items i
            JOIN itemTypes t ON t.itemTypeID=i.itemTypeID JOIN collectionItems ci ON ci.itemID=i.itemID
            WHERE ci.collectionID IN ({slots}) AND t.typeName IN ({','.join('?' for _ in types)})
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)'''
        items={r['itemID']:dict(r) for r in con.execute(query,tuple(sorted(ids))+types)}
        if not items: raise ValueError('No selected article types in this collection.')
        # Query membership in SQL; metadata from unrelated collections is not loaded.
        scoped=f'''SELECT DISTINCT itemID FROM collectionItems WHERE collectionID IN ({slots})'''
        for row in con.execute(f'''SELECT d.itemID,f.fieldName,v.value FROM itemData d
            JOIN fields f ON f.fieldID=d.fieldID JOIN itemDataValues v ON v.valueID=d.valueID
            WHERE d.itemID IN ({scoped}) AND f.fieldName IN ({','.join('?' for _ in FIELDS)})''',tuple(sorted(ids))+tuple(FIELDS)):
            if row['itemID'] in items: items[row['itemID']][row['fieldName']]=row['value'] or ''
        for r in items.values(): r['attachments']=[]
        for a in con.execute(f'''SELECT a.parentItemID,i.key,a.path,a.contentType FROM itemAttachments a
            JOIN items i ON i.itemID=a.itemID WHERE a.parentItemID IN ({scoped})
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)''',tuple(sorted(ids))):
            if a['parentItemID'] in items and (a['contentType']=='application/pdf' or (a['path'] or '').lower().endswith('.pdf')):
                items[a['parentItemID']]['attachments'].append(dict(key=a['key'],path=a['path'] or ''))
        for r in items.values():
            for f in FIELDS: r.setdefault(f,'')
            r['paper_id']='z%d_%s'%(r['libraryID'],r['key'])
            r['zotero_key']=r['key'];r['doi']=norm_doi(r['DOI'])
            r['human_rights']=r['rights'];r['human_archive']=r['archive']
            r['llm_ds']=r['callNumber'];r['llm_is']=r['archiveLocation'];r['priority']=r['libraryCatalog']
            r['year']=r['date'][:4];r['authors']=''
            r['collection_key']=chosen['collection_key'];r['collection_path']=chosen['collection_path']
        return sorted(items.values(),key=lambda r:r['paper_id']),dict(chosen,include_children=bool(include_children),included_collection_ids=sorted(ids))
    finally: con.close()


def file_check(path,kind):
    p=Path(path).expanduser()
    try:
        if not p.is_file() or p.stat().st_size==0: return False,'missing_or_empty_file'
        with p.open('rb') as f: head=f.read(1024)
        if kind=='pdf' and b'%PDF-' not in head: return False,'invalid_pdf_header'
        return True,'available'
    except OSError: return False,'unreadable_file'


def link_legacy_ids(rows,legacy_rows):
    """Allow existing exported item_ID filenames/overrides to remain usable."""
    by_doi=defaultdict(set);by_title=defaultdict(set)
    for r in legacy_rows:
        if r.get('doi'): by_doi[norm_doi(r['doi'])].add(r['paper_id'])
        by_title[norm_title(r.get('title',''))].add(r['paper_id'])
    for r in rows:
        candidates=by_doi.get(r['doi'],set()) or by_title.get(norm_title(r['title']),set())
        r['legacy_paper_id']=next(iter(candidates)) if len(candidates)==1 else ''


def resolve_files(rows,source,zotero_folder,pdf_folder='',text_folder='',overrides=None,linked_base=''):
    """Existence/readability/header check only; extraction quality is checked later."""
    if source not in ('zotero','pdf_folder','text_folder'): raise ValueError('Unknown SOURCE')
    overrides=overrides or {};folder_files=[]
    if source!='zotero':
        folder=Path(pdf_folder if source=='pdf_folder' else text_folder).expanduser()
        if not str(pdf_folder if source=='pdf_folder' else text_folder) or not folder.is_dir():
            raise ValueError('Set the selected PDF_FOLDER or TEXT_FOLDER to an existing folder.')
        exts={'.pdf'} if source=='pdf_folder' else {'.txt','.md'}
        folder_files=[p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in exts]
    for r in rows:
        aliases=[r['paper_id'],r['zotero_key'],r.get('legacy_paper_id',''),r['doi']]
        supplied={str(Path(overrides[a]).expanduser()) for a in aliases if a and overrides.get(a)}
        paths=[];reason='no_matching_file'
        if len(supplied)>1:
            r.update(file_available=False,file_status='conflicting_file_overrides',pdf_path='',text_path='',candidate_files='')
            continue
        if supplied:
            paths=[Path(next(iter(supplied)))];reason='invalid_override'
        elif source=='zotero':
            for a in r['attachments']:
                path=a['path']
                if path.startswith('storage:'): p=Path(zotero_folder)/'storage'/a['key']/path[8:]
                elif path.startswith('attachments:'):
                    if not linked_base: continue
                    p=Path(linked_base)/path[len('attachments:'):]
                else: p=Path(path)
                if path: paths.append(p)
        else:
            stems=set(a for a in aliases[:3] if a)
            title=norm_title(r['title'])
            paths=[p for p in folder_files if p.stem in stems or any(p.stem==a+'_cln' for a in stems)
                or (len(title)>=20 and title in norm_title(p.stem))]
        valid=[]
        for p in paths:
            kind='pdf' if p.suffix.lower()=='.pdf' else 'text'
            if p.suffix.lower() not in ('.pdf','.txt','.md'): reason='unsupported_file_type';continue
            ok,reason_check=file_check(p,kind)
            if ok: valid.append(str(p.resolve()))
            else: reason=reason_check
        valid=sorted(set(valid))
        if len(valid)==1:
            p=Path(valid[0]);r.update(file_available=True,file_status='available',
                pdf_path=str(p) if p.suffix.lower()=='.pdf' else '',text_path=str(p) if p.suffix.lower()!='.pdf' else '')
        else:
            r.update(file_available=False,file_status='multiple_accessible_files' if len(valid)>1 else reason,pdf_path='',text_path='')
        r['candidate_files']=';'.join(str(p) for p in paths)
        r['candidate_files_json']=json.dumps([str(p) for p in paths])
    return rows


def paper_identity(r):
    return 'doi:'+r['doi'] if r['doi'] else 'title:'+norm_title(r['title']) if r['title'] else 'item:'+r['paper_id']


def sample_records(records,*,per_class=10,subsets=3,seed=20260917,mode='disjoint',
                   human_field='rights',balance_field='rights',require_files=True,forced=None,previous_targets=None):
    """Balanced random core plus forced cases. Forced cases do not fill random quotas."""
    if type(per_class) is not int or per_class<1 or type(subsets) is not int or subsets<1: raise ValueError('Positive integer subset counts are required.')
    if mode not in ('disjoint','independent'): raise ValueError('SUBSET_MODE must be disjoint or independent.')
    allowed={'rights','archive','callNumber','archiveLocation'}
    if human_field not in ('rights','archive') or balance_field not in allowed: raise ValueError('Unsupported human or balance field.')
    rows=[dict(r) for r in records]
    for r in rows:
        r['human_label']=human_label(r.get(human_field,''))
        r['raw_human_label']=r.get(human_field,'')
        raw=r.get(balance_field,'')
        r['sampling_stratum']=human_label(raw) if balance_field in ('rights','archive') else (raw.strip().upper() if raw.strip().upper() in LABELS else '')
        r['balance_raw']=raw;r['duplicate_of']='';r['duplicate_conflict']=False
    duplicate_groups=defaultdict(list)
    for r in rows: duplicate_groups[paper_identity(r)].append(r)
    representatives={}
    for identity,group in duplicate_groups.items():
        conflict=len({r['human_label'] for r in group if r['human_label']})>1 or len({r['sampling_stratum'] for r in group if r['sampling_stratum']})>1
        ordered=sorted(group,key=lambda r:(not bool(r['human_label'] and r['sampling_stratum']),not r['file_available'],r['paper_id']))
        representative=ordered[0];representatives[identity]=representative
        for r in group:
            r['duplicate_conflict']=conflict
            if r is not representative: r['duplicate_of']=representative['paper_id']
    for r in rows:
        reasons=[]
        if r['duplicate_conflict']: reasons.append('conflicting_duplicate_labels')
        if r['duplicate_of']: reasons.append('duplicate_reference')
        if not r['human_label']: reasons.append('unknown_human_label')
        if not r['sampling_stratum']: reasons.append('unknown_sampling_class')
        if require_files and not r['file_available']: reasons.append(r['file_status'])
        r['random_eligible']=not reasons;r['eligibility_reason']=';'.join(reasons) or 'eligible'
    eligible=[r for r in rows if r['random_eligible']]
    strata={c:sorted([r for r in eligible if r['sampling_stratum']==c],key=lambda r:r['paper_id']) for c in LABELS}
    need=per_class*subsets if mode=='disjoint' else per_class
    shortage={c:len(rr) for c,rr in strata.items() if len(rr)<need}
    counts={c:len(rr) for c,rr in strata.items()}
    base_audit=dict(seed=seed,per_class=per_class,subsets=subsets,mode=mode,human_field=human_field,
        balance_field=balance_field,require_local_files=require_files,eligible_per_class=counts,
        collection_records=len(rows),eligible_unique_records=len(eligible),
        maximum_per_class=min(counts.values())//subsets if mode=='disjoint' else min(counts.values()),
        exclusion_counts=dict(Counter(r['eligibility_reason'] for r in rows if not r['random_eligible'])))
    if shortage:
        return dict(ready=False,error=f'Insufficient eligible papers: need {need} per class; available {counts}. Reduce PAPERS_PER_CLASS/NUMBER_OF_SUBSETS explicitly or recover files.',audit=base_audit,frame=rows,manifest=[],memberships=[])
    selected=defaultdict(list)
    for c,pool in strata.items():
        if mode=='disjoint':
            order=random.Random(sha(f'{seed}|{c}|disjoint')).sample(pool,per_class*subsets)
            for s in range(subsets):
                for r in order[s*per_class:(s+1)*per_class]: selected[r['paper_id']].append(f'subset_{s+1:02d}')
        else:
            for s in range(subsets):
                for r in random.Random(sha(f'{seed}|{c}|{s}')).sample(pool,per_class): selected[r['paper_id']].append(f'subset_{s+1:02d}')
    # Resolve forced identifiers within the scoped collection only; no whole-library fallback.
    requests=[dict(x) for x in (forced or [])]
    if previous_targets:
        for old in previous_targets:
            if old.get('task')=='IE' or any('random' not in g for g in old.get('groups','').split(';') if g):
                requests.append(dict(doi=old.get('doi',''),legacy_paper_id=old['paper_id'],tasks=[old['task']],
                    note='Previous targeted sample: '+old.get('groups',''),historical_evidence=old.get('historical_evidence','')))
    forced_map=defaultdict(lambda:dict(tasks=set(),notes=set(),evidence=set()))
    unresolved=[]
    for req in requests:
        identifiers={k:req[k] for k in ('doi','zotero_key','paper_id','legacy_paper_id') if req.get(k)}
        if not identifiers: raise ValueError('Each FORCED_PAPERS entry needs a DOI or Zotero item key.')
        tasks=req.get('tasks',['DS'])
        if not isinstance(tasks,list) or not tasks or set(tasks)-{'DS','IE'}: raise ValueError('Forced tasks must be a list containing DS and/or IE.')
        matches=[]
        # DOI is authoritative for previous exports; use exported ID only if DOI absent.
        for r in rows:
            if identifiers.get('doi'):
                hit=r['doi']==norm_doi(identifiers['doi'])
            else: hit=all(r.get(k,'')==v for k,v in identifiers.items())
            if hit: matches.append(r)
        identities={paper_identity(r) for r in matches}
        if len(identities)!=1:
            unresolved.append(dict(request=canonical(req),reason='not_in_selected_collection' if not matches else 'ambiguous_identifier'))
            continue
        r=representatives[next(iter(identities))]
        f=forced_map[r['paper_id']];f['tasks'].update(tasks)
        f['notes'].add(req.get('note','User requested inclusion'))
        if req.get('historical_evidence'): f['evidence'].add(req['historical_evidence'])
    if unresolved:
        return dict(ready=False,error='Some forced papers could not be identified inside the selected collection. See unresolved_forced.csv.',
            audit=base_audit,frame=rows,manifest=[],memberships=[],unresolved_forced=unresolved)
    manifest=[];membership=[]
    union_counts=Counter(r['sampling_stratum'] for r in eligible if r['paper_id'] in selected)
    description='Eligible unique records in the selected Zotero collection; human reference and sampling class recognized'
    if require_files: description+='; local input file available at selection time'
    for r in sorted(representatives.values(),key=lambda x:x['paper_id']):
        pid=r['paper_id'];f=forced_map.get(pid);random_ids=selected.get(pid,[])
        tasks=({'DS'} if random_ids else set()) | (f['tasks'] if f else set())
        for task in sorted(tasks):
            is_random=task=='DS' and bool(random_ids);is_forced=bool(f and task in f['tasks'])
            N=counts.get(r['sampling_stratum'],0)
            probability=(min(1,per_class*subsets/N) if mode=='disjoint' else 1-(1-per_class/N)**subsets) if is_random else ''
            row={k:v for k,v in r.items() if k!='attachments'}
            row.update(task=task,in_random_sample=int(is_random),forced_inclusion=int(is_forced),
                forced_reason='; '.join(sorted(f['notes'])) if is_forced else '',
                historical_evidence='; '.join(sorted(f['evidence'])) if is_forced else '',
                random_subset_ids=';'.join(random_ids) if is_random else '',subset_per_class=per_class,
                groups=';'.join((['DS_random_union'] if is_random else [])+(['targeted_forced'] if is_forced else [])),
                sampling_weight=1/probability if is_random else '',inclusion_probability=probability,
                stratum_N=N if is_random else '',stratum_n=union_counts[r['sampling_stratum']] if is_random else '',
                sampling_frame_description=description)
            manifest.append(row)
            for s in range(subsets):
                sid=f'subset_{s+1:02d}'
                if (is_random and sid in random_ids) or is_forced:
                    membership.append(dict(subset_id=sid,task=task,paper_id=pid,random=int(is_random and sid in random_ids),
                        forced=int(is_forced),sampling_stratum=r['sampling_stratum']))
    base_audit.update(unique_random_papers=len(selected),random_counts_union=dict(union_counts),
        unique_papers=len({r['paper_id'] for r in manifest}),unique_paper_tasks=len(manifest),
        forced_paper_tasks=sum(r['forced_inclusion'] for r in manifest),
        forced_unavailable=[r['paper_id']+':'+r['task'] for r in manifest if r['forced_inclusion'] and not r['file_available']],
        frame_description=description,weights='Inverse inclusion probability in union of random subsets; forced additions excluded unless also drawn randomly.')
    return dict(ready=True,audit=base_audit,frame=rows,manifest=manifest,memberships=membership)


def write_csv(path,rows,fields=None):
    fields=fields or sorted(set(k for r in rows for k in r if k!='attachments'))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def write_selection(result,out,settings,collection):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Selection folder is not empty. Reuse its frozen sample or choose a new experiment name.')
    audit=dict(result['audit'],created_utc=datetime.now(timezone.utc).isoformat(),collection=collection,
        settings=settings,settings_sha256=sha(canonical(settings)),ready=result['ready'],error=result.get('error',''))
    audit.update(python_version=platform.python_version(),sampler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_csv(out/'availability_frame.csv',result['frame'])
    audit['frame_sha256']=hashlib.sha256((out/'availability_frame.csv').read_bytes()).hexdigest()
    write_csv(out/'missing_or_excluded.csv',[r for r in result['frame'] if not r['random_eligible']])
    write_csv(out/'unresolved_forced.csv',result.get('unresolved_forced',[]),['request','reason'])
    label_counts=Counter((r['raw_human_label'],r['human_label'],r['balance_raw'],r['sampling_stratum']) for r in result['frame'])
    write_csv(out/'label_mapping.csv',[dict(raw_human_label=h,human_label=l,raw_balance=b,sampling_stratum=s,count=n) for (h,l,b,s),n in sorted(label_counts.items())])
    if result['ready']:
        write_csv(out/'sample_manifest.csv',result['manifest'])
        write_csv(out/'subset_membership.csv',result['memberships'])
        write_csv(out/'forced_cases.csv',[r for r in result['manifest'] if r['forced_inclusion']])
        members=defaultdict(set)
        for r in result['memberships']:
            if r['random']: members[r['subset_id']].add(r['paper_id'])
        overlap=[dict(subset_a=a,subset_b=b,shared_random_papers=len(members[a]&members[b])) for a in sorted(members) for b in sorted(members)]
        write_csv(out/'subset_overlap.csv',overlap)
        audit['manifest_sha256']=hashlib.sha256((out/'sample_manifest.csv').read_bytes()).hexdigest()
    (out/'sampling_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return audit
