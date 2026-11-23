#!/usr/bin/env python3
"""Rebuild the fixed revision sample from the supplied Zotero RDF (stdlib only).

Never infers a refusal from an UNDECIDED label. Never modifies the source RDF.
Run --help for the three input paths. Outputs are deterministic for a fixed seed.
Earlier RDF-based sample builder, retained because the regression tests use it.
The sample analysed in the paper was drawn with collection_sampling.py.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

NS = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'z': 'http://www.zotero.org/namespaces/export#',
    'dcterms': 'http://purl.org/dc/terms/', 'dc': 'http://purl.org/dc/elements/1.1/',
    'foaf': 'http://xmlns.com/foaf/0.1/', 'bib': 'http://purl.org/net/biblio#',
    'link': 'http://purl.org/rss/1.0/modules/link/',
    'prism': 'http://prismstandard.org/namespaces/1.2/basic/',
    'vcard': 'http://nwalsh.com/rdf/vCard#',
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)
ABOUT = '{%s}about' % NS['rdf']
RESOURCE = '{%s}resource' % NS['rdf']
LABELS = ['EXCLUDED', 'OK', 'POSSIBLE', 'REVIEW']
IE_REPORTED = {'10.1007/s12040-016-0786-7', '10.1007/s10668-024-05859-w'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm_title(text):
    return re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', text).casefold())


def norm_doi(text):
    return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi\s*:?\s*)', '', text.strip(), flags=re.I).lower().rstrip('.')


def human_label(value):
    """Match the repo's category mapping, without its geography-only TWN rule.

    REVIEW (EXCLUDED) is REVIEW in that four-class coding convention. The raw
    rights value is always retained. Unknown labels are never guessed.
    """
    v = value.upper().strip()
    if v.startswith(('EXCLUDED', 'GLOBAL')) or any(s in v for s in (
        'REFERENCE (PAPER)', 'REFERENCE (REVIEW)', 'DATA (GLOBAL)', 'PLANTS/SPECIES'
    )):
        return 'EXCLUDED'
    for label in ('OK', 'POSSIBLE', 'REVIEW'):
        if v.startswith(label):
            return label
    return ''


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def read_rdf(path):
    root = ET.parse(path).getroot()
    index = {e.get(ABOUT): e for e in root if e.get(ABOUT)}
    records = []
    for e in root:
        if e.findtext('z:itemType', namespaces=NS) != 'journalArticle':
            continue
        containers = list(e.findall('dcterms:isPartOf/*', NS))
        for link in e.findall('dcterms:isPartOf', NS):
            target = index.get(link.get(RESOURCE))
            if target is not None:
                containers.append(target)
        identifiers = list(e.findall('dc:identifier', NS))
        identifiers += [x for c in containers for x in c.findall('dc:identifier', NS)]
        dois = {norm_doi(x.text) for x in identifiers if (x.text or '').upper().startswith('DOI ')}
        if len(dois) > 1:
            raise ValueError('Ambiguous DOI for ' + str(e.get(ABOUT)))
        get = lambda p: (e.findtext(p, default='', namespaces=NS) or '').strip()
        uri = e.get(ABOUT)
        raw_human = get('dc:rights')
        attachments = [index.get(x.get(RESOURCE)) for x in e.findall('link:link', NS)]
        pdfs = [x for x in attachments if x is not None and x.findtext('link:type', namespaces=NS) == 'application/pdf']
        authors = [' '.join(filter(None, [p.findtext('foaf:givenName', namespaces=NS), p.findtext('foaf:surname', namespaces=NS)]))
                   for p in e.findall('bib:authors/rdf:Seq/rdf:li/foaf:Person', NS)]
        rec = dict(paper_id=uri.lstrip('#'), rdf_uri=uri, title=get('dc:title'),
                   doi=next(iter(dois), ''), year=get('dc:date')[:4], authors='; '.join(authors),
                   abstract=get('dcterms:abstract'), human_archive=get('z:archive'),
                   human_rights=raw_human, human_label=human_label(raw_human),
                   llm_is=get('dc:coverage'), llm_ds=get('dc:subject/dcterms:LCC/rdf:value'),
                   priority=get('z:libraryCatalog'), pdf_attachment_count=len(pdfs),
                   pdf_attachment_rdf_ids=';'.join(x.get(ABOUT, '') for x in pdfs))
        records.append(rec)
    if len({r['paper_id'] for r in records}) != len(records):
        raise ValueError('Duplicate RDF item identifiers')
    return root, index, records


def export_rdf(root, index, ids, groups, path):
    """Copy selected articles, linked containers, notes and attachment metadata."""
    selected = {('#' + x) for x in ids}
    keep = set(selected)
    todo = list(selected)
    while todo:
        node = index[todo.pop()]
        for x in node.iter():
            ref = x.get(RESOURCE)
            if ref in index and ref not in keep:
                target = index[ref]
                if target.findtext('z:itemType', namespaces=NS) == 'journalArticle' and ref not in selected:
                    continue
                keep.add(ref)
                todo.append(ref)
    new = ET.Element(root.tag, root.attrib)
    for e in root:
        if e.get(ABOUT) in keep:
            clone = copy.deepcopy(e)
            for parent in clone.iter():
                for child in list(parent):
                    ref = child.get(RESOURCE)
                    if ref in index and ref not in keep:
                        parent.remove(child)
            new.append(clone)
    for name, members in groups.items():
        members = sorted(set(members) & set(ids))
        if not members:
            continue
        c = ET.SubElement(new, '{%s}Collection' % NS['z'], {ABOUT: '#revision_' + name})
        ET.SubElement(c, '{%s}title' % NS['dc']).text = name
        for pid in members:
            ET.SubElement(c, '{%s}hasPart' % NS['dcterms'], {RESOURCE: '#' + pid})
    ET.indent(new, space='    ')
    ET.ElementTree(new).write(path, encoding='UTF-8', xml_declaration=True)


STOP = set('the a an and or of to in on for with by from as at is are was were this that these those study analysis using model models land use cover change changes future based'.split())


def vectors(records):
    counts = {}
    df = Counter()
    for r in records:
        words = [w for w in re.findall(r'[a-z]{3,}', (r['title'] + ' ' + r['abstract']).lower()) if w not in STOP]
        counts[r['paper_id']] = Counter(words)
        df.update(set(words))
    out = {}
    for pid, c in counts.items():
        v = {w: (1 + math.log(n)) * (1 + math.log((1 + len(records)) / (1 + df[w]))) for w, n in c.items()}
        length = math.sqrt(sum(x*x for x in v.values())) or 1
        out[pid] = {w: x / length for w, x in v.items()}
    return out


def geography(record):
    # Explicit text mentions, not nationality, sovereignty, or model provenance.
    text = (record['title'] + ' ' + record['abstract']).lower()
    if re.search(r'\btaiwan\b', text):
        return 'Taiwan'
    if re.search(r'\bindia\b|\bindian\b', text):
        return 'India'
    return ''


def match_controls(cases, eligible, vec, task):
    """Two unique metadata-based comparators per case; matching is provisional.

    Prefer the same human class. One mentions the case geography; the other does
    not explicitly mention it. Rank by title/abstract TF-IDF cosine, then year
    proximity, abstract-length proximity and stable identifier. Absence of a
    place-name is NOT established different geography.
    """
    used = {r['paper_id'] for r in cases}
    matches = []
    for case in sorted(cases, key=lambda r: r['paper_id']):
        geo = geography(case)
        for relation in ('same_mentioned_geography', 'no_same_geography_mention'):
            candidates = [r for r in eligible if r['paper_id'] not in used]
            if geo:
                subset = [r for r in candidates if (geography(r) == geo) == (relation == 'same_mentioned_geography')]
            else:
                subset = []
            fallback = not bool(subset)
            candidates = subset or candidates
            if not candidates:
                raise ValueError('Insufficient unique comparison records')
            def score(r):
                similarity = sum(val * vec[r['paper_id']].get(w, 0) for w, val in vec[case['paper_id']].items())
                year_gap = abs(int(r['year'] or 0) - int(case['year'] or 0))
                length_gap = abs(len(r['abstract'].split()) - len(case['abstract'].split()))
                return (r['human_label'] != case['human_label'], -similarity, year_gap, length_gap, r['paper_id'])
            chosen = min(candidates, key=score)
            used.add(chosen['paper_id'])
            similarity = -score(chosen)[1]
            matches.append(dict(task=task, case_id=case['paper_id'], case_title=case['title'],
                case_doi=case['doi'], control_id=chosen['paper_id'], control_title=chosen['title'],
                control_doi=chosen['doi'], intended_match=relation, case_geography_mention=geo,
                control_geography_mention=geography(chosen), geography_fallback=int(fallback),
                same_human_class=int(chosen['human_label']==case['human_label']),
                title_abstract_cosine=round(similarity, 6),
                matching_status='metadata_based; full_text_length_and_parsing_not_yet_checked'))
    return matches


def build(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory; never overwrite a frozen sample.')
    out.mkdir(parents=True, exist_ok=True)
    root, index, records = read_rdf(args.rdf)
    by_id = {r['paper_id']: r for r in records}
    by_doi = defaultdict(list)
    for r in records:
        if r['doi']:
            by_doi[r['doi']].append(r)
    frame = [r for r in records if r['llm_ds'] in LABELS + ['UNDECIDED']]
    if any(not r['human_label'] for r in frame):
        raise ValueError('Review unclassified human labels before sampling')
    frame_counts = Counter(r['human_label'] for r in frame)
    rng = random.Random(args.seed)
    random_ids = []
    for label in LABELS:
        candidates = sorted((r['paper_id'] for r in frame if r['human_label'] == label))
        random_ids += rng.sample(candidates, args.per_class)
    ds_cases = [r for r in frame if r['llm_ds'] == 'UNDECIDED']
    historical_llm = read_csv(args.llm_csv)
    ie_by_id = {}
    unmatched = []
    for row in historical_llm:
        matches = by_doi.get(norm_doi(row['DOI']), [])
        if len(matches) != 1:
            matches = [r for r in records if norm_title(r['title']) == norm_title(row['Title'])]
        if len(matches) != 1:
            unmatched.append(row['Title'])
        else:
            ie_by_id[matches[0]['paper_id']] = row
    # The supplied RDF is not exactly the old IE manifest: retain/report missing
    # records and draw controls only from the records that can be crosswalked.
    # The two reported refusal cases must still be identifiable below.
    if len(ie_by_id) + len(unmatched) != len(historical_llm):
        raise ValueError('Duplicate IE records in the DOI/title crosswalk')
    ie_cases = [r for r in records if r['doi'] in IE_REPORTED]
    if len(ie_cases) != 2 or any(r['paper_id'] not in ie_by_id for r in ie_cases):
        raise ValueError('Cannot identify both manuscript-reported IE refusal papers')
    vec = vectors(frame)
    ds_pairs = match_controls(ds_cases, [r for r in frame if r['llm_ds'] in LABELS], vec, 'DS')
    ie_eligible = [by_id[pid] for pid, row in ie_by_id.items() if (row.get('LLM Full Response') or '').strip()
                   and by_id[pid]['doi'] not in IE_REPORTED and pid in vec]
    ie_pairs = match_controls(ie_cases, ie_eligible, vec, 'IE')
    groups = {
        'DS_random_120': random_ids,
        'DS_unresolved_candidates': [r['paper_id'] for r in ds_cases],
        'DS_comparison_records': [p['control_id'] for p in ds_pairs],
        'IE_reported_refusal_cases': [r['paper_id'] for r in ie_cases],
        'IE_comparison_records': [p['control_id'] for p in ie_pairs],
    }
    ds_ids = set().union(*(set(v) for k, v in groups.items() if k.startswith('DS_')))
    ie_ids = set().union(*(set(v) for k, v in groups.items() if k.startswith('IE_')))
    manifest = []
    for task, ids in [('DS', ds_ids), ('IE', ie_ids)]:
        for pid in sorted(ids):
            rec = dict(by_id[pid])
            rec.pop('abstract')
            roles = [k for k, members in groups.items() if k.startswith(task+'_') and pid in members]
            is_random = task == 'DS' and pid in random_ids
            rec.update(task=task, groups=';'.join(roles), in_random_sample=int(is_random),
                stratum_N=frame_counts[rec['human_label']] if is_random else '',
                stratum_n=args.per_class if is_random else '',
                sampling_weight=frame_counts[rec['human_label']]/args.per_class if is_random else '',
                historical_evidence=('manuscript_reported_IE_refusal;raw_error_unavailable' if task == 'IE' and rec['doi'] in IE_REPORTED
                   else 'DS_UNDECIDED;cause_unverified' if task == 'DS' and rec['llm_ds'] == 'UNDECIDED'
                   else 'label_or_output_available;not_a_confirmed_refusal'),
                text_path='', pdf_path='', zotero_key='')
            manifest.append(rec)
    write_csv(out/'sample_manifest.csv', manifest)
    write_csv(out/'control_matches.csv', ds_pairs+ie_pairs)
    write_csv(out/'refusal_candidates.csv', [dict(r, evidence='UNDECIDED is not a refusal diagnosis',
        also_reported_IE_refusal=int(r['doi'] in IE_REPORTED)) for r in ds_cases])
    mapping = Counter((r['human_rights'], r['human_label']) for r in frame)
    write_csv(out/'human_label_mapping.csv', [dict(raw_rights=a, normalized=b, count=n) for (a,b),n in sorted(mapping.items())])
    write_csv(out/'sampling_frame.csv', [{k:v for k,v in r.items() if k!='abstract'} for r in frame])
    write_csv(out/'ie_reference_llm.csv', [ie_by_id[pid] for pid in sorted(ie_ids)])
    human = read_csv(args.human_csv)
    selected_dois = {by_id[pid]['doi'] for pid in ie_ids}
    write_csv(out/'ie_reference_human.csv', [r for r in human if norm_doi(r['DOI']) in selected_dois])
    export_rdf(root,index,ds_ids,groups,out/'DS_sample.rdf')
    export_rdf(root,index,ie_ids,groups,out/'IE_sample.rdf')
    export_rdf(root,index,ds_ids|ie_ids,groups,out/'Experiment_sample.rdf')
    audit = dict(seed=args.seed, per_class=args.per_class, source_sha256={Path(p).name:sha256(p) for p in [args.rdf,args.llm_csv,args.human_csv]},
        journal_articles=len(records), ds_label_counts=dict(Counter(r['llm_ds'] for r in records)),
        reconstructed_DS_frame=len(frame), human_strata=dict(frame_counts),
        random_papers=len(random_ids), DS_unresolved=len(ds_cases), DS_control_records=len(ds_pairs),
        DS_unique=len(ds_ids), IE_unique=len(ie_ids), unique_documents=len(ds_ids|ie_ids),
        task_document_pairs=len(manifest), planned_calls_four_models_three_runs=len(manifest)*12,
        random_overlap_with_unresolved=len(set(random_ids)&set(groups['DS_unresolved_candidates'])),
        random_overlap_with_controls=len(set(random_ids)&set(groups['DS_comparison_records'])),
        ie_csv_crosswalk_count=len(ie_by_id),ie_csv_unmatched_titles=unmatched,
        explicit_refusal_logs_in_export=False,
        notes=[
            'The 2088 reconstructed DS outcomes do not identify the exact historical 2084 attempt manifest.',
            'The random frame includes all 11 UNDECIDED cases; missing historical outputs are not silently excluded.',
            'No PDF bytes/local paths or original eight-character Zotero keys are present in the RDF.',
            'Eleven UNDECIDED records are candidates, not a recovered list of seven confirmed refusals.',
            'Matching uses metadata only; absence of a geography mention is not proof of different study geography.',
            'REVIEW (EXCLUDED) maps to REVIEW to preserve the repository four-class convention.',
            'The repository geography-only TWN exclusion rule is not applied; no DS-frame record uses it.',
            'Balanced-class results require weights to describe this reconstructed frame, not the 11688-record initial pool.',
        ])
    (out/'sample_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rdf',required=True)
    p.add_argument('--llm-csv',required=True)
    p.add_argument('--human-csv',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--per-class',type=int,default=30)
    build(p.parse_args())
