"""Local fixtures only: collection scope, availability, sampling, subset analysis."""
import argparse
import contextlib
import copy
import io
import json
import random
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import collection_sampling as samp
import revision_experiment as exp


def database_fixture(root):
    db=root/'zotero.sqlite'
    con=sqlite3.connect(db)
    con.executescript('''
      CREATE TABLE collections(collectionID INTEGER PRIMARY KEY,collectionName TEXT,parentCollectionID INTEGER,libraryID INTEGER,key TEXT);
      CREATE TABLE collectionItems(collectionID INTEGER,itemID INTEGER);
      CREATE TABLE deletedCollections(collectionID INTEGER);
      CREATE TABLE deletedItems(itemID INTEGER);
      CREATE TABLE items(itemID INTEGER PRIMARY KEY,key TEXT,libraryID INTEGER,itemTypeID INTEGER);
      CREATE TABLE itemTypes(itemTypeID INTEGER,typeName TEXT);
      CREATE TABLE fields(fieldID INTEGER,fieldName TEXT);
      CREATE TABLE itemData(itemID INTEGER,fieldID INTEGER,valueID INTEGER);
      CREATE TABLE itemDataValues(valueID INTEGER PRIMARY KEY,value TEXT);
      CREATE TABLE itemAttachments(itemID INTEGER,parentItemID INTEGER,path TEXT,contentType TEXT);
      INSERT INTO itemTypes VALUES(1,'journalArticle'),(2,'attachment'),(3,'note');
      INSERT INTO collections VALUES(1,'DS',NULL,1,'DSKEY001'),(2,'Child',1,1,'CHILD001'),
        (3,'Unrelated',NULL,1,'OTHER001'),(4,'DS',NULL,2,'DSKEY001'),(5,'Trashed',1,1,'TRASH001');
      INSERT INTO deletedCollections VALUES(5);
    ''')
    fields={'title':1,'DOI':2,'rights':3,'archive':4,'callNumber':5,'archiveLocation':6,'libraryCatalog':7,'date':8}
    con.executemany('INSERT INTO fields VALUES(?,?)',[(v,k) for k,v in fields.items()])
    def add(i,collection,label='OK',available=True):
        key=f'P{i:07d}'
        con.execute('INSERT INTO items VALUES(?,?,1,1)',(i,key))
        con.execute('INSERT INTO collectionItems VALUES(?,?)',(collection,i))
        for name,value in dict(title=f'Complete article title for publication number {i}',DOI=f'10.1/p{i}',rights=label,archive=label,callNumber=label).items():
            vid=i*100+fields[name]
            con.execute('INSERT INTO itemDataValues VALUES(?,?)',(vid,value))
            con.execute('INSERT INTO itemData VALUES(?,?,?)',(i,fields[name],vid))
        ak=f'A{i:07d}';aid=i+10000
        con.execute('INSERT INTO items VALUES(?,?,1,2)',(aid,ak))
        con.execute('INSERT INTO itemAttachments VALUES(?,?,?,?)',(aid,i,'storage:paper.pdf','application/pdf'))
        if available:
            p=root/'storage'/ak/'paper.pdf';p.parent.mkdir(parents=True)
            p.write_bytes(b'%PDF-1.7\nFixture: availability test only, no extraction.')
        return key
    for c,label in enumerate(samp.LABELS):
        for j in range(12): add(c*100+j+1,1 if j<8 else 2,label,available=j!=0)
    add(999,3,'OK');add(888,5,'OK');add(777,1,'OK');con.execute('INSERT INTO deletedItems VALUES(777)')
    con.execute('INSERT INTO collectionItems VALUES(2,2)') # Same item in parent and child.
    con.commit();con.close()
    return db


def ready_records(n=12):
    result=[]
    for label in samp.LABELS:
        for j in range(n):
            result.append(dict(paper_id=f'{label}_{j}',zotero_key=f'{label}_{j}',doi=f'10.1/{label.lower()}{j}',
                title=f'{label} article {j}',rights=label,archive=label,callNumber=label,archiveLocation=label,
                human_rights=label,human_archive=label,file_available=True,file_status='available',pdf_path='/fixture.pdf',
                text_path='',attachments=[],legacy_paper_id=''))
    return result


class CollectionTests(unittest.TestCase):
    def test_scope_children_duplicates_trash_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);db=database_fixture(root);before=db.read_bytes()
            listed=samp.list_collections(db)
            self.assertNotIn('TRASH001',{r['collection_key'] for r in listed})
            with self.assertRaisesRegex(ValueError,'ambiguous'): samp.read_collection(db,'DSKEY001')
            parent,_=samp.read_collection(db,'DSKEY001',1,False)
            all_rows,scope=samp.read_collection(db,'DSKEY001',1,True)
            self.assertEqual(len(parent),32);self.assertEqual(len(all_rows),48)
            self.assertNotIn('z1_P0000999',{r['paper_id'] for r in all_rows})
            self.assertEqual(scope['included_collection_ids'],[1,2])
            self.assertEqual(before,db.read_bytes())

    def test_local_pdf_filter_and_missing_pdf_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);db=database_fixture(root)
            rows,_=samp.read_collection(db,'DSKEY001',1)
            samp.resolve_files(rows,'zotero',root)
            self.assertEqual(sum(r['file_available'] for r in rows),44)
            result=samp.sample_records(rows,per_class=2,subsets=3)
            self.assertTrue(result['ready'])
            self.assertEqual(len(result['manifest']),24)
            self.assertTrue(all(r['file_available'] for r in result['manifest']))
            self.assertEqual(result['audit']['eligible_per_class'],dict.fromkeys(samp.LABELS,11))
            self.assertEqual(sum(not r['random_eligible'] for r in result['frame']),4)

    def test_disjoint_balance_and_seed_reproducibility(self):
        rows=ready_records();a=samp.sample_records(rows,per_class=2,subsets=3,seed=7)
        shuffled=rows[:];random.Random(4).shuffle(shuffled)
        b=samp.sample_records(shuffled,per_class=2,subsets=3,seed=7)
        self.assertEqual(a['manifest'],b['manifest'])
        per=Counter((r['subset_id'],r['sampling_stratum']) for r in a['memberships'] if r['random'])
        self.assertEqual(set(per.values()),{2});self.assertEqual(len(per),12)
        self.assertTrue(all(len(r['random_subset_ids'].split(';'))==1 for r in a['manifest']))
        self.assertTrue(all(r['sampling_weight']==2 for r in a['manifest']))
        c=samp.sample_records(rows,per_class=2,subsets=3,seed=8)
        self.assertNotEqual({r['paper_id'] for r in a['manifest']},{r['paper_id'] for r in c['manifest']})

    def test_independent_draws_overlap_without_duplicate_calls(self):
        result=samp.sample_records(ready_records(4),per_class=3,subsets=3,mode='independent')
        self.assertTrue(result['ready'])
        self.assertEqual(sum(r['random'] for r in result['memberships']),36)
        self.assertLessEqual(len(result['manifest']),16)
        self.assertTrue(any(';' in r['random_subset_ids'] for r in result['manifest']))
        self.assertTrue(all(abs(r['sampling_weight']-1/(1-(1-3/4)**3))<1e-10 for r in result['manifest']))

    def test_forced_cases_do_not_fill_random_quotas_and_reuse_calls(self):
        rows=ready_records(4);rows[0]['file_available']=False;rows[0]['file_status']='missing_or_empty_file'
        force=[{'doi':rows[0]['doi'],'tasks':['DS','IE'],'note':'target'}]
        result=samp.sample_records(rows,per_class=1,subsets=3,forced=force)
        self.assertTrue(result['ready']);self.assertEqual(result['audit']['unique_random_papers'],12)
        forced=[r for r in result['manifest'] if r['forced_inclusion']]
        self.assertEqual(len(forced),2)
        self.assertTrue(all(not r['in_random_sample'] and not r['file_available'] for r in forced))
        self.assertEqual(len(result['manifest']),14)
        self.assertEqual(sum(r['forced'] for r in result['memberships']),6)
        both=samp.sample_records(ready_records(3),per_class=1,subsets=3,forced=[{'doi':'10.1/ok0','tasks':['DS']}])
        hit=[r for r in both['manifest'] if r['doi']=='10.1/ok0']
        self.assertEqual(len(hit),1);self.assertEqual(hit[0]['in_random_sample'],1);self.assertEqual(hit[0]['forced_inclusion'],1)

    def test_insufficient_classes_and_unknown_forced_case_stop(self):
        result=samp.sample_records(ready_records(2),per_class=1,subsets=3)
        self.assertFalse(result['ready']);self.assertEqual(result['manifest'],[])
        result=samp.sample_records(ready_records(),forced=[{'doi':'10.1/missing'}],per_class=1)
        self.assertFalse(result['ready']);self.assertEqual(len(result['unresolved_forced']),1)

    def test_unknown_labels_duplicates_and_availability_switch(self):
        rows=ready_records();rows[0]['rights']='UNDECIDED';rows[1]['file_available']=False;rows[1]['file_status']='missing'
        rows.append(dict(rows[2],paper_id='another_item',zotero_key='OTHER'))
        result=samp.sample_records(rows,per_class=1,require_files=True)
        self.assertEqual(result['audit']['eligible_per_class']['EXCLUDED'],10)
        switched=samp.sample_records(rows,per_class=1,require_files=False)
        self.assertEqual(switched['audit']['eligible_per_class']['EXCLUDED'],11)
        rows[-1]['rights']='OK'
        conflicted=samp.sample_records(rows,per_class=1,require_files=False)
        self.assertEqual(sum(r['duplicate_conflict'] for r in conflicted['frame']),2)

    def test_folder_sources_legacy_names_ambiguity_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sub=root/'sub';sub.mkdir();r=ready_records(1)[0];r['legacy_paper_id']='item_123'
            pdf=sub/'item_123.pdf';pdf.write_bytes(b'%PDF-1.7 fixture')
            out=samp.resolve_files([copy.deepcopy(r)],'pdf_folder',root,pdf_folder=root)
            self.assertTrue(out[0]['file_available'])
            (root/(r['zotero_key']+'.pdf')).write_bytes(b'%PDF-1.7 duplicate')
            out=samp.resolve_files([copy.deepcopy(r)],'pdf_folder',root,pdf_folder=root)
            self.assertEqual(out[0]['file_status'],'multiple_accessible_files')
            out=samp.resolve_files([copy.deepcopy(r)],'pdf_folder',root,pdf_folder=root,overrides={r['doi']:str(pdf)})
            self.assertTrue(out[0]['file_available'])
            text=sub/'item_123_cln.txt';text.write_text('The actual paper text, not PDF paths. '*100)
            out=samp.resolve_files([copy.deepcopy(r)],'text_folder',root,text_folder=root)
            self.assertEqual(out[0]['text_path'],str(text));self.assertEqual(out[0]['pdf_path'],'')

    def test_audit_saved_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=samp.sample_records(ready_records(),per_class=2)
            out=Path(tmp)/'selection';audit=samp.write_selection(result,out,{'seed':1},{'collection_key':'DS'})
            self.assertTrue(audit['ready']);self.assertTrue((out/'availability_frame.csv').exists())
            with self.assertRaisesRegex(ValueError,'not empty'):samp.write_selection(result,out,{}, {})

    def test_subset_comparisons_exclude_forced_additions(self):
        selected=samp.sample_records(ready_records(4),per_class=1,subsets=3,
            forced=[{'doi':'10.1/ok0','tasks':['IE']}])
        inputs=selected['manifest'];results=[]
        for row in inputs:
            for model in ['a','b']:
                for rep in [1,2,3]:
                    results.append(dict(task=row['task'],paper_id=row['paper_id'],model_name=model,replicate=rep,
                        final_status='usable',human_label=row['human_label'],predicted_label=row['human_label']))
        with tempfile.TemporaryDirectory() as tmp:
            exp.write_subset_summaries(results,inputs,{'replicates':3},Path(tmp))
            metrics=exp.read_csv(Path(tmp)/'subset_screening_metrics.csv')
            self.assertEqual(len(metrics),18)
            self.assertTrue(all(r['expected_papers']=='4' and r['attempted']=='4' for r in metrics))
            pairs=exp.read_csv(Path(tmp)/'subset_model_comparisons.csv')
            self.assertEqual(len(pairs),3)
            self.assertTrue(all(float(r['mean_difference_correct_and_usable'])==0 for r in pairs))


if __name__=='__main__': unittest.main()
