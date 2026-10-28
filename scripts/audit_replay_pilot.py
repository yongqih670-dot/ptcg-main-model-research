"""Audit source structure and labels before choosing any training corpus."""
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import atomic_json,sha256,read_json,utcnow
from ptcg_lab.replay_data import extract_episode,actor_observation


def audit(folder):
    reports=[]
    download=read_json(folder/'pilot-download.json')
    catalog={c['cardId']:c for c in read_json(ROOT/'ptcg_lab/public_cg/metadata.json')['cards']}
    for source in download['episodes']:
        path=folder/source['filename']
        if sha256(path)!=source['sha256']:raise ValueError('Downloaded replay changed or is incomplete')
        e=read_json(path);records,summary=extract_episode(e,'playground-2026-10-20/'+path.stem)
        contexts=Counter();single=Counter();decks={}
        for seat,deck in summary['own_decks'].items():
            unknown=[c for c in deck if c not in catalog]
            if unknown:raise ValueError('Unmapped card IDs in source deck')
            decks[str(seat)]={'hash':__import__('hashlib').sha256(json.dumps(sorted(deck)).encode()).hexdigest(),
                'pokemon':dict(Counter(catalog[c]['name'] for c in deck if catalog[c]['cardType']==0)),
                'ids':dict(Counter(deck))}
        fingerprints=Counter(r['observation_sha256'] for r in records)
        for r in records:
            sel=r['observation']['select'];contexts[str(r['context'])]+=1
            if sel['minCount']==sel['maxCount']==1 and len(sel['option'])>1:single[str(r['context'])]+=1
        # Mutating all omniscient visualization data must not change an actor
        # record. Also poison excluded top-level observation keys.
        poisoned=copy.deepcopy(e)
        for step in poisoned['steps']:
            for item in step:
                item['visualize']={'private_truth':'POISON'}
                item['observation']['search_begin_input']='POISON'
                item['observation']['hidden_truth']={'deck':[99999]}
        rerecords,_=extract_episode(poisoned,summary['group_id'])
        assert records==rerecords
        reports.append({**summary,'own_decks':decks,'source_file':path.name,'sha256':sha256(path),
            'module_version':e['module_version'],'configuration':e['configuration'],
            'nonforced_single_choice_by_context':dict(single),'duplicate_visible_observations':sum(n-1 for n in fingerprints.values()),
            'all_rounds_and_seats_same_split':len({r['split'] for r in records})==1,
            'omniscient_and_opaque_input_invariance':True})
    report={'at':utcnow(),'episodes':reports,'total_decisions':sum(r['decision_count'] for r in reports),
        'training_started':False,'limits':'Schema and index legality only; old/new DLL equality and semantic replay compatibility still need checking. This tiny high-rating pilot is not a representative dataset.'}
    atomic_json(folder/'pilot-audit.json',report)
    print(json.dumps(report,ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();audit(a.folder)
