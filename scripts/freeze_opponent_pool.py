"""Freeze public development policies and untouched, prespecified deck variants.

Run once before further tuning; no outcomes are consulted in selecting mutations.
Final variants are prohibited in development or teacher configurations.
"""
import copy
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import atomic_json,read_json,sha256,utcnow

SPECS=[
    ('lucario','fighting_mega','cggonzalezsc_lucario',[(6,1152,'energy-to-search'),(1197,1192,'disruption-to-draw')]),
    ('alakazam','psychic_counters','woominyo_alakazam',[(1081,1086,'removal-to-basic-setup'),(1428,1079,'supporter-to-evolution')]),
    ('archaludon','metal_tank','masami_archaludon',[(1244,8,'stadium-to-energy'),(1147,57,'healing-to-backup-relicanth')]),
    ('ppo','grass_acceleration','dipam_unown_gradiant_ppo',[(1,1231,'energy-to-evolution-supporter'),(1261,1097,'stadium-to-recovery')]),
]


def identity(folder):
    return {str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def valid(deck,cards):
    assert len(deck)==60 and all(c in cards for c in deck)
    names=Counter(cards[c]['name'] for c in deck if cards[c]['cardType']!=5)
    assert max(names.values())<=4
    assert sum(cards[c]['aceSpec'] for c in deck)<=1
    assert any(cards[c]['cardType']==0 and cards[c]['basic'] for c in deck)


def main():
    out=ROOT/'research/opponents'
    if (out/'final_holdout_v1.json').exists(): raise RuntimeError('Holdout already frozen; do not overwrite')
    cards={c['cardId']:c for c in read_json(ROOT/'ptcg_lab/public_cg/metadata.json')['cards']}
    development=[];holdout=[]
    for name,archetype,lineage,changes in SPECS:
        source=ROOT/'agents/public'/name
        deck=list(map(int,(source/'deck.csv').read_text(encoding='utf-8-sig').split()))
        common={'kind':'public_rule','lineage':lineage,'archetype':archetype}
        development.append({**common,'id':name,'path':str(source.relative_to(ROOT)).replace('\\','/'),
                            'weight':.25,'files':identity(source)})
        for number,(remove,add,rationale) in enumerate(changes,1):
            changed=deck.copy();changed.remove(remove);changed.append(add);valid(changed,cards)
            slug=f'{name}-holdout-{number}'
            folder=ROOT/'agents/holdout'/slug
            shutil.copytree(source,folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            (folder/'deck.csv').write_text('\n'.join(map(str,changed))+'\n',encoding='utf-8')
            atomic_json(folder/'variant.json',{'base':name,'remove':remove,'add':add,'rationale':rationale,
                'selection_basis':'prespecified resource trade-off, no matchup results consulted',
                'at':utcnow(),'parent_deck_sha256':sha256(source/'deck.csv')})
            holdout.append({**common,'id':slug,'path':str(folder.relative_to(ROOT)).replace('\\','/'),
                'weight':.125,'mutation':{'remove':remove,'add':add,'rationale':rationale},
                'files':identity(folder),'validation':'static deck legality only; no BO3 outcomes observed'})
    atomic_json(out/'development_v1.json',{'version':1,'role':'development','created_at':utcnow(),
        'policy_weighting':'four archetypes equally weighted','opponents':development})
    atomic_json(out/'final_holdout_v1.json',{'version':1,'role':'final_holdout','created_at':utcnow(),
        'status':'frozen_untouched','policy_weighting':'four archetypes equally weighted; two variants each',
        'independent_public_lineages':4,'archetypes':4,'opponent_count':8,
        'limitations':'Deck variants share their parent policy; this tests modest distribution shift, not eight independent algorithms.',
        'access_rule':'Only protocol/deck checks without outcome-driven selection, then final candidate and release_anchor evaluation. Never teacher generation or tuning.',
        'opponents':holdout})
    print(json.dumps({'development':len(development),'holdout':len(holdout),'independent_lineages':4,'holdout_games_played':0}))


if __name__=='__main__':main()
