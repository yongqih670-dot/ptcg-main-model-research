"""Offline checks and fixed BO3 bootstrap only: no network, model or training."""
from collections import Counter
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ptcg_lab.volume_statistics import primary,secondary
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
groups={};total=0
for role,count in [('smoke',50),('primary',3200),('development',2400)]:
    config=read(f'configs/TR12/early-stop-{role}.json')
    assert config['reference_only'] and not config['execution_authorized']
    rows=[json.loads(line) for line in (ROOT/f'evidence/TR12/{role}-outcomes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==count and {r['series_id'] for r in rows}==set(range(count))
    assert set().union(*(set(r) for r in rows))=={'series_id','cell_id','candidate_seat','result','statuses','policy_invalid_count','internal_random_fallback_count'}
    part={}
    for cell in config['matchups']:
        selected=sorted([r for r in rows if r['cell_id']==cell['cell_id']],key=lambda r:r['series_id'])
        n=cell['series_count'];assert len(selected)==n and Counter(r['candidate_seat'] for r in selected)=={0:n//2,1:n//2}
        assert all(r['statuses']==['DONE','DONE'] and r['result'] in ['win','draw','loss'] and r['policy_invalid_count']==0 and r['internal_random_fallback_count']==0 for r in selected)
        part[cell['cell_id']]=[{'win':1.,'draw':.5,'loss':0.}[r['result']] for r in selected]
    groups[role]=part;total+=len(rows)
assert total==5650
assert read('configs/TR12/early-stop-primary.json')['bootstrap_seed']==2026102801
dev=read('configs/TR12/early-stop-development.json');assert dev['bootstrap_seed']==2026102802
opponents=[c['reference']['id'] for c in dev['matchups'][::3]]
actual={'primary':primary(groups['primary']),'secondary':secondary(groups['development'],opponents)}
expected=read('research/autonomy/TR12-result.json')
def equal(a,b):
    if isinstance(a,dict):assert isinstance(b,dict) and set(a)==set(b);[equal(v,b[k]) for k,v in a.items()]
    elif isinstance(a,list):assert isinstance(b,list) and len(a)==len(b);[equal(x,y) for x,y in zip(a,b)]
    elif isinstance(a,bool):assert type(b) is bool and a==b
    elif isinstance(a,(int,float)):assert math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)
    else:assert a==b,(a,b)
for key in actual:equal(actual[key],expected[key])
assert not expected['primary']['learning_signal_passed'] and not expected['original_training_target_complete'] and not expected['promotion_eligible']
print(json.dumps({'verified_BO3':total,'bootstrap_samples':20000,'primary':actual['primary'],
    'all_reported_development_statistics_match':True,'network_model_training_or_game_execution':False},ensure_ascii=False))
