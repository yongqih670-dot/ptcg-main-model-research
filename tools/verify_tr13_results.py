"""Offline TR13 BO3 statistics, no model, network, games or training."""
from pathlib import Path
from collections import Counter
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ptcg_lab.volume_statistics import bootstrap_cells
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
cfg=read('configs/TR13/statistics.json');result=read('research/autonomy/TR13-result.json')
def rows(name,n):
 r=[json.loads(s) for s in (ROOT/f'evidence/TR13/{name}-outcomes.jsonl').read_text().splitlines()]
 assert len(r)==n and len({x['series_id'] for x in r})==n
 assert all(x['statuses']==['DONE','DONE'] and x['policy_invalid_count']==0 and x['internal_random_fallback_count']==0 and x['result'] in ['win','draw','loss'] for x in r)
 return r
def scores(r):return [{'win':1.,'draw':.5,'loss':0.}[x['result']] for x in r]
groups={}
for label,name in [('parent','parent-primary'),('endpoint','primary')]:
 r=rows(name,1600);assert Counter(x['candidate_seat'] for x in r)=={0:800,1:800};groups[label]=scores(r)
means,samples=bootstrap_cells(groups,{k:1600 for k in groups},cfg['bootstrap_seed'])
assert np.allclose(np.quantile(samples['endpoint']-samples['parent'],[.025,.975]),result['primary']['ci95'],rtol=0,atol=1e-12)
assert abs(means['endpoint']-means['parent']-result['primary']['gain'])<1e-12
panel={};data={'parent':rows('parent-development',800),'endpoint':rows('development',800)}
for cell in cfg['opponent_cells']:
 for label in ['parent','endpoint']:
  r=[x for x in data[label] if x['cell_id']==cell];assert len(r)==100 and Counter(x['candidate_seat'] for x in r)=={0:50,1:50};panel[label+'|'+cell]=scores(r)
means,samples=bootstrap_cells(panel,{k:100 for k in panel},cfg['bootstrap_seed'])
assert means==result['development']['cell_scores']
delta=sum((samples['endpoint|'+c]-samples['parent|'+c])/8 for c in cfg['opponent_cells'])
assert np.allclose(np.quantile(delta,[.025,.975]),result['development']['ci95'],rtol=0,atol=1e-12)
rows('reliability',50)
print(json.dumps({'verified_BO3':4850,'new_endpoint_BO3':2450,'reused_parent_BO3':2400,'fixed_bootstrap_samples':20000,'reported_statistics_match':True,'training_or_game_execution':False}))
