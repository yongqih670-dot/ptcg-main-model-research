"""Freeze reusable v15 BO3 evidence and all endpoint opponent cells before training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_json,read_series,atomic_json,sha256,utcnow

def ref(path):return {'path':path,'sha256':sha256(ROOT/path)}

primary='runs/20261029T041238Z_tr12-early-stop-primary_306458'
development='runs/20261029T051052Z_tr12-early-stop-development_277b56'
pc=read_json(ROOT/primary/'config.json');dc=read_json(ROOT/development/'config.json')
parents=read_series(ROOT/primary/'series.jsonl');panel=read_series(ROOT/development/'series.jsonl')
rows=[r for r in parents if r['cell_id']=='endpoint-vs-ppo']
assert len(rows)==1600 and sum(r['candidate_seat']==0 for r in rows)==800
cells=[c for c in dc['matchups'] if c['cell_id'].startswith('endpoint-vs-')]
assert len(cells)==8
for c in cells:
 subset=[r for r in panel if r['cell_id']==c['cell_id']]
 assert len(subset)==100 and sum(r['candidate_seat']==0 for r in subset)==50
for r in [*rows,*[r for r in panel if r['cell_id'].startswith('endpoint-vs-')]]:
 assert r['statuses']==['DONE','DONE'] and r['result'] in ['win','loss','draw'] and not r['error']
 assert r['policy_invalid_count']==0 and r['internal_random_fallback_count']==0
value={'at':utcnow(),'phase':'TR13','training_started':False,
 'primary_reference':{'BO3':1600,'candidate_seats':[800,800],'cell':'endpoint-vs-ppo',
                      'config':ref(primary+'/config.json'),'rows':ref(primary+'/series.jsonl')},
 'development_reference':{'BO3':800,'config':ref(development+'/config.json'),'rows':ref(development+'/series.jsonl')},
 'endpoint_primary_cell':{**next(c for c in pc['matchups'] if c['cell_id']=='endpoint-vs-ppo'),'candidate':'FIXED_TR13_ENDPOINT_TO_BE_HASH_BOUND_AFTER_VOLUME_COMPLETION'},
 'endpoint_development_cells':[{**c,'candidate':'FIXED_TR13_ENDPOINT_TO_BE_HASH_BOUND_AFTER_VOLUME_COMPLETION'} for c in cells],
 'new_endpoint_reliability_BO3':50,'new_endpoint_BO3_total':2450,'policy_seed':2026103000,
 'bootstrap_samples':20000,'bootstrap_seed':2026103001,
 'primary_gate':{'gain_at_least':.05,'95CI_lower_strictly_positive':True},
 'native_environment_seed_not_exposed':True,'unpaired_complete_BO3_bootstrap':True,
 'no_sample_extension_checkpoint_selection_final_pool_or_automatic_promotion':True}
dest=ROOT/'research/autonomy/TR13_EVALUATION_FROZEN.json'
if dest.exists():raise FileExistsError(dest)
atomic_json(dest,value)
print({'evaluation':ref(dest.relative_to(ROOT).as_posix()),'reused_parent_BO3':2400,'new_endpoint_BO3':2450})
