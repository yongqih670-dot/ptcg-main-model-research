"""Create fixed development smoke/screen configs using the frozen v2 panel."""
import argparse
import copy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_json,atomic_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--training-run',required=True)
    p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    if not all(c.isalnum() or c=='_' for c in a.prefix):raise ValueError('Unsafe prefix')
    spec=read_json(a.spec);source=read_json(ROOT/'configs/evaluation/h003_catboost_screen.json')
    cfg=copy.deepcopy(source);cfg['policy_seed']=a.seed;cfg['training_run_id']=a.training_run
    cfg['primary_change']='Full-token conditional Transformer BC versus frozen PPO anchor; same deck, fixed development_v2 opponents and weights'
    cfg['model_spec_sha256']=sha256(a.spec)
    for cell in cfg['matchups']:
        if cell['cell_id'].startswith('catboost-'):
            cell['cell_id']=cell['cell_id'].replace('catboost-','transformer-',1);cell['candidate']=spec
    for contrast in cfg['contrasts']:contrast['candidate_cell']=contrast['candidate_cell'].replace('catboost-','transformer-',1)
    screen=ROOT/'configs/evaluation'/f'{a.prefix}_screen.json'
    if screen.exists():raise ValueError('Do not overwrite prepared panel')
    atomic_json(screen,cfg)
    smoke=copy.deepcopy(cfg);smoke.pop('contrasts');smoke['matchups']=[c for c in smoke['matchups'] if c['cell_id'].startswith('transformer-')]
    for cell,n in zip(smoke['matchups'],(14,12,12,12)):cell['series_count']=n
    smoke.update(series_count=50,evaluation_stage='engineering_smoke',policy_seed=a.seed-1,
        success_criteria='50 complete official BO3; both seats; no errors/invalid/timeouts/internal random fallback; no strength promotion claim')
    atomic_json(ROOT/'configs/evaluation'/f'{a.prefix}_smoke.json',smoke)
    print(str(screen))


if __name__=='__main__':main()
