"""Prespecified opponent-stratified, unpaired BO3 bootstrap (not engine RNG pairing)."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptcg_lab.io import read_json, read_series, atomic_json
from ptcg_lab.evaluate import summarize


def bootstrap_contrast(groups, contrasts):
    rng=np.random.default_rng(20261023)
    differences=np.zeros(20000)
    point=0.0
    if abs(sum(c['weight'] for c in contrasts)-1)>=1e-12:
        raise ValueError('Frozen contrast weights must sum to one')
    for contrast in contrasts:
        arrays=[]
        for side in ('candidate_cell','anchor_cell'):
            values=groups[contrast[side]]
            if not values:raise ValueError('Missing complete BO3 comparison cell')
            arrays.append(np.array([{'win':1,'draw':.5,'loss':0,'error':0}[r['result']]
                                    for r in values]))
        a,b=arrays
        differences+=contrast['weight']*(rng.choice(a,(20000,len(a))).mean(1)-rng.choice(b,(20000,len(b))).mean(1))
        point+=contrast['weight']*(a.mean()-b.mean())
    return {'gain':float(point),'ci95':np.quantile(differences,[.025,.975]).tolist(),
            'bootstrap_samples':20000,'method':'opponent-stratified independent complete BO3 resampling',
            'weights':contrasts}


def classification(contrast):
    return ('improved' if contrast['ci95'][0]>0 else
            'regressed' if contrast['ci95'][1]<0 else 'inconclusive')


def panel_conclusions(groups, contrasts, original_primary_contrasts=None):
    full=bootstrap_contrast(groups,contrasts)
    report={'contrast':full,'classification':classification(full)}
    if original_primary_contrasts:
        original=bootstrap_contrast(groups,original_primary_contrasts)
        report['original_primary_contrast']=original
        report['original_primary_classification']=classification(original)
        report['both_preregistered_panels_positive']=all(c['ci95'][0]>0 for c in [full,original])
        if 'regressed' in [classification(full),classification(original)]:
            report['classification']='regressed'
        elif not report['both_preregistered_panels_positive']:
            report['classification']='inconclusive'
    return report


def analyze(run):
    config = read_json(run/'config.json')
    rows = read_series(run/'series.jsonl')
    groups = defaultdict(list)
    for row in rows: groups[row.get('cell_id','default')].append(row)
    report = {'run_id':run.name, 'complete':len(rows)==config['series_count'],
              'cells':{name:summarize(values) for name,values in groups.items()},
              'promotion_eligible':False, 'reason':'screening only; no independent confirmation'}
    contrasts=config.get('contrasts',[])
    if contrasts and report['complete']:
        report.update(panel_conclusions(groups,contrasts,config.get('original_primary_contrasts')))
    if any(r['result']=='error' or r['policy_invalid_count'] for r in rows):
        report['classification']='invalid'
    atomic_json(run/'panel-analysis.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    print(json.dumps(analyze(args.run)))
