"""Exact preregistered unpaired complete-BO3 learning diagnosis; never promote."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_json,read_series,atomic_json,sha256,utcnow
from ptcg_lab.evaluate import summarize
from ptcg_lab.continuous_diagnostic_stats import contrast

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=ROOT/'runs'/a.run
assert run.resolve().parent==ROOT/'runs'
c=read_json(run/'config.json');rows=read_series(run/'series.jsonl');m=read_json(run/'metrics.json')
assert c['evaluation_stage']=='basic_learning_diagnostic' and c['hypothesis_id']=='TR11'
assert m['complete'] and c['series_count']==800 and len(rows)==800
assert {r['series_id'] for r in rows}==set(range(800))
groups={name:sorted([r for r in rows if r['cell_id']==name],key=lambda r:r['series_id']) for name in ['parent-vs-ppo','endpoint-vs-ppo']}
assert all(len(g)==400 and Counter(r['candidate_seat'] for r in g)=={0:200,1:200} for g in groups.values())
bad=[r for r in rows if r['statuses']!=['DONE','DONE'] or r['result'] not in ['win','draw','loss']
    or r['policy_invalid_count'] or r.get('internal_random_fallback_count',0)]
assert c['policy_seed']==231101 and c['bootstrap_seed']==231102 and c['bootstrap_samples']==20000
scores={k:np.array([{'win':1.,'draw':.5,'loss':0.,'error':0.}[r['result']] for r in g]) for k,g in groups.items()}
parent=scores['parent-vs-ppo'];endpoint=scores['endpoint-vs-ppo']
result=contrast(parent,endpoint);gain=result['gain'];ci=result['ci95']
learning=not bad and result['learning_signal_passed']
classification='invalid' if bad else 'regressed' if ci[1]<0 else 'inconclusive'
report={'at':utcnow(),'run_id':run.name,'complete':True,'classification':classification,
    'learning_signal_passed':learning,'contrast':{'gain':gain,'ci95':ci,'bootstrap_seed':231102,
        'bootstrap_samples':20000,'method':'independent completeBO3 resampling on the single frozen legal training opponent'},
    'cells':{k:summarize(v) for k,v in groups.items()},'parent_score':float(parent.mean()),
    'endpoint_score':float(endpoint.mean()),'preregistered_learning_gate':'gain>=0.05 and CI95lower>0',
    'invalid_series':len(bad),'promotion_eligible':False,'final_or_general_strength_claim':False,
    'native_environment_seed_not_exposed':True,'no_additional_samples_or_model_selection':True,
    'reason':'Single fixed-opponent learnability diagnosis; no independent multi-opponent promotion confirmation.',
    'source_series_sha256':sha256(run/'series.jsonl'),'config_sha256':sha256(run/'config.json')}
atomic_json(run/'panel-analysis.json',report);atomic_json(run/'basic-learning-analysis.json',report)
print({'classification':classification,'learning_signal_passed':learning,'gain':gain,'ci95':ci,
    'parent_score':float(parent.mean()),'endpoint_score':float(endpoint.mean()),'promotion_eligible':False})
