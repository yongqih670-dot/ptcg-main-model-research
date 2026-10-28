"""Bind precommitted evaluation templates to completed PPO output hashes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_json,atomic_json,sha256,utcnow


def config_hash(config):return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def bind_candidate(template,spec,identity):
    config=copy.deepcopy(template)
    allowed=[{'bind':'completed_TR01_model_spec'},{'bind':'completed_'+identity+'_model_spec'}]
    for cell in config['matchups']:
        if cell['candidate'] in allowed:cell['candidate']=copy.deepcopy(spec)
        elif 'bind' in cell['candidate']:raise ValueError('Unrecognized candidate placeholder')
    return config


def smoke_allocation(count):
    if count==4:return [14,12,12,12]
    if count!=8:raise ValueError('Unregistered reliability panel size')
    return [8,6,6,6,6,6,6,6]


def bind_config(state,identity,config,path,reason):
    row=state['research_stage']['hypotheses'][identity];pr=ROOT/row['preregistration_path']
    if sha256(pr)!=row['preregistration_sha256']:raise ValueError('Preregistration changed')
    plan=read_json(pr);old_hash=sha256(pr)
    archive=pr.with_name(pr.stem+'.'+old_hash[:12]+'.json')
    if not archive.exists():shutil.copy2(pr,archive)
    if config_hash(config) in plan['config_hashes']:raise ValueError('Config binding already exists')
    if path.exists():raise ValueError('Never overwrite an evaluation config')
    atomic_json(path,config);plan['config_hashes'].append(config_hash(config))
    plan.setdefault('bindings',[]).append({'at':utcnow(),'previous_preregistration_sha256':old_hash,
        'config_path':path.relative_to(ROOT).as_posix(),'config_sha256':sha256(path),'reason':reason})
    atomic_json(pr,plan);row['preregistration_sha256']=sha256(pr)


def main():
    p=argparse.ArgumentParser();p.add_argument('--hypothesis',required=True);p.add_argument('--spec',type=Path)
    p.add_argument('--confirmation-of');a=p.parse_args()
    state=read_json(ROOT/'research/autonomy/state.json')
    if state.get('active_job') or (ROOT/'research/autonomy/job.lock').exists():raise ValueError('Wait for current job before binding')
    row=state['research_stage']['hypotheses'][a.hypothesis];plan=read_json(ROOT/row['preregistration_path'])
    if row['status']!='started':raise ValueError('Hypothesis is not running')
    template=ROOT/plan['evaluation_template_path']
    if sha256(template)!=plan['evaluation_template_sha256']:raise ValueError('Evaluation template changed')
    prefix=a.hypothesis.lower()
    if a.confirmation_of:
        run=ROOT/'runs'/a.confirmation_of
        if run.name not in row['run_ids']:raise ValueError('Screen outside hypothesis')
        panel=read_json(run/'panel-analysis.json')
        if not panel['complete'] or panel.get('classification')!='improved':raise ValueError('Screen does not justify confirmatory compute')
        cfg=read_json(run/'config.json')
        cfg.update(screening_run_id=run.name,evaluation_stage='independent_confirmation',policy_seed=cfg['policy_seed']+1)
        for cell in cfg['matchups']:cell['series_count']=100
        cfg['series_count']=sum(c['series_count'] for c in cfg['matchups'])
        cfg['success_criteria']='Predeclared independent 100 BO3 per arm/opponent; unchanged policy, deck and weights; all validate_confirmation checks required'
        dest=ROOT/'configs/evaluation'/f'{prefix}_confirmation.json'
        bind_config(state,a.hypothesis,cfg,dest,'Resolve predeclared screening_run_id and independent confirmation sample size; no strategy or weighting changes')
    else:
        spec=read_json(a.spec);training=ROOT/'runs'/spec['training_run_id'];metrics=read_json(training/'metrics.json')
        if training.name not in row['run_ids'] or not metrics['complete'] or metrics['model_sha256']!=spec['model_sha256']:
            raise ValueError('Candidate is not completed hypothesis training output')
        if sha256(ROOT/spec['path']/'model.pt')!=spec['model_sha256']:raise ValueError('Candidate weights changed')
        if plan.get('evaluation_manifest'):
            frozen=ROOT/plan['evaluation_manifest']['path']
            if sha256(frozen)!=plan['evaluation_manifest']['sha256']:raise ValueError('Frozen opponent manifest changed')
            for opponent in read_json(frozen)['opponents']:
                for name,digest in opponent['files'].items():
                    if sha256(ROOT/opponent['path']/name)!=digest:raise ValueError('Frozen development opponent changed')
        cfg=bind_candidate(read_json(template),spec,a.hypothesis)
        cfg.update(training_run_id=training.name,model_spec_sha256=sha256(a.spec))
        dest=ROOT/'configs/evaluation'/f'{prefix}_screen.json'
        bind_config(state,a.hypothesis,cfg,dest,'Resolve completed training artifact only into unchanged precommitted screen template')
        smoke=copy.deepcopy(cfg);smoke.pop('contrasts');smoke['matchups']=[x for x in smoke['matchups'] if x['cell_id'].startswith('candidate-')]
        smoke.pop('original_primary_contrasts',None)
        for cell,n in zip(smoke['matchups'],smoke_allocation(len(smoke['matchups']))):cell['series_count']=n
        smoke.update(series_count=50,evaluation_stage='engineering_smoke',policy_seed=cfg['policy_seed']-1,
            hypothesis_id=a.hypothesis+'_SMOKE',budget_hypothesis_id=a.hypothesis,
            success_criteria='50 complete official BO3; zero errors, invalid actions, timeouts or internal fallback. No strength selection.')
        atomic_json(ROOT/'configs/evaluation'/f'{prefix}_smoke.json',smoke)
        pr=ROOT/'research/autonomy/preregistrations'/f'{a.hypothesis}_SMOKE.json'
        atomic_json(pr,{'hypothesis_id':smoke['hypothesis_id'],'research':False,'final':False,'diagnostic':False,
            'allowed_job_kinds':['evaluation'],'max_raw_runs':1 if a.hypothesis=='TR10' else 3,'config_hashes':[config_hash(smoke)],'at':utcnow(),
            'scope':'Predeclared reliability-only check. Outcome scores cannot select/tune or promote policy.'})
        state['research_stage']['engineering_tasks'][smoke['hypothesis_id']]={'status':'registered','run_ids':[],
            'diagnostic':False,'preregistration_path':pr.relative_to(ROOT).as_posix(),'preregistration_sha256':sha256(pr)}
    state['updated_at']=utcnow();atomic_json(ROOT/'research/autonomy/state.json',state)
    print(dest)


if __name__=='__main__':main()
