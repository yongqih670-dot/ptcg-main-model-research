"""Freeze one completed PPO research run into a reviewed inference bundle."""
import argparse
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_json,atomic_json,sha256,utcnow


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--name',required=True);args=p.parse_args()
    if not all(c.isalnum() or c=='-' for c in args.name):raise ValueError('Unsafe candidate name')
    run=ROOT/'runs'/args.run
    if run.parent!=ROOT/'runs' or not (run/'metrics.json').exists():raise ValueError('Unknown training run')
    metrics=read_json(run/'metrics.json');cfg=read_json(run/'config.json');experiment=read_json(run/'experiment.json')
    hybrid=cfg.get('check_variant') in ('scientific_continue','fresh_scientific','derived_scientific','continuous_mainline','volume_extension') and cfg.get('job_kind')=='remote_train_ppo'
    if cfg.get('check_variant')=='volume_extension' and (cfg.get('hypothesis_id')!='TR12' or metrics.get('unique_PPO_trained_decisions',0)<369941):
        raise ValueError('TR12 export requires its fixed completed actual endpoint')
    if cfg.get('check_variant')=='continuous_mainline' and cfg.get('hypothesis_id')!='TR11':
        raise ValueError('Continuous export is only the authorized TR11 diagnostic')
    if not metrics.get('complete') or (cfg['job_kind']!='train_ppo' and not hybrid) or experiment['kind']!='research':
        raise ValueError('Export for evaluation requires completed research training')
    if hybrid:
        original=(ROOT/cfg['training_config_template']).resolve()
        if ROOT not in original.parents or sha256(original)!=cfg['training_config_template_sha256']:
            raise ValueError('Original scientific training configuration changed')
        if not (metrics.get('scientific_continuation') or metrics.get('scientific_training')) or not (run/'hybrid-materialization.json').exists():
            raise ValueError('Completed hybrid candidate requires durable raw-byte reconstruction proof')
        cfg=read_json(original)
    actual_decisions=metrics.get('unique_PPO_trained_decisions')
    if actual_decisions is not None and (type(actual_decisions) is not int or actual_decisions<cfg['target_decisions']):
        raise ValueError('Actual unique PPO training target is incomplete')
    if sha256(run/'model.pt')!=metrics['model_sha256']:raise ValueError('Training model changed')
    parent=run/'source'/cfg['bundle_path'];dest=ROOT/'agents/learned'/args.name
    if dest.exists():raise ValueError('Never overwrite an existing candidate')
    frozen=read_json(run/'source-manifest.json')
    for name in ('ptcg_lab/rl_core.py','ptcg_lab/token_features.py','ptcg_lab/rl_agent.py','ptcg_lab/replay_data.py'):
        expected=frozen.get(name,frozen.get(name.replace('/','\\')))
        if sha256(ROOT/name)!=expected:raise ValueError('Inference code changed since training: '+name)
    shutil.copytree(parent,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(run/'model.pt',dest/'model.pt')
    spec={'id':args.name,'kind':'transformer_rl','path':str(dest.relative_to(ROOT)).replace('\\','/'),
        'lineage':'dipam_unown_gradiant_ppo','parameters':{'_actor_seat_mode':'zero'},
        'model_sha256':sha256(dest/'model.pt'),'training_run_id':run.name,
        'training_config_sha256':sha256(run/'config.json'),'feature_schema':'full-inherited-blind-conditional-v3'}
    atomic_json(dest/'spec.json',spec)
    atomic_json(dest/'ppo-provenance.json',{'at':utcnow(),'parent_model_sha256':cfg['parent_model_sha256'],
        'training_run_id':run.name,
        'trained_decisions':actual_decisions if actual_decisions is not None else metrics['trained_decisions'],
        'unique_PPO_trained_decisions':actual_decisions,'legacy_consumed_decisions':metrics['trained_decisions'],
        'decision_accounting':metrics.get('decision_accounting','legacy_consumed_count_actual_unique_unspecified'),
        'updates':metrics['updates'],
        'deck_sha256':sha256(dest/'deck.csv'),'model_sha256':spec['model_sha256'],
        'inference_adapter_sha256':sha256(ROOT/'ptcg_lab/rl_agent.py'),
        'qualification':'unqualified candidate; independent BO3 evaluation still required'})
    print(str(dest/'spec.json'))


if __name__=='__main__':main()
