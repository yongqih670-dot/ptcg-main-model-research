"""Fresh newest-state GPU restore plus one official BO3 timing-window regression."""
import argparse
import os
from pathlib import Path
import time
import torch
from .io import read_json,atomic_json,sha256,utcnow
from .server_workspace import owned_root,inside
from .retained_source import resolve
from .tr13_source import read_full
from .tr13_acceptance_reuse import check_state,scientific_sources
from .rl_core import load_full_actor,optimizer_for,restore_rng,rng_state
from .framework_v2_training import semantic_digest
from .ppo_accounting import checkpoint_decisions
from .training_control import TrainingControl
from .rl_sampling import cpu_replica,CPUActorRNG
from .rl_league import League
from .rl_collector import Engine,collect_series
from .tensor_io import atomic_torch_save


def main():
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);a=p.parse_args()
    ticket=read_json(a.ticket);root=owned_root(ticket['project_root'],ticket['project_id']);job=inside(root,ticket['job_directory'])
    manifest=read_json(job/'training-manifest.raw.json');test=manifest.get('TR13_timeout_acceptance')
    if not test or not manifest.get('engineering_only') or ticket.get('new_diagnostic_identity')!='TR13':
        raise ValueError('Only the registered failed-source timing/recovery difference is allowed')
    package=Path(__file__).resolve().parent.parent;difference=test['difference_document']
    if scientific_sources(package)!=difference['new_scientific_sources']:raise ValueError('Unreviewed training difference')
    for name,digest in difference['new_runtime_sources'].items():
        if sha256(package/name)!=digest:raise ValueError('Unreviewed timing/queue implementation')
    source,previous=resolve(root,ticket['project_id'],manifest['existing_restore_source'])
    config=read_json(source/'payload-config.json');spec=manifest['source_v2']
    atomic_json(job/'progress.json',{'at':utcnow(),'stage':'verifying_latest_full_v17_state'})
    value,point,_=read_full(source,source.name,manifest['existing_restore_source']['lease_epoch'],spec['index_sha256'])
    if point!=spec['point'] or source.name!=difference['source_run_id']:raise ValueError('Latest positive source point differs')
    check_state(value,test['basis_acceptance']);actual=checkpoint_decisions(value)
    control=TrainingControl(job,job.name,ticket['lease_epoch'],float(os.environ['PTCG_JOB_DEADLINE_UNIX']),
        lambda stage,**kw:atomic_json(job/'progress.json',{'at':utcnow(),'stage':stage,**kw}))
    torch.set_num_threads(2);control.wait(120);torch.backends.cuda.matmul.allow_tf32=False
    model,features,_=load_full_actor(job/'control'/config['bundle_path'],config['parent_model_sha256'],'cuda')
    optimizer=optimizer_for(model,config['ppo']['actor_learning_rate'],config['ppo']['value_head_learning_rate'])
    model.load_state_dict(value['model']);optimizer.load_state_dict(value['optimizer']);restore_rng(value['rng'])
    for found,expected in [(model.state_dict(),value['model']),(optimizer.state_dict(),value['optimizer']),(rng_state(),value['rng'])]:
        if semantic_digest(found)!=semantic_digest(expected):raise ValueError('Latest CUDA model/optimizer/allRNG restore is not exact')
    restored={'generation':point['generation'],'cursor':point['cursor'],'actual_unique':actual,
        'model_state_revision':value['model_state_revision'],'full_CPU_and_GPU_values_RNG_exact':True,
        'semantic_sha256':value['state_semantic_sha256']}
    atomic_json(job/'latest-GPU-state-recovery.json',{'at':utcnow(),'complete':True,'fresh_process':True,'latest_only':True,
        'source_run_id':source.name,'source_v2_index_sha256':spec['index_sha256'],'checked_points':[restored],
        'optimizer_steps_executed':0,'scientific_decision_credit':0})
    sampler=cpu_replica(model);stream=CPUActorRNG(value['cpu_sampler_rng']);scratch=job/'timing-BO3';scratch.mkdir()
    deck=list(map(int,(job/'control'/config['bundle_path']/'deck.csv').read_text(encoding='utf-8-sig').split()))
    league_config=dict(config['league']);league_config.pop('retained_history',None)
    league=League(job/'control',scratch,league_config,model,features,deck,opponent_template=sampler)
    league.freeze(value['version']);seat=1;opponent,assignment=league.select(0,config['seed']+point['cursor']['next_series_id'])
    if assignment.get('opponent_id')!='ppo' or not assignment.get('fixed'):raise ValueError('Timing regression must retain the actual fixed PPO opponent')
    engine=Engine(root,job/'control',job/'timing-engine.log',sha256(job/'engine-manifest.json'))
    windows=0;inserted=False
    def maintenance():
        nonlocal windows,inserted
        control.wait(120,gpu_required=False);windows+=1
        if windows==3:
            atomic_json(job/'progress.json',{'at':utcnow(),'stage':'official_BO3_maintenance_boundary_regression','actual_unique':actual})
            time.sleep(2.5);inserted=True
    try:
        trajectories,summary=collect_series(engine,sampler,features,deck,opponent,seat,point['cursor']['next_series_id'],str(value['version']),
            control=maintenance,actor_chooser=stream.choose)
    finally:engine.close()
    diagnostic=summary['timing_diagnostics']
    if summary['statuses']!=['DONE','DONE'] or not inserted or diagnostic['maintenance_seconds']<2.5:
        raise ValueError('Official BO3/maintenance regression did not complete')
    if diagnostic['configuration']!={'actTimeout':0,'runTimeout':6000,'bo':3,'version':2}:
        raise ValueError('Official scientific time rules changed')
    atomic_torch_save(job/'timing-BO3-original.pt',{'trajectories':trajectories,'summary':summary,'scientific_decision_credit':0},replace=False)
    atomic_json(job/'TR13-timeout-recovery-acceptance.json',{'at':utcnow(),'complete':True,'engineering_only':True,
        'source_run_id':source.name,'source_v2_index_sha256':spec['index_sha256'],'latest_point':point,
        'latest_GPU_recovery':restored,'actual_unique':actual,'scientific_decision_credit':0,'optimizer_steps_executed':0,
        'official_complete_BO3':1,'original_PPO_opponent':assignment,'timing_diagnostics':diagnostic,
        'official_action_and_total_run_timeouts_unchanged':True,'failure_not_swallowed':True,
        'test_original_bytes_sha256':sha256(job/'timing-BO3-original.pt'),'targeted_difference':test['difference_reference'],
        'no_parent_state_mutation':True})


if __name__=='__main__':main()
