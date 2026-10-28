"""CPU-only acceptance of a frozen rollout actor from a terminal checkpoint."""
import argparse
import copy
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .io import read_json,atomic_json,sha256,utcnow
from .server_workspace import owned_root,inside
from .retained_source import resolve,checkpoint_reference
from .checkpoint_refs import read_index
from .tensor_io import load_torch,atomic_torch_save
from .rl_core import load_full_actor,evaluate_decisions,choose_decision
from .rl_collector import Engine,collect_series
from .rl_league import League
from .training_control import TrainingControl,ControlledTrainingStop
from .rl_resume import resume_journal


def cpu_rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}


def restore_cpu(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])


def next_action(actor,features,deck,observation,stream):
    tracker=features.Tracker(deck);tracker.update(observation)
    action,decision=stream.choose(actor,features,observation,tracker,0)
    if decision is None:raise ValueError('Nonforced real visible observation required')
    return {'action':action,'logp':decision['old_logp'],'value':decision['old_value']}


def run(ticket_path,fresh=False):
    from .rl_sampling import cpu_replica,CPUActorRNG
    ticket=read_json(ticket_path);root=owned_root(ticket['project_root'],ticket['project_id']);job=inside(root,ticket['job_directory'])
    if os.environ.get('PTCG_JOB_RUN_ID')!=job.name or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.device_count()!=0:
        raise ValueError('CPU sampler acceptance must hide every GPU')
    torch.set_num_threads(2);deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    manifest=read_json(job/'training-manifest.raw.json')
    if manifest['mode']!='cpu_sampler_check' or not manifest.get('no_training_updates'):
        raise ValueError('CPU acceptance cannot perform PPO')
    control=TrainingControl(job,job.name,ticket['lease_epoch'],deadline,
        lambda stage,**kw:atomic_json(job/'progress.json',{'at':utcnow(),'stage':stage,**kw}))
    def stage(name,**kw):
        control.wait(30,gpu_required=False)
        atomic_json(job/'progress.json',{'at':utcnow(),'stage':name,**kw})
    bundle=job/'control'/manifest['training_config']['bundle_path']
    actor,features,_=load_full_actor(bundle,manifest['training_config']['parent_model_sha256'],'cpu')
    deck=list(map(int,(bundle/'deck.csv').read_text(encoding='utf-8-sig').split()))
    if fresh:
        reference=load_torch(job/'CPU-sampling-reference.pt',map_location='cpu',weights_only=False)
        actor.load_state_dict(reference['model'],strict=True);actor=cpu_replica(actor)
        journal=resume_journal(job,{'version':reference['source_version'],'trained_decisions':0,'next_series_id':0})
        if not torch.equal(journal['cpu_sampler_rng_after'],reference['cpu_sampler_rng']):raise ValueError('CPU stream journal replay differs')
        restore_cpu(reference['probe_rng'])
        torch.manual_seed(760777);torch.rand(19)  # Stand in for unrelated opponent initialization.
        global_before=torch.get_rng_state().clone();stream=CPUActorRNG(journal['cpu_sampler_rng_after'])
        actual=next_action(actor,features,deck,reference['observation'],stream)
        if actual!=reference['expected_next_action']:raise ValueError('Fresh CPU process next action/prob/value differs')
        if not torch.equal(torch.get_rng_state(),global_before):raise ValueError('CPU learner perturbed opponent/global RNG')
        with torch.no_grad():logp,values,_=evaluate_decisions(actor,reference['fixture'],features.b.STOP)
        if not torch.equal(logp.cpu(),reference['fixture_logp']) or not torch.equal(values.cpu(),reference['fixture_values']):
            raise ValueError('Fresh CPU conditional outputs differ')
        if torch.cuda.is_initialized():raise ValueError('CPU recovery initialized CUDA')
        from .server_io_check import same
        source,_=resolve(root,ticket['project_id'],manifest['existing_restore_source'])
        point=checkpoint_reference(source,manifest['existing_restore_source'])
        original=load_torch(source/point['checkpoint']['path'],map_location='cpu',weights_only=False)
        if not same(original['model'],reference['model']) or not same(original['optimizer'],reference['optimizer']) or not same(original['rng'],reference['source_rng_payload_unchanged']):
            raise ValueError('Full source model/optimizer/RNG payload did not survive CPU check')
        atomic_json(job/'CPU-sampling-fresh.json',{'at':utcnow(),'passed':True,'complete':True,
            'model_tensors':len(reference['model']),'optimizer_states_preserved':len(reference['optimizer']['state']),
            'same_CPU_next_action_logp_value_and_fixture_exact':True,'independent_of_opponent_global_Torch_RNG':True,
            'production_CPU_stream_journal_replay_exact':True,'full_source_model_optimizer_and_all_RNG_payloads_exact':True,
            'CUDA_initialized':False,'PPO_steps':0})
        return
    stage('reading_terminal_full_checkpoint_and_tail')
    source,original=resolve(root,ticket['project_id'],manifest['existing_restore_source'])
    if manifest.get('source_postcommit_recovery'):
        from .scientific_resume import validate_postcommit_recovery
        validate_postcommit_recovery(source,manifest)
    if original.get('training_config')!=manifest['training_config']:raise ValueError('CPU acceptance learning recipe differs from source')
    point=checkpoint_reference(source,manifest['existing_restore_source'])
    before_index=sha256(source/'checkpoint-commits/index.json')
    checked=read_index(source,source.name,manifest['existing_restore_source']['lease_epoch'])
    if checked['commits'][-1]!=point:raise ValueError('Published source point changed')
    saved=load_torch(source/point['checkpoint']['path'],map_location='cpu',weights_only=False)
    expected_optimizer=97 if saved['version'] else 0
    if len(saved['model'])!=127 or len(saved['optimizer']['state'])!=expected_optimizer or set(saved['rng'])!={'python','numpy','torch','cuda'}:
        raise ValueError('Incomplete real model/optimizer/RNG payload')
    if any(not torch.isfinite(v).all() for v in saved['model'].values()):raise ValueError('Nonfinite source model')
    actor.load_state_dict(saved['model'],strict=True);restore_cpu(saved['rng']);actor=cpu_replica(actor)
    fixture=[]
    for row in point['pending']:
        if not row['path'].startswith('rollouts/') or not row['path'].endswith('.pt'):continue
        data=load_torch(source/row['path'],map_location='cpu',weights_only=False)
        if data['policy_version']!=saved['version']:raise ValueError('Off-policy acceptance fixture')
        fixture.extend(data['decisions'][:48-len(fixture)])
        if len(fixture)==48:break
    lo,hi=features.b.OFFSETS['oracle']
    if any(step['rows'][:,lo:hi].any() for d in fixture for step in d['steps']):raise ValueError('Oracle in CPU fixture')
    gpu_fixture=len(fixture)==48
    logp_error=value_error=None
    if gpu_fixture:
        with torch.no_grad():logp,values,_=evaluate_decisions(actor,fixture,features.b.STOP)
        logp_error=float(np.max(np.abs(logp.numpy()-np.asarray([d['old_logp'] for d in fixture]))))
        value_error=float(np.max(np.abs(values.numpy()-np.asarray([d['old_value'] for d in fixture]))))
        if logp_error>1e-4 or value_error>2e-4 or not torch.isfinite(logp).all() or not torch.isfinite(values).all():
            raise ValueError('Original CPU/GPU conditional probability/value tolerance exceeded')
    # This job has no GPU. The real CPU workload must progress despite a GPU
    # availability pause; manual stop/pause tests use separate owned fixtures.
    resource=job/'resource-training-control.json'
    if resource.exists():raise ValueError('Do not overwrite a preexisting resource command')
    atomic_json(resource,{'run_id':job.name,'lease_epoch':ticket['lease_epoch'],'sequence':0,'command':'pause',
        'reason':'Declared CPU-only acceptance: simulate unavailableGPU while CUDA is hidden'})
    stage('actual_CPU_complete_BO3_under_GPU_availability_pause')
    league_path=source/'league.json'
    if sha256(league_path)!=manifest['source_league_sha256']:raise ValueError('Original frozen history catalog differs')
    catalog=read_json(league_path);league_config=copy.deepcopy(manifest['training_config']['league'])
    for fixed in league_config['fixed_opponents']:
        fixed['files']={k.replace('\\','/'):v for k,v in fixed['files'].items()}
    league_config['retained_history']={'root':str(source),'members':{e['path']:e['sha256'] for e in catalog['snapshots']}}
    atomic_json(job/'league.json',catalog)
    league=League(job/'control',job,league_config,actor,features,deck,opponent_template=actor)
    league.freeze(saved['version']);engine=Engine(root,job/'control',job/'engine.log',sha256(job/'engine-manifest.json'))
    counts=[];first_observation=None;stream=CPUActorRNG(saved.get('cpu_sampler_rng',saved['rng']['torch']))
    try:
        for sid in range(2):
            opponent,assignment=league.select(1-sid,760100+sid)
            trajectories,summary=collect_series(engine,actor,features,deck,opponent,sid,sid,str(saved['version']),
                control=lambda:control.wait(45,gpu_required=False),actor_chooser=stream.choose)
            if summary['statuses']!=['DONE','DONE'] or summary['rounds'] not in (2,3):raise ValueError('CPU BO3 incomplete')
            decisions=[d for trajectory in trajectories for d in trajectory['decisions']]
            if not gpu_fixture:fixture.extend(decisions[:48-len(fixture)])
            if any(step['rows'][:,lo:hi].any() for d in decisions for step in d['steps']):raise ValueError('CPU actor oracle leakage')
            if not all(math.isfinite(d['old_logp']) and math.isfinite(d['old_value']) for d in decisions):raise ValueError('Nonfinite CPU trajectory')
            summary.update(assignment=assignment,round_rewards=[t['reward'] for t in trajectories],learner_model_sha256=league.entries[-1]['sha256'])
            shard=f"rollouts/v{saved['version']:04d}-s{sid:06d}.pt"
            atomic_torch_save(job/shard,{'decisions':decisions,'summary':summary,'series_id':sid,'policy_version':saved['version'],
                'rng_after':{**cpu_rng(),'cuda':[]},'cpu_sampler_rng_after':stream.state()})
            atomic_json(job/'series'/f'{sid:06d}.json',{**summary,'trajectory_path':shard,'trajectory_sha256':sha256(job/shard)})
            if first_observation is None:first_observation=summary['sample_observations'][0]
            counts.append({'learner_seat':sid,'decisions':len(decisions),'rounds':summary['rounds'],'assignment':assignment})
    finally:engine.close()
    if len(fixture)!=48:raise ValueError('At least48 real current-policy decisions required')
    if not gpu_fixture:
        with torch.no_grad():logp,values,_=evaluate_decisions(actor,fixture,features.b.STOP)
    stage('actual_CPU_RNG_and_manual_stop_checks')
    negative=inside(root,job/'negative-stop-probe');negative.mkdir()
    nc=TrainingControl(negative,negative.name,ticket['lease_epoch'],time.time()+10,lambda *a,**k:None)
    atomic_json(negative/'manual-training-control.json',{'run_id':negative.name,'lease_epoch':ticket['lease_epoch'],
        'sequence':0,'command':'stop','reason':'Declared isolated negative acceptance fixture'})
    rejected=False
    try:nc.wait(gpu_required=False)
    except ControlledTrainingStop:rejected=True
    if not rejected:raise ValueError('CPU collection bypassed a manual stop')
    journal=resume_journal(job,{'version':saved['version'],'trained_decisions':0,'next_series_id':0})
    if len(journal['decisions'])!=sum(r['decisions'] for r in counts) or journal['next_series_id']!=2 or not torch.equal(journal['cpu_sampler_rng_after'],stream.state()):
        raise ValueError('Production CPU trajectory/RNG journal did not reconstruct both BO3')
    state=cpu_rng();stream_before=stream.state();expected=next_action(actor,features,deck,first_observation,stream);restore_cpu(state)
    atomic_torch_save(job/'CPU-sampling-reference.pt',{'model':actor.state_dict(),'optimizer':saved['optimizer'],
        'source_rng_payload_unchanged':saved['rng'],'probe_rng':state,'expected_next_action':expected,
        'cpu_sampler_rng':stream_before,'source_version':saved['version'],
        'observation':first_observation,'fixture':fixture,'fixture_logp':logp,'fixture_values':values})
    # Preserve the actual full fixture and both complete BO3 before these
    # disposable engineering replicas can ever be considered for retirement.
    from .archive_outbox import ArchiveOutbox
    outbox=ArchiveOutbox(job,job.name,ticket['lease_epoch'])
    for relative in ['CPU-sampling-reference.pt',*[f"rollouts/v{saved['version']:04d}-s{i:06d}.pt" for i in range(2)],
                     *[f'series/{i:06d}.json' for i in range(2)]]:
        outbox.queue(relative)
    if torch.cuda.is_initialized():raise ValueError('CPU sampler initialized CUDA')
    if sha256(source/'checkpoint-commits/index.json')!=before_index or sha256(source/point['checkpoint']['path'])!=point['checkpoint']['sha256']:
        raise ValueError('Read-only original source changed')
    atomic_json(job/'CPU-sampling-audit.json',{'at':utcnow(),'passed':True,'complete':True,'engineering_only':True,
        'source_run_id':source.name,'source_point':point,'full_model_tensors':127,'optimizer_states_preserved':expected_optimizer,
        'CPU_GPU_old_logp_checks':48 if gpu_fixture else 0,'CPU_GPU_logp_tolerance':1e-4,'CPU_GPU_value_tolerance':2e-4,
        'CPU_GPU_logp_max_error':logp_error,'CPU_GPU_value_max_error':value_error,'complete_CPU_BO3':counts,
        'CPU_GPU_parity_required_in_GPU_bootstrap_before_any_optimizer_step':not gpu_fixture,
        'full_current_policy_CPU_fixture_ready':True,'fixture_reference_sha256':sha256(job/'CPU-sampling-reference.pt'),
        'CPU_collection_runs_with_GPU_availability_pause':True,'manual_stop_rejected_in_separate_owned_case':True,
        'dedicated_CPU_sampling_stream_and_production_journal_verified':True,
        'CUDA_initialized':False,'GPU_visible_devices':0,'PPO_updates':0,'source_unchanged':True,
        'no_cross_backend_bit_exact_action_sequence_claim':True,'promotion_eligible':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);p.add_argument('--fresh',action='store_true')
    a=p.parse_args();run(a.ticket,a.fresh)
