"""Actual new-disk restore/simulator/identity/capacity acceptance, zero updates."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from .io import atomic_json,read_json,sha256,utcnow
from .server_workspace import owned_root,inside,admit
from .storage_profile import OLD,NEW,PROJECT

FIRST='20261026T131453Z_tr09-epochs4-CPU32_382e9e'


def equal_nested(a,b,torch,np):
    if torch.is_tensor(a):
        if not torch.equal(a.detach().cpu(),b.detach().cpu()):raise ValueError('Restored tensor differs')
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError('Restored array differs')
    elif isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Restored dictionary keys differ')
        for k in a:equal_nested(a[k],b[k],torch,np)
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise ValueError('Restored sequence differs')
        for x,y in zip(a,b):equal_nested(x,y,torch,np)
    elif a!=b:raise ValueError('Restored scalar differs')


def restore_check(ticket,fresh=False):
    import numpy as np
    import torch
    from .rl_core import load_full_actor,optimizer_for,restore_rng,evaluate_decisions
    from .tensor_io import load_torch,atomic_torch_save
    from .checkpoint_refs import read_index
    from .archive_prefix import load_prefix
    from .archive_state import validate_state
    from .sampling_parity import validate_outputs
    from .rl_sampling import CPUActorRNG
    root=owned_root(NEW,PROJECT);job=inside(root,ticket['job_directory']);source=root/'jobs'/FIRST
    if Path(sys.prefix).resolve()!=root/'.venv':raise ValueError('Relocated environment prefix differs')
    gpu=ticket.get('gpu_restore_required',True);device='cuda' if gpu else 'cpu'
    if gpu:
        if torch.cuda.device_count()!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!=ticket['gpu_uuid']:raise ValueError('Exactly one authorized GPU must be visible')
    elif torch.cuda.device_count()!=0 or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU acceptance must hide CUDA')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=read_json(source/'training-manifest.raw.json');original=read_json(source/'payload-config.json')
    bundle=source/'control'/manifest['training_config']['bundle_path']
    model,features,inheritance=load_full_actor(bundle,manifest['training_config']['parent_model_sha256'],device,reset_value=False)
    optimizer=optimizer_for(model)
    def check_saved(saved):
        if len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97:raise ValueError('Incomplete full training state')
        if set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=1:raise ValueError('Incomplete global RNG')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        equal_nested(model.state_dict(),saved['model'],torch,np);equal_nested(optimizer.state_dict(),saved['optimizer'],torch,np)
        if any(t.dtype!=torch.uint8 or t.ndim!=1 or not t.numel() for t in saved['rng']['cuda']):raise ValueError('CUDA RNG byte payload incomplete')
        applied_rng=saved['rng'] if gpu else {**saved['rng'],'cuda':[]}
        restore_rng(applied_rng)
        expected=(random.random(),np.random.randint(0,1000000,8),torch.rand(8),torch.rand(16,device=device).cpu())
        restore_rng(applied_rng)
        actual=(random.random(),np.random.randint(0,1000000,8),torch.rand(8),torch.rand(16,device=device).cpu())
        equal_nested(expected,actual,torch,np)
        stream=CPUActorRNG(saved['cpu_sampler_rng']);a=torch.rand(16,generator=stream.generator)
        stream=CPUActorRNG(saved['cpu_sampler_rng']);b=torch.rand(16,generator=stream.generator)
        equal_nested(a,b,torch,np)
        return expected
    if fresh:
        proof=load_torch(job/'restored-training-state.pt',map_location='cpu',weights_only=False)
        draws=check_saved(proof['saved']);equal_nested(draws,proof['expected_draws'],torch,np)
        with torch.no_grad():lp,val,_=evaluate_decisions(model,proof['fixture'],features.b.STOP)
        torch.testing.assert_close(lp.cpu(),proof['GPU_logp'],rtol=0,atol=0)
        torch.testing.assert_close(val.cpu(),proof['GPU_values'],rtol=0,atol=0)
        atomic_json(job/'fresh-process-restore.json',{'complete':True,'all127_model_and97_optimizer_exact':True,
            'all_CPU_RNG_streams_reproduced':True,'CUDA_RNG_actually_reproduced':gpu,'same_device_logp_value_bit_exact':True,
            'source_checkpoint_sha256':proof['source_checkpoint_sha256'],'training_updates':0})
        return
    started=time.monotonic();index=read_index(source,FIRST,47);checks=[];saved=None
    for point in index['commits']:
        saved=load_torch(source/point['checkpoint']['path'],map_location='cpu',weights_only=False)
        if {k:saved[k] for k in point['cursor']}!=point['cursor']:raise ValueError('Recovery cursor differs')
        draws=check_saved(saved);checks.append({'generation':point['generation'],'checkpoint':point['checkpoint'],
            'full_model_optimizer_and_CPU_rng_restored':True,'CUDA_RNG_actually_reproduced':gpu,'pending_reference_files':len(point['pending'])})
    point=index['commits'][-1]
    if point['cursor']!={'version':2,'trained_decisions':16679,'next_series_id':130}:raise ValueError('Original TR09 cursor changed')
    if sha256(source/'checkpoint.pt')!=point['checkpoint']['sha256']:raise ValueError('Hot checkpoint differs from published point')
    prefix=load_prefix(source/point['prefix']['path'],point['prefix']['sha256'])
    state=validate_state(saved['archive_state'],prefix,point['prefix']['sha256'])
    pending=[];BO3=0;games=0
    for row in state['live_expected']:
        data=load_torch(source/row['trajectory_path'],map_location='cpu',weights_only=False)
        summary=read_json(source/f"series/{row['series_id']:06d}.json")
        if data['policy_version']!=2 or data['series_id']!=row['series_id'] or summary['rounds'] not in (2,3):raise ValueError('Pending full BO3 identity differs')
        if data.get('cpu_sampler_rng_after') is None:raise ValueError('Pending sampling stream absent')
        CPUActorRNG(data['cpu_sampler_rng_after'])
        pending.extend(data['decisions']);BO3+=1;games+=summary['rounds']
    if len(pending)!=4112 or BO3!=27:raise ValueError('Original complete pending tail differs')
    fixture=copy.deepcopy(pending[:48]);lo,hi=features.b.OFFSETS['oracle'];conditional=0
    for decision in pending:
        picked=set()
        for step in decision['steps']:
            if step['rows'][:,lo:hi].any() or not np.isfinite(step['rows']).all():raise ValueError('Hidden/nonfinite actor feature')
            target=step['target'];n=step['n']
            if type(target) is not int or not 0<=target<=n or target<n and target in picked:raise ValueError('Invalid conditional target')
            if target<n:picked.add(target)
            conditional+=1
    # CPU replica is the complete current actor, never fresh parent weights.
    from .rl_sampling import cpu_replica
    cpu=cpu_replica(model)
    with torch.no_grad():clp,cv,_=evaluate_decisions(cpu,fixture,features.b.STOP);glp,gv,_=evaluate_decisions(model,fixture,features.b.STOP)
    cp=validate_outputs(fixture,clp,cv);gp=validate_outputs(fixture,glp,gv)
    torch.testing.assert_close(glp.cpu(),clp,rtol=0,atol=1e-4);torch.testing.assert_close(gv.cpu(),cv,rtol=0,atol=2e-4)
    clean=torch.from_numpy(fixture[0]['steps'][0]['rows']).to(device);poison=clean.clone();poison[:,lo:hi]=123
    with torch.no_grad():a=model.forward_eval(clean);b=model.forward_eval(poison)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    league=read_json(source/'league.json')
    for member in league['snapshots']:
        if sha256(source/member['path'])!=member['sha256']:raise ValueError('Frozen history SHA differs')
    if sum(u['decisions'] for u in saved['updates'])!=16679 or len(saved['updates'])!=2:raise ValueError('PPO ledger changed')
    # Fresh-process equality uses a separately serialized engineering proof only.
    draws=check_saved(saved)
    atomic_torch_save(job/'restored-training-state.pt',{'saved':saved,'expected_draws':draws,'fixture':fixture,
        'GPU_logp':glp.detach().cpu(),'GPU_values':gv.detach().cpu(),'source_checkpoint_sha256':point['checkpoint']['sha256']})
    atomic_json(job/'full-restore.json',{'complete':True,'model_tensors':127,'optimizer_states':97,
        'two_actual_recovery_points':checks,'trained_decisions':16679,'PPO_updates':2,
        'pending_decisions':len(pending),'pending_complete_BO3':BO3,'pending_individual_games':games,
        'all_CPU_RNG_streams_reproduced':True,'CUDA_RNG_actually_reproduced':gpu,'CUDA_RNG_full_original_bytes_preserved':True,
        'actor_oracle_poison_exact':True,'conditional_steps_checked':conditional,
        'CPU_rollout_parity':cp,'restored_device_rollout_parity':gp,'device':device,
        'CPU_device_logp_max_abs':float((clp-glp.cpu()).abs().max()),'CPU_device_value_max_abs':float((cv-gv.cpu()).abs().max()),'frozen_historical_members':len(league['snapshots']),
        'parent_actor_inheritance':inheritance,'seconds':time.monotonic()-started,'new_PPO_updates':0,
        'source_scientific_data_modified':False,'source_index_sha256':sha256(source/'checkpoint-commits/index.json')})
    if not gpu and ticket.get('cpu_throughput_probe'):
        from .rl_collector import Engine,collect_series
        from .rl_league import FrozenOpponent
        member=league['snapshots'][-2] if len(league['snapshots'])>1 else league['snapshots'][0]
        opponent_model=copy.deepcopy(cpu).eval();opponent_model.load_state_dict(load_torch(source/member['path'],map_location='cpu',weights_only=True),strict=True)
        for parameter in opponent_model.parameters():parameter.requires_grad_(False)
        opponent_model.freeze_tables()
        deck=list(map(int,(bundle/'deck.csv').read_text(encoding='utf-8-sig').split()));results=[]
        for threads in [2,4]:
            torch.set_num_threads(threads);stream=CPUActorRNG(saved['cpu_sampler_rng'])
            opponent=FrozenOpponent(opponent_model,features,deck,dict(member))
            engine=Engine(root,Path(__file__).resolve().parents[1],job/f'cpu-throughput-{threads}.log',sha256(job/'engine-manifest.json'))
            t=time.monotonic()
            def control():
                if time.time()>ticket['deadline_unix']-180:raise TimeoutError('Bounded CPU benchmark deadline')
            try:trajectories,summary=collect_series(engine,cpu,features,deck,opponent,0,threads,2,control=control,actor_chooser=stream.choose)
            finally:engine.close()
            elapsed=time.monotonic()-t;decisions=[d for trajectory in trajectories for d in trajectory['decisions']]
            atomic_torch_save(job/f'engineering-selfplay-threads{threads}.pt',{'role':'engineering_throughput_only_not_PPO_training',
                'trajectories':trajectories,'summary':summary,'cpu_sampler_rng_after':stream.state()})
            results.append({'CPU_threads':threads,'total_cgroup_CPU_quota':ticket['cpu_quota_cores'],
                'complete_BO3':1,'individual_games':len(trajectories),'sampled_decisions':len(decisions),'trained_decisions':0,
                'PPO_updates':0,'wall_seconds_including_engine_start':elapsed,'decisions_per_second':len(decisions)/elapsed,
                'collector_seconds':summary['seconds'],'learner_inference_seconds':summary['learner_inference_seconds'],
                'engine_decisions':summary['engine_decisions'],'fixed_frozen_history':member,'strength_evidence':False})
        torch.set_num_threads(2)
        atomic_json(job/'CPU-throughput-benchmark.json',{'complete':True,'engineering_only':True,'new_PPO_training_decisions':0,
            'not_a_paired_dealing_or_strength_evaluation':True,'runs':results})


def main():
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);p.add_argument('--fresh',action='store_true');args=p.parse_args()
    ticket=read_json(args.ticket);root=owned_root(NEW,PROJECT);job=inside(root,ticket['job_directory'])
    if ticket['run_id']!=os.environ.get('PTCG_JOB_RUN_ID') or ticket['kind']!='storage_acceptance':raise ValueError('Controlled acceptance identity differs')
    if args.fresh:restore_check(ticket,True);return
    def on_stop(*_):raise SystemExit(0)
    signal.signal(signal.SIGTERM,on_stop)
    from .server_trial_worker import kernel_snapshot
    protection=kernel_snapshot(ticket);atomic_json(job/'hard-protection.json',protection)
    blocked=[]
    for path in ['/etc/ptcg-forbidden',OLD+'/would-write','/mnt/HDD2_3TB/TRAINING_USER/other-project']:
        try:inside(root,path)
        except ValueError:blocked.append(path)
        else:raise ValueError('Directory isolation failed')
    try:admit(root,50*2**30)
    except OSError:capacity_stop=True
    else:raise ValueError('50GiB overforecast was not blocked')
    capacity=admit(root,512*2**20)
    from .systemd_job import inspect_service,identity,stop_verified
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();handle=identity(inspect_service(ticket['unit']),boot)
    try:stop_verified(dict(handle,invocation_id='0'*32))
    except RuntimeError:wrong_identity_rejected=True
    else:raise ValueError('Wrong job identity was allowed to stop')
    from kaggle_environments import make
    from kaggle_environments.envs.cabt.cg import sim
    sim.get_lib();make('cabt',configuration={'version':2,'bo':3})
    base=Path(sim.__file__).parent
    atomic_json(job/'engine-manifest.json',{'platform':'linux','files':{name:{'path':str(base/name),'sha256':sha256(base/name)} for name in ['libcg.so','sim.py','game.py']}})
    os.environ['PTCG_ENGINE_MANIFEST_PATH']=str(job/'engine-manifest.json')
    # Exercise the real simulator IPC and public-view action path, no model update.
    from .rl_collector import Engine
    from .agents import validate_action
    source=root/'jobs'/FIRST;manifest=read_json(source/'training-manifest.raw.json')
    deck=list(map(int,(source/'control'/manifest['training_config']['bundle_path']/'deck.csv').read_text(encoding='utf-8-sig').split()))
    engine=Engine(root,Path(__file__).resolve().parents[1],job/'engine.log',sha256(job/'engine-manifest.json'))
    simulator_decisions=0
    try:
        engine.send({'command':'bo3','decks':[deck,deck]})
        while True:
            if time.time()>ticket['deadline_unix']-180:raise TimeoutError('Simulator check deadline')
            message=engine.read()
            if message['event']=='series_end':
                if message['statuses']!=['DONE','DONE']:raise ValueError('Official simulator BO3 failed')
                break
            obs=message['observation'];sel=obs['select'];action=list(range(sel['minCount']))
            if not set(obs).issubset({'current','select','logs'}):raise ValueError('Non-public actor observation')
            validate_action(obs,action);engine.send({'action':action});simulator_decisions+=1
    finally:engine.close()
    atomic_json(job/'simulator-check.json',{'complete':True,'actual_official_BO3':1,'statuses':message['statuses'],
        'scores':message['scores'],'simulator_decisions':simulator_decisions,'engineering_only_not_strength_or_training':True})
    restore_check(ticket)
    result=subprocess.run([sys.executable,'-B','-m','ptcg_lab.storage_acceptance_worker','--ticket',args.ticket,'--fresh'],
        cwd=Path(__file__).resolve().parents[1],check=True,timeout=max(30,min(300,ticket['deadline_unix']-time.time()-90)))
    proof={'at':utcnow(),'complete':True,'phase':'awaiting_verified_service_stop','training_updates':0,
        'directory_isolation_rejections':blocked,'overforecast_50GiB_stop_passed':capacity_stop,
        'wrong_invocation_stop_rejected':wrong_identity_rejected,'kernel_limits':protection,'capacity':capacity,
        'full_restore':read_json(job/'full-restore.json'),'fresh_process_restore':read_json(job/'fresh-process-restore.json'),
        'simulator':read_json(job/'simulator-check.json'),'source_root_preserved':OLD,'target_root':NEW}
    if (job/'CPU-throughput-benchmark.json').exists():proof['CPU_throughput']=read_json(job/'CPU-throughput-benchmark.json')
    atomic_json(job/'acceptance-metrics.json',proof)
    # The controller now validates and stops this exact own invocation. A finite
    # fallback exits without extending the existing kernel or budget deadline.
    stop_at=min(time.time()+40,ticket['deadline_unix']-65)
    while time.time()<stop_at:time.sleep(.2)


if __name__=='__main__':main()
