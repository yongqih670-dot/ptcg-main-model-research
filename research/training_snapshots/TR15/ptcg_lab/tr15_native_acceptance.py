"""Finite disposable native/selfplay/CUDA recovery differences, zero science credit."""
import argparse,copy,os,time
from pathlib import Path
import numpy as np
import torch
from .io import read_json,atomic_json,sha256,utcnow
from .tensor_io import atomic_torch_save,load_torch
from .framework_v2_training import NativeCheckpoints,semantic_digest
from .tr15_framework import admitted_switches,SWITCHES,validate_report
from .rl_core import load_full_actor,optimizer_for,restore_rng,rng_state,ppo_step
from .rl_resume import resume_journal
from .training_control import TrainingControl,ControlledTrainingStop
from . import ppo_transaction as tx

def backend(job,ticket,config):
    return NativeCheckpoints(job,ticket['lease_epoch'],config['framework_v2_checkpoint_cadence'],
        capacity_root=Path(ticket['project_root']),retained_bytes=config['readonly_retained_bytes'],optimizations=SWITCHES)

def snapshot(job,ticket,config):
    b=backend(job,ticket,config)
    try:return b.restore()
    finally:b.close()

def control_for(job,ticket):
    return TrainingControl(job,job.name,ticket['lease_epoch'],float(os.environ['PTCG_JOB_DEADLINE_UNIX']),lambda *a,**k:None,optimizations=SWITCHES)

def cache_control(job,ticket):
    from .framework_v3_runtime import JsonEvidence
    path=job/'engineering-cache-probe.json';cache=JsonEvidence(job,read_json)
    atomic_json(path,{'value':1});assert cache.get(path)=={'value':1}
    atomic_json(path,{'value':2});assert cache.get(path)=={'value':2}
    manual=job/'manual-training-control.json';old=read_json(manual) if manual.exists() else None
    if old and old['command']!='run':raise ValueError('Existing user pause or stop must not be overridden')
    control=control_for(job,ticket);control.wait(90,gpu_required=False)
    seq=old['sequence']+1 if old else 0
    def publish(command,n):
        atomic_json(manual,{'run_id':job.name,'lease_epoch':ticket['lease_epoch'],'sequence':n,
                           'command':command,'reason':'TR15 isolated engineering control difference','at':utcnow()})
    publish('stop',seq)
    try:control.wait(90,gpu_required=False)
    except ControlledTrainingStop:pass
    else:raise ValueError('Cached controller missed explicit stop')
    publish('run',seq+1);control.wait(90,gpu_required=False)
    # Only our exact test row is replaced. A concurrent user request wins.
    import threading
    publish('pause',seq+2)
    failures=[]
    def release():
        time.sleep(.6)
        row=read_json(manual)
        if row['reason']!='TR15 isolated engineering control difference' or row['sequence']!=seq+2:
            failures.append('Concurrent control request');return
        publish('run',seq+3)
    thread=threading.Thread(target=release);thread.start();began=time.monotonic()
    control.wait(90,gpu_required=False);elapsed=time.monotonic()-began;thread.join()
    if failures or elapsed<.5:raise ValueError('Cached pause/resume control failed')
    result={'complete':True,'metadata_atomic_replacement_invalidates_cache':True,'explicit_stop_detected':True,
            'pause_resume_detected':True,'manual_pause_elapsed_seconds':elapsed,
            'manual_test_pause_is_charged_not_shared_wait':True}
    atomic_json(job/'TR15-native-cache-control.json',result)

def gpu_cursor(job,ticket,config,cold):
    from .tr15_cuda_comparison import compare_tensors
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    control=control_for(job,ticket);control.wait(90)
    saved,point=snapshot(job,ticket,config)
    if saved['version']!=41 or saved['partial_ppo'] is None or saved['partial_ppo']['positive_optimizer_steps']!=1:
        raise ValueError('Real driver must durably stop after exactly one positive original PPO step')
    journal=resume_journal(job,saved,config['archive_restore'],None);decisions=journal['decisions']
    if len(decisions)<512:raise ValueError('Actual complete native BO3 PPO fixture is too small')
    folder=job/'engineering-gpu-cursor';folder.mkdir(exist_ok=True)
    model,features,_=load_full_actor(job/'control'/config['bundle_path'],config['parent_model_sha256'],'cuda')
    settings=config['ppo']
    def one():
        optimizer=optimizer_for(model,settings['actor_learning_rate'],settings['value_head_learning_rate'])
        model.load_state_dict(saved['model']);optimizer.load_state_dict(copy.deepcopy(saved['optimizer']));restore_rng(saved['rng'])
        partial=copy.deepcopy(saved['partial_ppo'])
        cut={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),'partial':partial}
        if semantic_digest(cut)!=semantic_digest({'model':saved['model'],'optimizer':saved['optimizer'],'rng':saved['rng'],'partial':saved['partial_ppo']}):
            raise ValueError('Saved CUDA parameters/optimizer/all RNG/permutation did not restore exactly')
        chosen=tx.next_batch(partial,settings['epochs']);control.wait(90)
        began=time.monotonic()
        metrics=ppo_step(model,optimizer,[decisions[i] for i in chosen],partial['advantages'][chosen],partial['returns'][chosen],features.b.STOP,
            clip=settings['clip'],entropy_coef=settings['entropy_coefficient'],value_coef=settings['value_loss_coefficient'],
            max_grad_norm=settings['max_grad_norm'],target_kl=settings['target_kl'],normalize_advantages=False,
            critic_updates_shared_trunk=settings.get('critic_updates_shared_trunk',True),record_gradient_components=False)
        tx.acknowledge(partial,chosen,metrics,settings['epochs'],saved.get('inherited_pending_remaining',0))
        # Capture the arithmetic result before cooperative telemetry calls.
        value=copy.deepcopy({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'optimizer':optimizer.state_dict(),'rng':rng_state(),'partial':partial})
        control.after_gpu_work(time.monotonic()-began)
        return value
    if not cold:
        expected=one();atomic_torch_save(folder/'expected.pt',expected,replace=False)
        repeat=one();atomic_torch_save(folder/'repeat.pt',repeat,replace=False)
        atomic_json(folder/'cut-binding.json',{'generation':point['generation'],'state_semantic_sha256':saved['state_semantic_sha256'],
                    'source_snapshot_sha256':saved['league']['snapshots'][-1]['sha256'],'pending_decisions':len(decisions)})
    else:
        binding=read_json(folder/'cut-binding.json')
        if binding['generation']!=point['generation'] or binding['state_semantic_sha256']!=saved['state_semantic_sha256']:
            raise ValueError('Actual driver cut state changed before cold comparison')
        actual=one();expected=load_torch(folder/'expected.pt',map_location='cpu',weights_only=False)
        repeat=load_torch(folder/'repeat.pt',map_location='cpu',weights_only=False)
        discrete=lambda v:{'rng':v['rng'],'partial':{k:x for k,x in v['partial'].items() if k!='logs'}}
        exact=semantic_digest(discrete(actual))==semantic_digest(discrete(expected))==semantic_digest(discrete(repeat))
        numbers=compare_tensors({k:expected[k] for k in ('model','optimizer')},
             {k:actual[k] for k in ('model','optimizer')},{k:repeat[k] for k in ('model','optimizer')})
        logs_ok=True;max_log=0.
        for a,b,c in zip(expected['partial']['logs'],actual['partial']['logs'],repeat['partial']['logs']):
            if set(a)!=set(b) or set(a)!=set(c):logs_ok=False;continue
            for k in a:
                if isinstance(a[k],float):
                    delta=abs(a[k]-b[k]);max_log=max(max_log,delta)
                    bound=min(1e-4*max(1.,abs(a[k])),max(8*abs(a[k]-c[k]),1e-7*max(1.,abs(a[k]))))
                    if not np.isfinite(delta) or delta>bound:logs_ok=False
                elif a[k]!=b[k] or a[k]!=c[k]:logs_ok=False
        result={'complete':bool(exact and numbers['passed'] and logs_ok),'checkpoint_model_optimizer_all_RNG_permutation_exact':True,
            'RNG_permutation_input_counters_exact':exact,'PPO_continuation_within_measured_GPU_repeatability':numbers['passed'] and logs_ok,
            'fresh_CUDA_process_after_positive_step':True,'scientific_decision_credit':0,'numerical_comparison':numbers,
            'uninterrupted_repeat_bitwise_exact':semantic_digest(expected)==semantic_digest(repeat),'maximum_log_abs_difference':max_log}
        atomic_json(job/'TR15-cursor-comparison.json',result)
        if not result['complete']:raise ValueError('Actual new-backend CUDA partial recovery difference failed')

def run_step(job,ticket,config,step):
    from .tr15_train import run
    before,point=snapshot(job,ticket,config) if (job/'recovery-v2.json').exists() else (None,None)
    # The inherited-parent fixture/reference belongs only to the pristine
    # parent. Later cuts use exact latest-state guards and the new cold proof.
    run(dict(config,_engineering_step=step,derived_resume_runtime_audit=step=='collect'),job)
    after,point=snapshot(job,ticket,config)
    if step=='collect':
        parent=load_torch(job/'checkpoint.pt',map_location='cpu',weights_only=False)
        if after['version']!=41 or after['partial_ppo'] is not None or after['unique_PPO_trained_decisions']!=307037:
            raise ValueError('Engineering collection changed parent model version or unique count')
        if semantic_digest(after['model'])!=semantic_digest(parent['model']) or semantic_digest(after['optimizer'])!=semantic_digest(parent['optimizer']):
            raise ValueError('Zero-update native collection changed parent parameters or optimizer')
        cache_control(job,ticket)
    elif step=='cut':
        if after['version']!=41 or after['partial_ppo']['positive_optimizer_steps']!=1:raise ValueError('Partial cut not persisted')
        gpu_cursor(job,ticket,config,False)
    elif step=='finish':
        if before['version']!=41 or before['partial_ppo'] is None or after['version']!=42 or after['partial_ppo'] is not None:
            raise ValueError('Actual partial driver did not complete exactly one original PPO update')
        if after['league']['snapshots'][-1]['version']!=42:raise ValueError('Full update did not refresh frozen self')
        if after['league']['snapshots'][-2]['sha256']!=before['league']['snapshots'][-1]['sha256']:raise ValueError('Old frozen opponent changed')
    elif step=='probe':
        if after['version']!=42 or after['partial_ppo'] is not None:raise ValueError('Postupdate BO3 probe performed optimization')
    atomic_json(job/('TR15-native-'+step+'.json'),{'complete':True,'step':step,'version':after['version'],
        'partial_PPO':after['partial_ppo'] is not None,'generation':point['generation'],
        'engineering_copy_unique_count':after['unique_PPO_trained_decisions'],'scientific_decision_credit':0,
        'snapshot_audit':read_json(job/'selfplay-snapshot-audit.json'),
        **({'zero_update_GPU_comparison':read_json(job/'derived-gpu-restore.json')} if step=='collect' else {})})

def recovery_driver_step(job,ticket,config,step):
    """The real driver restores once; independent cold tests check full bytes.

    Reuse hash-bound metadata here instead of rehashing the same819MiB graph
    before and after every already verified driver call.
    """
    from .tr15_train import run
    if not config.get('TR15_engineering_recovery'):raise ValueError('Registered isolated recovery required')
    before=read_json(job/'league.json')['snapshots'][-1]
    run(dict(config,_engineering_step=step,derived_resume_runtime_audit=False),job)
    point=read_json(job/'recovery-v2.json')['points'][-1];progress=read_json(job/'progress.json')
    audit=read_json(job/'selfplay-snapshot-audit.json')
    expected={'restore_cut':('engineering_partial_restore_committed',41,True),
              'finish':('update_committed',42,False),'probe':('engineering_probe_complete',42,False)}[step]
    if (progress['stage'],progress['version'],progress['partial_PPO_in_progress'])!=expected or point['cursor']['version']!=expected[1]:
        raise ValueError('Actual resumed driver did not reach the exact finite boundary')
    if step=='finish':
        if before['version']!=41 or not any(e['event']=='full_PPO_commit_snapshot_refresh' and e['version']==42 and e['snapshot_matches_committed_actor'] for e in audit['events']):
            raise ValueError('Full-update immutable snapshot refresh is incomplete')
        entries=read_json(job/'league.json')['snapshots']
        if entries[-2]['sha256']!=before['sha256']:raise ValueError('Preupdate frozen self changed')
    if step=='restore_cut':cache_control(job,ticket)
    report={'complete':True,'step':step,'version':expected[1],'partial_PPO':expected[2],'generation':point['generation'],
        'engineering_copy_unique_count':progress['unique_PPO_trained_decisions'],'scientific_decision_credit':0,
        'snapshot_audit':audit,'fresh_independent_full_CUDA_checks':'warm/cold plus final two-point cold verification',
        'latest_exact_GPU_restore':read_json(job/'latest-exact-GPU-restore.json')}
    name='cut' if step=='restore_cut' else step
    atomic_json(job/('TR15-native-'+name+'.json'),report)

def final(job,ticket,config):
    from .tr15_cold_recovery import main as cold_main
    import sys
    original=sys.argv;sys.argv=['tr15_cold_recovery','--ticket',str(job/'ticket.json')]
    try:cold_main()
    finally:sys.argv=original
    from .archive_outbox import ArchiveOutbox
    outbox=ArchiveOutbox(job,job.name,ticket['lease_epoch'],optimizations=SWITCHES)
    # Full recovery graph and frozen members were queued by the real writer.
    # Mutable metadata is copied with complete chunk hashes by archive_small.
    while outbox.pending()['pending_objects']:
        if time.time()>float(os.environ['PTCG_JOB_DEADLINE_UNIX'])-45:raise TimeoutError('Actual complete engineering archive ACK did not drain')
        time.sleep(.25)
    rows=[read_json(p) for p in sorted((job/'series').glob('[0-9]'*6+'.json'))]
    if not rows or not {41,42}<=set(r['assignment']['opponent_policy_version'] for r in rows):raise ValueError('Whole BO3 before and after refresh are required')
    for row in rows:
        assignment=row['assignment']
        if not assignment['snapshot_fixed_for_whole_BO3'] or assignment['bucket']!='latest_frozen_learner' or assignment['fixed']:
            raise ValueError('Native selfplay assignment changed')
        timing=row['timing_diagnostics']['configuration']
        if row['statuses']!=['DONE','DONE'] or timing['runTimeout']!=6000 or timing['actTimeout']!=0:
            raise ValueError('Official BO3 status or runtime timeout changed')
    comparison=read_json(job/'TR15-cursor-comparison.json');cold=read_json(job/'framework-v2-cold-recovery.json')
    partial_events=read_json(job/'TR15-native-finish.json')['snapshot_audit']['events']
    if not any(e['event']=='partial_restore_adopt_preupdate_opponent' for e in partial_events):raise ValueError('Actual partial restore self opponent was not proved')
    from .tr15_acceptance_reuse import runtime_fingerprint,canonical_hash,format_signature
    saved,point=snapshot(job,ticket,config)
    recovery=config.get('TR15_engineering_recovery')
    if recovery:parent_semantic=recovery['pristine_parent_semantic_sha256']
    else:
        parent,_,_=__import__('ptcg_lab.tr15_source',fromlist=['read_full']).read_full(
            Path(ticket['project_root'])/'jobs'/read_json(job/'training-manifest.raw.json')['existing_restore_source']['run_id'],
            config['segment_source_run_id'],read_json(job/'training-manifest.raw.json')['existing_restore_source']['lease_epoch'],
            read_json(job/'training-manifest.raw.json')['source_v2']['index_sha256'])
        parent_semantic=parent['state_semantic_sha256']
    environment=runtime_fingerprint(ticket['project_root'],ticket['gpu_uuid'])
    result={'at':utcnow(),'complete':True,'engineering_only':True,'scientific_decision_credit':0,
        'parent_acceptance_passed':True,'parent_source_unique':307037,'parent_source_version':41,
        'parent_zero_update_GPU_restore':read_json(job/'TR15-native-collect.json'),
        **{k:comparison[k] for k in ('checkpoint_model_optimizer_all_RNG_permutation_exact','RNG_permutation_input_counters_exact','PPO_continuation_within_measured_GPU_repeatability')},
        'native_selfplay_whole_BO3_snapshot_fixed':True,'full_update_snapshot_refresh_verified':True,
        'partial_restore_retains_preupdate_snapshot':True,'two_recovery_points_cold_restored':len(cold['two_points'])==2,
        'complete_archive_ACK_verified':True,'cache_invalidation_stop_verified':read_json(job/'TR15-native-cache-control.json')['complete'],
        'framework_v3_manifest_sha256':config['framework_v3_manifest_sha256'],'native_BO3_count':len(rows),
        'numerical_comparison':comparison,'runtime_environment':environment,'runtime_environment_sha256':canonical_hash(environment),
        'model_format_sha256':canonical_hash(format_signature(saved)),
        'pristine_scientific_parent_semantic_sha256':parent_semantic,
        'reused_zero_update_evidence':recovery,
        'disposable_engineering_copy_must_never_parent_science':True,'AMP_enabled':False,
        'GPU_performance_improvement_not_measured':True,'promotion_eligible':False}
    validate_report(result);atomic_json(job/'TR15-native-selfplay-acceptance.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);p.add_argument('--step',required=True);args=p.parse_args()
    ticket=read_json(args.ticket);job=Path(ticket['job_directory']);config=read_json(job/'payload-config.json')
    if not config.get('TR15_native_acceptance') or ticket.get('new_diagnostic_identity')!='TR15':raise ValueError('Disposable native acceptance only')
    admitted_switches(config,job)
    if args.step=='bootstrap':
        from .scientific_resume import audit_entry
        audit_entry(str(job/'ticket.json'))
    elif args.step=='warm':gpu_cursor(job,ticket,config,False)
    elif args.step=='cold':gpu_cursor(job,ticket,config,True)
    elif args.step=='final':final(job,ticket,config)
    elif config.get('TR15_engineering_recovery') and args.step in ('restore_cut','finish','probe'):recovery_driver_step(job,ticket,config,args.step)
    elif args.step in ('collect','cut','finish','probe'):run_step(job,ticket,config,args.step)
    else:raise ValueError('Unknown finite native step')

if __name__=='__main__':main()
