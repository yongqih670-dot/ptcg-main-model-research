"""Bounded official-BO3 PPO: complete rounds, immutable opponents, durable shards."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .io import atomic_json,read_json,sha256,utcnow
from .rl_core import load_full_actor,optimizer_for,gae,ppo_step,checkpoint,restore_rng,rng_state
from .rl_collector import Engine,collect_series
from .rl_league import League
from .rl_journal import committed_fixed_results
from .tensor_io import atomic_torch_save,load_torch
from .rl_resume import resume_journal,advance_resume_state
from .training_checkpoint import load_training_checkpoint,initialize_recovery,publish_training_checkpoint
from .training_control import TrainingControl,ControlledTrainingStop
from .ppo_accounting import ACCOUNTING,checkpoint_decisions,update_decisions
from . import phase_timing
from .framework_v2_training import NativeCheckpoints, advance_native_state, ensure_two_points
from . import ppo_transaction as transaction


def atomic_torch(path,value):
    return atomic_torch_save(path,value)


def run(config,output,bootstrap_only=False):
    from .cooperative_shutdown import CooperativeShutdown
    shutdown=CooperativeShutdown()
    output=Path(output);source=Path(__file__).resolve().parents[1];project=Path(config['project_path'])
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Training requires controlled runner')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    control=None
    if config.get('cooperative_resource_control'):
        control=TrainingControl(output,output.name,int(os.environ['PTCG_JOB_LEASE_EPOCH']),deadline,
            lambda stage,**kw:atomic_json(output/'progress.json',{'at':utcnow(),'stage':stage,**kw}))
        control.wait(90)
    torch.set_num_threads(2);torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    model,features,inheritance=load_full_actor(source/config['bundle_path'],config['parent_model_sha256'],'cuda',
        device_guard=(lambda:control.wait(90)) if control else None)
    settings=config['ppo'];optimizer=optimizer_for(model,settings['actor_learning_rate'],settings['value_head_learning_rate'])
    sampling_device=config.get('sampling_device','cuda')
    if sampling_device not in ('cpu','cuda'):raise ValueError('Unreviewed sampling device')
    cp=output/'checkpoint.pt';version=0;trained=0;unique_trained=0;series_id=0;updates=[];shards=[];decisions=[];saved=None
    archive_spec=config.get('archive_restore');archive_state=None;origin_prefix=None
    if config.get('fresh_compact_bootstrap'):
        from .fresh_compact import initialize
        if archive_spec:raise ValueError('Fresh compact cannot replace inherited archive settings')
        archive_spec=initialize(output,config,model,optimizer)
    lease_epoch=int(os.environ['PTCG_JOB_LEASE_EPOCH']) if archive_spec else None
    outbox=None;archive_options=config.get('online_archive')
    if archive_options:
        if not archive_spec or lease_epoch is None:raise ValueError('Online archive requires compact fenced recovery')
        from .archive_outbox import ArchiveOutbox
        outbox=ArchiveOutbox(output,output.name,lease_epoch)
        outbox.backpressured(archive_options['maximum_pending_bytes'])
    if config.get('hypothesis_id')!='TR13' or config['target_decisions']!=206935:
        raise ValueError('Prospective driver is limited to the directly authorized TR13 endpoint')
    profile=config['framework_v2_checkpoint_cadence']
    backend=NativeCheckpoints(output,lease_epoch,profile,capacity_root=project,
        retained_bytes=config.get('readonly_retained_bytes',0))
    checkpoint_interval=profile['trigger_after_seconds']
    partial=None;model_revision=0;inherited_pending=6807
    if (output/'recovery-v2.json').exists():
        restored,point=backend.restore()
        loaded={'saved':restored,'archive_spec':archive_spec,'digest':None,
                'recovery':{'generation':point['generation'],'framework_v2':point}}
    else:loaded=load_training_checkpoint(output,archive_spec,lease_epoch)
    saved=loaded['saved'];archive_spec=loaded['archive_spec']
    if saved:
        partial=saved.get('partial_ppo');model_revision=saved.get('model_state_revision',0)
        inherited_pending=saved.get('inherited_pending_remaining',6807)
        if saved.get('staged_v2_reference_map'):
            for ref in saved['staged_v2_reference_map'].values():backend.objects.evidence.verify(ref,fresh=True)
            backend.bridge.reference_map=dict(saved['staged_v2_reference_map'])
    if saved is not None:
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        if config.get('require_exact_restored_GPU_state'):
            from .framework_v2_training import semantic_digest
            for current,expected in [(model.state_dict(),saved['model']),(optimizer.state_dict(),saved['optimizer']),(rng_state(),saved['rng'])]:
                if semantic_digest(current)!=semantic_digest(expected):raise ValueError('Latest complete model/optimizer/allRNG CUDA restore differs')
            atomic_json(output/'latest-exact-GPU-restore.json',{'at':utcnow(),'complete':True,'source_run_id':config['segment_source_run_id'],
                'version':saved['version'],'actual_unique':checkpoint_decisions(saved),'model_revision':saved.get('model_state_revision'),
                'model_optimizer_all_RNG_exact':True,'optimizer_steps_executed':0})
        version=saved['version'];trained=saved['trained_decisions'];series_id=saved['next_series_id'];updates=saved['updates']
        unique_trained=checkpoint_decisions(saved)
    if config.get('fresh_compact_bootstrap'):
        from .fresh_compact import audit_gpu_restart
        if control:control.wait(90)
        audit_gpu_restart(output,model,optimizer,saved,os.environ['CUDA_VISIBLE_DEVICES'],bootstrap_only)
    deck=list(map(int,(source/config['bundle_path']/'deck.csv').read_text(encoding='utf-8-sig').split()))
    sampler=model;cpu_stream=None
    if sampling_device=='cpu':
        from .rl_sampling import cpu_replica,CPUActorRNG
        sampler=cpu_replica(model)
    league=League(source,output,config['league'],model,features,deck,
                  opponent_template=sampler if sampling_device=='cpu' else None);league.freeze(version)
    (output/'rollouts').mkdir(exist_ok=True);(output/'series').mkdir(exist_ok=True)
    # The same journal loader is exercised by the read-only local migration
    # audit; compact mode reads archived metadata plus the required live tail.
    if archive_spec and saved is None:raise ValueError('Compact training requires a verified inherited checkpoint')
    resume=resume_journal(output,saved or {'version':0,'trained_decisions':0,'next_series_id':0},archive_spec,
                          loaded['digest'])
    decisions=resume['decisions'];shards=resume['shards'];series_id=resume['next_series_id']
    archive_state=resume['archive_state'];origin_prefix=resume['origin_prefix']
    new_tail=transaction.has_postcheckpoint_tail(saved,resume)
    if partial is not None and new_tail:raise ValueError('Sampling cannot append to a partially optimized PPO input')
    if saved and not new_tail:restore_rng(saved['rng'])
    elif resume['rng_after'] is not None:restore_rng(resume['rng_after'])
    if sampling_device=='cpu':
        stream_state=resume.get('cpu_sampler_rng_after')
        if saved and not new_tail:
            stream_state=saved.get('cpu_sampler_rng',torch.get_rng_state())
        if stream_state is None:stream_state=(saved or {}).get('cpu_sampler_rng',torch.get_rng_state())
        cpu_stream=CPUActorRNG(stream_state)
    if config.get('derived_resume_runtime_audit'):
        from .scientific_resume import audit_runtime
        if control:control.wait(90)
        audit_runtime(output,model,optimizer,saved,decisions,features,os.environ['CUDA_VISIBLE_DEVICES'],
            fixture_reference=config.get('CPU_backend_fixture'),cpu_rollout_values=config.get('CPU_rollout_value_parity_required',False))
    inflight=output/'inflight.json'
    if inflight.exists():
        interrupted=read_json(inflight)
        sid=interrupted['series_id']
        if not (output/'series'/f'{sid:06d}.json').exists():
            atomic_json(output/'series'/f'{sid:06d}-interrupted.json',{**interrupted,'status':'interrupted_unfinished_not_training_data','at':utcnow()})
            series_id=max(series_id,sid+1)
        inflight.unlink()
    # The durable trajectory/summary commits precede league.observe(). Rebuild
    # from complete records so a crash in that window cannot lose or double
    # count an opponent result. Interrupted games never enter these statistics.
    if saved is not None:
        recovered_stats=resume['fixed_statistics']
        if recovered_stats!=league.stats:
            atomic_json(output/'league-recovery.json',{'at':utcnow(),'previous':league.stats,
                'recovered':recovered_stats,'source':'hash-verified committed complete BO3'})
            league.stats=recovered_stats;league.save()
    if config.get('require_exact_restored_GPU_state'):
        expected_rng=resume['rng_after'] if new_tail else saved['rng']
        if semantic_digest(rng_state())!=semantic_digest(expected_rng):raise ValueError('Inherited full RNG after complete tail differs')
        expected_CPU=resume.get('cpu_sampler_rng_after') if new_tail else saved.get('cpu_sampler_rng')
        if cpu_stream is not None and expected_CPU is not None and semantic_digest(cpu_stream.state())!=semantic_digest(expected_CPU):
            raise ValueError('Inherited CPU sampler RNG differs')
        inherited_tail=config.get('complete_postcheckpoint_tail')
        if inherited_tail and (len(decisions)!=inherited_tail['pending_decisions'] or series_id!=inherited_tail['next_series_id_after_interruption_exclusion']):
            raise ValueError('Complete tail or interrupted cursor was not preserved')
        atomic_json(output/'complete-inherited-state-restore.json',{'at':utcnow(),'complete':True,
            'source_run_id':config['segment_source_run_id'],'actual_unique':unique_trained,'version':version,
            'pending_decisions':len(decisions),'next_series_id':series_id,'all_rng_and_CPU_sampler_exact':True,
            'optimizer_steps_executed':0,'no_parent_state_restart':True})
    started=time.monotonic();collect_seconds=0.;update_seconds=0.;last_cp=time.monotonic();save_seconds=0.;recovery=loaded['recovery']
    if archive_spec and recovery is None:
        t=time.monotonic();recovery=initialize_recovery(output,saved,archive_spec,archive_state,lease_epoch);save_seconds=time.monotonic()-t
    if outbox:
        if not config.get('staged_v2_bootstrap'):outbox.queue_recovery(recovery)
        for entry in league.entries[-1:]:
            if league.member_path(entry)==output/entry['path']:outbox.queue(entry['path'],entry['sha256'])
    def progress(stage,**extra):
        if stage!='checkpointed_for_control':shutdown.check()
        if stage in ('paused_for_resource','yielding_for_shared_capacity') and backend.cadence.must_checkpoint(0,checkpoint_reserve()):save()
        phase_timing.publish(stage=stage,unique_trained=unique_trained,
            segment_start_actual_unique=config.get('segment_start_actual_unique',0),pending_decisions=len(decisions),version=version)
        atomic_json(output/'progress.json',{'at':utcnow(),'stage':stage,'version':version,'trained_decisions':trained,
            'unique_PPO_trained_decisions':unique_trained,'decision_accounting':ACCOUNTING,
            'pending_decisions':len(decisions),'next_series_id':series_id,'seconds_this_attempt':time.monotonic()-started,
            'model_state_revision':model_revision,'partial_PPO_in_progress':partial is not None,
            'partial_optimizer_steps_executed':partial['positive_optimizer_steps'] if partial is not None else 0,
            'remaining_job_seconds':max(0,deadline-time.time()),'last_checkpoint_seconds':save_seconds,
            'sampling_device':sampling_device,'PPO_device':str(next(model.parameters()).device),
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'recovery_generation':recovery['generation'] if recovery else None,**extra})
    unstaged_bytes=0
    def checkpoint_reserve():
        model_bytes=64*2**20 if backend.sealed_state!=(version,model_revision) else 0
        predicted=2*(unstaged_bytes+model_bytes)/(4*2**20)
        return max(60,save_seconds*1.5+10,predicted*1.3+10)
    def execution_reserve():return checkpoint_reserve()+config.get('segment_completion_reserve_seconds',1200)
    @phase_timing.timed('checkpoint_save_total')
    def save():
        nonlocal last_cp,archive_state,save_seconds,recovery,unstaged_bytes
        began=time.monotonic()
        extra={'cpu_sampler_rng':cpu_stream.state()} if cpu_stream is not None else {}
        # This cache lives for exactly one serialized training-writer safe
        # point. Every referenced unique file is fully hashed; duplicate reads
        # reuse that hash only while inode,size,mtime,UID and mode still match.
        # It is never saved in a checkpoint, reused by another save, or used to
        # skip the fresh hash immediately preceding an ACK-gated unlink.
        verified={} if config.get('checkpoint_transaction_hash_reuse') else None
        if archive_spec:
            checkpoint_cursor=series_id+int(inflight.exists())
            candidate=advance_native_state(backend,archive_state,origin_prefix,archive_spec,
                {'version':version,'trained_decisions':trained,'next_series_id':checkpoint_cursor},output.name,lease_epoch)
            extra.update(archive_state=candidate,archive_origin_prefix_sha256=archive_spec['prefix_sha256'])
        value={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),
               'unique_PPO_trained_decisions':unique_trained,'decision_accounting':ACCOUNTING,
               'version':version,'trained_decisions':trained,'next_series_id':checkpoint_cursor,'updates':updates,
               'partial_ppo':partial,'model_state_revision':model_revision,
               'inflight_BO3_excluded_on_restore':read_json(inflight) if inflight.exists() else None,
               'inherited_pending_remaining':inherited_pending,**extra}
        recovery=backend.save(value,archive_spec,read_json(output/'league.json'),model_revision=model_revision)
        if archive_spec:archive_state=candidate
        if outbox:
            outbox.queue_recovery(recovery)
            # Retain all originals throughout TR13; no unreferenced-object eviction here.
        last_cp=time.monotonic();save_seconds=last_cp-began;unstaged_bytes=0
    save()
    def safe_point(gpu_required=True, next_operation_seconds=2):
        shutdown.check()
        if backend.cadence.decision(next_operation_seconds,checkpoint_reserve())!='continue':save()
        if control:control.wait(execution_reserve(),gpu_required=gpu_required)
        if backend.cadence.decision(next_operation_seconds,checkpoint_reserve())!='continue':save()
    if control:control.progress=progress
    phase_timing.base('driver_control_and_journal')
    if config.get('volume_parent_acceptance_only'):
        if config.get('hypothesis_id')!='TR13' or config.get('continuous_parent_actual_decisions')!=106935:
            raise ValueError('TR12 zero-update acceptance scope differs')
        if config.get('complete_postcheckpoint_tail'):
            if len(decisions)!=config['complete_postcheckpoint_tail']['pending_decisions'] or series_id!=config['complete_postcheckpoint_tail']['next_series_id_after_interruption_exclusion']:
                raise ValueError('Acceptance lost complete tail or interrupted cursor')
            # Prove both newly published recovery points reference the complete
            # live tail. These are zero-update full-state saves, not samples.
            save();save()
        ensure_two_points(backend,save)
        if outbox:
            outbox.queue_recovery(recovery)
            for ref in (config.get('complete_postcheckpoint_tail') or {}).get('references',[]):
                outbox.queue(ref['path'],ref['sha256'])
            for entry in league.entries:outbox.queue(entry['path'],entry['sha256'])
            while outbox.pending()['pending_objects']:
                if time.time()>deadline-45:raise TimeoutError('TR12 full-state acceptance archive deadline')
                time.sleep(.2)
        atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,
            'optimizer_steps_executed':0,'new_sampled_decisions':0,'unique_PPO_trained_decisions':unique_trained,
            'new_unique_PPO_trained_decisions':0,'full_parent_archive_ACKs_verified':True,
            'pending_decisions_preserved':len(decisions),'promotion_eligible':False})
        progress('parent_acceptance_complete');return
    if config.get('parent_acceptance_only'):
        if config.get('continuous_parent_actual_decisions')!=32782 or decisions or version!=5:
            raise ValueError('Parent-only acceptance cannot use incomplete input or a changed cursor')
        from .continuous_mainline import preserve_milestones
        preserve_milestones(output,{**config,'decision_milestones':[32782]},unique_trained,recovery,league,outbox)
        if outbox:
            for entry in league.entries:outbox.queue(entry['path'],entry['sha256'])
            while outbox.pending()['pending_objects']:
                if time.time()>deadline-45:raise TimeoutError('Parent-only complete archive acceptance deadline')
                time.sleep(.2)
        atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,
            'optimizer_steps_executed':0,'new_sampled_decisions':0,'unique_PPO_trained_decisions':unique_trained,
            'new_unique_PPO_trained_decisions':0,'full_parent_archive_ACKs_verified':True,'promotion_eligible':False})
        progress('parent_acceptance_complete');return
    if bootstrap_only:
        if not config.get('fresh_compact_bootstrap'):raise ValueError('Bootstrap-only requires explicit zero-history origin')
        progress('fresh_bootstrap_complete');return
    optimizing=False
    new_complete_BO3=0;new_complete_games=0;new_sampled_decisions=0
    engine=Engine(project,source,output/'engine.log',config['engine_manifest_sha256'])
    try:
        while unique_trained<config['target_decisions']:
            target=min(settings['rollout_complete_learner_decisions'],config['target_decisions']-unique_trained)
            while partial is None and len(decisions)<target:
                safe_point(gpu_required=sampling_device!='cpu',next_operation_seconds=10)
                if outbox:
                    paused,pending=outbox.backpressured(archive_options['maximum_pending_bytes'])
                    if paused:
                        save()
                        while paused:
                            progress('archive_backpressure_wait',archive_pending=pending)
                            if time.time()>deadline-checkpoint_reserve():
                                save();progress('checkpointed_archive_backpressure',archive_pending=pending);return
                            safe_point(gpu_required=False,next_operation_seconds=0)
                            time.sleep(1)
                            paused,pending=outbox.backpressured(archive_options['maximum_pending_bytes'])
                if time.time()>deadline-checkpoint_reserve():
                    save();progress('checkpointed_at_soft_deadline');return
                seat=series_id%2;opponent,assignment=league.select(1-seat,config['seed']+series_id)
                atomic_json(inflight,{'series_id':series_id,'policy_version':version,'learner_seat':seat,'assignment':assignment,'at':utcnow()})
                t=time.monotonic()
                trajectories,summary=collect_series(engine,sampler,features,deck,opponent,seat,series_id,str(version),
                    progress=lambda counts:progress('collecting',**counts),control=lambda:safe_point(gpu_required=sampling_device!='cpu'),
                    actor_chooser=cpu_stream.choose if cpu_stream is not None else None)
                collect_seconds+=time.monotonic()-t
                batch=[d for trajectory in trajectories for d in trajectory['decisions']]
                new_complete_BO3+=1;new_complete_games+=len(trajectories);new_sampled_decisions+=len(batch)
                summary.update(assignment=assignment,round_rewards=[t['reward'] for t in trajectories],
                    learner_model_sha256=league.entries[-1]['sha256'])
                shard=f'rollouts/v{version:04d}-s{series_id:06d}.pt'
                shard_data={'decisions':batch,'rng_after':rng_state(),'series_id':series_id,'policy_version':version,'summary':summary}
                if cpu_stream is not None:shard_data['cpu_sampler_rng_after']=cpu_stream.state()
                atomic_torch(output/shard,shard_data)
                summary['trajectory_sha256']=sha256(output/shard);summary['trajectory_path']=shard
                atomic_json(output/'series'/f'{series_id:06d}.json',summary)
                unstaged_bytes+=(output/shard).stat().st_size+(output/'series'/f'{series_id:06d}.json').stat().st_size
                if outbox:outbox.queue_series(summary)
                league.observe(assignment,summary['rewards'][seat]);inflight.unlink()
                decisions.extend(batch);shards.append(shard);series_id+=1
                progress('series_committed')
                if time.monotonic()-last_cp>checkpoint_interval:save()
            # Rollouts remain on-policy; persist update input before optimization.
            save()
            if time.time()>deadline-checkpoint_reserve():progress('checkpointed_before_update_at_soft_deadline');return
            consumed_for_barrier=list(recovery['pending']) if outbox and config.get('consumed_reference_barrier_required') else None
            if config.get('CPU_GPU_batch_parity_required') and partial is None:
                if sampling_device!='cpu':raise ValueError('CPU/GPU parity requires CPU sampling')
                if control:control.wait(execution_reserve())
                from .sampling_parity import before_optimizer
                before_optimizer(model,decisions,features.b.STOP,output,version)
            progress('optimizing');t=time.monotonic();optimizing=True
            if partial is None:
                advantages,returns=gae(decisions,settings['gamma'],settings['gae_lambda'])
                advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
                partial=transaction.begin(decisions,advantages,returns,settings['minibatch_initial'])
            transaction.validate(partial,len(decisions),settings['epochs'])
            expected_advantages,expected_returns=gae(decisions,settings['gamma'],settings['gae_lambda'])
            expected_advantages=(expected_advantages-expected_advantages.mean())/(expected_advantages.std()+1e-8)
            if not np.array_equal(partial['advantages'],expected_advantages) or not np.array_equal(partial['returns'],expected_returns):
                raise ValueError('Partial PPO cached GAE no longer matches its immutable decisions')
            while not partial['finished']:
                safe_point(next_operation_seconds=30)
                if time.time()>deadline-checkpoint_reserve():
                    save();progress('partial_update_committed_at_soft_deadline');return
                chosen=transaction.next_batch(partial,settings['epochs'])
                epoch=partial['epoch'];offset=partial['offset']
                step_began=time.monotonic()
                try:
                    metrics=ppo_step(model,optimizer,[decisions[i] for i in chosen],partial['advantages'][chosen],partial['returns'][chosen],features.b.STOP,
                        clip=settings['clip'],entropy_coef=settings['entropy_coefficient'],value_coef=settings['value_loss_coefficient'],
                        max_grad_norm=settings['max_grad_norm'],target_kl=settings['target_kl'],normalize_advantages=False,
                        critic_updates_shared_trunk=settings.get('critic_updates_shared_trunk',True),
                        record_gradient_components=bool(config.get('continuous_parent_actual_decisions') is not None and epoch==0 and offset==0))
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                    alternatives=[x for x in settings['minibatch_memory_fallbacks'] if x<partial['minibatch']]
                    if not alternatives:raise
                    partial['minibatch']=max(alternatives);progress('minibatch_memory_fallback',minibatch=partial['minibatch']);continue
                gpu_work_seconds=time.monotonic()-step_began
                transaction.acknowledge(partial,chosen,metrics,settings['epochs'],inherited_pending)
                first_positive=False
                if not metrics['skipped_for_kl']:
                    model_revision+=1;first_positive=partial['positive_optimizer_steps']==1
                if first_positive or backend.cadence.must_checkpoint(30,checkpoint_reserve()):save()
                if control:control.after_gpu_work(gpu_work_seconds)
                if torch.cuda.max_memory_allocated()/2**30>10:raise RuntimeError('VRAM policy limit exceeded')
            logs=partial['logs'];minibatch=partial['minibatch']
            update_seconds+=time.monotonic()-t
            row={'version':version+1,'decisions':len(decisions),'series_ids':[int(Path(p).stem.split('-s')[1]) for p in shards],
                'terminal_rounds':sum(d['terminal'] for d in decisions),'minibatch':minibatch,'steps':logs,
                'actor_lr':settings['actor_learning_rate'],'value_lr':settings['value_head_learning_rate'],'at':utcnow()}
            row['unique_PPO_trained_decisions']=update_decisions(row)
            row['inherited_pending_first_epoch_coverage']=partial['inherited_first_epoch_coverage']
            row['new_sampling_first_epoch_coverage']=row['unique_PPO_trained_decisions']-row['inherited_pending_first_epoch_coverage']
            row['inherited_pending_not_covered']=inherited_pending-row['inherited_pending_first_epoch_coverage']
            row['positive_optimizer_steps']=partial['positive_optimizer_steps']
            inherited_pending=0;partial=None
            unique_trained+=row['unique_PPO_trained_decisions']
            updates.append(row);trained+=len(decisions);version+=1;decisions=[];shards=[];optimizing=False
            if sampling_device=='cpu':sampler=cpu_replica(model)
            league.freeze(version)
            save()
            if consumed_for_barrier is not None and not config.get('framework_v2_checkpoint_cadence'):
                from .archive_barrier import retire_before_sampling
                retire_before_sampling(outbox,consumed_for_barrier,
                    {'version':version,'trained_decisions':trained,'next_series_id':series_id},save,
                    deadline,checkpoint_reserve,progress,control)
            elif outbox and config.get('retire_consumed_with_second_checkpoint'):
                # Both retained recovery points advance to the committed update
                # before another rollout grows the bounded remote working set.
                # No optimizer step, sample or RNG draw occurs between saves.
                save()
            league.freeze(version);atomic_json(output/'training-updates.json',updates)
            if outbox:outbox.queue(league.entries[-1]['path'],league.entries[-1]['sha256'])
            if config.get('decision_milestones'):
                from .continuous_mainline import preserve_milestones
                preserve_milestones(output,config,unique_trained,recovery,league,outbox)
            if (config.get('continuous_parent_actual_decisions') is not None and len(updates)>=2
                    and all(update_decisions(r)==0 for r in updates[-2:])):
                progress('checkpointed_for_control',control_stop_reason='two_consecutive_zero_coverage_updates')
                return
            progress('update_committed')
        path=output/'model.pt';atomic_torch(path,{k:v.detach().cpu() for k,v in model.state_dict().items()})
        if outbox:outbox.queue('model.pt',sha256(path))
        metrics={'complete':True,'training_only':True,'model_sha256':sha256(path),'trained_decisions':trained,
            'unique_PPO_trained_decisions':unique_trained,'decision_accounting':ACCOUNTING,
            'updates':version,'completed_series':len(list((output/'series').glob('[0-9][0-9][0-9][0-9][0-9][0-9].json'))),
            'collect_seconds_this_attempt':collect_seconds,'update_seconds_this_attempt':update_seconds,
            'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30,'inheritance':inheritance,
            'league_snapshots':league.entries,'promotion_eligible':False,'validation_required':'independent fixed BO3 development comparison'}
        if config.get('continuous_parent_actual_decisions') is not None:
            metrics.update(inherited_actual_unique_PPO_decisions=config['continuous_parent_actual_decisions'],
                new_actual_unique_PPO_decisions=unique_trained-config['continuous_parent_actual_decisions'],
                inherited_PPO_updates=config.get('continuous_parent_PPO_updates',5),
                new_PPO_updates=version-config.get('continuous_parent_PPO_updates',5),
                new_complete_BO3=new_complete_BO3,new_complete_games=new_complete_games,
                new_sampled_decisions=new_sampled_decisions,
                inherited_history_must_not_be_added_to_project_volume_again=True)
        atomic_json(output/'metrics.json',metrics);progress('complete')
    except ControlledTrainingStop as exc:
        save();progress('checkpointed_for_control',partial_PPO_preserved=partial is not None,
                        control_stop_reason=str(exc),unfinished_BO3_excluded=inflight.exists())
        return
    except Exception as exc:
        if not shutdown.requested:raise
        save();progress('checkpointed_for_control',partial_PPO_preserved=partial is not None,
            control_stop_reason='Owned termination interrupted a subordinate: '+type(exc).__name__+': '+str(exc),
            unfinished_BO3_excluded=inflight.exists())
        return
    finally:
        engine.close();backend.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bootstrap-only',action='store_true')
    a=p.parse_args();config=read_json(a.config)
    phase_timing.configure(a.output,config.get('phase_timing'),'train')
    try:run(config,a.output,a.bootstrap_only)
    finally:phase_timing.publish(force=True,process_finalizing=True)


if __name__=='__main__':main()
