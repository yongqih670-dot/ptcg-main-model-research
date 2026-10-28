"""Actual resource-capped server checks and single short PPO payload."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .io import atomic_json,read_json,sha256,utcnow
from .server_workspace import owned_root,inside,admit
from .staged_environment import kernel_snapshot


def materialize(root,job,manifest):
    parent=manifest.get('parent_capsule');input_parent=manifest.get('input_capsule')
    capsule=job
    if parent:
        capsule=inside(root,root/'jobs'/parent['run_id'])
        if sha256(capsule/'training-manifest.raw.json')!=parent['manifest_sha256']:
            raise ValueError('Parent capsule changed')
        if sha256(capsule/'trial-metrics.json')!=parent['remote_basic_metrics_sha256']:
            raise ValueError('Parent basic correctness result changed')
        passed=read_json(capsule/'trial-metrics.json')
        if not passed.get('basic_cpu_correctness_passed'):raise ValueError('Parent basic checks did not pass')
        original=read_json(capsule/'training-manifest.raw.json')
    elif input_parent:
        capsule=inside(root,root/'jobs'/input_parent['run_id'])
        if sha256(capsule/'training-manifest.raw.json')!=input_parent['manifest_sha256']:
            raise ValueError('Read-only input capsule changed')
        original=manifest  # Reuse only pinned immutable blobs; repeat every acceptance check.
    else:original=manifest
    blob_capsule=capsule;blob_manifest=read_json(blob_capsule/'training-manifest.raw.json') if (blob_capsule/'training-manifest.raw.json').exists() else original
    seen=set()
    while blob_manifest.get('input_capsule'):
        spec=blob_manifest['input_capsule']
        if spec['run_id'] in seen or len(seen)>=4:raise ValueError('Input capsule ancestry cycle/depth')
        seen.add(spec['run_id']);blob_capsule=inside(root,root/'jobs'/spec['run_id'])
        if sha256(blob_capsule/'training-manifest.raw.json')!=spec['manifest_sha256']:
            raise ValueError('Ancestor input capsule changed')
        blob_manifest=read_json(blob_capsule/'training-manifest.raw.json')
    for item in original['file_map']:
        source=inside(root,blob_capsule/'stage'/(item['sha256']+'.blob'))
        if source.stat().st_size!=item['bytes'] or sha256(source)!=item['sha256']:
            raise ValueError('Immutable training input blob changed')
        target=inside(root,job/'control'/item['path']);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise ValueError('Input materialization would overwrite a source')
        os.link(source,target)
    return original,capsule


def gpu_fixture_check(ticket):
    import numpy as np
    import torch
    from .rl_core import load_full_actor
    from .tensor_io import load_torch
    root=owned_root(ticket['project_root'],ticket['project_id']);job=inside(root,ticket['job_directory'])
    manifest=read_json(job/'training-manifest.raw.json');parent=inside(root,root/'jobs'/manifest['parent_capsule']['run_id'])
    fixture=parent/'actor-parity-fixture.pt'
    if sha256(fixture)!=manifest['basic_fixture_sha256']:raise ValueError('CPU fixture changed')
    if torch.cuda.device_count()!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!=ticket['gpu_uuid']:
        raise ValueError('Trial does not expose exactly the selected authorized GPU')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model,features,inheritance=load_full_actor(job/'control/agents/frozen/ppo-zero-anchor',manifest['parent_model_sha256'],'cuda')
    sample=load_torch(fixture,map_location='cpu',weights_only=False);rows=sample['rows']
    x=torch.from_numpy(rows).cuda();dirty=x.clone();lo,hi=features.b.OFFSETS['oracle'];dirty[:,lo:hi]=123
    with torch.no_grad():
        scores,values=model.forward_eval(x);poisoned=model.forward_eval(dirty)
        torch.testing.assert_close(scores.cpu(),sample['scores'],rtol=1e-4,atol=2e-4)
        torch.testing.assert_close(values.cpu(),sample['values'],rtol=1e-4,atol=2e-4)
        for clean,changed in zip((scores,values),poisoned):torch.testing.assert_close(clean,changed,rtol=0,atol=0)
    atomic_json(job/'gpu-fixture-check.json',{'at':utcnow(),'passed':True,'single_visible_gpu':True,
        'original_full_actor_inherited':inheritance,'cpu_gpu_tolerances':{'rtol':1e-4,'atol':2e-4},
        'actor_and_critic_oracle_poison_equal':True,'fixture_sha256':sha256(fixture),'gpu_uuid':ticket['gpu_uuid']})


def checkpoint_roundtrip(ticket):
    import random
    import numpy as np
    import torch
    from .rl_core import load_full_actor,optimizer_for,restore_rng
    from .tensor_io import load_torch
    root=owned_root(ticket['project_root'],ticket['project_id']);job=inside(root,ticket['job_directory'])
    manifest=read_json(job/'training-manifest.raw.json');saved=load_torch(job/'checkpoint.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if torch.cuda.device_count()!=1:raise ValueError('Roundtrip must stay on one GPU')
    model,_,_=load_full_actor(job/'control/agents/frozen/ppo-zero-anchor',manifest['parent_model_sha256'],'cuda')
    model.load_state_dict(saved['model'],strict=True);optimizer=optimizer_for(model);optimizer.load_state_dict(saved['optimizer'])
    if len(saved['model'])!=127 or len(optimizer.state)!=97 or set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=1:
        raise ValueError('Full checkpoint fields are incomplete')
    for name,value in model.state_dict().items():
        if not torch.equal(value.cpu(),saved['model'][name]) or not torch.isfinite(value).all():
            raise ValueError('Restored model tensor differs or is nonfinite')
    current_states=optimizer.state_dict()['state']
    if set(current_states)!=set(saved['optimizer']['state']):raise ValueError('Restored optimizer parameter identities differ')
    for state_id,current in current_states.items():
        old=saved['optimizer']['state'][state_id]
        if set(current)!=set(old):raise ValueError('Restored optimizer state keys differ')
        for key,value in current.items():
            if torch.is_tensor(value):
                if not torch.equal(value.cpu(),old[key]) or not torch.isfinite(value).all():raise ValueError('Restored optimizer tensor differs or is nonfinite')
            elif value!=old[key]:raise ValueError('Restored optimizer scalar differs')
    def draws():return random.random(),np.random.randint(0,1000000,8),torch.rand(8),torch.rand(16,device='cuda').cpu()
    restore_rng(saved['rng']);first=draws();restore_rng(saved['rng']);second=draws()
    if first[0]!=second[0] or not np.array_equal(first[1],second[1]) or not torch.equal(first[2],second[2]) or not torch.equal(first[3],second[3]):
        raise ValueError('Restored RNG sequence differs')
    atomic_json(job/'checkpoint-roundtrip.json',{'at':utcnow(),'passed':True,'model_tensors':127,'optimizer_states':97,
        'python_numpy_cpu_and_cuda_rng_reproduced':True,'version':saved['version'],'trained_decisions':saved['trained_decisions'],
        'next_series_id':saved['next_series_id'],'checkpoint_sha256':sha256(job/'checkpoint.pt'),
        'fresh_process_restore_on_same_selected_gpu':True,'no_training_update_or_source_checkpoint_write':True})


def run(ticket_path):
    ticket=read_json(ticket_path);root=owned_root(ticket['project_root'],ticket['project_id']);job=inside(root,ticket['job_directory'])
    if os.environ.get('PTCG_JOB_RUN_ID')!=ticket['run_id'] or job.name!=ticket['run_id']:
        raise ValueError('Server payload identity differs')
    if sha256(job/'training-manifest.raw.json')!=ticket['manifest_sha256']:raise ValueError('Trial input manifest changed')
    if Path(sys.prefix).resolve()!=root/'.venv':raise ValueError('Payload interpreter is not private')
    manifest=read_json(job/'training-manifest.raw.json');mode=ticket['mode'];deadline=ticket['deadline_unix']
    active_clock=None
    if ticket.get('active_clock_scope'):
        from .active_clock import ServiceClock
        scope=ticket['active_clock_scope']
        atomic_json(job/'active-clock-settings.json',{'run_id':job.name,'lease_epoch':ticket['lease_epoch'],'scope':scope})
        active_clock=ServiceClock(job,job.name,ticket['lease_epoch'],scope);active_clock.start()
    if mode in ('basic_cpu_check','compact_cpu_restore','compact_cpu_existing','live_archive_check','journal_io_check','cpu_sampler_check') and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU acceptance must hide CUDA')
    protection=kernel_snapshot(ticket);atomic_json(job/'hard-protection.json',{'at':utcnow(),'run_id':job.name,'kernel':protection,
        'mode':mode,'io_hostwide_background_only':True,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES')})
    child=None;child_ticks=None;stop=False;peak_workspace=0;minimum_free=None
    initial_capacity=admit(root,0);own_gpu_peak_mib=0;resource_samples=0
    from .resource_journal import capacity_profile,append as append_resource_record
    runtime=ticket.get('TR11_runtime_scope')
    if ticket.get('TR12_runtime_scope'):
        from .volume_scope import validate_remote
        maximum=validate_remote(ticket['TR12_runtime_scope'],mode,ticket.get('new_diagnostic_identity'))
        journal_profile=capacity_profile(86400 if active_clock else maximum,extended_TR12=True,wait_exempt=bool(active_clock))
    elif runtime:
        from .continuous_runtime import validate_remote_scope
        maximum=validate_remote_scope(runtime,mode,ticket.get('new_diagnostic_identity'))
        journal_profile=capacity_profile(maximum,extended_TR11=True)
    else:journal_profile=capacity_profile()
    atomic_json(job/'resource-journal-profile.json',journal_profile)
    def interrupted(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def capacity():
        nonlocal peak_workspace,minimum_free
        record=admit(root,0);peak_workspace=max(peak_workspace,record['workspace_bytes'])
        minimum_free=record['free_bytes'] if minimum_free is None else min(minimum_free,record['free_bytes'])
        return record
    def ticks(pid):return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    def stop_child():
        if child is None or child.poll() is not None:return
        if ticks(child.pid)!=child_ticks or os.getpgid(child.pid)!=child.pid:
            raise RuntimeError('Child identity changed; refuse signal')
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            if ticks(child.pid)!=child_ticks:raise RuntimeError('Child identity changed before final stop')
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
    def gpu_sample():
        nonlocal own_gpu_peak_mib
        if mode not in ('short_gpu_trial','gpu_existing_restore','scientific_continue','fresh_scientific','derived_scientific'):return None
        group=Path('/proc/self/cgroup').read_text().strip().split(':',2)[2]
        members={int(x) for x in (Path('/sys/fs/cgroup')/group.lstrip('/')/'cgroup.procs').read_text().split()}
        rows=subprocess.run(['nvidia-smi','-i',ticket['gpu_uuid'],'--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,check=True).stdout.splitlines()
        own=0;foreign=[]
        for line in rows:
            parts=[x.strip() for x in line.split(',')]
            if len(parts)!=3 or parts[0]!=ticket['gpu_uuid']:continue
            if int(parts[1]) in members:
                if parts[2].isdigit():own+=int(parts[2])
            else:foreign.append(int(parts[1]))
        device=subprocess.run(['nvidia-smi','-i',ticket['gpu_uuid'],'--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,check=True).stdout.strip()
        own_gpu_peak_mib=max(own_gpu_peak_mib,own)
        return {'own_service_process_gpu_memory_mib':own,'device_memory_and_utilization_background':device,
                'foreign_compute_pids_observed':foreign}
    def execute(argv,child_deadline):
        nonlocal child,child_ticks,resource_samples
        env=dict(os.environ,PTCG_JOB_DEADLINE_UNIX=str(child_deadline),
                 PTCG_ENGINE_MANIFEST_PATH=str(job/'engine-manifest.json'))
        probe=None
        if manifest.get('cooperative_pause_probe_required') and not (job/'cooperative-pause-acceptance.json').exists():
            from .cooperative_probe import begin
            begin(job,ticket)
        with (job/'logs/payload.log').open('ab') as log:
            child=subprocess.Popen(argv,cwd=job/'control',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child_ticks=ticks(child.pid);last=0
            if active_clock:active_clock.register(child.pid,child_ticks)
            if manifest.get('cooperative_pause_probe_required') and not (job/'cooperative-pause-acceptance.json').exists():
                from .cooperative_probe import EarlyPauseProbe
                probe=EarlyPauseProbe(job,ticket,child.pid)
            try:
                while child.poll() is None:
                    if active_clock:
                        active_clock.check()
                        from .active_clock import remaining_from_record
                        if remaining_from_record(job,job.name,ticket['lease_epoch'])<=70:
                            raise TimeoutError('Active execution limit reached final shutdown reserve')
                    if stop or time.time()>deadline-75:raise TimeoutError('Owned payload stop/deadline')
                    if time.monotonic()-last>=10:
                        cooperative=manifest.get('cooperative_resource_control') is True
                        try:gpu=gpu_sample()
                        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,ValueError) as exc:
                            if not cooperative:raise
                            gpu={'monitoring_unavailable':type(exc).__name__+': '+str(exc),'own_service_process_gpu_memory_mib':None}
                        if cooperative and gpu is not None:
                            from .training_control import publish_resource_control
                            publish_resource_control(job,job.name,ticket['lease_epoch'],gpu)
                        if probe:
                            group=Path('/proc/self/cgroup').read_text().strip().split(':',2)[2]
                            events=(Path('/sys/fs/cgroup')/group.lstrip('/')/'cgroup.events').read_text()
                            probe.observe(dict(line.split() for line in events.splitlines())['frozen'])
                        sample={'at':utcnow(),'capacity':capacity(),'kernel':kernel_snapshot(ticket),
                            'child_pid':child.pid,'child_start_ticks':child_ticks,'gpu':gpu}
                        atomic_json(job/'resources.json',sample)
                        append_resource_record(root,job,sample,maximum_bytes=journal_profile['maximum_journal_bytes'])
                        resource_samples+=1
                        last=time.monotonic()
                    time.sleep(.5)
                if child.returncode:raise RuntimeError('Server payload failed; inspect logs/payload.log')
            finally:stop_child()
        child=None;return
    try:
        capacity();original,capsule=materialize(root,job,manifest)
        from kaggle_environments.envs.cabt.cg import sim
        sim.get_lib();base=Path(sim.__file__).parent
        engine={'platform':'linux','files':{name:{'path':str(base/name),'sha256':sha256(base/name)} for name in ['libcg.so','sim.py','game.py']}}
        atomic_json(job/'engine-manifest.json',engine);engine_hash=sha256(job/'engine-manifest.json')
        source=job/'control';python=str(root/'.venv/bin/python')
        if mode=='cpu_sampler_check':
            from .public_fragment_retirement import retire,finish_terminal_consumed_retirement
            from .fresh_experiment import retire_caches
            retire(root,job,manifest['public_fragment_retirement'])
            retire_caches(root,job,ticket['project_id'],manifest['terminal_cache_retirement'])
            if manifest.get('source_consumed_cache_retirement_acks'):
                finish_terminal_consumed_retirement(root,job,ticket['project_id'],manifest['existing_restore_source'],
                    manifest['source_postcommit_recovery'],manifest['source_consumed_cache_retirement_acks'])
            admit(root,manifest['post_retirement_forecast_bytes'])
            command=[python,'-B','-m','ptcg_lab.cpu_sampling_check','--ticket',str(ticket_path)]
            execute(command,deadline-100);execute(command+['--fresh'],deadline-70)
            report=read_json(job/'CPU-sampling-audit.json');fresh=read_json(job/'CPU-sampling-fresh.json')
            if not report.get('passed') or not fresh.get('same_CPU_next_action_logp_value_and_fixture_exact') or report.get('GPU_visible_devices')!=0 or report.get('PPO_updates')!=0:
                raise ValueError('Actual CPU sampler/recovery acceptance failed')
            metrics={'at':utcnow(),'complete':True,'engineering_only':True,'CPU_sampler_actual_acceptance_passed':True,
                'audit':report,'fresh_CPU_process':fresh,'engine':engine,'training_updates_executed':0,
                'model_selection_forbidden':True,'promotion_eligible':False}
        elif mode=='journal_io_check':
            execute([python,'-B','-m','ptcg_lab.server_io_check','--ticket',str(ticket_path)],deadline-80)
            report=read_json(job/'journal-io-audit.json')
            if not report.get('passed') or report.get('optimizer_steps')!=0 or report.get('GPU_visible_devices')!=0:
                raise ValueError('Actual CPU journal/IO check failed')
            metrics={'at':utcnow(),'complete':True,'engineering_only':True,'actual_journal_and_checkpoint_IO_passed':True,
                'audit':report,'engine':engine,'training_updates_executed':0,'model_selection_forbidden':True,'promotion_eligible':False}
        elif mode=='basic_cpu_check':
            config=original['audit_config'];config.update(project_path=str(root),engine_manifest_sha256=engine_hash,
                audit_device='cpu',write_actor_fixture=True,replay_path=str(source/'inputs/parity-replay.json.gz'))
            atomic_json(job/'payload-config.json',config)
            command=[python,'-B','-m','ptcg_lab.rl_audit','--config',str(job/'payload-config.json'),'--output',str(job)]
            execute(command,deadline-100);execute(command,deadline-100)
            report=read_json(job/'audit.json')
            if (not report.get('passed') or not report.get('complete') or not report['recovery']['exact_next_update_match']
                    or report['trajectory']['real_official_bo3']!=2 or report['parity']['greedy_action_checks']!=48):
                raise ValueError('Full basic CPU correctness acceptance failed')
            metrics={'at':utcnow(),'complete':True,'engineering_only':True,'basic_cpu_correctness_passed':True,
                'parent_model_sha256':config['parent_model_sha256'],'audit_sha256':sha256(job/'audit.json'),
                'actor_fixture_sha256':sha256(job/'actor-parity-fixture.pt'),'actual_cpu_checkpoint_restart_update_equal':True,
                'audit':report,'gpu_training_verified':False,'source_manifest_sha256':ticket['manifest_sha256'],
                'engine':engine,'promotion_eligible':False}
        elif mode=='fresh_scientific':
            from .fresh_experiment import prepare
            config=prepare(root,job,ticket,manifest,engine_hash)
            command=[python,'-B','-m','ptcg_lab.train_ppo','--config',str(job/'payload-config.json'),'--output',str(job)]
            execute(command+['--bootstrap-only'],deadline-600)
            pause_proof=read_json(job/'cooperative-pause-acceptance.json')
            bootstrap=read_json(job/'fresh-gpu-bootstrap.json')
            if not pause_proof.get('complete') or not bootstrap.get('complete'):raise ValueError('Early pause and full fresh actor bootstrap are required before scientific updates')
            execute(command,deadline-600)
            training=read_json(job/'metrics.json') if (job/'metrics.json').exists() else {}
            restored=read_json(job/'fresh-gpu-restore.json')
            if not restored.get('same_GPU_fresh_process_rng_exact'):raise ValueError('Fresh same-GPU model/optimizer/RNG restore failed')
            metrics={**training,'at':utcnow(),'complete':bool(training.get('complete')),'engineering_only':False,
                'training_only':True,'scientific_training':True,'scientific_continuation':False,
                'fresh_initialization':read_json(job/'fresh-experiment-inheritance.json'),
                'cooperative_pause_acceptance':pause_proof,'fresh_GPU_restore':restored,
                'gpu_index':ticket['gpu_index'],'gpu_uuid':ticket['gpu_uuid'],
                'actual_config':config,'last_training_progress':read_json(job/'progress.json'),
                'engine':engine,'promotion_eligible':False,'validation_required':'Local independent complete-BO3 screen and confirmation'}
        elif mode=='derived_scientific':
            if manifest.get('new_diagnostic_identity')=='TR12':
                from .volume_extension import prepare
            elif manifest.get('new_diagnostic_identity')=='TR11':
                from .continuous_mainline import prepare
            else:
                from .scientific_resume import prepare
            config=prepare(root,job,ticket,manifest,engine_hash)
            phase_deadline=deadline-(60 if manifest.get('parent_acceptance_only') else 600)
            execute([python,'-B','-m','ptcg_lab.scientific_resume','--ticket',str(ticket_path)],phase_deadline)
            execute([python,'-B','-m','ptcg_lab.train_ppo','--config',str(job/'payload-config.json'),'--output',str(job)],phase_deadline)
            training=read_json(job/'metrics.json') if (job/'metrics.json').exists() else {}
            restored=read_json(job/'derived-gpu-restore.json')
            if not restored.get('same_GPU_fresh_process_rng_and_outputs_exact'):raise ValueError('Derived actual GPU restoration failed')
            metrics={**training,'at':utcnow(),'complete':bool(training.get('complete')),'engineering_only':False,
                'training_only':True,'scientific_training':True,'scientific_continuation':True,
                'derived_inheritance':read_json(job/'derived-inheritance.json'),'derived_GPU_restore':restored,
                'gpu_index':ticket['gpu_index'],'gpu_uuid':ticket['gpu_uuid'],'actual_config':config,
                'last_training_progress':read_json(job/'progress.json'),'engine':engine,'promotion_eligible':False,
                'validation_required':'Local independent complete-BO3 screen and confirmation'}
            if manifest.get('parent_acceptance_only'):
                if training.get('optimizer_steps_executed')!=0 or not training.get('full_parent_archive_ACKs_verified'):
                    raise ValueError('Parent acceptance cannot train or finish before complete local archive ACKs')
                metrics.update(engineering_only=True,training_only=False,scientific_training=False,
                    scientific_continuation=False,parent_acceptance_passed=True,model_selection_forbidden=True)
        elif mode=='scientific_continue':
            from .hybrid_continue import prepare
            config=prepare(root,job,ticket,manifest,engine_hash)
            execute([python,'-B','-m','ptcg_lab.train_ppo','--config',str(job/'payload-config.json'),'--output',str(job)],deadline-600)
            training=read_json(job/'metrics.json') if (job/'metrics.json').exists() else {}
            progress_record=read_json(job/'progress.json')
            metrics={**training,'at':utcnow(),'complete':bool(training.get('complete')),
                'engineering_only':False,'training_only':True,'scientific_continuation':True,
                'original_TR04_inheritance':read_json(job/'continuation-inheritance.json'),
                'last_training_progress':progress_record,'engine':engine,'promotion_eligible':False,
                'validation_required':'Local independent frozen BO3 comparison; no promotion from training metrics'}
        elif mode=='live_archive_check' and manifest.get('continuous_archive_reuse'):
            execute([python,'-B','-m','ptcg_lab.continuous_archive_accept','--ticket',str(ticket_path)],deadline-60)
            report=read_json(job/'archive-reuse-audit.json')
            if not report['complete'] or not report['full_parent_archive_ACKs_verified']:raise ValueError('Full reused-parent ACK acceptance failed')
            metrics={**report,'at':utcnow(),'engine':engine,'model_selection_forbidden':True,'promotion_eligible':False,
                'training_updates_executed':0,'full_long_unattended_recovery_and_archive_acceptance':False}
        elif mode=='live_archive_check':
            execute([python,'-B','-m','ptcg_lab.archive_live_check','--ticket',str(ticket_path)],deadline-80)
            report=read_json(job/'archive-live-audit.json')
            if not report.get('passed'):raise ValueError('Actual archive protocol acceptance failed')
            metrics={'at':utcnow(),'complete':True,'engineering_only':True,'actual_live_archive_passed':True,
                'archive_live_audit':report,'engine':engine,'model_selection_forbidden':True,
                'training_updates_executed':0,'full_long_unattended_recovery_and_archive_acceptance':False}
            if manifest.get('ack_lag_barrier_check'):
                if report.get('actual_ACK_lag_barrier_passed') is not True:raise ValueError('Actual production barrier check missing')
                metrics['actual_ACK_lag_barrier_passed']=True
        elif mode=='gpu_existing_restore':
            module='ptcg_lab.prepared_parent_verify' if manifest.get('prepared_parent_verify') else 'ptcg_lab.gpu_restore'
            command=[python,'-B','-m',module,'--ticket',str(ticket_path)]
            execute(command,deadline-80);execute(command+['--fresh'],deadline-60)
            if manifest.get('prepared_parent_verify'):
                from .archive_outbox import ArchiveOutbox
                box=ArchiveOutbox(job,job.name,ticket['lease_epoch'])
                while box.pending()['pending_objects']:
                    if time.time()>deadline-35:raise TimeoutError('Prepared inspection archive ACK deadline')
                    time.sleep(.2)
                restored=read_json(job/'derived-gpu-restore.json')
                if not restored.get('same_GPU_fresh_process_rng_and_outputs_exact'):raise ValueError('Prepared two-process numerical restore failed')
                report={'passed':True,'fresh_process_verified':True,'same_GPU_outputs_and_rng_exact':True,
                    'prepared_parent_readonly_inspection':True}
                atomic_json(job/'gpu-restore-audit.json',report)
            else:report=read_json(job/'gpu-restore-audit.json')
            if not report.get('passed') or not report.get('fresh_process_verified') or not report.get('same_GPU_outputs_and_rng_exact'):
                raise ValueError('Original TR04 GPU checkpoint/RNG fresh-process restore failed')
            metrics={'at':utcnow(),'complete':True,'engineering_only':True,'original_TR04_CUDA_restore_verified':True,
                'gpu_restore_audit':report,'engine':engine,'model_selection_forbidden':True,'promotion_eligible':False,
                'training_updates_executed':0,'full_long_unattended_recovery_and_archive_acceptance':False}
            if manifest.get('prepared_parent_verify'):
                metrics.pop('original_TR04_CUDA_restore_verified',None)
                metrics.update(parent_acceptance_passed=True,optimizer_steps_executed=0,new_unique_PPO_trained_decisions=0,
                    full_parent_archive_ACKs_verified=True,original_preparation_still_failed=True)
        elif mode in ('compact_cpu_restore','compact_cpu_existing'):
            command=[python,'-B','-m','ptcg_lab.restore_capsule','--ticket',str(ticket_path)]
            if mode=='compact_cpu_restore':execute(command,deadline-100)
            else:
                source_path=job/'existing-source.json';atomic_json(source_path,manifest['existing_restore_source'])
                command+=['--existing-source-json',str(source_path)]
            execute(command+['--fresh'],deadline-100)
            report=read_json(job/'restore-audit.json')
            if not report.get('complete') or not report.get('linux_full_compact_restore_verified') or not report.get('fresh_process_verified'):
                raise ValueError('Actual Linux compact checkpoint restore failed')
            metrics={'at':utcnow(),'complete':True,'engineering_only':True,'linux_full_compact_restore_verified':True,
                'restore_audit':report,'engine':engine,'promotion_eligible':False,'gpu_training_verified':False,
                'full_long_unattended_recovery_and_archive_acceptance':False,'model_selection_forbidden':True}
        else:
            # The GPU-only fixture subprocess exits before the learner starts.
            execute([python,'-B','-m','ptcg_lab.server_trial_worker','--ticket',str(ticket_path),'--gpu-fixture-only'],deadline-100)
            config=original['training_config'];config.update(project_path=str(root),engine_manifest_sha256=engine_hash,
                target_decisions=1000000000,hypothesis_id='ENGINEERING_SHORT_TRIAL_NOT_TR04',
                engineering_only=True,model_selection_forbidden=True)
            for spec in config['league']['fixed_opponents']:
                spec['files']={name.replace('\\','/'):digest for name,digest in spec['files'].items()}
            config['ppo']['rollout_complete_learner_decisions']=256
            atomic_json(job/'payload-config.json',config)
            child_started=time.monotonic()
            execute([python,'-B','-m','ptcg_lab.train_ppo','--config',str(job/'payload-config.json'),'--output',str(job)],deadline-120)
            child_seconds=time.monotonic()-child_started
            execute([python,'-B','-m','ptcg_lab.server_trial_worker','--ticket',str(ticket_path),'--checkpoint-roundtrip-only'],deadline-60)
            restored=read_json(job/'checkpoint-roundtrip.json')
            summaries=[read_json(path) for path in sorted((job/'series').glob('[0-9]'*6+'.json'))]
            loop_progress=read_json(job/'progress.json')
            if child_seconds<300 or loop_progress['seconds_this_attempt']<300 or restored['version']<1 or not summaries or any(row['statuses']!=['DONE','DONE'] for row in summaries):
                raise ValueError('Short trial lacked300s, a PPO update, or normal complete BO3')
            if not restored['passed'] or restored['model_tensors']!=127 or restored['optimizer_states']!=97:
                raise ValueError('Incomplete full actor/optimizer/single-GPU RNG checkpoint')
            metrics={'at':utcnow(),'complete':True,'engineering_only':True,'short_gpu_trial_passed':True,
                'gpu_uuid':ticket['gpu_uuid'],'gpu_index':ticket['gpu_index'],'learner_process_seconds':child_seconds,
                'training_loop_seconds':loop_progress['seconds_this_attempt'],'completed_bo3':len(summaries),
                'bo3_per_minute':len(summaries)/child_seconds*60,'trained_decisions':restored['trained_decisions'],
                'ppo_updates':restored['version'],'checkpoint_sha256':restored['checkpoint_sha256'],
                'model_tensor_count':restored['model_tensors'],'optimizer_states':restored['optimizer_states'],
                'learner_cuda_peak_allocated_bytes':loop_progress['cuda_peak_allocated_bytes'],
                'learner_cuda_peak_reserved_bytes':loop_progress['cuda_peak_reserved_bytes'],
                'checkpoint_contains_python_numpy_torch_and_cuda_rng':True,'parent_model_sha256':manifest['parent_model_sha256'],
                'fresh_gpu_checkpoint_restore':restored,
                'gpu_basic_fixture':read_json(job/'gpu-fixture-check.json'),'league':read_json(job/'league.json'),
                'actual_config':config,'parent_capsule':manifest['parent_capsule'],'engine':engine,
                'scientific_strength_or_promotion_evidence':False,'model_selection_forbidden':True,
                'full_long_unattended_recovery_and_archive_acceptance':False,'all_raw_outputs_retained_on_server':True}
        final_capacity=capacity();final_kernel=kernel_snapshot(ticket)
        metrics.update(peak_workspace_bytes=peak_workspace,minimum_free_bytes=minimum_free,
                       kernel_initial=protection,kernel_final=final_kernel,
                       cgroup_memory_peak_bytes=int(final_kernel['memory.peak']),own_service_gpu_memory_peak_mib=own_gpu_peak_mib,
                       project_growth_bytes=final_capacity['workspace_bytes']-initial_capacity['workspace_bytes'],
                       resource_samples=resource_samples,promotion_eligible=False)
        atomic_json(job/'trial-metrics.json',metrics)
    except Exception as exc:
        atomic_json(job/'failure.json',{'at':utcnow(),'error':type(exc).__name__+': '+str(exc),
            'peak_workspace_bytes':peak_workspace,'minimum_free_bytes':minimum_free,'no_promotion_claim':True})
        raise
    finally:
        stop_child()
        if active_clock:active_clock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ticket',required=True);parser.add_argument('--gpu-fixture-only',action='store_true');parser.add_argument('--checkpoint-roundtrip-only',action='store_true')
    args=parser.parse_args()
    if args.gpu_fixture_only:gpu_fixture_check(read_json(args.ticket))
    elif args.checkpoint_roundtrip_only:checkpoint_roundtrip(read_json(args.ticket))
    else:run(args.ticket)


if __name__=='__main__':main()
