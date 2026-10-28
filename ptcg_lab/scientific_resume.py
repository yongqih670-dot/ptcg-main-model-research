"""One registered original continuation, preserving samples, attempts and RNG."""
import copy
import os
from pathlib import Path
import time
from .io import atomic_json, read_json, sha256, utcnow
from .archive_prefix import canonical_hash


def validate_postcommit_recovery(source,manifest):
    """An exact terminal failure after two full publications is recoverable.

    This does not relabel the failed attempt or authorize another attempt.
    It accepts only the recorded TR07 eviction-receipt disk-reserve failure.
    """
    proof=manifest.get('source_postcommit_recovery')
    if not proof or source.name!='20261025T212721Z_tr07-v01_2f8e6f':
        raise ValueError('Unreviewed post-publication recovery source')
    log=source/'logs/payload.log'
    if sha256(log)!=proof['payload_log_sha256']:
        raise ValueError('Original post-publication failure bytes changed')
    text=log.read_text(encoding='utf-8')
    if 'outbox.evict_consumed(' not in text or 'OSError: Hybrid filesystem reserve reached' not in text:
        raise ValueError('Source did not fail in the exact archived-replica retirement path')
    index=read_json(source/'checkpoint-commits/index.json');points=index['commits']
    if len(points)!=2 or [x['checkpoint']['sha256'] for x in points]!=proof['checkpoint_hashes']:
        raise ValueError('Both original published checkpoints must match the complete local forensic proof')
    if any(p['cursor']!=proof['cursor'] or p['pending'] for p in points):
        raise ValueError('Post-publication recovery cursor or consumed-tail coverage differs')
    if proof['cursor']!={'version':2,'trained_decisions':16646,'next_series_id':102}:
        raise ValueError('This proof cannot authorize another incomplete or partial update')
    return proof


def validate_scope(project, config, deadline, current_run_id=None):
    project = Path(project)
    original = project/'runs'/config['original_training_run_id']
    names={'TR05':'20261025T174630Z_tr05-value-head-only_0d9af6',
           'TR07':'20261025T212721Z_tr07-v01_2f8e6f',
           'TR08':'20261026T085757Z_tr08-actor-lr3e6-CPU-sampling_868ff0',
           'TR09':'20261026T131453Z_tr09-epochs4-CPU32_382e9e',
           'TR10':'20261027T073939Z_tr10-lambda1-diagnostic_df4705'}
    hypothesis=config.get('hypothesis_id')
    if hypothesis not in names or original.name != names[hypothesis]:
        raise ValueError('Continuation must name the registered first original attempt')
    previous = read_json(original/'config.json')
    attempts = read_json(original/'attempts.json')
    if (previous.get('check_variant'), previous.get('hypothesis_id')) != ('fresh_scientific', hypothesis):
        raise ValueError('Continuation parent is not the original fresh hypothesis')
    if len(attempts) != 1 or abs(attempts[0]['seconds']-config['prior_training_seconds']) > .01:
        raise ValueError('Original training expenditure changed or is incomplete')
    if hypothesis=='TR10':
        from .ppo_accounting import ACCOUNTING
        proof=config['source_clean_checkpoint_recovery'];ref=proof['local_audit'];path=(project/ref['path']).resolve()
        if project.resolve() not in path.parents or sha256(path)!=ref['sha256']:
            raise ValueError('Actual full TR10 cold recovery audit changed')
        audit=read_json(path)
        if (audit.get('classification')!='TR10_original_checkpoint_full_recovery_audited'
                or not audit.get('complete') or audit.get('source_run_id')!=original.name
                or not audit.get('all_full_model_optimizer_RNG_checkpoint_payloads_verified')
                or not audit.get('actor_information_isolation_passed') or audit.get('remaining_original_attempts')!=1
                or audit.get('decision_accounting')!=ACCOUNTING or config.get('actual_decision_accounting')!=ACCOUNTING
                or attempts[0].get('status')!='checkpointed' or attempts[0].get('exit_code')!=0
                or abs(audit['first_attempt_seconds']-attempts[0]['seconds'])>.01
                or audit['pending_decisions']<48):
            raise ValueError('Original full-state TR10, exact actual coverage and unspent second required')
        for key in ['source_progress_sha256','source_index_sha256','source_payload_config_sha256']:
            if proof[key]!=audit[key]:raise ValueError('Full audited TR10 terminal source changed')
        point=audit['source_point'];metrics=read_json(original/'metrics.json')
        if (metrics.get('controlled_outcome')!='checkpointed' or metrics.get('complete') is not False
                or any(metrics['last_training_progress'].get(k)!=v for k,v in point['cursor'].items())
                or config['existing_restore_source']['checkpoint_reference']!=point['checkpoint']
                or config['existing_restore_source']['cursor']!=point['cursor']
                or config['source_league_sha256']!=audit['source_league_sha256']
                or config.get('source_inflight')!=audit['source_inflight']
                or any(config.get(k) for k in ['additional_complete_tail','source_transport_recovery',
                    'source_postcommit_recovery','terminal_cache_retirement','numeric_reference'])):
            raise ValueError('TR10 cannot change its full audited source or borrow another recovery scope')
    if hypothesis=='TR08':
        if attempts[0]['status']!='failed' or abs(attempts[0]['seconds']-5330.86)>.01:
            raise ValueError('TR08 original transport-failed attempt expenditure differs')
        proof=config['source_transport_recovery'];audit=read_json(project/proof['local_audit']['path'])
        if sha256(project/proof['local_audit']['path'])!=proof['local_audit']['sha256'] or not audit.get('complete') or audit.get('source_run_id')!=original.name:
            raise ValueError('Full TR08 terminal forensic audit changed')
        if sha256(original/'stderr-01.log')!=audit['first_error_sha256'] or 'SSH response deadline for archive_chunk' not in (original/'stderr-01.log').read_text(encoding='utf-8'):
            raise ValueError('Different original controller failure')
        if config['additional_complete_tail']!=audit['additional_complete_tail']:
            raise ValueError('Complete post-checkpoint BO3 tail was omitted or changed')
    if hypothesis=='TR09':
        from .tr09_recovery_analysis import validate_clean_contract,proved_wrapper_failure
        proof=config['source_clean_checkpoint_recovery'];ref=proof['local_audit'];path=project/ref['path']
        if project.resolve() not in path.resolve().parents or sha256(path)!=ref['sha256']:
            raise ValueError('Actual full clean TR09 recovery audit changed')
        audit=read_json(path)
        if (audit.get('classification')!='TR09_original_checkpoint_full_recovery_audited'
                or not audit.get('complete') or audit.get('source_run_id')!=original.name
                or not audit.get('all_full_model_optimizer_RNG_checkpoint_payloads_verified')
                or not audit.get('reference_fixture_never_enters_training')):
            raise ValueError('Actual clean full-state TR09 evidence required')
        raw=project/'runs'/audit['archive_run_id']/'archive';wrapper_proof=proved_wrapper_failure(original,raw)
        if wrapper_proof!=audit.get('controller_post_checkpoint_failure_proof'):raise ValueError('Original wrapper-only failure proof changed')
        # The exact archived wrapper failure occurred before controller metrics
        # publication. Its authenticated payload report is the original evidence;
        # do not manufacture metrics.json or relabel the failed first attempt.
        metrics_path=original/'metrics.json'
        if not metrics_path.exists() and wrapper_proof is None:
            raise ValueError('Missing controller metrics without proved wrapper failure')
        metrics=read_json(metrics_path if metrics_path.exists() else raw/'trial-metrics.json')
        validate_clean_contract(original.name,previous,attempts,metrics,
            read_json(raw/'progress.json'),audit['source_point']['cursor'],allow_proved_wrapper_failure=wrapper_proof is not None)
        if config['numeric_reference']!=audit['numeric_reference'] or config.get('additional_complete_tail') or config.get('source_transport_recovery') or config.get('terminal_cache_retirement'):
            raise ValueError('TR09 clean continuation cannot borrow old failure/tail/cache scopes')
        numeric=project/audit['numeric_reference']['path']
        if project.resolve() not in numeric.resolve().parents or numeric.stat().st_size!=audit['numeric_reference']['bytes'] or sha256(numeric)!=audit['numeric_reference']['sha256']:
            raise ValueError('Exact recovery-only numeric reference changed')
        if config['existing_restore_source']['checkpoint_reference']!=audit['source_point']['checkpoint'] or config['existing_restore_source']['cursor']!=audit['source_point']['cursor']:
            raise ValueError('Continuation changed its fully audited checkpoint')
    if (config.get('training_attempt_index'), config.get('maximum_attempts')) != (2, 1):
        raise ValueError('Only the original second training attempt remains')
    seconds = deadline-time.time()
    if not 0 < seconds <= 7200.5 or attempts[0]['seconds']+seconds > 14400.5:
        raise ValueError('Original per-attempt/total training time exceeded')
    from .remote_lease import validate_no_unreconciled_remote
    validate_no_unreconciled_remote(project, {'remote_job':read_json(original/'remote-job.json')})
    for run in (project/'runs').iterdir():
        path = run/'config.json'
        if path.exists() and read_json(path).get('check_variant') == 'derived_scientific' and read_json(path).get('hypothesis_id')==hypothesis:
            if run.name==current_run_id and read_json(path)==config:continue
            raise ValueError('Original second attempt already has a raw run')


def prepare(root, job, ticket, manifest, engine_hash):
    from .retained_source import resolve, checkpoint_reference
    from .server_workspace import inside, admit
    from .checkpoint_refs import _copy_immutable, commit_checkpoint
    from .training_checkpoint import load_training_checkpoint
    source, original = resolve(root, ticket['project_id'], manifest['existing_restore_source'])
    point = checkpoint_reference(source, manifest['existing_restore_source'])
    if original['mode'] != 'fresh_scientific' or manifest['training_config'] != original['training_config']:
        raise ValueError('Continuation changed the scientific configuration or parent role')
    clean={'checkpointed_for_control','checkpointed_at_soft_deadline','checkpointed_before_update_at_soft_deadline'}
    if manifest['training_config'].get('hypothesis_id')=='TR09':
        proof=manifest.get('source_clean_checkpoint_recovery')
        if source.name!='20261026T131453Z_tr09-epochs4-CPU32_382e9e' or not proof or not manifest.get('numeric_reference'):
            raise ValueError('Only the original clean TR09 and exact recovery reference are admitted')
        for relative,key in [('progress.json','source_progress_sha256'),('checkpoint-commits/index.json','source_index_sha256'),('payload-config.json','source_payload_config_sha256')]:
            if sha256(source/relative)!=proof[key]:raise ValueError('Clean terminal TR09 bytes changed')
        progress=read_json(source/'progress.json')
        phases=clean|{'partial_update_abandoned_at_deadline','partial_update_abandoned_for_control'}
        if progress.get('stage') not in phases or any(progress.get(k)!=v for k,v in point['cursor'].items()) or sha256(source/'checkpoint.pt')!=point['checkpoint']['sha256']:
            raise ValueError('Clean TR09 published full point differs from terminal cursor')
        if manifest.get('source_transport_recovery') or manifest.get('source_postcommit_recovery') or manifest.get('additional_complete_tail'):
            raise ValueError('TR09 cannot reinterpret an old transport or postcommit failure')
    elif manifest['training_config'].get('hypothesis_id')=='TR10':
        from .ppo_accounting import ACCOUNTING,checkpoint_decisions
        proof=manifest.get('source_clean_checkpoint_recovery')
        if (source.name!='20261027T073939Z_tr10-lambda1-diagnostic_df4705' or not proof
                or manifest.get('actual_decision_accounting')!=ACCOUNTING):
            raise ValueError('Only exact audited original TR10 actual-count continuation admitted')
        for relative,key in [('progress.json','source_progress_sha256'),('checkpoint-commits/index.json','source_index_sha256'),('payload-config.json','source_payload_config_sha256')]:
            if sha256(source/relative)!=proof[key]:raise ValueError('Clean terminal TR10 bytes changed')
        progress=read_json(source/'progress.json')
        if (progress.get('stage') not in clean or any(progress.get(k)!=v for k,v in point['cursor'].items())
                or sha256(source/'checkpoint.pt')!=point['checkpoint']['sha256']):
            raise ValueError('Original TR10 published state differs from its terminal cursor')
        from .tensor_io import load_torch
        if checkpoint_decisions(load_torch(source/point['checkpoint']['path'],map_location='cpu',weights_only=False))>=32768:
            raise ValueError('Completed actual target cannot use incomplete continuation')
    elif read_json(source/'progress.json').get('stage') not in clean:
        if manifest.get('source_transport_recovery'):
            proof=manifest['source_transport_recovery']
            if source.name!='20261026T085757Z_tr08-actor-lr3e6-CPU-sampling_868ff0' or manifest['training_config'].get('hypothesis_id')!='TR08':
                raise ValueError('Unreviewed transport recovery source')
            for relative,key in [('progress.json','source_progress_sha256'),('checkpoint-commits/index.json','source_index_sha256'),('payload-config.json','source_payload_config_sha256')]:
                if sha256(source/relative)!=proof[key]:raise ValueError('Cold terminal TR08 source changed')
            progress=read_json(source/'progress.json')
            if (progress['stage'],progress['version'],progress['trained_decisions'],progress['pending_decisions'],progress['next_series_id'])!=('collecting',2,16523,6015,141):
                raise ValueError('Only the exact collecting-state transport failure is recoverable')
        else:validate_postcommit_recovery(source,manifest)
    if manifest.get('terminal_cache_retirement'):
        if manifest['training_config'].get('hypothesis_id')!='TR08' or not manifest.get('source_transport_recovery'):
            raise ValueError('Only the exact TR08 continuation may use this terminal-cache plan')
        allowed={'20261025T212721Z_tr07-v01_2f8e6f','20261025T233324Z_tr07-CPU-sampler-startup-repair_bb99ae'}
        if any(p['source']['run_id'] not in allowed for p in manifest['terminal_cache_retirement']):
            raise ValueError('Unreviewed old terminal cache owner')
        from .fresh_experiment import retire_caches
        retire_caches(root,job,ticket['project_id'],manifest['terminal_cache_retirement'])
    admit(root, manifest.get('post_retirement_forecast_bytes',manifest['reserved_future_bytes']))
    for ref, relative in [(point['checkpoint'], 'checkpoint.pt'), (point['prefix'], 'archive-prefix.json'),
                          *[(r, r['path']) for r in point['pending']],
                          *[(r,r['path']) for r in manifest.get('additional_complete_tail',[])]]:
        src = inside(root, source/ref['path'])
        if src.stat().st_size != ref['bytes'] or sha256(src) != ref['sha256']:
            raise ValueError('Inherited checkpoint or complete BO3 bytes changed')
        dst = inside(root, job/relative);dst.parent.mkdir(parents=True,exist_ok=True)
        # Both names are owned and immutable, within this same project. Future
        # checkpoint saves use atomic replacement; tail eviction unlinks only
        # the derived name. The terminal parent's bytes remain available.
        before = src.stat()
        os.link(src,dst)
        after = src.stat();linked = dst.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or (linked.st_dev,linked.st_ino)!=(before.st_dev,before.st_ino):
            raise ValueError('Inherited immutable link identity changed')
    league_path = inside(root, source/'league.json')
    if sha256(league_path) != manifest['source_league_sha256']:
        raise ValueError('Original full league changed')
    league = read_json(league_path)
    members = {}
    for row in league['snapshots']:
        path = inside(root, source/row['path'])
        if sha256(path) != row['sha256']:raise ValueError('Historical opponent bytes changed')
        members[row['path']] = row['sha256']
    if not members:raise ValueError('Original frozen league is missing')
    atomic_json(job/'league.json', league)
    inflight = manifest.get('source_inflight')
    if inflight:
        path = inside(root, source/'inflight.json')
        if sha256(path) != inflight['sha256'] or path.stat().st_size != inflight['bytes']:
            raise ValueError('Interrupted BO3 record changed')
        _copy_immutable(path, inside(root, job/'inflight.json'))
    prefix = read_json(job/'archive-prefix.json')
    spec = {'prefix_path':'archive-prefix.json', 'prefix_sha256':point['prefix']['sha256'],
            'initial_checkpoint_sha256':prefix['checkpoint']['sha256']}
    # A derived owner gets its own generation zero. Copy the complete payload;
    # never adopt a hot checkpoint or rewrite the immutable scientific origin.
    derived = commit_checkpoint(job, job.name, ticket['lease_epoch'], 0, 'checkpoint.pt',
                                'archive-prefix.json', [r['path'] for r in point['pending']], point['cursor'])
    load_training_checkpoint(job, spec, ticket['lease_epoch'])
    config = copy.deepcopy(manifest['training_config'])
    config.update(project_path=str(root), engine_manifest_sha256=engine_hash,
        archive_restore=spec, checkpoint_interval_seconds=600,
        retire_consumed_with_second_checkpoint=True, cooperative_resource_control=True,
        derived_resume_runtime_audit=True, online_archive={'maximum_pending_bytes':512*2**20})
    if manifest.get('sampling_device')=='cpu':
        if config.get('hypothesis_id') not in ('TR07','TR08','TR09','TR10') or not manifest.get('CPU_sampler_acceptance_sha256'):
            raise ValueError('CPU collection requires exact recorded server acceptance')
        config.update(sampling_device='cpu',checkpoint_transaction_hash_reuse=True)
        if config['hypothesis_id']=='TR10':
            from .ppo_accounting import ACCOUNTING
            if (not manifest.get('consumed_reference_barrier_required') or not manifest.get('ACK_lag_barrier_acceptance_sha256')
                    or manifest.get('actual_decision_accounting')!=ACCOUNTING):
                raise ValueError('TR10 original archive barrier and actual counting remain mandatory')
            config.update(CPU_GPU_batch_parity_required=True,CPU_rollout_value_parity_required=True,
                consumed_reference_barrier_required=True,actual_decision_accounting=ACCOUNTING)
        elif config['hypothesis_id']=='TR09':
            if not manifest.get('consumed_reference_barrier_required') or not manifest.get('ACK_lag_barrier_acceptance_sha256'):
                raise ValueError('Original TR09 ACK retirement barrier remains mandatory')
            numeric=manifest['numeric_reference'];fixture=inside(root,job/'stage'/(numeric['sha256']+'.blob'))
            if fixture.stat().st_size!=numeric['bytes'] or sha256(fixture)!=numeric['sha256']:
                raise ValueError('TR09 exact numeric reference transport changed')
            from .tensor_io import load_torch
            value=load_torch(fixture,map_location='cpu',weights_only=False)
            if (value.get('role')!='TR09_engineering_recovery_reference_only_not_training'
                    or value.get('source_run_id')!=source.name or value.get('source_checkpoint_sha256')!=point['checkpoint']['sha256']):
                raise ValueError('Numeric reference cannot replace current scientific data')
            config.update(CPU_backend_fixture=str(fixture),CPU_GPU_batch_parity_required=True,
                CPU_rollout_value_parity_required=True,consumed_reference_barrier_required=True)
        elif config['hypothesis_id']=='TR08':
            if not manifest.get('source_transport_recovery') or len(manifest.get('additional_complete_tail',[]))!=18:
                raise ValueError('TR08 requires its full original transport-failure tail')
            config.update(CPU_GPU_batch_parity_required=True,CPU_rollout_value_parity_required=True)
        else:
            cpu_source,cpu_manifest=resolve(root,ticket['project_id'],manifest['CPU_backend_source'])
            fixture=inside(root,cpu_source/'CPU-sampling-reference.pt')
            cpu_audit=read_json(cpu_source/'CPU-sampling-audit.json')
            if (cpu_manifest.get('mode')!='cpu_sampler_check' or not cpu_audit.get('passed')
                    or cpu_audit.get('source_run_id')!=source.name
                    or sha256(fixture)!=manifest['CPU_backend_fixture_sha256']):
                raise ValueError('Actual terminal CPU backend acceptance or current-model fixture differs')
            config['CPU_backend_fixture']=str(fixture)
    config['league']['retained_history'] = {'root':str(source), 'members':members}
    for fixed in config['league']['fixed_opponents']:
        fixed['files'] = {k.replace('\\','/'):v for k,v in fixed['files'].items()}
    atomic_json(job/'payload-config.json', config)
    atomic_json(job/'derived-inheritance.json', {'at':utcnow(), 'hypothesis_id':config['hypothesis_id'],
        'original_training_attempt_index':2, 'source_run_id':source.name,
        'source_point':point, 'derived_point':derived, 'prefix_unchanged':True,
        'full_checkpoint_and_pending_bytes_unchanged':True, 'all_history_members':members,
        'source_postcommit_recovery':manifest.get('source_postcommit_recovery'),
        'source_transport_recovery':manifest.get('source_transport_recovery'),
        'source_clean_checkpoint_recovery':manifest.get('source_clean_checkpoint_recovery'),
        'additional_complete_tail':manifest.get('additional_complete_tail',[]),
        'no_fresh_initializer':True, 'interrupted_BO3_excluded':bool(inflight),
        'total_training_wall_seconds_max':14400, 'per_attempt_seconds_max':7200})
    return config


def audit_runtime(output, model, optimizer, saved, decisions, features, gpu_uuid, reference=False, fixture_reference=None,cpu_rollout_values=False):
    import numpy as np
    import torch
    from .rl_core import rng_state, restore_rng, evaluate_decisions
    from .fresh_compact import _draws
    from .tensor_io import load_torch, atomic_torch_save
    output = Path(output)
    if len(saved['model']) != 127 or len(saved['optimizer']['state']) not in (0,97):
        raise ValueError('Full model/optimizer coverage differs')
    current = model.state_dict()
    if set(current) != set(saved['model']):raise ValueError('Inherited model keys differ')
    for key, value in current.items():
        if not torch.equal(value.cpu(), saved['model'][key]) or not torch.isfinite(value).all():
            raise ValueError('Inherited model tensor differs')
    observed = optimizer.state_dict()
    if observed['param_groups'] != saved['optimizer']['param_groups'] or set(observed['state']) != set(saved['optimizer']['state']):
        raise ValueError('Inherited optimizer groups/keys differ')
    for key, values in observed['state'].items():
        old = saved['optimizer']['state'][key]
        if set(values) != set(old):raise ValueError('Inherited optimizer slots differ')
        for name, value in values.items():
            if torch.is_tensor(value):
                if not torch.equal(value.cpu(), old[name]) or not torch.isfinite(value).all():raise ValueError('Inherited optimizer tensor differs')
            elif value != old[name]:raise ValueError('Inherited optimizer scalar differs')
    state = rng_state();draws = _draws(state)
    cpu_reference=None
    if fixture_reference is not None:
        cpu_reference=load_torch(fixture_reference,map_location='cpu',weights_only=False)
        if set(cpu_reference['model'])!=set(saved['model']) or any(not torch.equal(v,saved['model'][k]) for k,v in cpu_reference['model'].items()):
            raise ValueError('CPU backend reference is not the exact inherited current model')
    fixture = cpu_reference['fixture'] if cpu_reference is not None else decisions[:48]
    if len(fixture)!=48:raise ValueError('At least48 actual current-policy decisions required for strict recovery parity')
    with torch.no_grad():logp, values, _ = evaluate_decisions(model, fixture, features.b.STOP)
    error = float(np.max(np.abs(logp.cpu().numpy()-np.asarray([d['old_logp'] for d in fixture]))))
    if error > 1e-4 or not torch.isfinite(logp).all() or not torch.isfinite(values).all():
        raise ValueError('Derived conditional action logp parity exceeded original1e-4')
    value_error=None
    if cpu_rollout_values:
        value_error=float(np.max(np.abs(values.cpu().numpy()-np.asarray([d['old_value'] for d in fixture]))))
        if not np.isfinite(value_error) or value_error>2e-4:raise ValueError('CPU rollout value parity exceeded original2e-4')
    if cpu_reference is not None:
        value_error=float((values.cpu()-cpu_reference['fixture_values']).abs().max())
        if float((logp.cpu()-cpu_reference['fixture_logp']).abs().max())>1e-4 or value_error>2e-4:
            raise ValueError('Actual CPU/GPU conditional probability/value parity exceeded original tolerance')
    result = {'draws':draws, 'logp':logp.cpu(), 'values':values.cpu(), 'gpu_uuid':gpu_uuid,
              'cursor':{k:saved[k] for k in ['version','trained_decisions','next_series_id']}}
    path = output/'derived-gpu-reference.pt'
    if reference:atomic_torch_save(path, result, replace=False)
    else:
        previous = load_torch(path, map_location='cpu', weights_only=False)
        a = previous['draws'];b = draws
        if (previous['gpu_uuid'] != gpu_uuid or previous['cursor'] != result['cursor'] or a['python'] != b['python']
            or not np.array_equal(a['numpy'], b['numpy']) or not torch.equal(a['torch'], b['torch'])
            or not torch.equal(a['cuda'], b['cuda']) or not torch.equal(previous['logp'], result['logp'])
            or not torch.equal(previous['values'], result['values'])):
            raise ValueError('Derived same-GPU fresh-process output/RNG differs')
    restore_rng(state)
    atomic_json(output/('derived-gpu-bootstrap.json' if reference else 'derived-gpu-restore.json'),
        {'at':utcnow(), 'complete':True, 'model_tensors':127, 'optimizer_states':len(observed['state']),
         'conditional_logp_checks':48, 'conditional_logp_max_error':error, 'conditional_logp_tolerance':1e-4,
         'CPU_GPU_fixture_reference_used':cpu_reference is not None,'CPU_GPU_value_max_error':value_error,
         'CPU_GPU_value_tolerance':2e-4 if cpu_reference is not None or cpu_rollout_values else None,
         'cursor':result['cursor'], 'pending_decisions':len(decisions), 'gpu_uuid':gpu_uuid,
         'same_GPU_fresh_process_rng_and_outputs_exact':not reference, 'optimizer_steps_executed':0,
         'sampling_rng_restored_after_probe':True})


def audit_entry(ticket_path):
    import torch
    from .server_workspace import owned_root, inside
    from .training_control import TrainingControl
    from .training_checkpoint import load_training_checkpoint
    from .rl_resume import resume_journal
    from .rl_core import load_full_actor, optimizer_for, restore_rng
    ticket = read_json(ticket_path)
    root = owned_root(ticket['project_root'], ticket['project_id']);job = inside(root, ticket['job_directory'])
    if os.environ.get('PTCG_JOB_RUN_ID') != job.name or os.environ.get('CUDA_VISIBLE_DEVICES') != ticket['gpu_uuid']:
        raise ValueError('Derived GPU audit identity differs')
    control = TrainingControl(job, job.name, ticket['lease_epoch'], float(os.environ['PTCG_JOB_DEADLINE_UNIX']),
        lambda stage,**kw:atomic_json(job/'progress.json', {'at':utcnow(), 'stage':stage, **kw}))
    control.wait(90);config = read_json(job/'payload-config.json')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model, features, _ = load_full_actor(job/'control'/config['bundle_path'], config['parent_model_sha256'], 'cuda', device_guard=lambda:control.wait(90))
    if torch.cuda.device_count() != 1:raise ValueError('Derived audit must see exactly one GPU')
    settings = config['ppo'];optimizer = optimizer_for(model, settings['actor_learning_rate'], settings['value_head_learning_rate'])
    loaded = load_training_checkpoint(job, config['archive_restore'], ticket['lease_epoch']);saved = loaded['saved']
    model.load_state_dict(saved['model'], strict=True);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    journal = resume_journal(job, saved, loaded['archive_spec'], loaded['digest'])
    if journal['rng_after'] is not None:restore_rng(journal['rng_after'])
    control.wait(90)
    audit_runtime(job, model, optimizer, saved, journal['decisions'], features, ticket['gpu_uuid'], reference=True,
        fixture_reference=config.get('CPU_backend_fixture'),cpu_rollout_values=config.get('CPU_rollout_value_parity_required',False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser();parser.add_argument('--ticket', required=True)
    audit_entry(parser.parse_args().ticket)
