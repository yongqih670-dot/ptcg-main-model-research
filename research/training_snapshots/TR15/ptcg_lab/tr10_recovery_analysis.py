"""Read-only, local cold recovery audit of the original first TR10 attempt."""
import hashlib
import json
import os
from pathlib import Path
import random
import time

from .io import atomic_json, read_json, sha256, utcnow
from .ppo_accounting import ACCOUNTING, checkpoint_decisions, update_decisions

FIRST = '20261027T073939Z_tr10-lambda1-diagnostic_df4705'
CLEAN = {'checkpointed_for_control', 'checkpointed_at_soft_deadline',
         'checkpointed_before_update_at_soft_deadline'}


def validate_contract(config, attempts, metrics, progress, point):
    if (config.get('hypothesis_id'), config.get('check_variant'), config.get('training_attempt_index')) != ('TR10', 'fresh_scientific', 1):
        raise ValueError('Only original first TR10 is recoverable here')
    if (len(attempts) != 1 or attempts[0].get('status') != 'checkpointed'
            or attempts[0].get('exit_code') != 0 or not 0 < attempts[0]['seconds'] <= 7201):
        raise ValueError('Require one actually clean terminal attempt with all cost retained')
    if (metrics.get('complete') is not False or progress.get('stage') not in CLEAN
            or any(progress.get(k) != v for k, v in point['cursor'].items())):
        raise ValueError('Terminal checkpoint and controller outcome differ')


def run(config, output):
    import numpy as np
    import torch
    from .archive_ack import safe_target, validate_ack
    from .archive_compressed import verify_archive
    from .checkpoint_refs import read_index
    from .remote_lease import validate_no_unreconciled_remote
    from .rl_core import load_full_actor, optimizer_for, evaluate_decisions
    from .rl_resume import resume_journal
    from .sampling_parity import validate_outputs
    from .tensor_io import load_torch
    from .training_checkpoint import load_training_checkpoint

    root = Path(config['project_path']).resolve()
    output = Path(output).resolve()
    deadline = float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if os.environ.get('PTCG_JOB_RUN_ID') != output.name or not 0 < deadline-time.time() <= 600.5:
        raise ValueError('One controlled bounded local audit required')
    prior=config.get('local_audit_prior_run_ids',[])
    if prior:
        if prior!=['20261027T093218Z_tr10-full-cold-recovery_d78614']:
            raise ValueError('Only the recorded zero-update local schema repair admitted')
        spent=sum(a['seconds'] for rid in prior for a in read_json(root/'runs'/rid/'attempts.json'))
        if spent+deadline-time.time()>600.5:
            raise ValueError('Original cumulative600s local audit scope exhausted')
    source = root/'runs'/FIRST
    lease = read_json(source/'remote-job.json')
    validate_no_unreconciled_remote(root, {'remote_job': lease})
    ref = config['terminal_observation']
    if sha256(root/ref['path']) != ref['sha256']:
        raise ValueError('Exact final observation changed')
    observation = read_json(root/ref['path'])
    if observation['run_id'] != FIRST or observation['identity'] != lease['identity']:
        raise ValueError('Final observation source identity differs')
    files = observation['files']
    points = files['checkpoint-commits/index.json']['data']['commits']
    previous = read_json(source/'config.json')
    progress = files['progress.json']['data']
    validate_contract(previous, read_json(source/'attempts.json'), read_json(source/'metrics.json'), progress, points[-1])
    archive = source/'online-archive'
    acks = {}
    for path in archive.rglob('acks/*.json'):
        ack = read_json(path)
        validate_ack(ack, FIRST, lease['lease_epoch'], ack['object'], ack['sha256'], ack['bytes'])
        key = (ack['object'], ack['sha256'])
        if key in acks and acks[key] != ack:
            raise ValueError('Ambiguous durable archive identity')
        acks[key] = ack
    raw = (root/'runs'/prior[0]/'recovery-source'/FIRST) if prior else output/'recovery-source'/FIRST
    if not prior:raw.mkdir(parents=True, exist_ok=False)
    elif not raw.is_dir() or root not in raw.resolve().parents:raise ValueError('Original verified reconstruction absent')
    copied = {}

    def bounded():
        if time.time() > deadline-25:
            raise TimeoutError('Original local audit boundary reached')

    def materialize(item):
        bounded()
        relative, digest, size = item['path'], item['sha256'], item['bytes']
        if relative in copied:
            if copied[relative]['sha256'] != digest:
                raise ValueError('Conflicting inherited path')
            return
        if sum(r['bytes'] for r in copied.values()) + size > 768*2**20:
            raise ValueError('Local recovery materialization bound reached')
        ack = acks[(relative, digest)]
        if ack['bytes'] != size:
            raise ValueError('Inherited archive size differs')
        path = safe_target(raw, relative)
        if prior:
            if not path.is_file() or path.stat().st_size!=size or sha256(path)!=digest:
                raise ValueError('Retained original reconstruction differs')
            verify_archive(archive,ack)
            copied[relative]=dict(item)
            return
        payload = verify_archive(archive, ack, collect=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        if sha256(path) != digest:
            raise ValueError('Local inherited bytes changed during publication')
        copied[relative] = dict(item)

    # Recover exact Linux JSON bytes, not Windows text-mode newline variants.
    for relative in ['checkpoint-commits/index.json', 'payload-config.json', 'league.json', 'progress.json', 'inflight.json']:
        if relative not in files:
            continue
        item = files[relative]
        payload = (json.dumps(item['data'], ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
        if hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError('Observed metadata cannot reproduce exact original bytes: '+relative)
        path = safe_target(raw, relative); path.parent.mkdir(parents=True, exist_ok=True)
        if prior:
            if path.read_bytes()!=payload:raise ValueError('Original metadata reconstruction changed')
            continue
        with path.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    league = read_json(raw/'league.json')
    for point in points:
        for item in [point['checkpoint'], point['prefix'], *point['pending']]:
            materialize(item)
    for member in league['snapshots']:
        ack = acks[(member['path'], member['sha256'])]
        materialize(dict(path=member['path'], sha256=member['sha256'], bytes=ack['bytes']))
    index = read_index(raw, FIRST, lease['lease_epoch'])
    for point in index['commits']:
        saved = load_torch(raw/point['checkpoint']['path'], map_location='cpu', weights_only=False)
        if ({k:saved[k] for k in point['cursor']} != point['cursor'] or len(saved['model']) != 127
                or len(saved['optimizer']['state']) != 97 or set(saved['rng']) != {'python','numpy','torch','cuda'}
                or len(saved['rng']['cuda']) != 1 or 'cpu_sampler_rng' not in saved):
            raise ValueError('Incomplete model, optimizer, cursor or RNG')
        for value in [*saved['model'].values(), *[v for slot in saved['optimizer']['state'].values() for v in slot.values() if torch.is_tensor(v)]]:
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite restored tensor')
        for key in ['torch']:
            torch.Generator().set_state(saved['rng'][key])
        torch.Generator().set_state(saved['cpu_sampler_rng'])
        random.Random().setstate(saved['rng']['python'])
        np.random.RandomState().set_state(saved['rng']['numpy'])
        if any(v.dtype != torch.uint8 or v.ndim != 1 or not v.numel() for v in saved['rng']['cuda']):
            raise ValueError('Incomplete CUDA RNG payload')
        unique = checkpoint_decisions(saved)
    actual = read_json(raw/'payload-config.json')
    # A fresh job's payload-config precedes in-process fresh initialization.
    # Its immutable published prefix carries the actual zero-history origin.
    prefix=read_json(raw/points[-1]['prefix']['path'])
    spec=dict(prefix_path=points[-1]['prefix']['path'],prefix_sha256=points[-1]['prefix']['sha256'],
        initial_checkpoint_sha256=prefix['checkpoint']['sha256'])
    if saved['archive_origin_prefix_sha256']!=spec['prefix_sha256']:
        raise ValueError('Published compact origin differs from full checkpoint')
    loaded = load_training_checkpoint(raw, spec, lease['lease_epoch'])
    saved = loaded['saved']
    journal = resume_journal(raw, saved, loaded['archive_spec'], loaded['digest'])
    decisions = journal['decisions']
    if len(decisions) != progress['pending_decisions'] or len(decisions) < 48:
        raise ValueError('Complete pending current-policy tail with recovery fixture required')
    summaries = []
    for record in [*journal['archive_state']['archived'], *journal['archive_state']['live_expected']]:
        bounded()
        for relative, digest in [(record['trajectory_path'], record['trajectory_sha256']),
                                 (f"series/{record['series_id']:06d}.json", record['summary_sha256'])]:
            ack = acks[(relative, digest)]
            payload = verify_archive(archive, ack, collect=relative.startswith('series/'))
            if relative.startswith('series/'):
                summaries.append(json.loads(payload))
    if (len(summaries) != saved['next_series_id'] or len({r['series_id'] for r in summaries}) != len(summaries)
            or any(r['statuses'] != ['DONE','DONE'] or r['rounds'] not in (2,3) for r in summaries)
            or sum(r['learner_decisions'] for r in summaries) != saved['trained_decisions'] + len(decisions)):
        raise ValueError('Incomplete BO3 or sample ledger')
    torch.set_num_threads(2)
    model, features, _ = load_full_actor(root/actual['bundle_path'], actual['parent_model_sha256'], 'cpu', reset_value=False)
    model.load_state_dict(saved['model'], strict=True); model.eval()
    optimizer = optimizer_for(model, actual['ppo']['actor_learning_rate'], actual['ppo']['value_head_learning_rate'])
    optimizer.load_state_dict(saved['optimizer'])
    if optimizer.state_dict()['param_groups'] != saved['optimizer']['param_groups']:
        raise ValueError('Restored optimizer groups differ')
    lo, hi = features.b.OFFSETS['oracle']
    for decision in decisions:
        if any(step['rows'][:,lo:hi].any() or not np.isfinite(step['rows']).all() for step in decision['steps']):
            raise ValueError('Actor information isolation failed')
    with torch.no_grad():
        logp, values, _ = evaluate_decisions(model, decisions[:48], features.b.STOP)
    parity = validate_outputs(decisions[:48], logp, values)
    bounded()
    report = dict(at=utcnow(), complete=True, engineering_only=True,
        classification='TR10_original_checkpoint_full_recovery_audited', source_run_id=FIRST,
        source_point=points[-1], source_manifest_sha256=sha256(source/'remote-training-manifest.json'),
        source_progress_sha256=files['progress.json']['sha256'], source_index_sha256=files['checkpoint-commits/index.json']['sha256'],
        source_payload_config_sha256=files['payload-config.json']['sha256'], source_league_sha256=files['league.json']['sha256'],
        source_inflight=({'sha256':files['inflight.json']['sha256'], 'bytes':(raw/'inflight.json').stat().st_size} if 'inflight.json' in files else None),
        full_model_tensors=127, optimizer_states=97, all_full_model_optimizer_RNG_checkpoint_payloads_verified=True,
        actor_information_isolation_passed=True, pending_CPU_probability_value_parity=parity,
        all_frozen_historical_members_verified=league['snapshots'], pending_decisions=len(decisions),
        complete_BO3=len(summaries), complete_games=sum(r['rounds'] for r in summaries),
        sampled_decisions=sum(r['learner_decisions'] for r in summaries), legacy_consumed_decisions=saved['trained_decisions'],
        unique_PPO_trained_decisions=checkpoint_decisions(saved), decision_accounting=ACCOUNTING,
        update_coverage=[dict(version=r['version'], consumed=r['decisions'], actual_unique=update_decisions(r)) for r in saved['updates']],
        PPO_updates=saved['version'], remaining_original_attempts=1, first_attempt_seconds=read_json(source/'attempts.json')[0]['seconds'],
        materialized_root=raw.relative_to(root).as_posix(), materialized_files=list(copied.values()),
        original_checkpoint_payload_and_legacy_cursor_unchanged=True, no_new_training_data_or_optimizer_step=True,
        source_observation=ref, scientific_time_attempt_cost_and_stop_history_preserved=True)
    atomic_json(output/'verified-recovery-report.json', report)
    atomic_json(output/'metrics.json', report)
    return report
