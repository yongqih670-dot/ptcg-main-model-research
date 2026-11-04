"""One final diagnostic from a completed full TR10 parent, never an old attempt."""
import copy
import os
from pathlib import Path
import time

from .io import atomic_json, read_json, sha256, utcnow
from .ppo_accounting import ACCOUNTING, checkpoint_decisions

PARENT = '20261027T094058Z_tr10-original-second-actual-counter_2063d8'
IDENTITY = 'TR11'
BASE = 32782
ADDITIONAL = 16384


def normalize_fixed_file_paths(config):
    """Match the established Linux continuation's path-key conversion."""
    from .archive_prefix import relative_path
    for fixed in config['league']['fixed_opponents']:
        normalized={}
        for name,digest in fixed['files'].items():
            canonical=relative_path(name.replace('\\','/'))
            if canonical in normalized:raise ValueError('Fixed opponent path aliases collide')
            normalized[canonical]=digest
        fixed['files']=normalized
    return config


def validate_training(training, parent):
    if training.get('hypothesis_id') != IDENTITY or training.get('target_decisions') != BASE+ADDITIONAL:
        raise ValueError('Only the frozen final diagnostic target is admitted')
    if training['ppo'] != parent['ppo'] or training['seed'] != parent['seed']:
        raise ValueError('No optimizer or sampling-seed changes in the basic learning diagnostic')
    if training['bundle_path'] != parent['bundle_path'] or training['parent_model_sha256'] != parent['parent_model_sha256']:
        raise ValueError('Full original actor lineage must remain bound')
    league = training['league']
    expected = {'fixed_diverse':1.0, 'historical_learner':0.0,
                'latest_frozen_learner':0.0, 'weakness_training_opponents':0.0}
    original = [x for x in parent['league']['fixed_opponents'] if x['id'] == 'ppo']
    if league['weights'] != expected or league['fixed_opponents'] != original or len(original) != 1:
        raise ValueError('Only the predeclared stable legal PPO opponent is admitted')
    if training.get('decision_milestones') != [BASE+8192, BASE+ADDITIONAL]:
        raise ValueError('Predeclared unique-decision milestones changed')


def validate_local(root, config, deadline, current_run_id=None):
    root = Path(root).resolve()
    acceptance=config.get('parent_acceptance_only') is True
    expected_id='CONTINUOUS_SERVER_FULL_PARENT_ACCEPTANCE' if acceptance else IDENTITY
    if (config.get('check_variant'), config.get('hypothesis_id'), config.get('original_training_run_id')) != ('continuous_mainline', expected_id, PARENT):
        raise ValueError('A new registered diagnostic is required, not an old continuation')
    from .continuous_runtime import local_limit,attempt_limit
    limit=300 if acceptance else local_limit(root,config) or 7200
    maximum=1 if acceptance else attempt_limit(root,config)
    index=config.get('training_attempt_index')
    if type(index) is not int or not 1<=index<=maximum:
        raise ValueError('Authorized TR11 attempt cap exhausted')
    if (config.get('maximum_attempts'), config.get('training_wall_seconds')) != (1, limit):
        raise ValueError('One new bounded training raw; no additional scientific attempt')
    if not 0 < deadline-time.time() <= limit+.5 or config.get('actual_decision_accounting') != ACCOUNTING:
        raise ValueError('Frozen deadline or actual-count contract differs')
    source = root/'runs'/PARENT
    from .remote_lease import validate_no_unreconciled_remote
    validate_no_unreconciled_remote(root, {'remote_job':read_json(source/'remote-job.json')})
    if not read_json(source/'metrics.json').get('complete'):
        raise ValueError('Completed full parent required')
    state = read_json(root/'research/autonomy/state.json')
    parent = state['research_stage']['hypotheses']['TR10']
    if parent['status'] != 'completed' or parent.get('remaining_original_training_attempts') != 0:
        raise ValueError('Old TR10 remains closed with no remaining attempt')
    row = state['research_stage']['hypotheses'].get(IDENTITY, {})
    if not row.get('diagnostic') or row.get('status') not in ('registered', 'started'):
        raise ValueError('Final diagnostic must be explicitly preregistered')
    prior=[]
    for rid in row.get('run_ids', []):
        p = root/'runs'/rid/'config.json'
        if p.exists() and read_json(p).get('job_kind') == 'remote_train_ppo' and rid != current_run_id:
            prior.append(rid)
    if len(prior)!=index-1 or prior!=config.get('TR11_prior_training_runs',[]):
        raise ValueError('Every prior training attempt must remain counted in exact order')
    if prior:
        audits=config.get('TR11_prior_zero_update_audits',[])
        if len(audits)!=len(prior):raise ValueError('Positive updates require complete checkpoint recovery, not parent restart')
        spent=0.
        for rid,ref in zip(prior,audits):
            directory=root/'runs'/rid
            validate_no_unreconciled_remote(root,{'remote_job':read_json(directory/'remote-job.json')})
            rows=read_json(directory/'attempts.json');spent+=sum(a['seconds'] for a in rows)
            if not rows or rows[-1]['status']!='failed':raise ValueError('No fresh restart of a complete or checkpointed scientific attempt')
            ap=(root/ref['path']).resolve()
            if root not in ap.parents or sha256(ap)!=ref['sha256']:raise ValueError('Zero-update proof changed')
            audit=read_json(ap)
            if (audit['run_id']!=rid or not audit['same_invocation_terminal']
                    or not audit['no_new_series_rollouts_inflight_updates_or_model']
                    or audit['new_actual_unique_PPO_decisions']!=0 or audit['new_games']!=0
                    or audit['parent_checkpoint_sha256']!=config['existing_restore_source']['checkpoint_reference']['sha256']):
                raise ValueError('Original parent restart only after positively proven zero-update startup failure')
        if abs(spent-config.get('prior_training_seconds',-1))>.01:raise ValueError('Prior attempt time cannot be reset')
    reference = read_json(source/'remote-training-manifest.json')['training_config']
    if sha256(root/config['training_config_template'])!=config['training_config_template_sha256']:
        raise ValueError('Frozen training configuration bytes changed')
    validate_training(read_json(root/config['training_config_template']), reference)
    proof = config['complete_parent_audit']
    path = (root/proof['path']).resolve()
    if root not in path.parents or sha256(path) != proof['sha256']:
        raise ValueError('Full parent audit changed')
    audit = read_json(path)
    if not audit.get('complete') or audit.get('classification') != 'complete_continuous_parent_verified':
        raise ValueError('Full-state parent audit has not passed')
    if audit['numeric_reference'] != config['numeric_reference'] or audit['parent_checkpoint_sha256'] != config['existing_restore_source']['checkpoint_reference']['sha256']:
        raise ValueError('Parent checkpoint or recovery-only fixture differs')
    numeric=(root/config['numeric_reference']['path']).resolve()
    if (root not in numeric.parents or sha256(numeric)!=config['numeric_reference']['sha256']
            or numeric.stat().st_size!=config['numeric_reference']['bytes']):
        raise ValueError('Exact recovery-only fixture bytes changed')
    for relative, digest in config['parent_evidence'].items():
        if sha256(source/relative) != digest:
            raise ValueError('Closed full parent evidence changed')


def prepare(root, job, ticket, manifest, engine_hash):
    from .retained_source import resolve, checkpoint_reference
    from .server_workspace import inside, admit
    from .checkpoint_refs import commit_checkpoint
    from .training_checkpoint import load_training_checkpoint
    from .tensor_io import load_torch
    source, previous = resolve(root, ticket['project_id'], manifest['existing_restore_source'])
    if source.name != PARENT or manifest.get('new_diagnostic_identity') != IDENTITY:
        raise ValueError('Only the fully closed predeclared parent may seed this new question')
    validate_training(manifest['training_config'], previous['training_config'])
    point = checkpoint_reference(source, manifest['existing_restore_source'])
    if point['pending'] or point['cursor']['version'] != 5:
        raise ValueError('Complete parent must have no pending training samples')
    saved = load_torch(source/point['checkpoint']['path'], map_location='cpu', weights_only=False)
    if checkpoint_decisions(saved) != BASE or len(saved['model']) != 127 or len(saved['optimizer']['state']) != 97:
        raise ValueError('Full actual parent state differs')
    progress = read_json(source/'progress.json')
    if progress.get('stage') != 'complete' or progress.get('pending_decisions') != 0:
        raise ValueError('Parent is not a completed original training endpoint')
    admit(root, manifest['reserved_future_bytes'])
    for ref, relative in [(point['checkpoint'], 'checkpoint.pt'), (point['prefix'], 'archive-prefix.json')]:
        src = inside(root, source/ref['path'])
        if sha256(src) != ref['sha256'] or src.stat().st_size != ref['bytes']:
            raise ValueError('Parent immutable object changed')
        os.link(src, inside(root, job/relative))
    league_path = inside(root, source/'league.json')
    if sha256(league_path) != manifest['source_league_sha256']:
        raise ValueError('Complete original league changed')
    league = read_json(league_path)
    runtime = read_json(source/'payload-config.json')
    retained = runtime['league'].get('retained_history')
    for row in league['snapshots']:
        base = inside(root, retained['root']) if retained and row['path'] in retained['members'] else source
        src = inside(root, base/row['path'])
        if sha256(src) != row['sha256']:
            raise ValueError('A frozen historical opponent changed')
        dst = inside(root, job/row['path']);dst.parent.mkdir(exist_ok=True)
        os.link(src, dst)
    atomic_json(job/'league.json', league)
    prefix = read_json(job/'archive-prefix.json')
    spec = {'prefix_path':'archive-prefix.json', 'prefix_sha256':point['prefix']['sha256'],
            'initial_checkpoint_sha256':prefix['checkpoint']['sha256']}
    commit_checkpoint(job, job.name, ticket['lease_epoch'], 0, 'checkpoint.pt', 'archive-prefix.json', [], point['cursor'])
    load_training_checkpoint(job, spec, ticket['lease_epoch'])
    numeric = manifest['numeric_reference']
    fixture = inside(root, job/'stage'/(numeric['sha256']+'.blob'))
    if sha256(fixture) != numeric['sha256'] or fixture.stat().st_size != numeric['bytes']:
        raise ValueError('Exact CPU numerical fixture changed')
    config = copy.deepcopy(manifest['training_config'])
    normalize_fixed_file_paths(config)
    config.update(project_path=str(root), engine_manifest_sha256=engine_hash, archive_restore=spec,
        checkpoint_interval_seconds=600, cooperative_resource_control=True, derived_resume_runtime_audit=True,
        online_archive={'maximum_pending_bytes':512*2**20}, sampling_device='cpu',
        checkpoint_transaction_hash_reuse=True, CPU_GPU_batch_parity_required=True,
        CPU_rollout_value_parity_required=True, CPU_backend_fixture=str(fixture),
        consumed_reference_barrier_required=True, actual_decision_accounting=ACCOUNTING,
        continuous_parent_actual_decisions=BASE,parent_acceptance_only=manifest.get('parent_acceptance_only') is True)
    atomic_json(job/'payload-config.json', config)
    atomic_json(job/'derived-inheritance.json', {'at':utcnow(), 'new_hypothesis_id':IDENTITY,
        'new_question_not_original_TR10_attempt':True, 'source_run_id':PARENT, 'source_point':point,
        'all_model_optimizer_RNG_CPU_sampler_and_league_inherited':True, 'no_value_or_optimizer_reset':True,
        'parent_unique_actual_decisions':BASE, 'new_unique_target':ADDITIONAL,
        'old_attempts_and_costs_unchanged':True})
    return config


def preserve_milestones(output, config, unique, recovery, league, outbox):
    """Pin full immutable optimizer/RNG checkpoints at predeclared crossings."""
    identity=config.get('hypothesis_id')
    if identity not in (IDENTITY,'TR12'):
        raise ValueError('Milestone pinning belongs only to the new diagnostic')
    if identity=='TR12' and config['decision_milestones']!=[369941]:
        raise ValueError('TR12 permits only its frozen final scientific endpoint')
    from .server_workspace import inside
    root = Path(config['project_path'])
    output = inside(root, output)
    for target in config['decision_milestones']:
        if unique < target:
            continue
        folder = inside(root, output/'milestones'/str(target))
        marker = folder/'manifest.json'
        if marker.exists():
            continue
        if recovery['pending'] or recovery['cursor']['version'] != league.entries[-1]['version']:
            raise ValueError('Only a full committed update with its frozen league can be a milestone')
        folder.mkdir(parents=True, exist_ok=False)
        refs = {}
        for kind, name in [('checkpoint', 'checkpoint.pt'), ('prefix', 'archive-prefix.json')]:
            ref = recovery[kind]
            src = inside(root, output/ref['path'])
            if sha256(src) != ref['sha256']:
                raise ValueError('Published milestone object changed')
            dst = folder/name
            os.link(src, dst)
            refs[kind] = {'path':dst.relative_to(output).as_posix(), 'sha256':ref['sha256'], 'bytes':ref['bytes']}
            if outbox:
                outbox.queue(ref['path'], ref['sha256'])
        league_file = folder/'league.json'
        atomic_json(league_file, {'snapshots':league.entries, 'fixed_training_results':league.stats,
                                 'weights':config['league']['weights']})
        if outbox:
            for entry in league.entries:
                outbox.queue(entry['path'], entry['sha256'])
        atomic_json(marker, {'at':utcnow(), 'requested_unique':target, 'actual_unique':unique,
            'new_unique_since_parent':unique-(49173 if identity=='TR12' else BASE), 'full_state':refs,
            'league_path':league_file.relative_to(output).as_posix(), 'league_sha256':sha256(league_file),
            'cursor':recovery['cursor'], 'pending':[], 'new_hypothesis_id':identity,
            'contains_full_optimizer_and_all_RNG':True, 'not_promotion_evidence':True})
