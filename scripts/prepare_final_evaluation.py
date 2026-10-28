"""Register a fixed final BO3 panel only after two independently confirmed seeds.

This prepares files; it never runs matches, changes the champion, or uploads.
The controlled runner still owns all resource, stop, and execution gates.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptcg_lab.io import read_json, atomic_json, sha256, utcnow
from ptcg_lab.stage_control import safe_path, validate_activation, validate_stage_job, activation_protected_hashes
from scripts.run_experiment import acquire_lock


def require_qualification_row(row):
    if (row.get('status') != 'completed' or row.get('classification') != 'improved'
            or row.get('promotion_eligible') is not True or not row.get('evaluation_run_id')
            or not row.get('evidence_sha256', {}).get('confirmation')):
        raise ValueError('A completed independent development confirmation is required')


def completed_training_config(root, training, spec):
    """Resolve the actual learning recipe, requiring raw hybrid output proof."""
    controller = read_json(training / 'config.json')
    metrics = read_json(training / 'metrics.json')
    if (read_json(training / 'experiment.json').get('kind') != 'research'
            or not metrics.get('complete') or sha256(training / 'config.json') != spec['training_config_sha256']
            or sha256(training / 'model.pt') != spec['model_sha256']
            or metrics.get('model_sha256') != spec['model_sha256']):
        raise ValueError('Candidate does not match complete research training bytes')
    recipe = controller; runtime = None
    if controller.get('job_kind') == 'remote_train_ppo':
        if (controller.get('check_variant') not in ('scientific_continue', 'fresh_scientific', 'derived_scientific')
                or controller.get('engineering_only') is not False
                or not (metrics.get('scientific_training') or metrics.get('scientific_continuation'))):
            raise ValueError('Engineering server weights cannot qualify')
        template = safe_path(root, controller['training_config_template'])
        if sha256(template) != controller['training_config_template_sha256']:
            raise ValueError('Pinned server learning recipe changed')
        recipe = read_json(template)
        manifest_path = training / 'remote-training-manifest.json'
        manifest = read_json(manifest_path)
        if (sha256(manifest_path) != metrics.get('remote_manifest_sha256')
                or manifest.get('mode') != controller['check_variant']
                or manifest.get('training_config') != recipe):
            raise ValueError('Actual server scientific manifest differs')
        from ptcg_lab.remote_lease import validate_no_unreconciled_remote
        lease = read_json(training / 'remote-job.json')
        validate_no_unreconciled_remote(root, {'remote_job': lease})
        terminal_receipt = read_json(safe_path(root, lease['terminal_receipt_path']))
        terminal = (terminal_receipt['journal_terminal_proof']
                    if terminal_receipt.get('terminal_proof_kind')=='exact_invocation_systemd_journal'
                    else terminal_receipt['service_properties'])
        if terminal.get('Result') != 'success' or terminal.get('ExecMainStatus') != '0':
            raise ValueError('Failed server invocation cannot qualify stale complete metrics')
        materialization = read_json(training / 'hybrid-materialization.json')
        ack = materialization['model_archive_ack']
        receipt = training / 'online-archive' / training.name / 'acks' / f"{ack['sequence']:08d}.json"
        if (read_json(receipt) != ack or ack['object'] != 'model.pt' or ack['run_id'] != training.name
                or ack['sha256'] != spec['model_sha256']
                or materialization.get('raw_model_sha256') != spec['model_sha256']):
            raise ValueError('Durable raw model reconstruction proof differs')
        from ptcg_lab.archive_compressed import verify_archive
        verify_archive(training / 'online-archive', ack)
        engine = metrics.get('engine', {})
        if engine.get('platform') != 'linux' or set(engine.get('files', {})) != {'libcg.so', 'sim.py', 'game.py'}:
            raise ValueError('Actual official Linux training engine evidence required')
        actual = metrics.get('actual_config')
        if not isinstance(actual, dict) or any(actual.get(k) != recipe.get(k) for k in
                ('seed', 'target_decisions', 'ppo', 'parent_model_sha256', 'bundle_path')):
            raise ValueError('Actual executed server learning recipe differs')
        sampling_device = actual.get('sampling_device', 'cuda')
        if sampling_device not in ('cpu', 'cuda'):
            raise ValueError('Actual sampling execution device is unknown')
        runtime = {'execution': 'server', 'sampling_device': sampling_device,
                   'engine_sha256': {k:v['sha256'] for k,v in engine['files'].items()}}
    elif controller.get('job_kind') != 'train_ppo':
        raise ValueError('Expected completed PPO research training')
    if (recipe.get('job_kind') != 'train_ppo' or type(recipe.get('target_decisions')) is not int
            or recipe['target_decisions'] <= 0 or metrics.get('trained_decisions', -1) < recipe['target_decisions']):
        raise ValueError('Full preregistered training decision budget has not completed')
    return recipe, runtime


def verified_candidate(root, state, identity):
    row = state['research_stage']['hypotheses'][identity]
    require_qualification_row(row)
    run = safe_path(root, 'runs/' + row['evaluation_run_id'])
    for key, name in (('panel', 'panel-analysis.json'), ('metrics', 'metrics.json'), ('confirmation', 'confirmation.json')):
        if sha256(run / name) != row['evidence_sha256'][key]:
            raise ValueError('Closed development evidence changed: ' + name)
    report_path = safe_path(root, row['report_path'])
    if sha256(report_path) != row['report_sha256']:
        raise ValueError('Development qualification report changed')
    report = read_json(report_path)
    if report.get('classification') != 'improved' or report.get('promotion_eligible') is not True:
        raise ValueError('Development report does not qualify candidate')
    # Recompute from raw BO3 rows, not merely a saved true/false flag.
    from scripts.validate_confirmation import validate
    result = validate(run)
    if not result['development_confirmation_passed'] or sha256(run / 'confirmation.json') != row['evidence_sha256']['confirmation']:
        raise ValueError('Raw development evidence no longer reproduces qualification')
    config = read_json(run / 'config.json')
    candidates = [c['candidate'] for c in config['matchups'] if c['cell_id'].startswith('candidate-')]
    if not candidates or any(c != candidates[0] for c in candidates):
        raise ValueError('Confirmation did not evaluate one fixed candidate')
    spec = candidates[0]
    if spec['kind'] != 'transformer_rl':
        raise ValueError('Expected the primary Transformer policy')
    training = safe_path(root, 'runs/' + spec['training_run_id'])
    if (training.name not in row['run_ids']
            or sha256(safe_path(root, spec['path']) / 'model.pt') != spec['model_sha256']):
        raise ValueError('Candidate does not match completed full-budget training')
    train_config, runtime = completed_training_config(root, training, spec)
    plan_path = safe_path(root, row['preregistration_path'])
    if sha256(plan_path) != row['preregistration_sha256']:
        raise ValueError('Training preregistration changed')
    return {'row': row, 'run': run, 'config': config, 'spec': spec, 'training': training,
            'training_config': train_config, 'runtime_identity': runtime, 'plan': read_json(plan_path),
            'qualification': {'path': row['report_path'], 'sha256': row['report_sha256']}}


def compare_seed_protocols(first, second):
    a = first['training_config']; b = second['training_config']
    if type(a.get('seed')) is not int or type(b.get('seed')) is not int or a['seed'] == b['seed']:
        raise ValueError('Second training seed must be distinct')
    ignored = {'seed', 'hypothesis_id', 'primary_change', 'success_criteria'}
    if {k:v for k,v in a.items() if k not in ignored} != {k:v for k,v in b.items() if k not in ignored}:
        raise ValueError('Second seed changed algorithm, training distribution, or sample budget')
    if first.get('runtime_identity') != second.get('runtime_identity'):
        raise ValueError('Second seed changed execution location or actual training engine')
    budget_fields = ('training_wall_seconds_max', 'training_attempt_seconds_max', 'training_attempts_max')
    def declared(plan, field):
        values = [plan[k] for k in (('training_attempts_max', 'max_training_attempts') if field == 'training_attempts_max' else (field,)) if k in plan]
        if not values or any(v != values[0] for v in values):
            raise ValueError('Missing or conflicting original training budget: ' + field)
        return values[0]
    for field in budget_fields:
        if declared(first['plan'], field) != declared(second['plan'], field):
            raise ValueError('Second seed must retain preregistered training time budget: ' + field)
    def panel_protocol(config):
        result = copy.deepcopy(config)
        result = {k:result[k] for k in ('matchups', 'contrasts', 'series_count', 'evaluation_stage')}
        for cell in result['matchups']:
            if cell['cell_id'].startswith('candidate-'):
                cell['candidate'] = {'bound_candidate': True}
        return result
    if panel_protocol(first['config']) != panel_protocol(second['config']):
        raise ValueError('Second seed must use the same independent confirmation panel and sample size')


def build_panel(candidate, anchor, holdout, policy, identity, per_policy, policy_seed):
    rules = policy['evaluation']; opponents = holdout['opponents']
    if not rules['final_series_per_policy_min'] <= per_policy <= rules['final_series_per_policy_max']:
        raise ValueError('Final sample size is outside policy')
    if (len(opponents) < rules['opponents_min'] or len({o['id'] for o in opponents}) != len(opponents)
            or len({o['archetype'] for o in opponents}) < rules['archetypes_min']
            or len({o['lineage'] for o in opponents}) < rules['independent_public_policy_sources_min']):
        raise ValueError('Insufficient distinct final opponents, archetypes, or policy sources')
    if per_policy % len(opponents) or (per_policy // len(opponents)) % 2:
        raise ValueError('Use equal integer opponent cells with balanced seats')
    weights = [o['weight'] for o in opponents]
    if any(type(w) not in (float, int) or not 0 < w <= 1 for w in weights) or abs(sum(weights) - 1.) > 1e-12:
        raise ValueError('Invalid frozen final weights')
    cells = []; contrasts = []
    for opponent in opponents:
        ref = {k:copy.deepcopy(opponent[k]) for k in ('id', 'kind', 'path', 'lineage', 'parameters') if k in opponent}
        names = {}
        for side, spec in (('candidate', candidate), ('anchor', anchor)):
            names[side] = side + '-vs-' + opponent['id']
            cells.append({'cell_id': names[side], 'candidate': copy.deepcopy(spec), 'reference': ref,
                          'series_count': per_policy // len(opponents)})
        contrasts.append({'candidate_cell': names['candidate'], 'anchor_cell': names['anchor'], 'weight': opponent['weight']})
    return {'job_kind': 'evaluation', 'hypothesis_id': identity, 'workers': 8,
            'series_count': 2 * per_policy, 'series_per_policy': per_policy, 'policy_seed': policy_seed,
            'save_all_replays': True, 'stop_on_internal_fallback': True,
            'evaluation_stage': 'final_holdout', 'matchups': cells, 'contrasts': contrasts,
            'primary_change': 'Independent final holdout distribution; both frozen policies and weights fixed before outcomes',
            'success_criteria': 'All unchanged final policy gates, including macro score, gain, BO3 CI, zero faults, Linux package and second-seed evidence; final panel alone never authorizes upload'}


def sample_design_evidence(panel, config, per_policy, minimum_gain):
    """Development-only planning approximation; never a final significance test."""
    if not panel.get('complete') or not 0 < minimum_gain < 1 or per_policy <= 0:
        raise ValueError('Invalid development sample-design evidence')
    variance = 0.; cells = []
    for contrast in config['contrasts']:
        weight = contrast['weight']
        for side in ('candidate_cell', 'anchor_cell'):
            name = contrast[side]; cell = panel['cells'][name]; n = cell['series_count']
            score = cell['score_with_errors_counted_as_losses']
            if n <= 0 or not 0 <= score <= 1:
                raise ValueError('Invalid development outcome summary')
            adjusted = (n * score + .5) / (n + 1)
            variance += weight * adjusted * (1 - adjusted)
            cells.append({'cell': name, 'series': n, 'observed_score': score,
                          'planning_score_half_count_adjusted': adjusted, 'weight': weight})
    z95 = 1.959963984540054; z80 = .8416212335729143
    required = math.ceil(variance * (z95 + z80) ** 2 / minimum_gain ** 2)
    signal = minimum_gain / math.sqrt(variance / per_policy)
    cdf = lambda x: .5 * (1 + math.erf(x / math.sqrt(2)))
    return {'development_run_id': panel['run_id'], 'cells': cells,
            'target_gain_for_planning': minimum_gain, 'fixed_series_per_policy': per_policy,
            'approximate_required_series_per_policy_for_80_percent_power': required,
            'approximate_power_at_fixed_sample': 1 - cdf(z95 - signal) + cdf(-z95 - signal),
            'method': 'Independent equal-allocation Bernoulli normal approximation using fixed development weights and half-count adjusted scores',
            'limits': 'Planning approximation only: final distribution may differ; draws and stratification affect variance. Actual final decision still uses fixed complete-BO3 bootstrap, effect-size and all other policy gates. No sample extension after final outcomes.'}


def prepare(first_id, second_id, identity, per_policy, policy_seed):
    if not identity or any(not (c.isalnum() or c == '_') for c in identity):
        raise ValueError('Invalid final hypothesis identifier')
    lock = ROOT / 'research/autonomy/job.lock'; owner = acquire_lock(lock)
    try:
        state_path = ROOT / 'research/autonomy/state.json'; state = read_json(state_path)
        policy = read_json(ROOT / 'research/autonomy/policy.json')
        validate_activation(ROOT, policy, state)
        from scripts.run_experiment import validate_cycle_open
        validate_cycle_open(state)
        if state.get('active_job') or any(r.get('status') == 'started' for r in state['research_stage']['hypotheses'].values()):
            raise ValueError('Finish and close current heavy job/hypothesis first')
        if identity in state['research_stage']['hypotheses']:
            raise ValueError('Never overwrite a registered hypothesis')
        if first_id == second_id:
            raise ValueError('Two distinct completed training hypotheses are required')
        first = verified_candidate(ROOT, state, first_id)
        second = verified_candidate(ROOT, state, second_id)
        compare_seed_protocols(first, second)
        # Hash relevant implementation paths, including visible feature encoding.
        first_source = {k.replace('\\', '/'):v for k,v in read_json(first['training'] / 'source-manifest.json').items()}
        second_source = {k.replace('\\', '/'):v for k,v in read_json(second['training'] / 'source-manifest.json').items()}
        relevant = ('train_ppo.py', 'rl_core.py', 'rl_collector.py', 'rl_engine_worker.py', 'rl_league.py',
                    'rl_journal.py', 'token_features.py', 'rl_agent.py', 'agents.py', 'public_agent.py', 'replay_data.py', 'io.py')
        for name in relevant:
            key = 'ptcg_lab/' + name
            if (first_source.get(key) is None or first_source[key] != second_source.get(key)
                    or sha256(ROOT / key) != first_source[key]):
                raise ValueError('Training/inference implementation differs: ' + key)
        env_a = read_json(first['training'] / 'environment.json'); env_b = read_json(second['training'] / 'environment.json')
        if any(env_a[k] != env_b[k] for k in ('files', 'packages')):
            raise ValueError('Training environments differ across seeds')
        protected_hashes = activation_protected_hashes(ROOT,state['research_stage']['activation_receipt'])
        anchor_path = 'research/opponents/release_anchor.json'; holdout_path = 'research/opponents/final_holdout_v2.json'
        for name in (anchor_path, holdout_path):
            if sha256(ROOT / name) != protected_hashes[name]:
                raise ValueError('Protected anchor or final holdout changed')
        anchor = read_json(ROOT / anchor_path); holdout = read_json(ROOT / holdout_path)
        for spec in [dict(anchor['spec'], files=anchor['files']), *holdout['opponents']]:
            base = safe_path(ROOT, spec['path'])
            for name, expected in spec['files'].items():
                if sha256(base / name.replace('\\', '/')) != expected:
                    raise ValueError('Frozen opponent artifact changed')
        # Merely copying holdout files into source snapshots is not use. A run
        # configured to act against any final variant is conservatively treated
        # as exposure, even if its result is incomplete or failed.
        forbidden = [o['path'].replace('\\', '/').lower() for o in holdout['opponents']]
        for config_path in (ROOT / 'runs').glob('*/config.json'):
            text = json.dumps(read_json(config_path)).replace('\\\\', '/').lower()
            if any(path in text for path in forbidden):
                raise ValueError('Final pool has prior run exposure; a fresh/corrected protocol is required')
        config = build_panel(first['spec'], anchor['spec'], holdout, policy, identity, per_policy, policy_seed)
        config.update(final_holdout_path=holdout_path, final_holdout_sha256=sha256(ROOT / holdout_path),
                      release_anchor_sha256=sha256(ROOT / anchor_path), candidate_training_run_id=first['training'].name,
                      replication_training_run_id=second['training'].name)
        design = sample_design_evidence(read_json(first['run'] / 'panel-analysis.json'), first['config'],
                                       per_policy, policy['evaluation']['final_weighted_gain_vs_anchor_min'])
        design['development_panel_sha256'] = sha256(first['run'] / 'panel-analysis.json')
        destination = ROOT / 'configs/evaluation' / (identity.lower() + '.json')
        plan_path = ROOT / 'research/autonomy/preregistrations' / (identity + '.json')
        if destination.exists() or plan_path.exists():
            raise ValueError('Do not overwrite an earlier final preparation')
        plan = {'hypothesis_id': identity, 'at': utcnow(), 'research': True, 'final': True, 'diagnostic': False,
                'max_raw_runs': 1, 'config_hashes': [hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()],
                'development_qualification': first['qualification'], 'second_seed_qualification': second['qualification'],
                'first_seed_is_release_candidate': True, 'second_seed_not_selected_by_final_scores': True,
                'series_per_policy': per_policy, 'holdout_sha256': sha256(ROOT / holdout_path),
                'sample_size_rationale': design,
                'no_optional_stopping_or_posthoc_sample_extension': True,
                'failure_rule': 'Retain failures; do not tune on or relabel this pool as untouched. Preserve candidate and reasons if gates fail.'}
        atomic_json(destination, config); atomic_json(plan_path, plan)
        state['research_stage']['hypotheses'][identity] = {'status': 'registered', 'run_ids': [], 'diagnostic': False,
            'preregistration_path': plan_path.relative_to(ROOT).as_posix(), 'preregistration_sha256': sha256(plan_path)}
        validate_stage_job(ROOT, config, policy, state, True, True)
        state['updated_at'] = utcnow(); atomic_json(state_path, state)
        return destination
    finally:
        if lock.exists():
            current = read_json(lock)
            if all(current.get(k) == owner[k] for k in ('pid', 'created')):
                lock.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--candidate-hypothesis', required=True)
    p.add_argument('--replication-hypothesis', required=True)
    p.add_argument('--hypothesis', required=True)
    p.add_argument('--series-per-policy', type=int, required=True)
    p.add_argument('--policy-seed', type=int, required=True)
    a = p.parse_args()
    print(prepare(a.candidate_hypothesis, a.replication_hypothesis, a.hypothesis, a.series_per_policy, a.policy_seed))
