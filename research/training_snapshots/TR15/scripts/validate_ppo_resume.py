"""Post-run audit of the preregistered production PPO resume exercise."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptcg_lab.io import read_json, atomic_json, sha256, utcnow
from ptcg_lab.rl_journal import committed_fixed_results


def validate(run):
    run = run.resolve()
    if run.parent != ROOT / 'runs':
        raise ValueError('Expected project run')
    state = read_json(ROOT / 'research/autonomy/state.json')
    if state.get('active_job') or (ROOT / 'research/autonomy/job.lock').exists():
        raise ValueError('Wait for controlled job completion')
    before = read_json(run / 'pre-resume-audit.json')
    attempts = read_json(run / 'attempts.json')
    metrics = read_json(run / 'metrics.json')
    updates = read_json(run / 'training-updates.json')
    records = {int(p.stem): read_json(p) for p in (run / 'series').glob('[0-9][0-9][0-9][0-9][0-9][0-9].json')}
    used = [sid for u in updates for sid in u['series_ids']]
    checks = {
        'engineering_only': read_json(run / 'experiment.json')['kind'] == 'engineering',
        'two_recorded_attempts': len(attempts) == 2 and [a['status'] for a in attempts] == ['checkpointed', 'completed'],
        'resumed_real_pending_data': before['progress']['pending_decisions'] > 0,
        'source_and_config_preserved': sha256(run / 'source-manifest.json') == before['source_manifest_sha256'] and sha256(run / 'config.json') == before['config_sha256'],
        'all_prior_committed_files_unchanged': all(sha256(run / p) == h for p, h in before['committed_files'].items()),
        'target_complete': metrics['complete'] and metrics['trained_decisions'] >= read_json(run / 'config.json')['target_decisions'],
        'complete_series_count_matches': len(records) == metrics['completed_series'],
        'each_complete_series_used_once_per_rollout': len(used) == len(set(used)) and set(used) == set(records),
        'no_policy_version_mixing': all(int(records[sid]['policy_version']) == u['version'] - 1 for u in updates for sid in u['series_ids']),
        'no_duplicate_optimizer_minibatches': all(len({(s['epoch'], s['offset']) for s in u['steps']}) == len(u['steps']) for u in updates),
        'decisions_reconcile': sum(u['decisions'] for u in updates) == metrics['trained_decisions'],
        'league_statistics_reconcile': committed_fixed_results(run) == read_json(run / 'league.json')['fixed_training_results'],
        'model_digest_matches': sha256(run / 'model.pt') == metrics['model_sha256'],
        'raw_research_count_unchanged': state['budget_used']['executed_research_experiments'] == before['budget']['executed_research_experiments'],
        'gpu_resume_cost_added': abs(state['budget_used']['gpu_job_wall_hours'] - before['budget']['gpu_job_wall_hours'] - sum(a['seconds'] for a in attempts[1:]) / 3600) < 1e-7,
    }
    result = {'at': utcnow(), 'run_id': run.name, 'checks': checks, 'passed': all(checks.values()),
              'attempts': attempts, 'complete_series': len(records), 'trained_decisions': metrics['trained_decisions'],
              'scope': 'Controlled soft checkpoint and same-run production resume; TR00 separately tested hard-stop and exact optimizer/RNG restoration. This test does not claim identical native-engine deals.',
              'weights_discarded_from_model_selection': True}
    atomic_json(run / 'resume-validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    args = parser.parse_args()
    import json
    result = validate(args.run)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
