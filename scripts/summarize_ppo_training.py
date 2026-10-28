"""Summarize recorded PPO optimization and public series metrics; never select."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptcg_lab.io import atomic_json, read_json, sha256, utcnow


def summarize(run):
    run = run.resolve()
    if run.parent != ROOT / 'runs':
        raise ValueError('Expected a project run')
    config = read_json(run / 'config.json')
    if config.get('job_kind') != 'train_ppo':
        raise ValueError('Expected PPO training')
    updates = read_json(run / 'training-updates.json')
    rows = []
    for update in updates:
        steps = update['steps']
        applied = [s for s in steps if not s['skipped_for_kl']]
        presentations = sum(s['batch'] for s in applied)
        rows.append({
            'version': update['version'], 'on_policy_decisions': update['decisions'],
            'complete_series': len(update['series_ids']),
            'optimizer_steps': len(applied), 'sample_presentations': presentations,
            'planned_presentations': update['decisions'] * config['ppo']['epochs'],
            'early_kl_stop': any(s['skipped_for_kl'] for s in steps),
            'max_observed_minibatch_kl': max(s['approx_kl'] for s in steps),
            'max_applied_minibatch_kl': max((s['approx_kl'] for s in applied), default=None),
            'weighted_value_loss': (sum(s['batch'] * s['value_loss'] for s in applied) / presentations
                                    if presentations else None),
            'weighted_entropy': (sum(s['batch'] * s['entropy'] for s in applied) / presentations
                                 if presentations else None),
        })
    assignments = Counter()
    identities = Counter()
    for path in (run / 'series').glob('[0-9][0-9][0-9][0-9][0-9][0-9].json'):
        row = read_json(path)
        assignments[str(row['assignment'].get('bucket'))] += 1
        identities[str(row['opponent_identity'].get('id'))] += 1
    metrics = read_json(run / 'metrics.json') if (run / 'metrics.json').exists() else {}
    result = {
        'at': utcnow(), 'run_id': run.name, 'complete': metrics.get('complete', False),
        'training_only': True, 'strength_conclusion': None, 'promotion_eligible': False,
        'config_sha256': sha256(run / 'config.json'),
        'updates_sha256': sha256(run / 'training-updates.json'),
        'updates': rows, 'league_assignment_counts': dict(assignments),
        'opponent_identity_counts': dict(identities),
        'on_policy_decisions': sum(r['on_policy_decisions'] for r in rows),
        'optimizer_steps': sum(r['optimizer_steps'] for r in rows),
        'sample_presentations': sum(r['sample_presentations'] for r in rows),
        'note': 'Training diagnostics only. BO3 screens and independent confirmations determine strength.',
    }
    atomic_json(run / 'optimization-summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    import json
    print(json.dumps(summarize(args.run), ensure_ascii=True, indent=2))
