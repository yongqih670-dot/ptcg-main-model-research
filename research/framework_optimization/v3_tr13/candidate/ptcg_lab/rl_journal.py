"""Recover training opponent statistics from committed complete BO3 records."""
from pathlib import Path
from .io import read_json, sha256


def committed_fixed_results(output):
    output = Path(output).resolve()
    stats = {}
    for path in sorted((output / 'series').glob('[0-9][0-9][0-9][0-9][0-9][0-9].json')):
        row = read_json(path)
        if row['series_id'] != int(path.stem) or row['statuses'] != ['DONE', 'DONE']:
            raise ValueError('Only correctly identified complete BO3 can enter league history')
        shard = (output / row['trajectory_path']).resolve()
        if output / 'rollouts' not in shard.parents:
            raise ValueError('Trajectory outside rollout journal')
        if sha256(shard) != row['trajectory_sha256']:
            raise ValueError('Committed trajectory changed')
        assignment = row['assignment']
        if assignment['fixed']:
            reward = row['rewards'][row['learner_seat']]
            record = stats.setdefault(assignment['opponent_id'], {'n': 0, 'losses': 0.})
            record['n'] += 1
            record['losses'] += 1 if reward < 0 else .5 if reward == 0 else 0
    return stats
