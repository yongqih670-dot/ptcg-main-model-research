"""Serializable minibatch cursor; a completed optimizer step is never replayed.

Scientific update accounting remains deferred until the original PPO update
finishes. The cursor records the already drawn permutation and next offset.
"""
import numpy as np
from pathlib import Path


def has_postcheckpoint_tail(saved, resume):
    expected={r['series_id'] for r in (saved or {}).get('archive_state',{}).get('live_expected',[])}
    return any(int(Path(p).stem.split('-s')[1]) not in expected for p in resume['shards'])


def begin(decisions, advantages, returns, minibatch):
    return {'schema': 1, 'decision_count': len(decisions), 'epoch': 0,
            'offset': 0, 'permutation': None, 'minibatch': minibatch,
            'advantages': advantages, 'returns': returns, 'logs': [],
            'inherited_first_epoch_coverage': 0, 'positive_optimizer_steps': 0,
            'finished': False}


def validate(value, count, epochs):
    if value.get('schema') != 1 or value['decision_count'] != count:
        raise ValueError('Partial PPO input identity differs')
    if not 0 <= value['epoch'] <= epochs or not 0 <= value['offset'] <= count:
        raise ValueError('Partial PPO cursor out of range')
    indices = value['permutation']
    if indices is not None and (len(indices) != count or
            not np.array_equal(np.sort(indices), np.arange(count))):
        raise ValueError('Partial PPO permutation is not without replacement')
    if value['positive_optimizer_steps']!=sum(not row['skipped_for_kl'] for row in value['logs']):
        raise ValueError('Partial PPO positive-step accounting differs')
    return value


def next_batch(value, epochs):
    if value['finished'] or value['epoch'] >= epochs:
        return None
    if value['permutation'] is None:
        value['permutation'] = np.random.permutation(value['decision_count'])
    return value['permutation'][value['offset']:value['offset']+value['minibatch']]


def acknowledge(value, chosen, metrics, epochs, inherited_count=0):
    expected = value['permutation'][value['offset']:value['offset']+value['minibatch']]
    if not np.array_equal(expected, chosen):
        raise ValueError('Optimizer batch/cursor differs')
    value['logs'].append({'epoch':value['epoch'], 'offset':value['offset'],
                          'batch':len(chosen), **metrics})
    if metrics['skipped_for_kl']:
        value['finished'] = True
        return
    value['positive_optimizer_steps'] += 1
    if value['epoch'] == 0:
        value['inherited_first_epoch_coverage'] += int(np.count_nonzero(chosen < inherited_count))
    value['offset'] += len(chosen)
    if value['offset'] == value['decision_count']:
        value['epoch'] += 1
        value['offset'] = 0
        value['permutation'] = None
        value['finished'] = value['epoch'] == epochs
