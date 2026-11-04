"""Count distinct rollout decisions that reached a committed PPO optimizer step.

The protected PPO loop draws a permutation without replacement in each epoch
and stops the entire update at the first KL skip. Its ordered minibatch log
therefore proves exact first-epoch coverage, including a partial first epoch.
"""


ACCOUNTING = 'unique_successful_first_epoch_coverage_v1'


def update_decisions(row):
    count = row['decisions']
    if type(count) is not int or count <= 0:
        raise ValueError('Positive rollout decision count required')
    epoch = offset = unique = 0
    skipped = False
    steps = row['steps']
    if not isinstance(steps, list) or not steps:
        raise ValueError('Committed update needs its ordered optimizer evidence')
    for step in steps:
        current, start, batch = (step[k] for k in ('epoch', 'offset', 'batch'))
        skip = step['skipped_for_kl']
        if (any(type(x) is not int for x in (current, start, batch))
                or type(skip) is not bool or skipped):
            raise ValueError('Malformed optimizer trace or steps after KL stop')
        if current == epoch + 1 and offset == count:
            epoch, offset = current, 0
        if current != epoch or start != offset or not 0 < batch <= count - offset:
            raise ValueError('Optimizer trace is not contiguous epoch coverage')
        if skip:
            skipped = True
        else:
            offset += batch
            if epoch == 0:
                unique += batch
    if not skipped and offset != count:
        raise ValueError('Uncommitted partial epoch cannot be counted')
    if 'unique_PPO_trained_decisions' in row and row['unique_PPO_trained_decisions'] != unique:
        raise ValueError('Recorded unique decision count differs from optimizer trace')
    return unique


def checkpoint_decisions(saved):
    rows = saved['updates']
    if len(rows) != saved['version'] or [r['version'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Checkpoint update history does not match its version')
    consumed = sum(r['decisions'] for r in rows)
    unique = sum(update_decisions(r) for r in rows)
    if consumed != saved['trained_decisions']:
        raise ValueError('Legacy consumed cursor differs from update history')
    if saved.get('unique_PPO_trained_decisions', unique) != unique:
        raise ValueError('Checkpoint unique count differs from optimizer history')
    return unique
