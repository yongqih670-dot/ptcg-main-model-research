"""Read-only scalar credit analysis of a preselected, committed hybrid sample."""
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from .archive_compressed import verify_archive
from .hybrid_materialization import owned_runner_process
from .io import atomic_json, read_json, sha256, utcnow
from scripts.diagnose_ppo_credit import aggregate, scalars


def compare_credit(decisions, gamma, baseline_lambda):
    baseline = scalars(decisions, gamma, baseline_lambda)
    monte_carlo = scalars(decisions, gamma, 1.)
    if not baseline:
        raise ValueError('Empty committed trajectory')
    # At lambda=1 on complete episodes, returns must telescope to reward-to-go.
    error = max(abs(r['gae_target'] - r['mc_return']) for r in monte_carlo)
    if error > 1e-8:
        raise ValueError('Complete-round Monte Carlo identity failed')
    return baseline, monte_carlo


def summarize(baseline, monte_carlo):
    a = np.array([r['advantage'] for r in baseline])
    b = np.array([r['advantage'] for r in monte_carlo])
    return {'baseline': aggregate(baseline), 'lambda_one': aggregate(monte_carlo),
            'raw_advantage_sign_disagreement_fraction': float(np.mean(np.sign(a) != np.sign(b))),
            'baseline_advantage_std': float(a.std()), 'lambda_one_advantage_std': float(b.std()),
            'normalization_scope': 'No sample-wise normalization; production uses complete rollout normalization.'}


def run(config, output):
    root = Path(config['project_path']).resolve()
    output = Path(output).resolve()
    deadline = float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if (os.environ.get('PTCG_JOB_RUN_ID') != output.name or output.parent != root/'runs'
            or deadline-time.time() > 300.5 or config['maximum_attempts'] != 1):
        raise RuntimeError('Expected bounded single local runner')
    state = read_json(root/'research/autonomy/state.json')
    until = min(time.time()+1, deadline-10)
    while not state.get('active_job') and time.time() < until:
        time.sleep(.05)
        state = read_json(root/'research/autonomy/state.json')
    if not owned_runner_process(state.get('active_job') or {}, output.name):
        raise RuntimeError('Runner identity mismatch')
    source = (root/'runs'/config['training_run_id']).resolve()
    if source.parent != root/'runs' or source == output:
        raise ValueError('Source outside project runs')
    for relative, digest in config['source_evidence_sha256'].items():
        if sha256(source/relative) != digest:
            raise ValueError('Immutable source changed: '+relative)
    if not read_json(source/'metrics.json')['complete']:
        raise ValueError('Analyze only completed training')
    updates = read_json(source/'remote-results/training-updates.json')
    selected = config['selected']
    expected = [{'version': u['version']-1, 'series_id': sid}
                for u in updates if u['version'] in (3, 4)
                for sid in sorted(u['series_ids'])[:8]]
    if selected != expected or len(selected) != 16:
        raise ValueError('Outcome-independent sample selection changed')
    archive = source/'online-archive'
    acknowledgments = [read_json(p) for p in sorted((archive/source.name/'acks').glob('*.json'))]
    by_object = {a['object']: a for a in acknowledgments}
    if len(by_object) != len(acknowledgments):
        raise ValueError('Ambiguous archive object')
    lease = read_json(source/'remote-job.json')
    import torch
    torch.set_num_threads(1)
    grouped = {}
    evidence = []
    total_bytes = 0
    for item in selected:
        if time.time() > deadline-15:
            raise TimeoutError('Credit analysis deadline')
        sid, version = item['series_id'], item['version']
        summary_ack = by_object[f'series/{sid:06d}.json']
        summary = json.loads(verify_archive(archive, summary_ack, collect=True))
        ack = by_object[summary['trajectory_path']]
        if (ack['run_id'] != source.name or ack['lease_epoch'] != lease['lease_epoch']
                or ack['sha256'] != summary['trajectory_sha256']
                or summary['statuses'] != ['DONE', 'DONE']):
            raise ValueError('Committed BO3 archive identity mismatch')
        total_bytes += ack['bytes']
        if total_bytes > 256*2**20:
            raise ValueError('Registered input byte bound exceeded')
        data = torch.load(io.BytesIO(verify_archive(archive, ack, collect=True)),
                          map_location='cpu', weights_only=False)
        decisions = data['decisions']
        if (data['series_id'] != sid or int(data['policy_version']) != version
                or any(d['series_id'] != sid or int(d['policy_version']) != version for d in decisions)
                or len(decisions) != summary['learner_decisions']):
            raise ValueError('Wrong committed decision identity')
        baseline, mc = compare_credit(decisions, config['gamma'], config['baseline_lambda'])
        rows = grouped.setdefault(str(version), ([], []))
        rows[0].extend(baseline)
        rows[1].extend(mc)
        evidence.append({**item, 'trajectory_sha256': ack['sha256'],
                         'summary_sha256': summary_ack['sha256'], 'decisions': len(decisions)})
        del data, decisions
    baseline = [r for pair in grouped.values() for r in pair[0]]
    mc = [r for pair in grouped.values() for r in pair[1]]
    report = {'at': utcnow(), 'complete': True, 'engineering_only': True,
              'training_run_id': source.name, 'verified_raw_input_bytes': total_bytes,
              'sample_rule': 'First eight committed series in each of updates three and four, fixed before reads.',
              'evidence': evidence, 'per_version': {k: summarize(*v) for k, v in grouped.items()},
              'pooled': summarize(baseline, mc), 'optimizer_steps': 0,
              'promotion_eligible': False, 'source_modified': False,
              'limits': ['Descriptive training sample; neither independent evaluation nor causal intervention.',
                         'Lambda one removes return bootstrap bias but can increase gradient variance.',
                         'Sign disagreement alone does not prove incorrect baseline advantages.',
                         'Reward-derived values are analysis labels, never actor inputs.',
                         'No new scientific hypothesis, sampled decisions or PPO updates.']}
    atomic_json(output/'metrics.json', report)
    return report
