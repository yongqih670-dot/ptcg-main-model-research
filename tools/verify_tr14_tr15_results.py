"""Recompute the published complete-BO3 results; no model, network or games."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptcg_lab.volume_statistics import bootstrap_cells


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def outcomes(phase, role, n):
    rows = [json.loads(line) for line in (ROOT / f'evidence/{phase}/{role}-outcomes.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows) == n and len({x['series_id'] for x in rows}) == n
    assert all(set(x) == {'series_id', 'cell_id', 'candidate_seat', 'result', 'statuses', 'policy_invalid_count', 'internal_random_fallback_count'} for x in rows)
    assert all(x['statuses'] == ['DONE', 'DONE'] and x['policy_invalid_count'] == x['internal_random_fallback_count'] == 0 and x['result'] in ('win', 'draw', 'loss') for x in rows)
    return rows


def scores(rows):
    return [{'win': 1., 'draw': .5, 'loss': 0.}[x['result']] for x in rows]


def verify(phase):
    config = read(f'configs/{phase}/statistics.json')
    result = read(f'research/autonomy/{phase}-result.json')
    assert config['bootstrap_samples'] == 20000
    parent = config['parent_outcomes_phase']
    groups = {}
    for label, stage in [('parent', parent), ('endpoint', phase)]:
        rows = outcomes(stage, 'primary', 1600)
        assert Counter(x['candidate_seat'] for x in rows) == {0: 800, 1: 800}
        groups[label] = scores(rows)
    means, samples = bootstrap_cells(groups, {k: 1600 for k in groups}, config['bootstrap_seed'])
    primary_ci = np.quantile(samples['endpoint'] - samples['parent'], [.025, .975])
    assert np.allclose(primary_ci, result['primary']['ci95'], rtol=0, atol=1e-12)
    for label in ('parent', 'endpoint'):
        assert abs(means[label] - result['primary'][label + '_score']) < 1e-12
    assert abs(means['endpoint'] - means['parent'] - result['primary']['gain']) < 1e-12
    assert result['primary']['learning_signal_passed'] == (result['primary']['gain'] >= .05 and primary_ci[0] > 0)
    panel = {}
    data = {'parent': outcomes(parent, 'development', 800), 'endpoint': outcomes(phase, 'development', 800)}
    for cell in config['opponent_cells']:
        for label in ('parent', 'endpoint'):
            rows = [x for x in data[label] if x['cell_id'] == cell]
            assert len(rows) == 100 and Counter(x['candidate_seat'] for x in rows) == {0: 50, 1: 50}
            panel[label + '|' + cell] = scores(rows)
    means, samples = bootstrap_cells(panel, {k: 100 for k in panel}, config['bootstrap_seed'])
    assert means == result['development']['cell_scores']
    delta = sum((samples['endpoint|' + c] - samples['parent|' + c]) / 8 for c in config['opponent_cells'])
    assert np.allclose(np.quantile(delta, [.025, .975]), result['development']['ci95'], rtol=0, atol=1e-12)
    for label in ('parent', 'endpoint'):
        point = sum(means[label + '|' + c] / 8 for c in config['opponent_cells'])
        assert abs(point - result['development'][label + '_score']) < 1e-12
    outcomes(phase, 'reliability', 50)
    for role, n in [('primary', 1600), ('development', 800), ('reliability', 50)]:
        metric = read(f'evidence/{phase}/{role}-metrics.json')
        rows = outcomes(phase, role, n)
        counts = Counter(x['result'] for x in rows)
        assert metric['complete'] and metric['series_count'] == n
        assert all(metric[k] == 0 for k in ('errors', 'timeouts', 'invalid_actions', 'internal_random_fallbacks'))
        assert metric['wins'] == counts['win'] and metric['draws'] == counts['draw'] and metric['losses'] == counts['loss']
    assert result['classification'] == 'inconclusive' and not result['promotion_eligible']
    assert result['new_endpoint_BO3'] == 2450 and result['parent_BO3_reused'] == 2400
    return {'phase': phase, 'new_endpoint_BO3': 2450, 'reused_parent_BO3': 2400,
            'reported_statistics_match': True, 'learning_gate_passed': False}


if __name__ == '__main__':
    result = [verify(name) for name in ('TR14', 'TR15')]
    print(json.dumps({'verified': result, 'new_endpoint_BO3_total': 4900,
                      'bootstrap_samples': 20000, 'models_loaded': False,
                      'training_or_games_started': False}, ensure_ascii=False))
