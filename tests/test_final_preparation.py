import copy
import unittest
from scripts.prepare_final_evaluation import build_panel, compare_seed_protocols, require_qualification_row, sample_design_evidence


class FinalPreparationTests(unittest.TestCase):
    def setUp(self):
        self.policy = {'evaluation': {'final_series_per_policy_min': 1600,
            'final_series_per_policy_max': 3200, 'opponents_min': 8,
            'archetypes_min': 4, 'independent_public_policy_sources_min': 3}}
        self.pool = {'opponents': [dict(id='fixture-'+str(i), kind='public_rule',
            path='fixture/'+str(i), lineage='lineage-'+str(i//2), archetype='type-'+str(i//2),
            weight=.125) for i in range(8)]}
        self.first = {'training_config': {'seed': 1, 'target_decisions': 262144,
            'ppo': {'actor_learning_rate': 1e-5, 'gae_lambda': .95}, 'hypothesis_id': 'FIRST'},
            'plan': {'training_wall_seconds_max': 10800, 'training_attempt_seconds_max': 7200,
                     'max_training_attempts': 2},
            'config': {'matchups': [{'cell_id': 'candidate-vs-fixture', 'candidate': {'model': 'a'},
                'reference': {'model': 'anchor'}, 'series_count': 100}],
                'contrasts': [{'weight': 1.}], 'series_count': 100, 'evaluation_stage': 'independent_confirmation'}}
        self.second = copy.deepcopy(self.first)
        self.second['training_config'].update(seed=2, hypothesis_id='SECOND')
        self.second['config']['matchups'][0]['candidate'] = {'model': 'b'}

    def test_two_seeds_keep_training_and_confirmation_protocol(self):
        compare_seed_protocols(self.first, self.second)
        for change in ({'seed': 1}, {'target_decisions': 32768}, {'ppo': {'actor_learning_rate': 3e-5, 'gae_lambda': .95}}):
            altered = copy.deepcopy(self.second); altered['training_config'].update(change)
            with self.assertRaises(ValueError): compare_seed_protocols(self.first, altered)
        for field in ('training_wall_seconds_max', 'training_attempt_seconds_max', 'max_training_attempts'):
            altered = copy.deepcopy(self.second); altered['plan'][field] += 1
            with self.assertRaises(ValueError): compare_seed_protocols(self.first, altered)
        altered = copy.deepcopy(self.second); altered['config']['matchups'][0]['series_count'] = 50
        with self.assertRaises(ValueError): compare_seed_protocols(self.first, altered)

    def test_final_counts_weights_and_original_candidate_preserved(self):
        candidate = {'id': 'original-qualified-seed'}; anchor = {'id': 'fixed-anchor'}
        before = copy.deepcopy(self.pool)
        config = build_panel(candidate, anchor, self.pool, self.policy, 'FINAL', 3200, 91)
        self.assertEqual(config['series_count'], 6400)
        self.assertEqual(len(config['matchups']), 16)
        self.assertTrue(all(c['series_count'] == 400 for c in config['matchups']))
        self.assertEqual(sum(c['weight'] for c in config['contrasts']), 1.)
        self.assertTrue(all(c['candidate'] == candidate for c in config['matchups'] if c['cell_id'].startswith('candidate-')))
        self.assertEqual(self.pool, before)

    def test_sample_and_diversity_cannot_be_reduced(self):
        for n in (1599, 3201, 1601):
            with self.assertRaises(ValueError): build_panel({}, {}, self.pool, self.policy, 'FINAL', n, 91)
        for field in ('id', 'lineage', 'archetype'):
            altered = copy.deepcopy(self.pool)
            for o in altered['opponents']: o[field] = 'same'
            with self.assertRaises(ValueError): build_panel({}, {}, altered, self.policy, 'FINAL', 3200, 91)
        altered = copy.deepcopy(self.pool); altered['opponents'][0]['weight'] = 0
        with self.assertRaises(ValueError): build_panel({}, {}, altered, self.policy, 'FINAL', 3200, 91)

    def test_screen_and_inconclusive_records_do_not_qualify(self):
        for row in ({}, {'status': 'started'}, {'status': 'completed', 'classification': 'inconclusive'},
                    {'status': 'completed', 'classification': 'improved', 'promotion_eligible': True,
                     'evaluation_run_id': 'a-screen-only', 'evidence_sha256': {}}):
            with self.assertRaises(ValueError): require_qualification_row(row)

    def test_design_uses_development_data_and_does_not_treat_it_as_final_result(self):
        panel = {'complete': True, 'run_id': 'fixture-development-only', 'cells': {
            'a': {'series_count': 100, 'score_with_errors_counted_as_losses': .85},
            'b': {'series_count': 100, 'score_with_errors_counted_as_losses': .82}}}
        config = {'contrasts': [{'candidate_cell': 'a', 'anchor_cell': 'b', 'weight': 1.}]}
        a = sample_design_evidence(panel, config, 1600, .03)
        b = sample_design_evidence(panel, config, 3200, .03)
        self.assertEqual(a['development_run_id'], 'fixture-development-only')
        self.assertGreater(b['approximate_power_at_fixed_sample'], a['approximate_power_at_fixed_sample'])
        self.assertEqual(a['approximate_required_series_per_policy_for_80_percent_power'], b['approximate_required_series_per_policy_for_80_percent_power'])
        self.assertNotIn('submission_qualified', b)


if __name__ == '__main__':
    unittest.main()
