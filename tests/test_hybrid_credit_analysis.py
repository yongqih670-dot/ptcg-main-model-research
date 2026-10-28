import unittest
from ptcg_lab.hybrid_credit_analysis import compare_credit, summarize


class CreditAnalysisTests(unittest.TestCase):
    def test_complete_opposite_rounds_telescope_without_outcome_leakage(self):
        decisions = []
        for round_id, reward in [(1, 1.), (2, -1.)]:
            for i, value in enumerate([.7, -.2, .4]):
                decisions.append(dict(series_id=9, round=round_id, learner_seat=0,
                    policy_version=2, old_value=value, terminal=i == 2,
                    reward=reward if i == 2 else 0.))
        baseline, mc = compare_credit(decisions, .9, .95)
        self.assertEqual(len(mc), 6)
        for row, target in zip(mc, [.81, .9, 1., -.81, -.9, -1.]):
            self.assertAlmostEqual(row['gae_target'], target)
        report = summarize(baseline, mc)
        self.assertEqual(report['lambda_one']['represented_rounds'], 2)
        self.assertLess(report['lambda_one']['gae_target_rmse_from_mc'], 1e-12)

    def test_incomplete_round_cannot_be_reported_as_monte_carlo(self):
        with self.assertRaises(ValueError):
            compare_credit([dict(series_id=1, round=1, learner_seat=0,
                policy_version=2, old_value=.2, terminal=False, reward=0)], 1., .95)


if __name__ == '__main__':
    unittest.main()
