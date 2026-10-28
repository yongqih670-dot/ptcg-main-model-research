import unittest
from scripts.diagnose_ppo_credit import scalars, aggregate


def decision(round_id, value=0., terminal=False, reward=0.):
    return dict(series_id=1, round=round_id, learner_seat=0, policy_version='0',
                old_value=value, terminal=terminal, reward=reward)


class CreditDiagnosticTests(unittest.TestCase):
    def test_terminal_rewards_do_not_cross_round_boundaries(self):
        rows = scalars([decision(1), decision(1, terminal=True, reward=1),
                        decision(2), decision(2, terminal=True, reward=-1)], 1., .5)
        self.assertEqual([r['mc_return'] for r in rows], [1., 1., -1., -1.])
        self.assertEqual([r['gae_target'] for r in rows], [.5, 1., -.5, -1.])
        self.assertEqual([r['distance'] for r in rows], [1, 0, 1, 0])

    def test_perfect_visible_value_can_carry_credit_despite_small_direct_coefficient(self):
        rows = scalars([decision(1, value=1.) for _ in range(80)] +
                       [decision(1, value=1., terminal=True, reward=1.)], 1., .95)
        self.assertLess(rows[0]['terminal_reward_coefficient'], .05)
        self.assertTrue(all(abs(r['advantage']) < 1e-12 and r['gae_target'] == 1. for r in rows))
        self.assertIsNone(aggregate(rows)['mc_value_explained_variance'])

    def test_incomplete_and_silent_round_boundary_rejected(self):
        for ds in ([decision(1)], [decision(1), decision(2, terminal=True, reward=1)]):
            with self.assertRaises(ValueError):
                scalars(ds, 1., .95)


if __name__ == '__main__':
    unittest.main()
