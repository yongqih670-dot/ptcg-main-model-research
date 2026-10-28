import unittest
from ptcg_lab.continuous_diagnostic_stats import contrast


class DiagnosticStatsTests(unittest.TestCase):
    def test_exact_five_percentage_point_boundary_and_seed_are_preserved(self):
        r=contrast([1.]*360+[0.]*40,[1.]*380+[0.]*20)
        self.assertEqual(r['gain'],.05);self.assertTrue(r['learning_signal_passed'])
        self.assertEqual(r['bootstrap_seed'],231102)
    def test_no_gain_does_not_pass_and_sample_extension_is_rejected(self):
        r=contrast([.5]*400,[.5]*400)
        self.assertEqual(r['ci95'],[0.,0.]);self.assertFalse(r['learning_signal_passed'])
        with self.assertRaises(ValueError):contrast([.5]*401,[.5]*400)
