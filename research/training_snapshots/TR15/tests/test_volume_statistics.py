import unittest
from ptcg_lab.volume_statistics import primary,secondary

class StatisticsTests(unittest.TestCase):
    def test_primary_reports_exact_positive_and_negative_whole_series_signal(self):
        result=primary({'TR11-vs-ppo':[0.]*1600,'endpoint-vs-ppo':[1.]*1600})
        self.assertEqual(result['gain'],1.);self.assertEqual(result['ci95'],[1.,1.])
        self.assertTrue(result['learning_signal_passed'])
        result=primary({'TR11-vs-ppo':[1.]*1600,'endpoint-vs-ppo':[0.]*1600})
        self.assertFalse(result['learning_signal_passed']);self.assertEqual(result['ci95'],[-1.,-1.])
    def test_no_missing_or_extra_samples_and_no_error_as_loss(self):
        for rows in ([1.]*1599,[1.]*1601,[1.]*1599+[2.]):
            with self.assertRaises(ValueError):primary({'TR11-vs-ppo':[0.]*1600,'endpoint-vs-ppo':rows})
    def test_opponent_weights_and_three_strategies_are_preserved(self):
        opponents=['lucario','alakazam','archaludon','ppo','tr01','tr02','tr03','tr09']
        groups={f'{role}-vs-{o}':[value]*100 for o in opponents for role,value in [('endpoint',1.),('TR11',.5),('anchor',0.)]}
        result=secondary(groups,opponents)
        for name,n in [('full_eight',8),('original_four',4)]:
            panel=result['panels'][name];self.assertEqual(list(panel['weights'].values()),[1/n]*n)
            self.assertEqual(panel['contrasts']['endpoint-minus-TR11']['ci95'],[.5,.5])
            self.assertEqual(panel['contrasts']['endpoint-minus-anchor']['gain'],1.)

if __name__=='__main__':unittest.main()
