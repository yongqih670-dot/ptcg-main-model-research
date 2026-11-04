import random,unittest
from ptcg_lab.phase_timing import Timer

class PhaseTimingTests(unittest.TestCase):
    def test_nested_exclusive_partition_and_inclusive_not_additive(self):
        now=[0.];timer=Timer(lambda:now[0])
        with timer.span('save'):
            now[0]=2
            with timer.span('hash'):now[0]=5
            now[0]=7
        now[0]=10;report=timer.report()
        self.assertEqual(report['exclusive_seconds'],{'initialization_restore':3.,'save':4.,'hash':3.})
        self.assertEqual(report['inclusive_seconds_nonadditive'],{'hash':3.,'save':7.})
        self.assertEqual(report['exclusive_partition_residual_seconds'],0.)
    def test_exception_unwinds_and_does_not_draw_rng(self):
        before=random.getstate();timer=Timer()
        with self.assertRaises(ValueError):
            with timer.span('failure'):raise ValueError('expected')
        self.assertEqual(timer.stack,[]);self.assertEqual(random.getstate(),before)
        self.assertAlmostEqual(timer.report()['exclusive_partition_residual_seconds'],0.,places=10)
    def test_long_live_span_is_included_in_snapshot(self):
        now=[0.];timer=Timer(lambda:now[0])
        with timer.span('wait'):
            now[0]=50;report=timer.report()
            self.assertEqual(report['exclusive_seconds']['wait'],50)
            self.assertEqual(report['inclusive_seconds_nonadditive']['wait'],50)

