import tempfile
import unittest
from pathlib import Path
from ptcg_lab.io import atomic_json,append_series
from scripts.analyze_panel import analyze


class AnalysisTests(unittest.TestCase):
    def test_series_stratified_comparison_and_failure_gate(self):
        for failure in (False,True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                run=Path(d)
                atomic_json(run/'config.json',{'series_count':4,'contrasts':[
                    {'candidate_cell':'new','anchor_cell':'old','weight':1.0}]})
                for i in range(4):
                    result='error' if failure and i==0 else 'win' if i<2 else 'loss'
                    append_series(run/'series.jsonl',{
                        'series_id':i,'cell_id':'new' if i<2 else 'old','result':result,
                        'candidate_seat':i%2,'seconds':1,'policy_invalid_count':int(result=='error'),
                        'statuses':['ERROR'] if result=='error' else ['DONE','DONE'],'round_results':[2,0,0]})
                report=analyze(run)
                self.assertEqual(report['classification'],'invalid' if failure else 'improved')
                self.assertFalse(report['promotion_eligible'])
                self.assertEqual(report['contrast']['gain'],.5 if failure else 1.0)
                if not failure: self.assertEqual(report['contrast']['ci95'],[1.0,1.0])


if __name__=='__main__': unittest.main()
