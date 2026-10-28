import tempfile
import unittest
from pathlib import Path
from ptcg_lab.agents import ProtocolAgent, validate_action
from ptcg_lab.io import append_series, read_series
from ptcg_lab.evaluate import summarize, recover_rows
from ptcg_lab.io import atomic_json


class ProtocolTests(unittest.TestCase):
    def test_callable_instance_accepts_framework_configuration(self):
        agent=ProtocolAgent('random',[3]*60,0,7)
        self.assertEqual(agent({'select':None},{'bo':3}),[3]*60)

    def test_multiselect_contract(self):
        obs = {'select': {'minCount': 1, 'maxCount': 2, 'option': [0, 1, 2]}}
        validate_action(obs, [0, 2])
        for bad in ([], [0, 0], [-1], [3], [0, 1, 2], [True]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_action(obs, bad)

    def test_recovery_keeps_only_committed_complete_records(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'series.jsonl'
            append_series(path, {'series_id': 0})
            with path.open('ab') as f:
                f.write(b'{"series_id":')
            with self.assertRaises(ValueError):
                read_series(path)
            self.assertEqual(read_series(path, recover_tail=True), [{'series_id': 0}])
            append_series(path, {'series_id': 0})
            with self.assertRaises(ValueError):
                read_series(path)

    def test_failure_not_dropped_from_score(self):
        rows = [{'result':r, 'candidate_seat':i%2, 'seconds':1,
                 'policy_invalid_count':int(r=='error'), 'statuses':['ERROR'] if r=='error' else ['DONE','DONE'],
                 'round_results':[2,0,0]} for i,r in enumerate(['win','draw','loss','error'])]
        m = summarize(rows)
        self.assertEqual(m['series_count'],4)
        self.assertEqual(m['score_with_errors_counted_as_losses'],.375)
        self.assertEqual(m['invalid_actions'],1)

    def test_worker_result_recovery_is_exactly_once(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            atomic_json(path/'results/00002.json',{'series_id':2,'result':'loss'})
            rows=recover_rows(path)
            self.assertEqual(rows,[{'series_id':2,'result':'loss'}])
            self.assertEqual(recover_rows(path),rows)
            atomic_json(path/'results/00002.json',{'series_id':2,'result':'win'})
            with self.assertRaisesRegex(ValueError,'differs'):recover_rows(path)


if __name__ == '__main__':
    unittest.main()
