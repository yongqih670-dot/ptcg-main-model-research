import unittest
from ptcg_lab.server_trial_agent import choose_idle_gpu


class IdleGPUSelectionTests(unittest.TestCase):
    def rows(self):return [f'{i}, GPU-00000000-0000-0000-0000-{i:012d}, 2, 0' for i in range(8)]
    def test_authorized_priority_and_compute_users_are_enforced(self):
        rows=self.rows();self.assertEqual(choose_idle_gpu(rows,'')[0],'6')
        apps='GPU-00000000-0000-0000-0000-000000000006, 123'
        self.assertEqual(choose_idle_gpu(rows,apps)[0],'7')
        apps+=' GPU-00000000-0000-0000-0000-000000000007, 124'
        self.assertEqual(choose_idle_gpu(rows,apps)[0],'4')
    def test_no_permission_from_unapproved_empty_gpu_or_unknown_occupied_card(self):
        rows=[r if int(r.split(',')[0])<4 else r.replace(', 2, 0',', 500, 2') for r in self.rows()]
        self.assertIsNone(choose_idle_gpu(rows,''))
