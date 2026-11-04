import unittest
from ptcg_lab.local_cache_finalize import remaining_members


class PartialCacheProofTests(unittest.TestCase):
    def test_exact_prefix_receipts_leave_only_original_remaining_members(self):
        rows=[{'member':str(i),'bytes':i,'sha256':str(i)*64,'file_identity':[i]} for i in range(3)]
        self.assertEqual(remaining_members({'files':rows},[rows[0]]),rows[1:])
        for bad in [[rows[0],rows[0]],[{**rows[0],'sha256':'x'}],[{**rows[0],'member':'foreign'}]]:
            with self.assertRaises(ValueError):remaining_members({'files':rows},bad)

