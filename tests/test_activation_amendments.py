import tempfile
from pathlib import Path
import unittest
from ptcg_lab.io import atomic_json,sha256
from ptcg_lab.stage_control import activation_protected_hashes


class ActivationAmendmentTests(unittest.TestCase):
    def test_original_frozen_protections_survive_a_hash_bound_amendment(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'original.json';new=root/'amendment.json'
            hashes={'research/opponents/release_anchor.json':'fixed-anchor','research/opponents/final_holdout_v2.json':'fixed-holdout'}
            atomic_json(old,{'status':'complete','cycle_id':'same','before_sha256':hashes})
            atomic_json(new,{'status':'complete','cycle_id':'same','parent_activation_receipt':'original.json','parent_activation_receipt_sha256':sha256(old)})
            self.assertEqual(activation_protected_hashes(root,'amendment.json'),hashes)
            atomic_json(old,{'status':'complete','cycle_id':'same','before_sha256':{'anchor':'changed'}})
            with self.assertRaisesRegex(RuntimeError,'parent changed'):activation_protected_hashes(root,'amendment.json')
    def test_incomplete_or_new_cycle_parent_cannot_reset_protection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'original.json';new=root/'amendment.json'
            for status,cycle in [('preparing','same'),('complete','different')]:
                atomic_json(old,{'status':status,'cycle_id':cycle,'before_sha256':{'anchor':'a'}})
                atomic_json(new,{'status':'complete','cycle_id':'same','parent_activation_receipt':'original.json','parent_activation_receipt_sha256':sha256(old)})
                with self.assertRaises(RuntimeError):activation_protected_hashes(root,'amendment.json')
