import unittest
import numpy as np
import torch
from ptcg_lab.server_io_check import same
from scripts.run_experiment import compute_kind
import tests.test_shared_server_safety as fixtures


class ServerIOCheckTests(unittest.TestCase):
    def test_full_nested_model_optimizer_and_rng_comparison_detects_changes(self):
        a={'model':{'x':torch.arange(4)},'optimizer':{'group':[1,2]},'rng':('numpy',np.arange(8),torch.arange(8))}
        b={'model':{'x':torch.arange(4)},'optimizer':{'group':[1,2]},'rng':('numpy',np.arange(8),torch.arange(8))}
        self.assertTrue(same(a,b));b['rng'][1][3]=99;self.assertFalse(same(a,b))
        self.assertFalse(same({'a':[1]}, {'a':(1,)}))
        self.assertFalse(same(torch.arange(4),torch.arange(4).float()))
        self.assertFalse(same(np.arange(4),np.arange(4).astype(np.float32)))

    def test_acceptance_is_cpu_only_and_cannot_become_scientific_training(self):
        self.assertEqual(compute_kind({'job_kind':'remote_training_check'}),'cpu')
        f=fixtures.SharedServerSafetyTests('test_idle_detection_or_preparing_status_never_grants_resource_authorization');f.setUp()
        try:
            from ptcg_lab.shared_server_safety import validate_shared_server_job
            base={'job_kind':'remote_training_check','check_variant':'journal_io','engineering_only':True,
                  'no_training_updates':True,'workers':1,'maximum_attempts':1,'check_wall_seconds':900}
            for changed in [{'engineering_only':False},{'job_kind':'remote_train_ppo'},{'gpu_index':6},{'workers':2},{'maximum_attempts':2}]:
                with self.assertRaises(RuntimeError):validate_shared_server_job(f.root,dict(base,**changed),f.state)
        finally:f.tearDown()


if __name__=='__main__':unittest.main()
