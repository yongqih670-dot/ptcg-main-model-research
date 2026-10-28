import unittest
from ptcg_lab.storage_cadence import PROFILE,transfer_due,apply,capacity_signals


class StorageCadenceTests(unittest.TestCase):
    def test_batch_by_time_size_pressure_and_flush_terminal(self):
        self.assertFalse(transfer_due(PROFILE,100,599))
        self.assertTrue(transfer_due(PROFILE,100,600))
        self.assertTrue(transfer_due(PROFILE,2**30,0))
        self.assertTrue(transfer_due(PROFILE,1,0,terminal=True))
        self.assertTrue(transfer_due(PROFILE,1,0,workspace_warning=True))
        self.assertFalse(transfer_due(PROFILE,0,900,terminal=True))
    def test_legacy_config_unchanged_and_new_profile_cannot_relax_old_root(self):
        config={'consumed_reference_barrier_required':True}
        self.assertIs(apply(config,{},'/legacy'),config)
        with self.assertRaises(ValueError):apply(config,{'storage_cadence':PROFILE},'/legacy')
        with self.assertRaises(ValueError):transfer_due(dict(PROFILE,maximum_pending_bytes=20*2**30),10,0)
        with self.assertRaises(ValueError):transfer_due(PROFILE,10,-1)
    def test_warning_is_separate_from_hard_limit_and_missing_data_is_unknown(self):
        self.assertFalse(capacity_signals({'workspace_bytes':39*2**30,'free_bytes':100*2**30})['workspace_warning'])
        self.assertTrue(capacity_signals({'workspace_bytes':40*2**30,'free_bytes':100*2**30})['workspace_warning'])
        self.assertTrue(capacity_signals({'workspace_bytes':30*2**30,'forecast_growth_bytes':11*2**30,'free_bytes':30*2**30})['free_space_warning'])
        self.assertEqual(capacity_signals(None),{'capacity_observation_available':False})


if __name__=='__main__':unittest.main()
