import copy
import hashlib
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from ptcg_lab.storage_profile import validate,NEW,OLD,PROJECT,ticket_ceiling
from ptcg_lab.systemd_job import service_command


class StorageMigrationTests(unittest.TestCase):
    def setUp(self):
        authority={'authorized_target_root':NEW,'source_root_read_only':OLD,
            'server_total_project_working_set_hard_GiB':50,'server_free_disk_reserve_GiB':20,
            'old_copy_preserved_until_verified_and_evidence_archived':True}
        self.profile={'schema':'ptcg_HDD2_storage_v1','project_id':PROJECT,'root':NEW,
            'hard_max_bytes':50*2**30,'warning_bytes':40*2**30,'free_reserve_bytes':20*2**30,
            'retained_roots':[OLD],'authority':authority,
            'authority_canonical_sha256':hashlib.sha256(json.dumps(authority,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    def test_new_ceiling_requires_exact_authorized_new_root_and_retained_source(self):
        validate(self.profile,NEW)
        self.assertEqual(ticket_ceiling({'project_root':NEW,'storage_profile':self.profile}),50*2**30)
        self.assertEqual(ticket_ceiling({'project_root':OLD}),10*2**30)
        for change in [{'hard_max_bytes':51*2**30},{'retained_roots':[]},{'root':OLD},{'free_reserve_bytes':0}]:
            with self.assertRaises(ValueError):validate(dict(self.profile,**change),NEW)
        with self.assertRaises(ValueError):validate(self.profile,OLD)
    def test_no_GPU_and_kernel_limits_are_preserved_in_migration(self):
        ticket={'unit':'ptcg-hybrid-migrate-12345678.service','kind':'storage_migration','project_root':NEW,
            'job_directory':NEW+'/jobs/test','ticket_path':NEW+'/jobs/test/ticket.json','run_id':'test',
            'deadline_unix':1000,'memory_max_bytes':2*2**30,'workspace_hard_max_bytes':50*2**30,
            'storage_profile':self.profile,'cpu_quota_cores':2,'project_id':PROJECT}
        command=service_command(ticket,now=400)
        for expected in ['--setenv=CUDA_VISIBLE_DEVICES=','--property=CPUQuota=200%',
            '--property=MemoryMax='+str(2*2**30),'--property=RuntimeMaxSec=550',
            '--setenv=HOME='+NEW+'/tmp/home']:
            self.assertIn(expected,command)
        with self.assertRaises(ValueError):service_command(dict(ticket,workspace_hard_max_bytes=51*2**30),now=400)
    def test_incremental_capacity_keeps_old_copy_and_conservative_growth_reserve(self):
        from ptcg_lab.server_workspace import observed_bytes,_capacity_cache
        _capacity_cache.clear()
        with patch('ptcg_lab.server_workspace.owned_root',return_value=Path(OLD)),patch('ptcg_lab.server_workspace.logical_bytes',side_effect=[9*2**30,10*2**30,9*2**30,11*2**30]) as measure,patch('ptcg_lab.server_workspace.time.monotonic',side_effect=[100,105,111]):
            first=observed_bytes(NEW,self.profile);second=observed_bytes(NEW,self.profile);third=observed_bytes(NEW,self.profile)
        self.assertEqual(first,(19*2**30+256*2**20,9*2**30,False))
        self.assertEqual(second,(first[0],first[1],True));self.assertEqual(third[0],20*2**30+256*2**20)
        self.assertEqual(measure.call_count,4)


if __name__=='__main__':unittest.main()
