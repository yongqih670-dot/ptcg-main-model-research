from pathlib import Path
import os
import tempfile
import time
import unittest
from unittest.mock import patch
from ptcg_lab.io import atomic_json,sha256
from ptcg_lab.shared_server_safety import validate_shared_server_job,validate_gpu_choice


class SharedServerSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        path=self.root/'authorization.json'
        atomic_json(path,{'allowed_gpu_indices':[4,5,6,7],'maximum_concurrent_gpus':1})
        self.state={'shared_server_safety':{'authorization':{'path':'authorization.json','sha256':sha256(path)}}}
    def tearDown(self):self.tmp.cleanup()
    def test_idle_detection_or_preparing_status_never_grants_resource_authorization(self):
        for state in [{},{'shared_server_safety':{'gpu_idle':True}},{'execution_target':{'required':'hybrid','status':'active'}}]:
            with self.assertRaises(RuntimeError):validate_shared_server_job(self.root,{'job_kind':'server_environment'},state)
    def test_old_environment_job_cannot_bypass_new_io_acceptance(self):
        with self.assertRaisesRegex(RuntimeError,'acceptance is pending'):
            validate_shared_server_job(self.root,{'job_kind':'server_environment'},self.state)
    def test_only_small_registered_resource_probe_can_precede_acceptance(self):
        validate_shared_server_job(self.root,{'job_kind':'server_resource_audit','audit_total_io_bytes_max':32*2**20},self.state)
        with self.assertRaises(RuntimeError):validate_shared_server_job(self.root,{'job_kind':'server_resource_audit','audit_total_io_bytes_max':128*2**20},self.state)
    def test_partial_acceptance_or_changed_authorization_is_rejected(self):
        path=self.root/'acceptance.json';atomic_json(path,{'checks':{'cpu_kernel_limit':True,'memory_kernel_limit':True}})
        self.state['shared_server_safety']['acceptance']={'path':'acceptance.json','sha256':sha256(path)}
        with self.assertRaises(RuntimeError):validate_shared_server_job(self.root,{'job_kind':'server_environment'},self.state)
        atomic_json(self.root/'authorization.json',{'allowed_gpu_indices':[0]})
        with self.assertRaisesRegex(RuntimeError,'evidence changed'):validate_shared_server_job(self.root,{'job_kind':'server_environment'},self.state)
    def test_gpu_authorization_and_idle_condition_are_both_required(self):
        auth={'allowed_gpu_indices':[4,5,6,7],'maximum_concurrent_gpus':1}
        idle={'index':6,'uuid':'GPU-authorized','memory_used_mib':1,'utilization_percent':0}
        self.assertEqual(validate_gpu_choice(auth,idle,[]),'GPU-authorized')
        with self.assertRaises(RuntimeError):validate_gpu_choice(auth,dict(idle,index=0),[])
        with self.assertRaises(RuntimeError):validate_gpu_choice(auth,idle,[{'gpu_uuid':'GPU-authorized'}])
        with self.assertRaises(RuntimeError):validate_gpu_choice(auth,dict(idle,memory_used_mib=19738),[])
    def test_copied_profile_hash_cannot_authorize_an_unguarded_old_entrypoint(self):
        checks={k:True for k in ['directory_ownership_and_realpath','cpu_kernel_limit','memory_kernel_limit',
            'io_all_descendants_covered','io_throttle_exercised','io_excess_stops_owned_unit',
            'guard_failure_stops_owned_unit','disk_reserve_and_workset_stop','control_transfer_io_paced',
            'kernel_block_io_upper_bound_present','exited_child_kernel_block_io_retained',
            'zombie_has_independent_kernel_coverage','monitor_loss_stops_owned_unit',
            'clean_payload_completion','private_inprocess_pip_install','same_pid_exec_io_accounting','exit_barrier_acknowledged']}
        path=self.root/'acceptance.json'
        atomic_json(path,{'checks':checks,'authorization_sha256':self.state['shared_server_safety']['authorization']['sha256'],
                          'profile_sha256':'copied','approved_entrypoints':{}})
        self.state['shared_server_safety']['acceptance']={'path':'acceptance.json','sha256':sha256(path)}
        with self.assertRaisesRegex(RuntimeError,'no integrated guard acceptance'):
            validate_shared_server_job(self.root,{'job_kind':'server_environment','shared_guard_profile_sha256':'copied'},self.state)
    def test_server_json_writes_reject_realpath_escape_before_creating_directories(self):
        uid=os.getuid() if hasattr(os,'getuid') else self.root.stat().st_uid
        atomic_json(self.root/'.ptcg-owner.json',{'project_id':'test','uid':uid,'root':str(self.root.resolve())})
        with tempfile.TemporaryDirectory() as other,patch.dict(os.environ,{'PTCG_PROJECT_ROOT':str(self.root),'PTCG_PROJECT_ID':'test',
                'PTCG_JOB_DEADLINE_UNIX':str(time.time()+60)}):
            outside=Path(other)/'never-created/file.json'
            with self.assertRaises(ValueError):atomic_json(outside,{'bad':True})
            self.assertFalse(outside.parent.exists())
            atomic_json(self.root/'owned/ok.json',{'ok':True})
            self.assertTrue((self.root/'owned/ok.json').is_file())
