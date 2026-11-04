import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ptcg_lab.io import atomic_json,sha256
from ptcg_lab.remote_lease import validate_no_unreconciled_remote
from ptcg_lab.server_workspace import admit,inside


class HybridCapacityLeaseTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_installation_growth_and_free_reserve_are_admission_constraints(self):
        (self.root/'input.whl').write_bytes(b'x'*64)
        with patch('shutil.disk_usage',return_value=type('Usage',(),{'free':21*2**30})()):
            self.assertEqual(admit(self.root,100)['workspace_bytes'],64)
            with self.assertRaises(OSError):admit(self.root,2*2**30)
        with patch('ptcg_lab.server_workspace.logical_bytes',return_value=10*2**30-1),patch('shutil.disk_usage',return_value=type('Usage',(),{'free':100*2**30})()):
            with self.assertRaises(OSError):admit(self.root,2)
    def test_paths_cannot_traverse_outside_project(self):
        with self.assertRaises(ValueError):inside(self.root,self.root/'..'/'other')
        self.assertEqual(inside(self.root,self.root/'jobs'/'own'),self.root/'jobs'/'own')
    def test_dead_local_wrapper_never_clears_server_job(self):
        lease={'run_id':'run','attempt':1,'lease_epoch':3,'host':'host','unit':'own.service','identity':{'invocation_id':'a'}}
        with self.assertRaisesRegex(RuntimeError,'Unreconciled'):validate_no_unreconciled_remote(self.root,{'remote_job':lease})
    def test_only_hash_bound_same_identity_terminal_evidence_reopens_gate(self):
        properties={'unit':'own.service','LoadState':'loaded','InvocationID':'a'*32,
                    'ExecMainStartTimestampMonotonic':'1','ActiveState':'active','SubState':'exited'}
        handle={'unit':'own.service','invocation_id':'a'*32,'start_monotonic':'1','host_boot_id':'boot'}
        lease={'run_id':'run','attempt':1,'lease_epoch':3,'host':'host','unit':'own.service','identity':handle,'deadline_unix':100}
        receipt={**lease,'host_boot_id':'boot','service_properties':properties,'server_unix':200}
        path=self.root/'terminal.json';atomic_json(path,receipt)
        lease.update(terminal_receipt_path='terminal.json',terminal_receipt_sha256=sha256(path))
        validate_no_unreconciled_remote(self.root,{'remote_job':lease})
        receipt['lease_epoch']=2;atomic_json(path,receipt);lease['terminal_receipt_sha256']=sha256(path)
        with self.assertRaisesRegex(RuntimeError,'identity differs'):validate_no_unreconciled_remote(self.root,{'remote_job':lease})
    def test_missing_unit_before_deadline_or_running_invocation_never_reopens_gate(self):
        handle={'unit':'own.service','invocation_id':'a'*32,'start_monotonic':'1','host_boot_id':'boot'}
        lease={'run_id':'run','attempt':1,'lease_epoch':3,'host':'host','unit':'own.service','identity':handle,'deadline_unix':100}
        receipt={**lease,'host_boot_id':'boot','server_unix':110,'service_properties':{'LoadState':'not-found'}}
        path=self.root/'terminal.json';atomic_json(path,receipt)
        lease.update(terminal_receipt_path='terminal.json',terminal_receipt_sha256=sha256(path))
        with self.assertRaisesRegex(RuntimeError,'before deadline'):validate_no_unreconciled_remote(self.root,{'remote_job':lease})
        receipt['server_unix']=161;atomic_json(path,receipt);lease['terminal_receipt_sha256']=sha256(path)
        validate_no_unreconciled_remote(self.root,{'remote_job':lease})
        receipt['service_properties']={'unit':'own.service','LoadState':'loaded','InvocationID':'a'*32,
                    'ExecMainStartTimestampMonotonic':'1','ActiveState':'active','SubState':'running'}
        atomic_json(path,receipt);lease['terminal_receipt_sha256']=sha256(path)
        with self.assertRaisesRegex(RuntimeError,'not terminated'):validate_no_unreconciled_remote(self.root,{'remote_job':lease})
