from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from ptcg_lab.io import atomic_json
from ptcg_lab.io_pacing import BytePacer
from ptcg_lab.server_resource_audit import evaluate,audit_scope_seconds
from ptcg_lab.process_guardian import WorkerSession


class ResourceAuditTests(unittest.TestCase):
    def test_guard_revision_does_not_reset_original_cumulative_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,task,seconds in [('old','original',50),('repair','different-name',17),('other','unrelated',99)]:
                run=root/'runs'/name
                atomic_json(run/'config.json',{'job_kind':'server_resource_audit' if name!='other' else 'evaluation','hypothesis_id':task})
                atomic_json(run/'attempts.json',[{'seconds':seconds}])
            self.assertEqual(audit_scope_seconds(root),67)
            self.assertEqual(audit_scope_seconds(root,root/'runs/repair'),50)
    def test_remote_audit_sources_compile(self):
        source=Path(__file__).resolve().parents[1]/'ptcg_lab'
        for name in ['server_resource_agent.py','server_resource_worker.py','process_guardian.py']:
            compile((source/name).read_bytes(),name,'exec')
    def test_control_pacer_observes_cumulative_byte_deadlines(self):
        now=[0.]
        pacer=BytePacer(100,10**12,clock=lambda:now[0],sleep=lambda seconds:now.__setitem__(0,now[0]+seconds))
        pacer.consume(100);self.assertGreaterEqual(now[0],1)
        pacer.consume(200);self.assertGreaterEqual(now[0],3)
    def test_changed_worker_identity_cannot_receive_group_signal(self):
        session=object.__new__(WorkerSession)
        session.root={'pid':123,'uid':1005,'start_ticks':'1','process_group':123,'session':123}
        with patch('ptcg_lab.process_guardian.process_record',return_value={**session.root,'start_ticks':'2'}),patch('os.killpg',create=True) as kill:
            with self.assertRaises(RuntimeError):session.signal(19)
            kill.assert_not_called()
    def test_requested_properties_and_partial_logs_do_not_pass_real_acceptance(self):
        final={'artifacts':{},'terminal':True,'service_properties':{'Result':'exit-code'}}
        result=evaluate('io_stop',{},final)
        self.assertFalse(result['complete']);self.assertFalse(result['checks']['cpu_kernel_limit'])
        self.assertFalse(result['checks']['io_throttle_exercised'])
    def test_normal_completion_requires_private_install_and_exit_barrier(self):
        final={'artifacts':{},'terminal':True,'service_properties':{'Result':'success'}}
        result=evaluate('normal_completion',{},final)
        self.assertFalse(result['complete']);self.assertFalse(result['checks']['private_inprocess_pip_install'])
        self.assertFalse(result['checks']['exit_barrier_acknowledged'])
