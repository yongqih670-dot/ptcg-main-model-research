import copy
import unittest
from unittest.mock import patch
from ptcg_lab.systemd_job import service_command,terminal,identity,stop_verified


class SystemdJobTests(unittest.TestCase):
    def setUp(self):
        self.ticket={'unit':'ptcg-hybrid-test-12345678.service','kind':'environment_setup','project_root':'/home/user/ptcg',
                     'job_directory':'/home/user/ptcg/jobs/run','ticket_path':'/home/user/ptcg/jobs/run/ticket.json',
                     'run_id':'run','deadline_unix':1000,'memory_max_bytes':22*2**30,'workspace_hard_max_bytes':10*2**30}
    def test_worker_has_kernel_runtime_memory_and_group_stop_boundaries(self):
        command=service_command(self.ticket,now=400)
        self.assertIn('--property=RuntimeMaxSec=550',command);self.assertIn('--property=TimeoutStopSec=45s',command)
        self.assertIn('--property=KillMode=control-group',command);self.assertIn('--expand-environment=no',command)
        self.assertIn('--setenv=CUDA_VISIBLE_DEVICES=',command)
    def test_unowned_names_outside_paths_or_budgets_cannot_launch(self):
        for change in [{'unit':'other-user.service'},{'ticket_path':'/etc/passwd'},{'memory_max_bytes':23*2**30},
                       {'workspace_hard_max_bytes':11*2**30},{'deadline_unix':9000},{'kind':'evaluation'}]:
            with self.assertRaises(ValueError):service_command(dict(self.ticket,**change),now=400)
    def test_ppo_requires_venv_and_explicit_uuid(self):
        ticket=dict(self.ticket,kind='train_ppo')
        with self.assertRaises(ValueError):service_command(ticket,now=400)
        ticket.update(python_kind='venv',gpu_uuid='AUTHORIZED_GPU_UUID')
        command=service_command(ticket,now=400)
        self.assertIn('--setenv=CUDA_VISIBLE_DEVICES='+ticket['gpu_uuid'],command)
    def test_active_exited_is_terminal_but_running_or_deactivating_is_not(self):
        self.assertTrue(terminal({'ActiveState':'active','SubState':'exited'}))
        self.assertTrue(terminal({'LoadState':'not-found'}))
        self.assertFalse(terminal({'ActiveState':'active','SubState':'running'}))
        self.assertFalse(terminal({'ActiveState':'deactivating','SubState':'stop-sigterm'}))
    def test_changed_invocation_cannot_stop_any_service(self):
        properties={'unit':self.ticket['unit'],'LoadState':'loaded','InvocationID':'a'*32,'ExecMainStartTimestampMonotonic':'1'}
        handle=identity(properties,'boot');changed=dict(properties,InvocationID='b'*32)
        with patch('ptcg_lab.systemd_job.inspect_service',return_value=changed),patch('pathlib.Path.read_text',return_value='boot'),patch('subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError,'identity changed'):stop_verified(handle)
            run.assert_not_called()
