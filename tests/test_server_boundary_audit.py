import unittest
from unittest.mock import patch
from ptcg_lab.server_boundary_audit import REMOTE,WORKER,duration_seconds,ssh_command


class ServerBoundaryAuditTests(unittest.TestCase):
    def test_frozen_remote_programs_compile_without_executing(self):
        compile(WORKER,'boundary_worker.py','exec')
        compile(REMOTE.replace('PARAMETERS',repr('{}')),'boundary_bootstrap.py','exec')
    def test_actual_systemd_duration_formats_are_checked(self):
        self.assertEqual(duration_seconds('14s'),14)
        self.assertEqual(duration_seconds('1min 2s'),62)
        self.assertAlmostEqual(duration_seconds('14s 500ms'),14.5)
        self.assertEqual(duration_seconds('14000000'),14)
        with self.assertRaises(ValueError):duration_seconds('infinity')
    def test_ssh_uses_only_authorized_host_and_existing_key_without_reading_it(self):
        config={'host':'TRAINING_HOST','port':15134,'user':'TRAINING_USER',
                'identity_file':r'SSH_IDENTITY_FILE'}
        with patch('pathlib.Path.is_file',return_value=True),patch('pathlib.Path.read_bytes') as read:
            command=ssh_command(config);read.assert_not_called()
        self.assertIn('BatchMode=yes',command);self.assertIn('StrictHostKeyChecking=yes',command)
        with self.assertRaises(ValueError):ssh_command(dict(config,user='other'))
