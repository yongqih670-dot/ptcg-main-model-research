import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from ptcg_lab.io import atomic_json,sha256
from ptcg_lab.server_environment import archive_receipt,durable_bytes
from ptcg_lab.systemd_job import service_command


class ServerEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        path=Path(__file__).resolve().parents[1]/'ptcg_lab/server_environment_agent.py'
        spec=importlib.util.spec_from_file_location('isolated_agent_fixture',path);self.agent_module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'fcntl':types.SimpleNamespace()}):spec.loader.exec_module(self.agent_module)
        self.agent_module.ROOT=self.root
    def tearDown(self):self.tmp.cleanup()
    def agent(self,data=b'valid-byte-content'):
        job=self.root/'job';job.mkdir();(job/'stage').mkdir();(job/'requests').mkdir()
        agent=self.agent_module.Agent();agent.job=job;agent.started=True
        agent.manifest={'files':[{'file':'fixture.whl','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'expanded_bytes':len(data)}]}
        agent.reserved_future=0;agent.ticket={'manifest_sha256':'f'*64};agent.admit=lambda *_:None
        return agent
    def test_exact_binary_frame_requires_full_payload(self):
        self.assertEqual(self.agent_module.exact(io.BytesIO(b'abc'),3),b'abc')
        with self.assertRaises(EOFError):self.agent_module.exact(io.BytesIO(b'abc'),4)
    def test_small_clock_handshake_rejects_real_skew(self):
        agent=self.agent_module.Agent()
        with patch.object(self.agent_module.time,'time',return_value=100):
            self.assertEqual(agent.check_clock({'client_unix':99})['server_unix'],100)
            self.assertEqual(agent.clock_checked_at,100)
            with self.assertRaisesRegex(ValueError,'Clock skew'):
                agent.check_clock({'client_unix':90})
    def test_chunk_cursor_and_full_artifact_identity_before_install_request(self):
        data=b'valid-byte-content';agent=self.agent(data);agent.begin(0)
        with self.assertRaises(ValueError):agent.chunk({'index':0,'offset':1,'sha256':hashlib.sha256(data).hexdigest()},data)
        agent.chunk({'index':0,'offset':0,'sha256':hashlib.sha256(data).hexdigest()},data);agent.commit(0)
        self.assertEqual((agent.job/'stage/fixture.whl').read_bytes(),data)
        self.assertTrue((agent.job/'requests/000.json').exists())
    def test_corrupt_whole_artifact_never_becomes_an_install_request(self):
        agent=self.agent();agent.begin(0);data=b'x'*len(b'valid-byte-content')
        agent.chunk({'index':0,'offset':0,'sha256':hashlib.sha256(data).hexdigest()},data)
        with self.assertRaises(ValueError):agent.commit(0)
        self.assertFalse((agent.job/'requests/000.json').exists());self.assertFalse((agent.job/'stage/fixture.whl').exists())
    def test_archive_ack_requires_durable_exact_receipt_and_both_logs(self):
        data=b'install-ok';v=b'1\n';item={'name':'fixture','version':'1','bytes':10,'sha256':'a'*64}
        record={'index':0,'name':'fixture','version':'1','source_bytes':10,'source_sha256':'a'*64,
                'log_sha256':hashlib.sha256(data).hexdigest(),'version_log_sha256':hashlib.sha256(v).hexdigest()}
        path=self.root/'server-receipt.json';atomic_json(path,record)
        response={'receipt':record,'receipt_base64':base64.b64encode(path.read_bytes()).decode(),'receipt_sha256':sha256(path),
                  'logs':{'000.log':{'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()},
                          '000-version.log':{'base64':base64.b64encode(v).decode(),'sha256':hashlib.sha256(v).hexdigest()}}}
        ack=archive_receipt(self.root/'archive',response,item,0)
        self.assertEqual(ack['receipt_sha256'],sha256(self.root/'archive/receipt.json'))
        self.assertEqual(ack['local_commit_sha256'],sha256(self.root/'archive/commit.json'))
        response['logs']['000.log']['base64']=base64.b64encode(b'corrupt').decode()
        with self.assertRaises(ValueError):archive_receipt(self.root/'bad',response,item,0)
        self.assertFalse((self.root/'bad/commit.json').exists())
    def test_per_job_frozen_control_path_is_inside_owned_project(self):
        ticket={'unit':'ptcg-hybrid-env-12345678.service','kind':'environment_setup','project_root':'/home/u/ptcg',
            'job_directory':'/home/u/ptcg/jobs/run','ticket_path':'/home/u/ptcg/jobs/run/ticket.json',
            'control_root':'/home/u/ptcg/jobs/run/control','run_id':'run','deadline_unix':1000,
            'memory_max_bytes':4*2**30,'workspace_hard_max_bytes':10*2**30}
        self.assertIn('/home/u/ptcg/jobs/run/control/environment_worker.py',service_command(ticket,now=400))
        with self.assertRaises(ValueError):service_command(dict(ticket,control_root='/tmp/foreign'),now=400)
    def test_agent_and_worker_sources_compile(self):
        base=Path(__file__).resolve().parents[1]/'ptcg_lab'
        for name in ['server_environment_agent.py','server_environment_worker.py']:
            compile((base/name).read_bytes(),name,'exec')
