import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from ptcg_lab.server_trial_agent import make_agent,validate_blob
from ptcg_lab.server_trial_worker import materialize
from ptcg_lab.io import atomic_json,sha256


class ServerTrialTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        path=Path(__file__).resolve().parents[1]/'ptcg_lab/server_environment_agent.py'
        spec=importlib.util.spec_from_file_location('trial_control_base_fixture',path);module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'fcntl':types.SimpleNamespace()}):spec.loader.exec_module(module)
        module.ROOT=self.root;self.agent=make_agent(module.__dict__)
        job=self.root/'jobs/current';job.mkdir(parents=True);self.agent.job=job
        for name in ['stage','requests']:(job/name).mkdir()
        self.agent.admit=lambda *_:None;self.agent.reserved_future=0
        self.agent.ticket={'manifest_sha256':'a'*64,'mode':'short_gpu_trial','gpu_index':6,'gpu_uuid':'GPU-selected'}
    def tearDown(self):self.tmp.cleanup()
    def test_input_sha_required_before_service_start(self):
        data=b'full original actor input';digest=hashlib.sha256(data).hexdigest()
        self.agent.manifest={'files':[{'file':digest+'.blob','sha256':digest,'bytes':len(data),'expanded_bytes':0}]}
        self.agent.begin(0)
        self.agent.chunk({'index':0,'offset':0,'sha256':hashlib.sha256(b'x'*len(data)).hexdigest()},b'x'*len(data))
        with self.assertRaises(ValueError):self.agent.commit(0)
        self.assertFalse((self.agent.job/'stage'/(digest+'.blob')).exists())
    def test_legitimate_empty_init_module_transfers_and_commits(self):
        digest=hashlib.sha256(b'').hexdigest();item={'file':digest+'.blob','sha256':digest,'bytes':0,'expanded_bytes':0}
        validate_blob(item);self.agent.manifest={'files':[item]};self.agent.begin(0);self.agent.commit(0)
        self.assertEqual((self.agent.job/'stage'/(digest+'.blob')).read_bytes(),b'')
        with self.assertRaises(ValueError):validate_blob({**item,'sha256':'0'*64,'file':'0'*64+'.blob'})
    def test_gpu_rechecked_immediately_before_launch(self):
        self.agent.manifest={'files':[]}
        responses=[types.SimpleNamespace(stdout='6, GPU-selected, 4096, 0\n'),types.SimpleNamespace(stdout='GPU-selected, 1234\n')]
        with patch('ptcg_lab.server_trial_agent.subprocess.run',side_effect=responses):
            with self.assertRaisesRegex(RuntimeError,'ceased to be idle'):self.agent.start()
        self.assertFalse(self.agent.started)
    def test_artifact_reads_cannot_escape_project(self):
        with self.assertRaises(ValueError):self.agent.read_file({'path':'../../../../outside','offset':0,'count':64})
    def test_materialized_input_preserves_source_and_hash(self):
        job=self.agent.job;(job/'control').mkdir();data=b'public metadata';digest=hashlib.sha256(data).hexdigest()
        blob=job/'stage'/(digest+'.blob');blob.write_bytes(data)
        manifest={'file_map':[{'path':'public/data.json','bytes':len(data),'sha256':digest}]}
        materialize(self.root,job,manifest)
        self.assertEqual((job/'control/public/data.json').read_bytes(),data);self.assertEqual(blob.read_bytes(),data)
    def test_changed_blob_is_rejected_before_materialization(self):
        job=self.agent.job;digest='a'*64;(job/'stage'/(digest+'.blob')).write_bytes(b'changed')
        with self.assertRaises(ValueError):materialize(self.root,job,{'file_map':[{'path':'x','bytes':7,'sha256':digest}]})
        self.assertFalse((job/'control/x').exists())
    def test_accepted_parent_can_reuse_hash_bound_ancestor_inputs(self):
        blob_job=self.root/'jobs/blob';(blob_job/'stage').mkdir(parents=True)
        data=b'immutable original actor';digest=hashlib.sha256(data).hexdigest();(blob_job/'stage'/(digest+'.blob')).write_bytes(data)
        mapping=[{'path':'input/model.pt','bytes':len(data),'sha256':digest}]
        atomic_json(blob_job/'training-manifest.raw.json',{'file_map':mapping})
        passed=self.root/'jobs/passed';passed.mkdir()
        atomic_json(passed/'training-manifest.raw.json',{'file_map':mapping,'input_capsule':{'run_id':'blob','manifest_sha256':sha256(blob_job/'training-manifest.raw.json')}})
        atomic_json(passed/'trial-metrics.json',{'basic_cpu_correctness_passed':True})
        manifest={'parent_capsule':{'run_id':'passed','manifest_sha256':sha256(passed/'training-manifest.raw.json'),
            'remote_basic_metrics_sha256':sha256(passed/'trial-metrics.json')}}
        materialize(self.root,self.agent.job,manifest)
        self.assertEqual((self.agent.job/'control/input/model.pt').read_bytes(),data)
    def test_remote_gpu_trial_is_charged_as_gpu(self):
        path=Path(__file__).resolve().parents[1]/'scripts/run_experiment.py'
        spec=importlib.util.spec_from_file_location('trial_supervisor_fixture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.compute_kind({'job_kind':'remote_train_ppo'}),'gpu')
        self.assertEqual(module.compute_kind({'job_kind':'remote_training_check'}),'cpu')
