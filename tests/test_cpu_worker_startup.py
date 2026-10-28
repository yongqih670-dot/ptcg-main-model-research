import os,sys,tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from ptcg_lab.io import atomic_json,sha256
from ptcg_lab.server_trial_worker import run


class AdmissionReached(RuntimeError):pass


class CPUWorkerStartupTests(unittest.TestCase):
    def test_CPU_worker_calls_global_capacity_guard_before_workload(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job=root/'job';job.mkdir()
            atomic_json(job/'training-manifest.raw.json',{'mode':'cpu_sampler_check'})
            ticket={'project_root':str(root),'project_id':'test','job_directory':str(job),
                'run_id':job.name,'mode':'cpu_sampler_check','deadline_unix':9999999999,
                'manifest_sha256':sha256(job/'training-manifest.raw.json')}
            atomic_json(job/'ticket.json',ticket)
            with patch.dict(os.environ,{'PTCG_JOB_RUN_ID':job.name,'CUDA_VISIBLE_DEVICES':''}),\
                 patch('ptcg_lab.server_trial_worker.owned_root',return_value=root),\
                 patch.object(sys,'prefix',str(root/'.venv')),\
                 patch('ptcg_lab.server_trial_worker.kernel_snapshot',return_value={}),\
                 patch('ptcg_lab.server_trial_worker.admit',side_effect=AdmissionReached):
                with self.assertRaises(AdmissionReached):run(job/'ticket.json')


if __name__=='__main__':unittest.main()
