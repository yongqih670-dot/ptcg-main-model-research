import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from ptcg_lab.io import atomic_json
from ptcg_lab.processes import KillOnCloseJob
from scripts.run_experiment import allowed_seconds, verify_snapshot, validate_config_access,collect_metrics,compute_kind,validate_plateau,hypothesis_seconds,validate_cycle_open


class SupervisorTests(unittest.TestCase):
    def test_terminal_cycle_cannot_be_bypassed_by_job_kind(self):
        for status in ('plateau','budget_exhausted','stopped_by_user','submitted_verified'):
            with self.assertRaisesRegex(RuntimeError,'terminal'):validate_cycle_open({'status':status})
        validate_cycle_open({'status':'running'})
    def test_hypothesis_budget_combines_runs_and_orphan_caps(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'runs/a').mkdir(parents=True);(root/'runs/b').mkdir();(root/'runs/c').mkdir()
            for name,hyp in [('a','H'),('b','H'),('c','other')]:atomic_json(root/'runs'/name/'config.json',{'hypothesis_id':hyp})
            atomic_json(root/'runs/a/attempts.json',[{'attempt':1,'seconds':30}])
            atomic_json(root/'runs/c/attempts.json',[{'attempt':1,'seconds':999}])
            state={'recovery_accounting':[{'previous':{'run_id':'b','attempt':1,'limit_seconds':100}},
                                          {'previous':{'run_id':'a','attempt':1,'limit_seconds':100}}]}
            with patch('scripts.run_experiment.ROOT',root):self.assertEqual(hypothesis_seconds({'hypothesis_id':'H'},state),130)
    def test_gpu_reserve_and_plateau(self):
        p={'budget':{'gpu_job_wall_hours_max':60,'gpu_job_wall_hours_reserved_final':8},'execution':{'job_wall_hours_max':2},
           'stopping':{'consecutive_valid_experiments_without_promotion':5,'additional_low_cost_pivot_experiments_max':2}}
        s={'budget_used':{'gpu_job_wall_hours':52},'consecutive_valid_runs_without_promotion':5}
        self.assertEqual(compute_kind({'job_kind':'train_transformer'}),'gpu')
        self.assertEqual(allowed_seconds(p,s,False,300,'gpu'),0)
        self.assertEqual(allowed_seconds(p,s,True,300,'gpu'),300)
        with self.assertRaisesRegex(RuntimeError,'Plateau'):validate_plateau({},p,s,True,False)
        validate_plateau({'low_cost_pivot':True},p,s,True,False)
        s['started_pivot_runs']=2
        with self.assertRaisesRegex(RuntimeError,'allowance'):validate_plateau({'low_cost_pivot':True},p,s,True,False)
    def test_only_reviewed_jobs_and_explicit_completion(self):
        policy={'execution':{'cpu_workers_max':12}}
        with self.assertRaisesRegex(ValueError,'job kind'):validate_config_access({'job_kind':'arbitrary'},policy,False)
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);cfg={'job_kind':'prepare_bc'}
            self.assertFalse(collect_metrics(run,cfg)['complete'])
            atomic_json(run/'metrics.json',{'complete':False,'finished_shards':3})
            self.assertFalse(collect_metrics(run,cfg)['complete'])
            atomic_json(run/'metrics.json',{'complete':True,'finished_shards':4})
            self.assertTrue(collect_metrics(run,cfg)['complete'])
    def test_holdout_and_worker_limits(self):
        policy={'execution':{'cpu_workers_max':12}}
        config={'workers':8,'candidate':{'path':'agents/holdout/ppo-holdout-1'}}
        with self.assertRaisesRegex(ValueError,'holdout'):validate_config_access(config,policy,False)
        validate_config_access(config,policy,True)
        config['workers']=13
        with self.assertRaisesRegex(ValueError,'Worker'):validate_config_access(config,policy,True)

    def test_final_budget_reserve_cannot_be_spent_by_development(self):
        policy={'budget':{'cpu_job_wall_hours_max':60,'cpu_job_wall_hours_reserved_final':12},
                'execution':{'job_wall_hours_max':2}}
        state={'budget_used':{'cpu_job_wall_hours':48}}
        self.assertEqual(allowed_seconds(policy,state,False,120),0)
        self.assertEqual(allowed_seconds(policy,state,True,120),120)
        state['budget_used']['cpu_job_wall_hours']=60
        self.assertEqual(allowed_seconds(policy,state,True,120),0)

    def test_tampered_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'source').mkdir()
            (root/'source/main.py').write_text('tampered=True\n')
            atomic_json(root/'source-manifest.json',{'main.py':'wrong-hash'})
            with self.assertRaisesRegex(RuntimeError,'snapshot changed'):
                verify_snapshot(root)

    @unittest.skipUnless(os.name=='nt','Windows supervisor lifetime contract')
    def test_job_close_terminates_sleeping_child(self):
        job=KillOnCloseJob()
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
        try:
            job.assign(child.pid)
            self.assertIsNone(child.poll())
            job.close()
            child.wait(timeout=3)
            self.assertIsNotNone(child.poll())
        finally:
            job.close()
            if child.poll() is None:
                child.kill(); child.wait(timeout=3)


if __name__=='__main__':
    unittest.main()
