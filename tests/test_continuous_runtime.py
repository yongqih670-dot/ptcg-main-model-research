import copy
from pathlib import Path
import tempfile
import unittest
from ptcg_lab.continuous_runtime import local_limit,validate_remote_scope
from ptcg_lab.io import atomic_json,sha256


class RuntimeTests(unittest.TestCase):
    def test_explicit_three_attempt_grant_rejects_fourth_attempt_and_clock_reset(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'research/autonomy').mkdir(parents=True)
            runtime=root/'runtime.json';grant=root/'attempts.json'
            atomic_json(runtime,{'start_existing_TR11_after_all_gates':True,'TR11_training_wall_seconds_max':30000,
                'TR11_CPU_quota_cores':3,'new_scientific_attempts_added':0})
            atomic_json(grant,{'TR11_training_attempts_max':3,'TR11_research_raw_max':4,
                'project_research_raw_max':42,'original_absolute_remote_deadline_unix':100000})
            ref={'path':'attempts.json','sha256':sha256(grant)}
            atomic_json(root/'research/autonomy/policy.json',{'forward_authorizations':{'TR11_attempt3_raw42':ref}})
            c={'hypothesis_id':'TR11','check_variant':'continuous_mainline','job_kind':'remote_train_ppo',
                'engineering_only':False,'maximum_attempts':1,'training_attempt_index':3,
                'training_wall_seconds':30000,'hypothesis_max_training_attempts':3,
                'TR11_original_remote_deadline_unix':100000,'server_resource_profile':{'cpu_quota_cores':3},
                'TR11_runtime_authorization':{'path':'runtime.json','sha256':sha256(runtime)},
                'TR11_attempt_authorization':ref}
            self.assertEqual(local_limit(root,c),30000)
            for key,value in [('training_attempt_index',4),('TR11_original_remote_deadline_unix',100001),('hypothesis_max_training_attempts',4)]:
                changed=copy.deepcopy(c);changed[key]=value
                with self.assertRaises(ValueError):local_limit(root,changed)
    def test_extended_journal_is_fully_sized_and_default_limit_remains_unchanged(self):
        from ptcg_lab.resource_journal import capacity_profile
        self.assertEqual(capacity_profile()['maximum_journal_bytes'],8*2**20)
        with self.assertRaises(ValueError):capacity_profile(30000)
        p=capacity_profile(30000,extended_TR11=True)
        self.assertEqual(p['maximum_journal_bytes'],32*2**20)
        self.assertLess(p['planned_bytes_upper'],p['maximum_journal_bytes'])

    def test_extended_service_cannot_bypass_identity_stop_or_old_time_limits(self):
        from ptcg_lab.systemd_job import service_command
        scope={'seconds_max':30000,'CPU_quota_cores':3,'authorization_sha256':'a'*64}
        t={'unit':'ptcg-hybrid-test-12345678.service','kind':'train_ppo','python_kind':'venv',
            'project_root':'/home/user/ptcg','job_directory':'/home/user/ptcg/jobs/run',
            'ticket_path':'/home/user/ptcg/jobs/run/ticket.json','run_id':'run','deadline_unix':28800,
            'memory_max_bytes':22*2**30,'workspace_hard_max_bytes':10*2**30,'cpu_quota_cores':3,
            'gpu_uuid':'AUTHORIZED_GPU_UUID','mode':'derived_scientific',
            'new_diagnostic_identity':'TR11','TR11_runtime_scope':scope}
        cmd=service_command(t,now=100)
        self.assertIn('--property=CPUQuota=300%',cmd)
        self.assertIn('--property=KillMode=control-group',cmd)
        self.assertIn('--property=RuntimeMaxSec=28650',cmd)
        t.pop('TR11_runtime_scope')
        with self.assertRaises(ValueError):service_command(t,now=100)
    def test_new_scope_cannot_extend_old_experiments_or_extra_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=r/'approval.json'
            atomic_json(p,{'start_existing_TR11_after_all_gates':True,'TR11_training_wall_seconds_max':30000,
                'TR11_CPU_quota_cores':3,'new_scientific_attempts_added':0})
            c={'hypothesis_id':'TR11','check_variant':'continuous_mainline','job_kind':'remote_train_ppo',
                'engineering_only':False,'maximum_attempts':1,'training_attempt_index':1,
                'training_wall_seconds':30000,'server_resource_profile':{'cpu_quota_cores':3},
                'TR11_runtime_authorization':{'path':'approval.json','sha256':sha256(p)}}
            self.assertEqual(local_limit(r,c),30000)
            for key,value in [('hypothesis_id','TR10'),('maximum_attempts',2),('training_attempt_index',2),('engineering_only',True)]:
                altered=copy.deepcopy(c);altered[key]=value
                with self.assertRaises(ValueError):local_limit(r,altered)
            atomic_json(p,{'start_existing_TR11_after_all_gates':True})
            with self.assertRaises(ValueError):local_limit(r,c)
            self.assertIsNone(local_limit(r,{'hypothesis_id':'TR10'}))

    def test_remote_extension_is_exactly_one_TR11_role(self):
        scope={'seconds_max':30000,'CPU_quota_cores':3,'authorization_sha256':'a'*64}
        self.assertEqual(validate_remote_scope(scope,'derived_scientific','TR11'),30000)
        for mode,hyp in [('fresh_scientific','TR11'),('derived_scientific','TR10')]:
            with self.assertRaises(ValueError):validate_remote_scope(scope,mode,hyp)
