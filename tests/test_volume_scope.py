import copy
from pathlib import Path
import tempfile
import unittest
from ptcg_lab.io import read_json,atomic_json
from ptcg_lab.volume_scope import AUTH,AUTH_PATH,limits,validate,validate_raws,register,validate_remote
from ptcg_lab.storage_profile import validate as validate_storage,NEW,OLD,PROJECT
from ptcg_lab.archive_prefix import canonical_hash
from ptcg_lab.resource_journal import capacity_profile


class VolumeScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        source=Path(__file__).resolve().parents[1]/AUTH_PATH
        target=self.root/AUTH_PATH;target.parent.mkdir(parents=True);target.write_bytes(source.read_bytes())
        self.policy={'forward_authorizations':{'TR12_volume':AUTH}}
        self.state={'budget_used':{'gpu_job_wall_hours':23.,'cpu_job_wall_hours':10.},
            'forward_phases':{'TR12':{'status':'active','authority':AUTH,'baseline_budget_used':{'gpu_job_wall_hours':23.,'cpu_job_wall_hours':10.},
                'training_run_ids':[],'evaluation_run_ids':[],'engineering_run_ids':[]}}}
        self.cfg={'forward_phase':'TR12','hypothesis_id':'TR12','TR12_authorization':AUTH,
            'maximum_attempts':3,'job_kind':'remote_train_ppo','engineering_only':False}
    def tearDown(self):self.temp.cleanup()
    def test_prior_costs_are_carried_and_phase_caps_tighten_actual_job(self):
        self.state['budget_used']['gpu_job_wall_hours']+=83.5
        r=limits(self.root,self.cfg,'gpu',self.policy,self.state)
        self.assertEqual(r['job_seconds_max'],28800);self.assertEqual(r['phase_remaining_seconds'],1800)
        self.state['budget_used']['gpu_job_wall_hours']+=.5
        self.assertEqual(limits(self.root,self.cfg,'gpu',self.policy,self.state)['phase_remaining_seconds'],0)
        self.state['budget_used']['gpu_job_wall_hours']=22
        with self.assertRaises(ValueError):limits(self.root,self.cfg,'gpu',self.policy,self.state)
    def test_engineering_cpu_cost_and_cross_hypothesis_scope(self):
        c={**self.cfg,'hypothesis_id':'TR12_AUDIT','budget_hypothesis_id':'TR12','job_kind':'compact_audit'}
        self.state['budget_used']['cpu_job_wall_hours']+=11.75
        r=limits(self.root,c,'cpu',self.policy,self.state)
        self.assertEqual(r['job_seconds_max'],1800);self.assertEqual(r['phase_remaining_seconds'],900)
        with self.assertRaises(ValueError):validate(self.root,{**self.cfg,'hypothesis_id':'TR11'},self.policy,self.state)
        with self.assertRaises(ValueError):validate(self.root,{**self.cfg,'maximum_attempts':4},self.policy,self.state)
    def test_raw_roles_and_duplicates_cannot_bypass_phase_cap(self):
        p=self.state['forward_phases']['TR12'];p['training_run_ids']=[str(i) for i in range(16)]
        with self.assertRaises(ValueError):validate_raws(self.root,self.cfg,self.policy,self.state,True)
        e={**self.cfg,'job_kind':'evaluation'}
        validate_raws(self.root,e,self.policy,self.state,True)
        register(e,self.state,True,'eval')
        with self.assertRaises(ValueError):register(e,self.state,True,'eval')
        p['evaluation_run_ids']=['a','b','c','d']
        with self.assertRaises(ValueError):validate_raws(self.root,e,self.policy,self.state,True)
    def test_exact_remote_deadline_and_journal_scope(self):
        scope={'authorization_sha256':AUTH['sha256'],'seconds_max':28800,'CPU_quota_cores':3,
            'absolute_unique_target':369941,'phase_GPU_hours_max':84,'phase_CPU_hours_max':12}
        self.assertEqual(validate_remote(scope,'derived_scientific','TR12'),28800)
        with self.assertRaises(ValueError):validate_remote(scope,'derived_scientific','TR11')
        with self.assertRaises(ValueError):validate_remote({**scope,'seconds_max':28801},'derived_scientific','TR12')
        capacity_profile(28800,extended_TR12=True)
        with self.assertRaises(ValueError):capacity_profile(28800)
    def test_storage_keeps_old_root_and_twenty_GiB_reserve(self):
        authority=read_json(self.root/AUTH_PATH)
        value={'schema':'ptcg_HDD2_TR12_storage_v2','project_id':PROJECT,'root':NEW,
            'hard_max_bytes':80*2**30,'warning_bytes':64*2**30,'free_reserve_bytes':20*2**30,
            'retained_roots':[OLD],'authority':authority,'authority_file_sha256':AUTH['sha256'],
            'authority_canonical_sha256':canonical_hash(authority)}
        validate_storage(value,NEW)
        for altered in ({'retained_roots':[]},{'hard_max_bytes':81*2**30},{'free_reserve_bytes':19*2**30}):
            with self.assertRaises(ValueError):validate_storage({**value,**altered},NEW)

    def test_service_deadline_and_quota_are_enforced_and_not_shared_with_old_jobs(self):
        from ptcg_lab.systemd_job import service_command
        from ptcg_lab.server_trial_agent import trial_seconds_maximum
        scope={'authorization_sha256':AUTH['sha256'],'seconds_max':28800,'CPU_quota_cores':3,
            'absolute_unique_target':369941,'phase_GPU_hours_max':84,'phase_CPU_hours_max':12}
        t={'unit':'ptcg-hybrid-test-12345678.service','kind':'train_ppo','python_kind':'venv',
            'project_root':'/home/user/ptcg','job_directory':'/home/user/ptcg/jobs/run',
            'ticket_path':'/home/user/ptcg/jobs/run/ticket.json','run_id':'run','deadline_unix':28900,
            'memory_max_bytes':22*2**30,'workspace_hard_max_bytes':10*2**30,'cpu_quota_cores':3,
            'gpu_uuid':'AUTHORIZED_GPU_UUID','mode':'derived_scientific',
            'new_diagnostic_identity':'TR12','TR12_runtime_scope':scope}
        command=service_command(t,now=100)
        self.assertIn('--property=CPUQuota=300%',command)
        self.assertIn('--property=RuntimeMaxSec=28750',command)
        self.assertIn('--property=KillMode=control-group',command)
        with self.assertRaises(ValueError):service_command({**t,'deadline_unix':28901},now=100)
        with self.assertRaises(ValueError):service_command({**t,'new_diagnostic_identity':'TR11'},now=100)
        self.assertEqual(trial_seconds_maximum('derived_scientific'),7200)
        self.assertEqual(trial_seconds_maximum('derived_scientific',authorized_TR12=True),28800)
        with self.assertRaises(ValueError):trial_seconds_maximum('derived_scientific',authorized_training=True,authorized_TR12=True)

    def test_science_cannot_change_under_volume_extension(self):
        from ptcg_lab.volume_extension import validate_training
        root=Path(__file__).resolve().parents[1]
        before=read_json(root/'configs/training/tr11_continuous_basic_learning.json')
        after=read_json(root/'configs/training/tr12_volume_320768.json')
        validate_training(after,before)
        for key,value in [('seed',42),('target_decisions',400000)]:
            with self.assertRaises(ValueError):validate_training({**after,key:value},before)
        changed=copy.deepcopy(after);changed['ppo']['epochs']=4
        with self.assertRaises(ValueError):validate_training(changed,before)


if __name__=='__main__':unittest.main()
