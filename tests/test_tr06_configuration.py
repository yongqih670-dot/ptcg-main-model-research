import copy
import unittest
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
from ptcg_lab.io import atomic_json
from ptcg_lab.io import read_json
from ptcg_lab.fresh_experiment import validate_single_change


class EntropyOnlyTrainingTests(unittest.TestCase):
    def setUp(self):
        self.reference=read_json('configs/training/tr02_ppo.json')
        self.config=copy.deepcopy(self.reference)
        self.config.update(hypothesis_id='TR06',primary_change='entropy only')
        self.config['ppo']['entropy_coefficient']=.001
    def test_exact_single_entropy_change_retains_full_actor_and_other_settings(self):
        self.assertEqual(validate_single_change(self.config,self.reference),'TR06')
        self.assertEqual(self.config['parent_model_sha256'],self.reference['parent_model_sha256'])
    def test_other_learning_or_budget_or_parent_changes_are_rejected(self):
        for section,key,value in [('ppo','critic_updates_shared_trunk',False),('ppo','gae_lambda',.99),
                                   ('ppo','entropy_coefficient',.002),(None,'target_decisions',8192),
                                   (None,'parent_model_sha256','0'*64)]:
            c=copy.deepcopy(self.config);(c[section] if section else c)[key]=value
            with self.assertRaises(ValueError):validate_single_change(c,self.reference)
    def test_original_closed_TR05_configuration_remains_distinct(self):
        c=copy.deepcopy(self.reference);c['hypothesis_id']='TR05';c['ppo']['critic_updates_shared_trunk']=False
        self.assertEqual(validate_single_change(c,self.reference),'TR05')

    def test_TR07_value_weight_is_distinct_and_cannot_repeat_failed_entropy_trial(self):
        c=copy.deepcopy(self.reference);c['hypothesis_id']='TR07';c['ppo']['value_loss_coefficient']=.1
        self.assertEqual(validate_single_change(c,self.reference),'TR07')
        c['ppo']['entropy_coefficient']=.001
        with self.assertRaises(ValueError):validate_single_change(c,self.reference)

    def test_actual_controller_reaches_the_bound_SSH_entrypoint_without_connecting(self):
        from ptcg_lab.server_trial import run
        import os
        root=Path(__file__).resolve().parents[1]
        config=read_json(root/'configs/tr06-fresh-server.json')
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'controlled_abcdef';output.mkdir()
            atomic_json(output/'experiment.json',{'remote_lease_epoch':1})
            class ReachedSSH(Exception):pass
            def intercepted(c,path,log,deadline):
                self.assertEqual(path,root/'ptcg_lab/server_trial_agent.py')
                self.assertTrue(path.is_file());raise ReachedSSH()
            with patch.dict(os.environ,{'PTCG_JOB_RUN_ID':output.name,'PTCG_JOB_DEADLINE_UNIX':str(time.time()+60)}),patch('ptcg_lab.server_trial.SSHStream',side_effect=intercepted):
                with self.assertRaises(ReachedSSH):run(config,output)

    def test_preSSH_exception_repair_denies_any_remote_progress_or_third_attempt(self):
        from ptcg_lab.shared_server_safety import validate_pre_ssh_startup_failure
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'runs/first';run.mkdir(parents=True)
            atomic_json(run/'attempts.json',[{'status':'failed','seconds':.297}])
            atomic_json(run/'config.json',{'hypothesis_id':'TR06'})
            (run/'stderr-01.log').write_text("NameError: name 'package' is not defined",encoding='utf-8')
            config={'pre_ssh_startup_failure_run_id':'first','prior_training_seconds':.297}
            self.assertEqual(validate_pre_ssh_startup_failure(root,config),.297)
            atomic_json(run/'progress.json',{'stage':'initialized'})
            with self.assertRaises(RuntimeError):validate_pre_ssh_startup_failure(root,config)
