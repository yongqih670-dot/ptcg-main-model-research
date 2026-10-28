import copy
from pathlib import Path
import tempfile
import unittest
from ptcg_lab.io import atomic_json,read_json,sha256
from ptcg_lab.archive_compressed import receive_compressed_object
from scripts.prepare_final_evaluation import completed_training_config,compare_seed_protocols


class HybridFinalTrainingEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.run=self.root/'runs/completed';self.run.mkdir(parents=True)
        self.recipe={'job_kind':'train_ppo','seed':1,'target_decisions':32768,'ppo':{'value_loss_coefficient':.1}}
        self.template=self.root/'configs/learning.json';atomic_json(self.template,self.recipe)
        self.controller={'job_kind':'remote_train_ppo','check_variant':'fresh_scientific','engineering_only':False,
            'training_config_template':'configs/learning.json','training_config_template_sha256':sha256(self.template)}
        atomic_json(self.run/'config.json',self.controller);atomic_json(self.run/'experiment.json',{'kind':'research'})
        (self.run/'model.pt').write_bytes(b'unchanged model artifact bytes');digest=sha256(self.run/'model.pt')
        self.spec={'training_config_sha256':sha256(self.run/'config.json'),'model_sha256':digest}
        manifest={'mode':'fresh_scientific','training_config':self.recipe}
        atomic_json(self.run/'remote-training-manifest.json',manifest)
        self.metrics={'complete':True,'model_sha256':digest,'trained_decisions':32770,'scientific_training':True,
            'actual_config':copy.deepcopy(self.recipe),
            'remote_manifest_sha256':sha256(self.run/'remote-training-manifest.json'),
            'engine':{'platform':'linux','files':{n:{'sha256':str(i)*64} for i,n in enumerate(['libcg.so','sim.py','game.py'],1)}}}
        atomic_json(self.run/'metrics.json',self.metrics)
        data=(self.run/'model.pt').read_bytes()
        self.ack=receive_compressed_object(self.run/'online-archive',self.run.name,1,0,'model.pt',digest,len(data),[data],minimum_free_bytes=0)
        atomic_json(self.run/'hybrid-materialization.json',{'model_archive_ack':self.ack,'raw_model_sha256':digest})
        handle={'unit':'ptcg-hybrid-fixture.service','invocation_id':'a'*32,'start_monotonic':'100','host_boot_id':'test-boot'}
        lease={'run_id':self.run.name,'attempt':1,'lease_epoch':1,'host':'TRAINING_HOST','unit':handle['unit'],
            'deadline_unix':9999999999,'identity':handle}
        receipt={**lease,'host_boot_id':'test-boot','service_properties':{'unit':handle['unit'],'LoadState':'loaded',
            'ActiveState':'active','SubState':'exited','InvocationID':'a'*32,'ExecMainStartTimestampMonotonic':'100',
            'Result':'success','ExecMainStatus':'0'}}
        atomic_json(self.run/'remote-terminal.json',receipt)
        lease.update(terminal_receipt_path='runs/completed/remote-terminal.json',terminal_receipt_sha256=sha256(self.run/'remote-terminal.json'))
        atomic_json(self.run/'remote-job.json',lease)

    def tearDown(self):self.tmp.cleanup()

    def test_complete_hybrid_uses_its_pinned_learning_recipe_and_actual_engine(self):
        recipe,runtime=completed_training_config(self.root,self.run,self.spec)
        self.assertEqual(recipe,self.recipe);self.assertEqual(runtime['execution'],'server')
        self.assertEqual(set(runtime['engine_sha256']),{'libcg.so','sim.py','game.py'})
        self.assertNotIn('linux_cpu_verified',runtime)

    def test_engineering_incomplete_undertrained_and_nonresearch_cannot_qualify(self):
        for updates in [{'complete':False},{'trained_decisions':32767},{'scientific_training':False}]:
            atomic_json(self.run/'metrics.json',{**self.metrics,**updates})
            with self.assertRaises(ValueError):completed_training_config(self.root,self.run,self.spec)
        atomic_json(self.run/'metrics.json',self.metrics)
        atomic_json(self.run/'experiment.json',{'kind':'engineering'})
        with self.assertRaises(ValueError):completed_training_config(self.root,self.run,self.spec)

    def test_pinned_recipe_or_actual_server_manifest_changes_are_rejected(self):
        original=self.template.read_bytes();atomic_json(self.template,{**self.recipe,'seed':2})
        with self.assertRaises(ValueError):completed_training_config(self.root,self.run,self.spec)
        self.template.write_bytes(original)
        atomic_json(self.run/'remote-training-manifest.json',{'mode':'fresh_scientific','training_config':{**self.recipe,'seed':2}})
        with self.assertRaises(ValueError):completed_training_config(self.root,self.run,self.spec)

    def test_reported_actual_execution_recipe_must_match_the_pinned_recipe(self):
        bad=copy.deepcopy(self.metrics);bad['actual_config']['ppo']['value_loss_coefficient']=.5
        atomic_json(self.run/'metrics.json',bad)
        with self.assertRaises(ValueError):completed_training_config(self.root,self.run,self.spec)

    def test_raw_model_proof_is_reconstructed_instead_of_trusting_a_flag(self):
        from ptcg_lab.archive_ack import safe_target
        packed=safe_target(self.run/'online-archive',self.ack['archive_relative'])
        packed.write_bytes(b'changed compressed model bytes')
        with self.assertRaises(ValueError):completed_training_config(self.root,self.run,self.spec)

    def test_live_or_unreconciled_remote_service_cannot_qualify(self):
        lease=read_json(self.run/'remote-job.json');lease.pop('terminal_receipt_path')
        atomic_json(self.run/'remote-job.json',lease)
        with self.assertRaises(RuntimeError):completed_training_config(self.root,self.run,self.spec)

    def test_failed_service_cannot_qualify_stale_complete_metrics(self):
        receipt=read_json(self.run/'remote-terminal.json')
        receipt['service_properties'].update(ActiveState='failed',Result='exit-code',ExecMainStatus='1')
        atomic_json(self.run/'remote-terminal.json',receipt)
        lease=read_json(self.run/'remote-job.json');lease['terminal_receipt_sha256']=sha256(self.run/'remote-terminal.json')
        atomic_json(self.run/'remote-job.json',lease)
        with self.assertRaises(ValueError):completed_training_config(self.root,self.run,self.spec)

    def test_same_replication_budget_aliases_and_actual_engine_are_required(self):
        first={'training_config':self.recipe,'runtime_identity':{'execution':'server','engine_sha256':{'libcg.so':'1'*64}},
            'plan':{'training_wall_seconds_max':14400,'training_attempt_seconds_max':7200,'training_attempts_max':2},
            'config':{'matchups':[],'contrasts':[],'series_count':800,'evaluation_stage':'independent_confirmation'}}
        second=copy.deepcopy(first);second['training_config']['seed']=2
        second['plan']['max_training_attempts']=second['plan'].pop('training_attempts_max')
        compare_seed_protocols(first,second)
        second['plan']['training_attempts_max']=3
        with self.assertRaises(ValueError):compare_seed_protocols(first,second)
        second['plan'].pop('training_attempts_max');second['runtime_identity']['engine_sha256']['libcg.so']='2'*64
        with self.assertRaises(ValueError):compare_seed_protocols(first,second)

