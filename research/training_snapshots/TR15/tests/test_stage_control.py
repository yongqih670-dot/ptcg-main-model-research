import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from ptcg_lab.io import atomic_json, sha256
from ptcg_lab.stage_control import validate_stage_job, register_stage_run
from scripts.run_experiment import validate_config_access, validate_cycle_open


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.p={'status':'active','activation':{'executable':True},'stopping':{
            'mode':'bounded_hypothesis_stage','new_raw_research_runs_max':30,'new_raw_research_runs_regular_max':24,
            'new_hypotheses_max':12,'consecutive_valid_hypotheses_without_promotion':4,'additional_diagnostic_hypotheses_max_total':2},
            'budget':{'executed_research_experiments_max':40},'execution':{'cpu_workers_max':12}}
        self.c={'hypothesis_id':'TR01','job_kind':'evaluation'}
        self.plan={'hypothesis_id':'TR01','research':True,'final':False,'diagnostic':False,'max_raw_runs':3,
            'config_hashes':[hashlib.sha256(json.dumps(self.c,sort_keys=True,separators=(',',':')).encode()).hexdigest()]}
        self.s={'budget_used':{'executed_research_experiments':10},'consecutive_valid_runs_without_promotion':7,
            'research_stage':{'cycle_id':'v3','status':'active','activation_receipt':'receipt.json','baseline_raw_runs':10,
            'regular_raw_runs':0,'final_raw_runs':0,'consecutive_valid_hypotheses_without_promotion':0,
            'diagnostic_hypotheses_started':0,'hypotheses':{},'engineering_tasks':{}}}
        self.save()
    def tearDown(self):self.tmp.cleanup()
    def save(self):
        atomic_json(self.root/'research/autonomy/policy.json',self.p)
        atomic_json(self.root/'receipt.json',{'status':'complete','cycle_id':'v3','policy_sha256':sha256(self.root/'research/autonomy/policy.json')})
        atomic_json(self.root/'plan.json',self.plan)
        self.s['research_stage']['hypotheses']['TR01']={'status':'registered','run_ids':[],'diagnostic':self.plan['diagnostic'],
            'preregistration_path':'plan.json','preregistration_sha256':sha256(self.root/'plan.json')}
    def validate(self,**kw):validate_stage_job(self.root,self.c,self.p,self.s,True,False,**kw)
    def test_old_streak_retained_does_not_block_new_authorized_stage(self):
        self.validate();register_stage_run(self.c,self.p,self.s,True,False,'r1')
        self.assertEqual(self.s['consecutive_valid_runs_without_promotion'],7)
        self.assertEqual(self.s['research_stage']['regular_raw_runs'],1)
        self.validate(new=False,run_id='r1')
    def test_incomplete_activation_and_proposal_fail_closed(self):
        atomic_json(self.root/'receipt.json',{'status':'preparing'})
        with self.assertRaisesRegex(RuntimeError,'Activation'):self.validate()
        self.p['activation']['executable']=False
        with self.assertRaisesRegex(RuntimeError,'not active'):self.validate()
        with self.assertRaises(RuntimeError):validate_cycle_open({'status':'activating'})
    def test_raw_global_and_final_reserve(self):
        self.s['budget_used']['executed_research_experiments']=40
        with self.assertRaisesRegex(RuntimeError,'cap exhausted'):self.validate()
        self.s['budget_used']['executed_research_experiments']=34
        self.s['research_stage']['regular_raw_runs']=24
        with self.assertRaisesRegex(RuntimeError,'reserve'):self.validate()
    def test_plateau_and_diagnostic_cap(self):
        self.s['research_stage']['consecutive_valid_hypotheses_without_promotion']=4
        with self.assertRaisesRegex(RuntimeError,'Plateau'):self.validate()
        self.plan['diagnostic']=True;self.save();self.validate()
        self.s['research_stage']['diagnostic_hypotheses_started']=2
        with self.assertRaisesRegex(RuntimeError,'Diagnostic'):self.validate()
    def test_job_role_config_and_resume_cannot_bypass(self):
        with self.assertRaises(RuntimeError):validate_stage_job(self.root,self.c,self.p,self.s,False,False)
        with self.assertRaisesRegex(RuntimeError,'role'):validate_stage_job(self.root,self.c,self.p,self.s,True,True)
        with self.assertRaisesRegex(RuntimeError,'Resume'):self.validate(new=False,run_id='old')
        self.c['workers']=12
        with self.assertRaisesRegex(RuntimeError,'config'):self.validate()
    def test_hypothesis_caps_and_changed_preregistration(self):
        row=self.s['research_stage']['hypotheses']['TR01'];row['run_ids']=['a','b','c']
        with self.assertRaisesRegex(RuntimeError,'raw run cap'):self.validate()
        row['run_ids']=[]
        self.s['research_stage']['hypotheses'].update({str(i):{'run_ids':[str(i)],'status':'completed'} for i in range(12)})
        with self.assertRaisesRegex(RuntimeError,'hypothesis allowance'):self.validate()
        (self.root/'plan.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'Preregistration changed'):self.validate()
    def test_nested_training_holdout_rejected(self):
        c={'job_kind':'prepare_tokens','opponent_pool':{'entries':[{'path':'agents\\holdout\\x'}]}}
        with self.assertRaisesRegex(ValueError,'holdout'):validate_config_access(c,self.p,False)


if __name__=='__main__':unittest.main()
