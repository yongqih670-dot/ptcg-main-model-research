import copy,json,os,random,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from ptcg_lab.framework_v2_training import NativeCheckpoints,semantic_digest
from ptcg_lab.archive_state import initial_state
from ptcg_lab.archive_prefix import canonical_hash
from ptcg_lab.fo_v2.durable import full_hash
from ptcg_lab.io import atomic_json,read_json
PROFILE={'schema':'framework_v2_reliable_checkpoint_v1','maximum_reliable_commit_gap_seconds':800,'commit_reserve_seconds':60,'trigger_after_seconds':740,'semantics':'completed_commit_to_completed_commit'}

class NativeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'job';self.root.mkdir()
        prefix=dict(schema_version=1,checkpoint={'sha256':'a'*64,'version':1,'trained_decisions':10,'next_series_id':0},consumed=[],tail=[],consumed_records_root=canonical_hash([]),consumed_fixed_statistics={},full_fixed_statistics={})
        atomic_json(self.root/'prefix.json',prefix);digest=full_hash(self.root/'prefix.json')[0];self.spec={'prefix_path':'prefix.json','prefix_sha256':digest}
        self.value={'model':{'actor':torch.arange(6).reshape(2,3),'critic':torch.ones(2)},'optimizer':{'state':{0:{'step':torch.tensor(2.),'exp_avg':torch.ones(2)}},'param_groups':[]},
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':[]},'version':1,'trained_decisions':10,'unique_PPO_trained_decisions':10,
            'next_series_id':0,'updates':[],'partial_ppo':None,'cpu_sampler_rng':torch.get_rng_state(),'archive_origin_prefix_sha256':digest,'archive_state':initial_state(prefix,digest)}
        self.league={'snapshots':[],'statistics':{}};self.backend=NativeCheckpoints(self.root,1,PROFILE,hard_bytes=2*2**30,rate=2**40,optimizations={'journal_cache':True,'capacity_transaction':True})
    def tearDown(self):self.backend.close();self.temp.cleanup()
    def save(self,value=None,revision=0):return self.backend.save(self.value if value is None else value,self.spec,self.league,model_revision=revision)
    def check(self,value):
        actual={k:v for k,v in value.items() if k!='state_semantic_sha256'};self.assertEqual(semantic_digest(actual),value['state_semantic_sha256'])
        expected=dict(self.value,league=self.league,model_state_revision=0);self.assertEqual(semantic_digest(actual),semantic_digest(expected))
    def test_two_complete_points_cache_and_fresh_owner_restore(self):
        self.save();scans=self.backend.capacity.scans;self.save();self.save();self.assertEqual(self.backend.capacity.scans,scans+2)
        for point in (0,1):self.check(self.backend.restore(point)[0])
        self.backend.close();self.backend=NativeCheckpoints(self.root,1,PROFILE,hard_bytes=2*2**30,rate=2**40,optimizations={'journal_cache':True,'capacity_transaction':True})
        for point in (0,1):self.check(self.backend.restore(point)[0])
    def test_failure_before_manifest_never_publishes_new_point(self):
        self.save();self.save();prior=(self.root/'recovery-v2.json').read_bytes()
        self.backend.bridge.checkpoints.fault=lambda stage:(_ for _ in ()).throw(RuntimeError('injected publish crash')) if stage=='before_index' else None
        with self.assertRaises(RuntimeError):self.save()
        self.assertEqual((self.root/'recovery-v2.json').read_bytes(),prior);self.check(self.backend.restore()[0])
    def test_registered_write_scope_reuses_reconciled_capacity(self):
        from ptcg_lab.framework_v3_admission import install
        install(self.backend,self.root);scans=self.backend.capacity.scans
        self.save();self.save();self.save();self.assertEqual(self.backend.capacity.scans,scans)
        for point in (0,1):self.check(self.backend.restore(point)[0])
    def test_capacity_pressure_never_publishes_new_point(self):
        self.save();self.save();prior=(self.root/'recovery-v2.json').read_bytes();self.backend.capacity.disk_free=lambda:0
        with self.assertRaises(OSError):self.save()
        self.assertEqual((self.root/'recovery-v2.json').read_bytes(),prior)
    def test_partial_revision_and_rng_are_not_restarted(self):
        self.save();v=copy.deepcopy(self.value);v['partial_ppo']={'epoch':0,'offset':256,'permutation':np.array([2,0,1]),'positive_optimizer_steps':1,'logs':[{'batch':256}]};v['model']['critic']+=1
        self.save(v,revision=1);self.save(v,revision=1);restored,_=self.backend.restore()
        self.assertEqual(restored['model_state_revision'],1);self.assertEqual(restored['partial_ppo']['offset'],256);np.testing.assert_array_equal(restored['partial_ppo']['permutation'],[2,0,1]);self.assertTrue(torch.equal(v['model']['critic'],restored['model']['critic']))
    def test_corrupted_immutable_state_fails_fresh_restore(self):
        self.save();self.save();point=read_json(self.root/'recovery-v2.json')['points'][-1];p=self.root/point['tail']['path'];old=p.read_bytes();p.write_bytes(old[:-1]+bytes([old[-1]^1]))
        with self.assertRaises(ValueError):self.backend.restore()
