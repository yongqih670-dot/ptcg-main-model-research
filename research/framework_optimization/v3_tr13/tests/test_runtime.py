import copy,json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from ptcg_lab.framework_v3_runtime import JsonEvidence,ProgressPublisher,ExitEvidence,correlate,flags,from_config,capacity_type
from ptcg_lab.io import read_json,atomic_json

class CacheTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'job';self.root.mkdir();self.cache=JsonEvidence(self.root,read_json,maximum=4096)
    def tearDown(self):self.tmp.cleanup()
    def test_change_replacement_delete_and_return_alias(self):
        p=self.root/'control.json';atomic_json(p,{'sequence':1,'command':'run'});a=self.cache.get(p);a['command']='stop';self.assertEqual(self.cache.get(p)['command'],'run');self.assertEqual(self.cache.reads,1)
        atomic_json(p,{'sequence':2,'command':'stop'});self.assertEqual(self.cache.get(p)['command'],'stop');p.unlink();self.assertIsNone(self.cache.get(p,optional=True))
        atomic_json(p,{'sequence':3,'command':'run'});self.assertEqual(self.cache.get(p)['sequence'],3)
    def test_same_size_mtime_preserving_content_change(self):
        p=self.root/'x';p.write_text('{"v":1}');old=p.stat();self.cache.get(p);p.write_text('{"v":2}');os.utime(p,ns=(old.st_atime_ns,old.st_mtime_ns));self.assertEqual(self.cache.get(p)['v'],2)
    def test_change_during_read_fails_and_does_not_seal(self):
        p=self.root/'x';atomic_json(p,{'v':1})
        def changed(path):v=read_json(path);atomic_json(path,{'v':2});return v
        c=JsonEvidence(self.root,changed)
        with self.assertRaisesRegex(ValueError,'during complete read'):c.get(p)
        self.assertEqual(c.reads,0)
    def test_link_oversize_and_outside_root_rejected(self):
        p=self.root/'x';p.write_text('x'*4097)
        with self.assertRaises(ValueError):self.cache.get(p)
        with self.assertRaises(ValueError):self.cache.get(self.root.parent/'foreign.json')
    def test_stop_regression_semantics_survive_cached_decode(self):
        from ptcg_lab.training_control import TrainingControl,ControlledTrainingStop
        c=TrainingControl(self.root,'job',1,time.time()+1000,lambda *a,**k:None,optimizations={'control_cache':True})
        p=self.root/'manual-training-control.json';atomic_json(p,dict(run_id='job',lease_epoch=1,sequence=1,command='run',reason='ok'));c.wait(gpu_required=False)
        atomic_json(p,dict(run_id='job',lease_epoch=1,sequence=2,command='stop',reason='explicit'))
        with self.assertRaises(ControlledTrainingStop):c.wait(gpu_required=False)
        atomic_json(p,dict(run_id='job',lease_epoch=1,sequence=1,command='run',reason='old'))
        with self.assertRaises(ValueError):c.wait(gpu_required=False)
    def test_old_science_config_cannot_activate(self):
        atomic_json(self.root/'ticket.json',{'run_id':'job','lease_epoch':1})
        with self.assertRaises(ValueError):from_config({'framework_v3_optimizations':{'control_cache':True}},self.root)
        self.assertIsNone(from_config({},self.root))

class ProgressTests(unittest.TestCase):
    def test_bounded_nonterminal_and_immediate_stop_error_commit(self):
        clock=[0.];rows=[];p=ProgressPublisher(lambda name,row:rows.append(copy.deepcopy(row)),clock=lambda:clock[0])
        p('progress',{'stage':'collecting','n':1});clock[0]=.2;p('progress',{'stage':'collecting','n':2});self.assertEqual(len(rows),1)
        p('progress',{'stage':'checkpointed_for_control'});self.assertEqual(len(rows),2)
        clock[0]=1.2;p('progress',{'stage':'collecting','n':3});self.assertEqual(rows[-1]['n'],3)
        clock[0]=1.3;p('progress',{'stage':'collecting','n':4});p.flush();self.assertEqual(rows[-1]['n'],4)
    def test_publication_failure_is_not_swallowed(self):
        p=ProgressPublisher(lambda *a:(_ for _ in ()).throw(OSError('disk full')))
        with self.assertRaises(OSError):p('progress',{'stage':'error'})
    def test_interval_cannot_delay_beyond_one_second(self):
        with self.assertRaises(ValueError):ProgressPublisher(lambda *a:None,interval=2)

class ExitTests(unittest.TestCase):
    def test_correlated_missing_controller_and_EOF_are_unproven(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'job';root.mkdir();e=ExitEvidence(root,'job',2,'original-attempt','worker',process_identity={'pid':44,'created':3})
            p=e.record('peer_EOF',observed_cause='peer_stream_closed');row=read_json(p);result=correlate([row]);self.assertEqual(result['initiating_cause'],'unproven');self.assertIn('controller',result['missing_roles'])
            foreign=dict(row,lease_epoch=3)
            with self.assertRaises(ValueError):correlate([row,foreign])
            with self.assertRaises(ValueError):e.record('terminal',returncode=0)

class CapacityTests(unittest.TestCase):
    def test_external_growth_cannot_permanently_hide_in_cache(self):
        from ptcg_lab.fo_v2.durable import Capacity
        used=[0]
        class Base(Capacity):
            def _scan(self):return used[0]
        cls=capacity_type(Base)
        with tempfile.TemporaryDirectory() as d:
            c=cls(d,2*2**30,free_reserve=0,disk_free=lambda:10*2**30,reconcile_seconds=60)
            first=c.scans;t=c.reserve(1024);c.finish(t,1024);self.assertEqual(c.scans,first)
            used[0]=2*2**30;c.at-=61
            with self.assertRaises(OSError):c.reserve(1024)
            self.assertGreater(c.scans,first)
