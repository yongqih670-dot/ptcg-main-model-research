import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import supervisor

class SupervisorTests(unittest.TestCase):
    def fake(self,pid=22,created=3):
        p=Mock();p.pid=pid;p.create_time.return_value=created;p.children.return_value=[];p.is_running.return_value=True
        p.memory_info.return_value=SimpleNamespace(rss=10);p.io_counters.return_value=SimpleNamespace(read_bytes=5,write_bytes=7);return p
    def test_exit_race_and_PID_reuse_are_distinct(self):
        p=self.fake();p.create_time.side_effect=psutil.NoSuchProcess(22)
        with self.assertRaises(psutil.NoSuchProcess):supervisor.sample_family(p,3,{22:3},Mock())
        p=self.fake(created=4)
        with self.assertRaisesRegex(RuntimeError,'identity'):supervisor.sample_family(p,3,{22:3},Mock())
    def test_descendant_exit_is_optional_but_reuse_rejected(self):
        p=self.fake();d=self.fake(24);d.memory_info.side_effect=psutil.NoSuchProcess(24);p.children.return_value=[d]
        self.assertEqual(supervisor.sample_family(p,3,{22:3,24:3},Mock()),(10,[5,7]))
        d=self.fake(24,4);p.children.return_value=[d]
        with self.assertRaisesRegex(RuntimeError,'reused'):supervisor.sample_family(p,3,{22:3,24:3},Mock())
    def actual(self,code,race):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            context=patch.object(supervisor,'sample_family',side_effect=psutil.NoSuchProcess(0)) if race else patch.object(supervisor,'sample_family',wraps=supervisor.sample_family)
            with context:
                r=supervisor.run([sys.executable,'-B','-c',f'import time,sys;time.sleep(.05);sys.exit({code})'],cwd=d,console=d/'out',receipt=d/'receipt',progress=d/'progress',stop_file=d/'STOP',maximum_seconds=10)
            self.assertEqual(r['returncode'],code);self.assertEqual(r['exitcode_source'],'Popen.wait');self.assertEqual(r['status'],'terminal')
    def test_real_nonzero_exit_during_sample(self):self.actual(7,True)
    def test_real_zero_exit_during_sample(self):self.actual(0,True)
    def test_real_normal_exit(self):self.actual(0,False)
    def test_exited_handle_assignment_is_not_false_monitor_error(self):
        child=SimpleNamespace(_handle=1,poll=lambda:7);job=SimpleNamespace(handle=1,kernel=SimpleNamespace(AssignProcessToJobObject=lambda *a:False))
        self.assertFalse(supervisor.bind_job(child,job));child.poll=lambda:None
        with self.assertRaises(OSError):supervisor.bind_job(child,job)
