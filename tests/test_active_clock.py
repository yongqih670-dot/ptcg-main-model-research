import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from ptcg_lab import active_clock as ac
from ptcg_lab.io import atomic_json,read_json


class ActiveClockTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.job=Path(self.tmp.name)/'run';self.job.mkdir()
        now=time.time();self.scope={'spec':copy.deepcopy(ac.SPEC),'execution_limit_seconds':120,
            'job_started_unix':now-20,'hard_deadline_unix':now+160,'hard_window_seconds':180,'startup_wait_seconds':2}
        self.notices=[];self.clock=ac.ServiceClock(self.job,'run',7,self.scope,notifier=self.notices.append)
    def tearDown(self):self.tmp.cleanup()
    def record(self,credit=3,pid=42,epoch=7):
        self.clock.children[(42,'123')]=0.
        atomic_json(self.job/'resource-wait/42-123.json',{'run_id':'run','lease_epoch':epoch,'pid':pid,
            'start_ticks':'123','verified_wait_seconds':credit,'observed_monotonic':time.monotonic()})
    def test_wait_is_deducted_once_and_only_new_credit_notifies(self):
        self.record();one=self.clock.sample();self.clock.sample()
        self.assertEqual(one['verified_wait_seconds'],5)
        self.assertAlmostEqual(one['gross_seconds']-5,one['counted_execution_seconds'])
        self.assertEqual(len(self.notices),1)
        row=read_json(self.job/'resource-wait/42-123.json');row['verified_wait_seconds']=4
        atomic_json(self.job/'resource-wait/42-123.json',row);self.clock.sample()
        self.assertEqual(len(self.notices),2)
    def test_unregistered_process_wait_is_not_credited(self):
        self.record();self.clock.children.clear()
        self.assertEqual(self.clock.sample()['verified_wait_seconds'],2)
    def test_foreign_epoch_rejected(self):
        self.record(epoch=6)
        with self.assertRaises(ValueError):self.clock.sample()
    def test_foreign_pid_rejected(self):
        self.record(pid=43)
        with self.assertRaises(ValueError):self.clock.sample()
    def test_negative_credit_rejected(self):
        self.record(-1)
        with self.assertRaises(ValueError):self.clock.sample()
    def test_over_credit_rejected(self):
        self.record(1000)
        with self.assertRaises(ValueError):self.clock.sample()
    def test_nonfinite_credit_rejected(self):
        with self.assertRaises(ValueError):ac.finite(float('nan'))
    def test_credit_regression_rejected(self):
        self.record(4);self.clock.sample()
        row=read_json(self.job/'resource-wait/42-123.json');row['verified_wait_seconds']=3
        atomic_json(self.job/'resource-wait/42-123.json',row)
        with self.assertRaises(ValueError):self.clock.sample()
    def test_stale_health_fails_closed(self):
        row=self.clock.sample();row['observed_monotonic']-=31
        atomic_json(self.job/'active-clock-server.json',row)
        with self.assertRaises(RuntimeError):ac.remaining_from_record(self.job,'run',7)
    def test_attempt_credit_cannot_replay(self):
        atomic_json(self.job/'remote-job.json',{'attempt':2,'lease_epoch':7})
        atomic_json(self.job/'active-clock-local-02.json',{'run_id':'run','attempt':1,'lease_epoch':7,'verified_wait_seconds':2})
        with self.assertRaises(ValueError):ac.local_credit(self.job,'run',2,10)
    def test_clock_record_limits_active_work_before_gpu_operation(self):
        from ptcg_lab.training_control import TrainingControl,ControlledTrainingStop
        self.clock.sample()
        control=TrainingControl(self.job,'run',7,time.time()+180,lambda *a,**kw:None)
        control.wait_recorder=object()
        with patch.object(ac,'remaining_from_record',return_value=89):
            with self.assertRaises(ControlledTrainingStop):control.wait(90)
    def test_original_absolute_ceiling_still_wins(self):
        self.clock.scope['hard_deadline_unix']=time.time()+2
        self.assertLessEqual(self.clock.sample()['remaining_execution_seconds'],2)
    def test_unrecognized_authority_rejected(self):
        bad=copy.deepcopy(self.scope);bad['spec']['authorization']['sha256']='0'*64
        with self.assertRaises(ValueError):ac.validate_scope(bad)
    def test_no_productive_work_is_exempted(self):
        one=self.clock.sample();self.clock.origin-=5;two=self.clock.sample()
        self.assertGreaterEqual(two['counted_execution_seconds']-one['counted_execution_seconds'],5)
        self.assertEqual(two['verified_wait_seconds'],one['verified_wait_seconds'])


if __name__=='__main__':unittest.main()
