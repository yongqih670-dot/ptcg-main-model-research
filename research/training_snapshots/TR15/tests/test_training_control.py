from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from ptcg_lab.io import atomic_json,read_json
from ptcg_lab.training_control import TrainingControl,ControlledTrainingStop,publish_resource_control


class TrainingControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'run';self.root.mkdir()
        self.events=[];self.control=TrainingControl(self.root,'run',7,60,lambda stage,**kw:self.events.append((stage,kw)))
    def tearDown(self):self.tmp.cleanup()
    def row(self,command,sequence=0,**extra):return {'run_id':'run','lease_epoch':7,'command':command,'sequence':sequence,'reason':'test',**extra}
    def test_pause_resumes_without_freezing_a_process_or_resetting_deadline(self):
        atomic_json(self.root/'resource-training-control.json',self.row('pause'));clock=[0.]
        def sleep(seconds):
            clock[0]+=seconds;atomic_json(self.root/'resource-training-control.json',self.row('run',1))
        fake=types.SimpleNamespace(time=lambda:clock[0],monotonic=lambda:clock[0],sleep=sleep)
        with patch('ptcg_lab.training_control.time',fake):self.control.wait()
        self.assertEqual([x[0] for x in self.events],['paused_for_resource','resumed_after_resource_pause'])
        self.assertEqual(self.control.deadline,60);self.assertGreater(clock[0],0)
    def test_pause_obeys_original_deadline(self):
        atomic_json(self.root/'resource-training-control.json',self.row('pause'));clock=[0.]
        fake=types.SimpleNamespace(time=lambda:clock[0],monotonic=lambda:clock[0],sleep=lambda n:clock.__setitem__(0,clock[0]+n))
        with patch('ptcg_lab.training_control.time',fake):
            with self.assertRaises(ControlledTrainingStop):self.control.wait()
        self.assertLess(clock[0],16);self.assertEqual(self.control.deadline,60)
    def test_manual_pause_cannot_be_cleared_by_automatic_resource_resume(self):
        atomic_json(self.root/'manual-training-control.json',self.row('pause'))
        publish_resource_control(self.root,'run',7,{'foreign_compute_pids_observed':[]});clock=[0.];ticks=[]
        def sleep(n):
            clock[0]+=n;ticks.append(1)
            publish_resource_control(self.root,'run',7,{'foreign_compute_pids_observed':[]})
            if len(ticks)==2:atomic_json(self.root/'manual-training-control.json',self.row('run',1))
        with patch('ptcg_lab.training_control.time',types.SimpleNamespace(time=lambda:clock[0],monotonic=lambda:clock[0],sleep=sleep)):
            self.control.wait()
        self.assertEqual(len(ticks),2)
    def test_foreign_stale_conflicting_and_stop_commands_fail_closed(self):
        path=self.root/'resource-training-control.json';atomic_json(path,self.row('pause',4));self.control.request(path.name)
        for row in [self.row('run',3),self.row('run',4),self.row('run',5,lease_epoch=8),self.row('run',5,run_id='other')]:
            atomic_json(path,row)
            with self.assertRaises(ValueError):self.control.request(path.name)
        atomic_json(path,self.row('stop',5))
        with self.assertRaises(ControlledTrainingStop):self.control.wait()
    def test_telemetry_timeout_and_external_compute_pause_until_valid_observation(self):
        for sample in [{'monitoring_unavailable':'TimeoutExpired'}, {'foreign_compute_pids_observed':[123]}, {}]:
            row=publish_resource_control(self.root,'run',7,sample);self.assertEqual(row['command'],'pause')
        row=publish_resource_control(self.root,'run',7,{'foreign_compute_pids_observed':[]});self.assertEqual(row['command'],'run')
        self.assertEqual(row['sequence'],3)
        with self.assertRaises(ValueError):publish_resource_control(self.root,'run',8,{'foreign_compute_pids_observed':[]})
