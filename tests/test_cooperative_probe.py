from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ptcg_lab.cooperative_probe import begin,EarlyPauseProbe
from ptcg_lab.io import atomic_json,read_json


class CooperativeProbeTests(unittest.TestCase):
    def test_actual_pause_and_resume_events_required_without_freezing_clocks(self):
        with tempfile.TemporaryDirectory() as folder:
            job=Path(folder)/'run';job.mkdir();ticket={'run_id':'run','lease_epoch':7,'deadline_unix':1000}
            begin(job,ticket);probe=EarlyPauseProbe(job,ticket,10)
            row={'at':'now','run_id':'run','lease_epoch':7,'process_id':10,'sequence':0,'stage':'paused_for_resource'}
            atomic_json(job/'control-events/000000.json',row)
            with patch('ptcg_lab.cooperative_probe.time.monotonic',return_value=0):probe.observe('0')
            with patch('ptcg_lab.cooperative_probe.time.monotonic',return_value=6):probe.observe('0')
            self.assertEqual(read_json(job/'manual-training-control.json')['command'],'run');self.assertFalse(probe.complete)
            atomic_json(job/'control-events/000001.json',{**row,'sequence':1,'stage':'resumed_after_resource_pause'})
            with patch('ptcg_lab.cooperative_probe.time.monotonic',return_value=7):probe.observe('0')
            self.assertTrue(read_json(job/'cooperative-pause-acceptance.json')['complete'])
    def test_frozen_kernel_and_later_manual_command_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            job=Path(folder)/'run';job.mkdir();ticket={'run_id':'run','lease_epoch':7,'deadline_unix':1000};begin(job,ticket)
            atomic_json(job/'control-events/000000.json',{'run_id':'run','lease_epoch':7,'process_id':10,'sequence':0,'stage':'paused_for_resource'})
            probe=EarlyPauseProbe(job,ticket,10)
            with self.assertRaises(ValueError):probe.observe('1')
            with patch('ptcg_lab.cooperative_probe.time.monotonic',return_value=0):probe.observe('0')
            row=read_json(job/'manual-training-control.json');atomic_json(job/'manual-training-control.json',{**row,'sequence':2})
            with patch('ptcg_lab.cooperative_probe.time.monotonic',return_value=6):
                with self.assertRaises(ValueError):probe.observe('0')
