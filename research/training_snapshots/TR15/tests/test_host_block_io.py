from pathlib import Path
import tempfile
import unittest
from ptcg_lab.host_block_io import HostBlockCounters,parse_block_io


class HostBlockIOTests(unittest.TestCase):
    def test_whole_host_delta_includes_other_devices_conservatively(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'io.stat';boot=Path(temp)/'boot';boot.write_text('same-boot')
            path.write_text('8:0 rbytes=100 wbytes=200\n259:0 rbytes=50 wbytes=60\n7:0 \n')
            counter=HostBlockCounters('8:0',path,boot)
            path.write_text('8:0 rbytes=110 wbytes=220\n259:0 rbytes=80 wbytes=100\n7:0 \n')
            sample=counter.sample()
            self.assertEqual((sample['read_bytes'],sample['write_bytes']),(40,60))
            self.assertEqual(sample['project_device_delta'],[10,20])
            # No task or PID needs to remain alive for this persistent source.
            self.assertEqual(counter.sample()['write_bytes'],60)
    def test_missing_project_device_cannot_claim_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'io';boot=Path(temp)/'boot';boot.write_text('boot');path.write_text('8:1 rbytes=1 wbytes=2')
            with self.assertRaises(RuntimeError):HostBlockCounters('8:0',path,boot)
    def test_counter_regression_or_device_change_or_boot_change_fails_closed(self):
        for altered in ['8:0 rbytes=9 wbytes=20','8:1 rbytes=10 wbytes=20','8:0 rbytes=10 wbytes=20\n7:0 ']:
            with self.subTest(altered=altered),tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'io';boot=Path(temp)/'boot';boot.write_text('boot');path.write_text('8:0 rbytes=10 wbytes=20')
                counter=HostBlockCounters('8:0',path,boot);path.write_text(altered)
                with self.assertRaises(RuntimeError):counter.sample()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'io';boot=Path(temp)/'boot';boot.write_text('boot');path.write_text('8:0 rbytes=10 wbytes=20')
            counter=HostBlockCounters('8:0',path,boot);boot.write_text('new-boot')
            with self.assertRaises(RuntimeError):counter.sample()
    def test_unreadable_or_incomplete_critical_source_never_grants_continue(self):
        for text in ['', '8:0 rbytes=1', 'bad rbytes=1 wbytes=2','8:0 rbytes=-1 wbytes=2','8:0 rbytes=1 wbytes=2\n8:0 rbytes=1 wbytes=2']:
            with self.subTest(text=text),self.assertRaises((RuntimeError,ValueError)):parse_block_io(text)
