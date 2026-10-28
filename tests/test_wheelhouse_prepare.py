from pathlib import Path
import tempfile
import unittest
import zipfile
from ptcg_lab.wheelhouse_prepare import normalize_lock,wheel_record


class WheelhousePrepareTests(unittest.TestCase):
    def test_linux_cuda_requirement_is_retained_and_windows_marker_removed(self):
        h='0'*64
        text=f'torch==2.11.0 --hash=sha256:{h}\nnvidia-nccl-cu12==2.28.9; sys_platform == "linux" --hash=sha256:{h}\nwin-only==1; sys_platform == "win32" --hash=sha256:{h}'
        result,names=normalize_lock(text)
        self.assertIn('nvidia-nccl-cu12',names);self.assertNotIn('win-only',names)
        self.assertNotIn('sys_platform',result)
    def test_missing_hash_or_unpinned_input_is_rejected(self):
        for text in ['torch==2.11.0','torch>=2.11.0 --hash=sha256:'+'0'*64]:
            with self.assertRaises(ValueError):normalize_lock(text)
    def test_expansion_size_comes_from_wheel_members_not_download_size(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.whl'
            with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
                z.writestr('fixture.dist-info/METADATA','Name: fixture\nVersion: 1\n');z.writestr('payload.bin',b'0'*100000)
            record=wheel_record(p)
            self.assertGreater(record['expanded_bytes'],record['bytes']);self.assertEqual(record['name'],'fixture')
    def test_unsafe_wheel_cannot_be_used_for_installation_plan(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.whl'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','bad')
            with self.assertRaises(ValueError):wheel_record(p)
    def test_vendored_distribution_metadata_is_not_the_wheel_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'setuptools.whl'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('setuptools-78.1.0.dist-info/METADATA','Name: setuptools\nVersion: 78.1.0\n')
                z.writestr('setuptools/_vendor/fixture-1.dist-info/METADATA','Name: fixture\nVersion: 1\n')
            self.assertEqual(wheel_record(p)['name'],'setuptools')
