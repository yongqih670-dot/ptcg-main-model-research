import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from ptcg_lab.io import sha256
from ptcg_lab.linux_probe import (MARKER, QEMU_SHA512, checked_download, guest_script,
                                  make_payload, result_from_serial, safe_payload_name, verify_runtime_files,
                                  select_runtime_members,runtime_capacity)
from scripts.run_experiment import compute_kind, validate_config_access, validate_cycle_open


class LinuxProbeTests(unittest.TestCase):
    def listing(self,rows):
        return 'Archive header\n----------\n'+'\n\n'.join('Path = '+name+'\nSize = '+str(size)+'\nFolder = -' for name,size in rows)

    def test_select_cpu_runtime_before_extracting_all_architectures(self):
        rows=[('qemu-system-x86_64.exe',20),('qemu-img.exe',10),('lib.dll',5),('share/bios-256k.bin',3),
              ('share/vgabios-stdvga.bin',3),('share/keymaps/en-us',1),('qemu-system-aarch64.exe',700*2**20),('uninstall.exe',8)]
        result=select_runtime_members(self.listing(rows))
        self.assertEqual(sum(result.values()),42)
        self.assertNotIn('qemu-system-aarch64.exe',result);self.assertNotIn('uninstall.exe',result)

    def test_selected_extraction_rejects_duplicate_escape_missing_or_oversized(self):
        rows=[('qemu-system-x86_64.exe',20),('qemu-img.exe',10),('bios-256k.bin',3)]
        for extra in [('QEMU-IMG.EXE',1),('../escape.dll',1),('data.dll',601*2**20)]:
            with self.assertRaises(ValueError):select_runtime_members(self.listing(rows+[extra]))
        with self.assertRaises(ValueError):select_runtime_members(self.listing(rows[:-1]))

    def test_runtime_guard_counts_host_file_sizes_and_preserves_headroom(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);overlay=root/'guest.qcow2';overlay.write_bytes(b'x'*10)
            (root/'base.img').write_bytes(b'x'*20)
            with patch('ptcg_lab.linux_probe.RAM_RUNTIME_MAX',64*2**20+40),patch('ptcg_lab.linux_probe.RAM_OVERLAY_MAX',16*2**20+20):
                self.assertEqual(runtime_capacity(root,overlay)['runtime_logical_bytes'],30)
                overlay.write_bytes(b'x'*21)
                with self.assertRaises(OSError):runtime_capacity(root,overlay)
                overlay.write_bytes(b'x'*10);(root/'base.img').write_bytes(b'x'*31)
                with self.assertRaises(OSError):runtime_capacity(root,overlay)

    def test_ram_guest_keeps_pip_and_apt_temporary_data_in_size_limited_guest_ram(self):
        text=guest_script('http://PRIVATE_HOST:1234/','digest','nonce','run','archive',1234567,ram_guest=True)
        compile(text,'ram-guest-script','exec')
        self.assertIn('size=6G,noswap,mode=700',text);self.assertIn('TMPDIR=',text)
        self.assertIn('Dir::State::lists=',text);self.assertIn('noswap_observed',text)

    def test_oversized_cached_download_cannot_bypass_byte_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cache';path.write_bytes(b'oversized')
            with patch('urllib.request.urlopen',side_effect=AssertionError('no fetch')):
                with self.assertRaises(ValueError):checked_download('https://example.invalid',path,'sha256',sha256(path),100,2)

    def test_output_must_be_linux_unique_and_bound_to_job_and_archive(self):
        row={'nonce':'n','run_id':'r','archive_sha256':'a','platform':'Linux'}
        def line(x): return MARKER+base64.b64encode(json.dumps(x).encode()).decode()
        self.assertEqual(result_from_serial('noise\n'+line(row),'n','r','a'),row)
        for bad in [line(row)+'\n'+line(row),line(dict(row,platform='Windows')),line(dict(row,run_id='other')),line(dict(row,archive_sha256='other')),line(dict(row,nonce='old'))]:
            with self.assertRaises(ValueError): result_from_serial(bad,'n','r','a')

    def test_payload_is_allowlisted_and_includes_embeddings(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ['ptcg_lab/io.py','agents/a/model.pt','agents/a/data/embed_pca.npz','agents/b/main.py','secret.txt','agents/holdout/model.pt']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
            archive=root/'submission.tar.gz';archive.write_bytes(b'fake package for payload boundary test')
            config={'probe':{'candidate':{'path':'agents/a'},'reference':{'path':'agents/b'},'archive_path':str(archive),'archive_sha256':sha256(archive)}}
            data,identity=make_payload(root,config,'run',123)
            with zipfile.ZipFile(io.BytesIO(data)) as z: names=z.namelist()
            self.assertIn('source/agents/a/data/embed_pca.npz',names)
            self.assertNotIn('secret.txt',names);self.assertNotIn('source/agents/holdout/model.pt',names)
            self.assertEqual(identity['run_id'],'run')
            config['probe']['candidate']['path']='agents/../../secret'
            with self.assertRaises(ValueError):make_payload(root,config,'run',123)

    def test_corrupt_cached_binary_is_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'binary';p.write_bytes(b'corrupted')
            with patch('urllib.request.urlopen',side_effect=AssertionError('must not fetch')):
                with self.assertRaises(ValueError):checked_download('https://example.invalid',p,'sha256','0'*64,100,100)

    def test_runtime_resume_rejects_changed_or_extra_library(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'lib.dll';p.write_bytes(b'original')
            receipt={'installer_sha512':QEMU_SHA512,'files':{'lib.dll':sha256(p)}}
            verify_runtime_files(root,receipt)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_runtime_files(root,receipt)
            p.write_bytes(b'original');(root/'extra.dll').write_bytes(b'extra')
            with self.assertRaises(ValueError):verify_runtime_files(root,receipt)

    def test_guest_code_compiles_and_enforces_absolute_deadline(self):
        text=guest_script('http://PRIVATE_HOST:1234/','digest','nonce','run','archive',1234567)
        compile(text,'guest-script','exec')
        self.assertIn("C['deadline']-time.time()-90",text)
        self.assertIn("get('linux_execution_observed')",text)
        self.assertIn("get('cpu_only_observed')",text)

    def test_linux_cannot_escape_runner_or_holdout_guards(self):
        config={'job_kind':'linux_probe','workers':1,'probe':{'candidate':{'path':'agents/holdout/model'}}}
        self.assertEqual(compute_kind(config),'cpu')
        with self.assertRaises(ValueError):validate_config_access(config,{'execution':{'cpu_workers_max':12}},False)
        with self.assertRaises(RuntimeError):validate_cycle_open({'status':'plateau'})
        for name in ['../x','/x','C:/x','a\\x','a/../b','a//b']:
            self.assertFalse(safe_payload_name(name))


if __name__=='__main__':unittest.main()
