import hashlib
import io
import os
from pathlib import Path
import tempfile
import time
import unittest
import torch
from ptcg_lab.bounded_read import ReadContext
from ptcg_lab.io import atomic_json
from ptcg_lab.io_pacing import BytePacer
from ptcg_lab.tensor_io import load_torch


class BoundedReadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.root.chmod(0o700)
        uid=os.getuid() if hasattr(os,'getuid') else self.root.stat().st_uid
        atomic_json(self.root/'.ptcg-owner.json',{'project_id':'read','uid':uid,'root':str(self.root.resolve())})
        self.now=[0.];self.deadline=time.time()+120
        self.pacer=BytePacer(4*2**20,self.deadline,clock=lambda:self.now[0],sleep=lambda n:self.now.__setitem__(0,self.now[0]+n))
        self.context=ReadContext(self.root,'read',self.deadline,pacer=self.pacer)
    def tearDown(self):self.temp.cleanup()
    def test_seek_readinto_and_reread_are_all_accounted_in_bounded_chunks(self):
        data=bytes(range(256))*1024;path=self.root/'data';path.write_bytes(data)
        with self.context.open(path) as reader:
            self.assertFalse(hasattr(reader,'fileno'))
            self.assertEqual(reader.read(17),data[:17]);reader.seek(-100,io.SEEK_END)
            buffer=bytearray(200);self.assertEqual(reader.readinto(buffer),100)
            self.assertEqual(buffer[:100],data[-100:]);reader.seek(0)
            self.assertEqual(reader.read(),data);receipt=reader.receipt()
        self.assertEqual(receipt['bytes_read'],len(data)+117)
        self.assertLessEqual(receipt['maximum_read_chunk_bytes'],64*1024)
        self.assertGreaterEqual(self.now[0],receipt['bytes_read']/(4*2**20))
    def test_hash_uses_paced_reads_and_idle_time_cannot_grant_free_bandwidth(self):
        path=self.root/'data';data=b'h'*(128*1024);path.write_bytes(data)
        self.now[0]+=100;before=self.now[0]
        self.assertEqual(self.context.sha256(path),hashlib.sha256(data).hexdigest())
        self.assertGreaterEqual(self.now[0]-before,len(data)/(4*2**20))
    def test_changed_source_and_expired_deadline_are_not_silently_accepted(self):
        path=self.root/'data';path.write_bytes(b'original')
        with self.context.open(path) as reader:
            path.write_bytes(b'truncated')
            with self.assertRaises(RuntimeError):reader.read()
        context=ReadContext(self.root,'read',time.time()-1)
        with context.open(path) as reader:
            with self.assertRaises(TimeoutError):reader.read(1)
    def test_torch_zip_storage_and_python_metadata_load_through_paced_interface(self):
        path=self.root/'model.pt';value={'model':torch.arange(40000,dtype=torch.float32),'version':30,'metadata':('blind',True)}
        torch.save(value,path);saved=load_torch(path,read_context=self.context,map_location='cpu',weights_only=False)
        self.assertTrue(torch.equal(saved['model'],value['model']));self.assertEqual(saved['metadata'],value['metadata'])
        self.assertGreater(self.pacer.total,160000)
        with self.assertRaises(ValueError):load_torch(path,read_context=self.context,mmap=True,weights_only=False)
