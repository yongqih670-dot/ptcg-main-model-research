import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ptcg_lab.archive_ack import receive_object,validate_ack
from ptcg_lab.archive_compressed import receive_compressed_object,verify_archive
from ptcg_lab.archive_outbox import receive_one
from ptcg_lab.io import read_json
from tests import test_archive_outbox as outbox_fixture


class CompressedArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.raw=(bytes(range(256))+b'\0'*2048)*1700
        self.digest=hashlib.sha256(self.raw).hexdigest()
    def tearDown(self):self.tmp.cleanup()
    def receive(self,chunks=None):return receive_compressed_object(self.root,'run',1,0,'rollouts/v0000-s000000.pt',self.digest,len(self.raw),[self.raw] if chunks is None else chunks)
    def test_durable_gzip_reconstructs_original_and_ack_binds_both_hashes(self):
        ack=self.receive([self.raw[:34567],self.raw[34567:]])
        self.assertEqual(ack['schema_version'],2);self.assertLess(ack['storage']['bytes'],len(self.raw)//10)
        self.assertEqual(verify_archive(self.root,ack,collect=True),self.raw)
        self.assertEqual(read_json(self.root/'run/acks/00000000.json'),ack)
        changed={**ack,'storage':{**ack['storage'],'sha256':'0'*64}}
        with self.assertRaises(ValueError):validate_ack(changed,'run',1,ack['object'],self.digest,len(self.raw))
    def test_retry_does_not_consume_network_but_encoded_corruption_blocks_ack(self):
        ack=self.receive()
        def never():raise AssertionError('Already committed object must not be downloaded');yield b''
        self.assertEqual(self.receive(never()),ack)
        path=self.root/ack['archive_relative'];data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        with self.assertRaises(ValueError):self.receive(never())
    def test_wrong_raw_hash_never_publishes_file_or_receipt(self):
        with self.assertRaises(ValueError):self.receive([b'wrong bytes'])
        self.assertFalse((self.root/'run/acks/00000000.json').exists())
        self.assertFalse((self.root/'run/objects/rollouts/v0000-s000000.pt.gz').exists())
    def test_old_raw_receipt_is_preserved_and_not_reencoded(self):
        ack=receive_object(self.root,'run',1,0,'rollouts/v0000-s000000.pt',self.digest,len(self.raw),[self.raw])
        self.assertEqual(self.receive([]),ack);self.assertNotIn('storage',ack)
        self.assertEqual(verify_archive(self.root,ack,collect=True),self.raw)
    def test_disk_reserve_refuses_before_input_consumed(self):
        with patch('ptcg_lab.archive_compressed.shutil.disk_usage') as usage:
            usage.return_value.free=0
            with self.assertRaises(OSError):self.receive()
        self.assertFalse((self.root/'run/acks/00000000.json').exists())
    def test_empty_original_is_still_a_real_complete_gzip(self):
        ack=receive_compressed_object(self.root,'run',1,0,'empty',hashlib.sha256(b'').hexdigest(),0,[])
        self.assertEqual(verify_archive(self.root,ack,collect=True),b'');self.assertGreaterEqual(ack['storage']['bytes'],20)

    def test_new_lease_reuses_exact_compressed_bytes_without_reencoding_or_network(self):
        from ptcg_lab.archive_compressed import reuse_compressed_object
        import os
        old=self.receive();path=self.root/old['archive_relative'];before=path.read_bytes()
        row={'run_id':'new-run','lease_epoch':2,'sequence':0,'object':old['object'],'sha256':old['sha256'],'bytes':old['bytes']}
        ack=reuse_compressed_object(self.root,row,self.root,old)
        self.assertTrue(os.path.samefile(path,self.root/ack['archive_relative']))
        self.assertEqual(path.read_bytes(),before);self.assertEqual(ack['storage'],old['storage'])
        self.assertNotEqual(ack['commit_id'],old['commit_id']);self.assertEqual(verify_archive(self.root,ack,True),self.raw)
        self.assertEqual(reuse_compressed_object(self.root,row,self.root,old),ack)
        with self.assertRaises(ValueError):reuse_compressed_object(self.root,{**row,'sha256':'0'*64},self.root,old)


class CompressedOutboxTests(unittest.TestCase):
    def test_server_accepts_compressed_durable_ack_after_raw_reconstruction(self):
        fixture=outbox_fixture.ArchiveOutboxTests();fixture.setUp()
        try:
            ack=receive_one(fixture.stream,fixture.archive,fixture.rows[0],storage_codec='gzip')
            self.assertEqual(verify_archive(fixture.archive,ack,collect=True),(fixture.root/fixture.rows[0]['object']).read_bytes())
            self.assertTrue(fixture.box.accept_ack(ack)['replayed'])
            self.assertEqual(fixture.box.pending()['pending_objects'],1)
        finally:fixture.tearDown()
