import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from ptcg_lab.archive_compressed import receive_compressed_object
from ptcg_lab.io import sha256
from ptcg_lab.server_archive import materialize_compressed_reuse


class CompressedReuseTests(unittest.TestCase):
    def test_one_verified_decode_can_back_two_aliases_without_new_wire_data(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'prior';out=root/'new';out.mkdir();raw=b'full original training state\x00'*400
            digest=hashlib.sha256(raw).hexdigest();ack=receive_compressed_object(base,'run',1,0,'state.pt',digest,len(raw),[raw])
            ap=base/'run/acks/00000000.json';spec={'path':ap.relative_to(root).as_posix(),'sha256':sha256(ap)}
            rows=[{'path':name,'sha256':digest,'bytes':len(raw),'archive_root':'prior','ACK':spec} for name in ['checkpoint.pt','checkpoint-commits/objects/cp.pt']]
            aliases,decoded=materialize_compressed_reuse(root,out,rows,time.time()+60)
            self.assertEqual(len(decoded),1);self.assertEqual(len(aliases),2)
            self.assertEqual(Path(aliases['checkpoint.pt']['local_source']).read_bytes(),raw)
            self.assertTrue(aliases['checkpoint.pt']['accept_new_source_identity'])
    def test_changed_ACK_is_refused_before_decoding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';out.mkdir();ap=root/'ack.json';ap.write_text('{}')
            row={'path':'state','bytes':1,'sha256':'0'*64,'archive_root':'prior','ACK':{'path':'ack.json','sha256':'1'*64}}
            with self.assertRaises(ValueError):materialize_compressed_reuse(root,out,[row],time.time()+30)


if __name__=='__main__':unittest.main()
