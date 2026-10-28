import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ptcg_lab.archive_compressed import receive_compressed_object,verify_archive
from ptcg_lab.archive_outbox import receive_one
from ptcg_lab.archive_transfer_measurement import MeasuredStream
from ptcg_lab.continuous_archive_inventory import verify_existing
from ptcg_lab.io import read_json


class ReuseTests(unittest.TestCase):
    def test_full_gzip_reuse_never_requests_remote_payload_and_rebinds_ACK(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=b'full original bytes'*1000;digest=hashlib.sha256(data).hexdigest()
            old=root/'old';old.mkdir();ack=receive_compressed_object(old,'previous',1,0,'league/learner-0000.pt',digest,len(data),[data],minimum_free_bytes=0)
            class Stream:
                def request(self,op,params,**kw):
                    if op!='archive_ack':raise AssertionError('Must not download verified original bytes')
                    self.ack=params['ack'];return {'acknowledged':True}
            wire=Stream();measured=MeasuredStream(wire)
            row={'run_id':'new-owner','lease_epoch':2,'sequence':0,'object':'league/learner-0000.pt','sha256':digest,'bytes':len(data)}
            new=root/'new';new.mkdir()
            import ptcg_lab.archive_compressed as compressed
            receiver=compressed.receive_compressed_object
            with patch.object(compressed,'receive_compressed_object',side_effect=lambda *a,**k:receiver(*a,**k,minimum_free_bytes=0)):
                result=receive_one(measured,new,row,{'archive_root':old,'ack':ack},storage_codec='gzip')
            self.assertEqual(result['run_id'],'new-owner');self.assertEqual(result['lease_epoch'],2)
            self.assertEqual(measured.measurement['archive_chunk_requests'],0)
            self.assertEqual(measured.measurement['archive_ACK_original_bytes'],len(data))
            self.assertEqual(verify_archive(new,result,True),data)

    def test_inventory_rejects_corrupted_encoded_bytes_instead_of_trusting_ACK(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'runs/old/online-archive';base.mkdir(parents=True)
            data=b'original';digest=hashlib.sha256(data).hexdigest()
            ack=receive_compressed_object(base,'old',1,0,'model.pt',digest,len(data),[data],minimum_free_bytes=0)
            encoded=base/ack['archive_relative'];encoded.write_bytes(encoded.read_bytes()[:-1]+b'!')
            verified,missing,rejected=verify_existing(root,[{'object':'model.pt','sha256':digest,'bytes':len(data)}],time.time()+30)
            self.assertEqual(verified,[]);self.assertEqual(len(missing),1);self.assertEqual(len(rejected),1)


if __name__=='__main__':unittest.main()
