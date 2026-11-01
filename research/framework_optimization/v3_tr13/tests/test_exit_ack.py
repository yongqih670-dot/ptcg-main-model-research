import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from ptcg_lab.framework_v3_exit import RoleObserver,ObservedStream,ObservedCall
from ptcg_lab.archive_outbox import ArchiveOutbox
from ptcg_lab.archive_ack import receive_object
from ptcg_lab.io import sha256,atomic_json

class ExitAckTests(unittest.TestCase):
    def test_transport_failure_keeps_primary_when_evidence_also_fails(self):
        sink=Mock();sink.record.side_effect=OSError('evidence unavailable')
        c=RoleObserver(sink);r=RoleObserver(sink);stream=Mock();stream.close.side_effect=EOFError('real EOF')
        with self.assertRaisesRegex(EOFError,'real EOF'):ObservedStream(stream,c,r).close()
        self.assertTrue(c.ended);self.assertTrue(r.ended)
    def test_successful_transport_with_broken_evidence_fails_gate(self):
        sink=Mock();sink.record.side_effect=OSError('evidence unavailable')
        with self.assertRaisesRegex(RuntimeError,'Persistent'):ObservedStream(Mock(),RoleObserver(sink),RoleObserver(sink)).close()
    def test_worker_exception_preserved_and_eof_does_not_become_exit_code(self):
        sink=Mock();observer=RoleObserver(sink)
        with self.assertRaises(EOFError):
            with ObservedCall(observer):raise EOFError('closed')
        row=sink.record.call_args.kwargs;self.assertEqual(row['observed_cause'],'peer_stream_closed');self.assertNotIn('returncode',row)
    def test_missing_delayed_bad_conflicting_ACK(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'job';root.mkdir();p=root/'model.pt';p.write_bytes(b'owned immutable payload')
            out=ArchiveOutbox(root,'job',1,optimizations={'outbox_cache':True});row=out.queue('model.pt',sha256(p))
            self.assertEqual(out.pending()['pending_objects'],1);self.assertEqual(out.pending()['pending_objects'],1)
            ack=receive_object(Path(d)/'archive','job',1,0,'model.pt',row['sha256'],row['bytes'],[p.read_bytes()])
            bad=copy.deepcopy(ack);bad['lease_epoch']=2
            with self.assertRaises(ValueError):out.accept_ack(bad)
            out.accept_ack(ack);self.assertEqual(out.pending()['pending_objects'],0)
            path=root/'archive-acks/00000000.json';bad=copy.deepcopy(ack);bad['bytes']+=1;atomic_json(path,bad)
            with self.assertRaises(ValueError):out.pending()
            path.unlink();self.assertEqual(out.pending()['pending_objects'],1)

