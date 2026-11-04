import unittest
from unittest.mock import patch
from tests import test_archive_outbox as fixtures
from ptcg_lab.archive_live_check import controller_faults


class LiveControllerTests(unittest.TestCase):
    def test_real_fixture_fault_flow_keeps_raw_proof_and_idempotent_receipt(self):
        case=fixtures.ArchiveOutboxTests();case.setUp()
        class Endpoint:
            def request(self,operation,data=None,**kwargs):
                if operation=='status':return {'terminal':False,'progress.json':{'stage':'archive_backpressure_wait'}}
                try:return case.stream.request(operation,data,**kwargs)
                except ValueError as exc:raise RuntimeError(str(exc)) from exc
        try:
            with patch('ptcg_lab.archive_live_check.time.sleep'):
                result=controller_faults(Endpoint(),case.archive,'run',1,lambda *a,**k:None)
            self.assertTrue(result['passed']);self.assertGreater(result['real_wire_chunks_read'],1)
            self.assertEqual(case.box.pending()['pending_objects'],1)
            self.assertEqual(result['raw_sha256'],case.rows[0]['sha256'])
        finally:case.tearDown()
