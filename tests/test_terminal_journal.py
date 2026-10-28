import copy,time,tempfile
from pathlib import Path
import unittest
from ptcg_lab.terminal_journal import journal_terminal,journal_failed_exit
from ptcg_lab.server_archive import terminal_proven,REMOTE
from ptcg_lab.io import atomic_json,sha256
from ptcg_lab.remote_lease import validate_no_unreconciled_remote


class TerminalJournalTests(unittest.TestCase):
    def test_collected_failed_unit_requires_positive_exit_failure_and_original_empty_group_receipt(self):
        handle,rows=self.fixture();rows[-1]['MESSAGE_ID']='d9b373ed55a64feb8242e02dbe79a49c'
        prior={'InvocationID':handle['invocation_id'],'ExecMainStartTimestampMonotonic':handle['start_monotonic'],
            'ActiveState':'failed','Result':'exit-code','ExecMainStatus':'1','ControlGroup':''}
        proof=journal_failed_exit(rows,handle,1005,2000)
        self.assertEqual(proof['ExecMainStatus'],'1');self.assertNotIn('stopped_unix',proof)
        missing={'LoadState':'not-found'}
        self.assertTrue(terminal_proven(missing,handle,3000,prior,2000,rows,1005,True))
        self.assertFalse(terminal_proven(missing,handle,3000,prior,2000,rows,1005))
        for old in [{},{**prior,'ControlGroup':'/still-present'},{**prior,'ExecMainStatus':'0'},{**prior,'InvocationID':'d'*32}]:
            self.assertFalse(terminal_proven(missing,handle,3000,old,2000,rows,1005,True))
        for key,value in [('_UID','0'),('USER_INVOCATION_ID','c'*32),('EXIT_STATUS','0')]:
            wrong=copy.deepcopy(rows);wrong[1][key]=value
            with self.assertRaises(ValueError):journal_failed_exit(wrong,handle,1005,2000)
        with self.assertRaises(ValueError):journal_failed_exit(rows[:-1],handle,1005,2000)
    def fixture(self):
        handle={'unit':'ptcg-hybrid-trial-abcdef-43.service','invocation_id':'a'*32,
            'host_boot_id':'b'*32,'start_monotonic':'1000000'}
        common={'USER_UNIT':handle['unit'],'USER_INVOCATION_ID':handle['invocation_id'],'_BOOT_ID':handle['host_boot_id'],
            '_UID':'1005','_COMM':'systemd','_EXE':'/usr/lib/systemd/systemd'}
        rows=[{**common,'MESSAGE_ID':mid,'__MONOTONIC_TIMESTAMP':str(mono),'__REALTIME_TIMESTAMP':str(1_000_000_000+mono),'__CURSOR':str(mono)}
            for mid,mono in [('39f53479d3a045ac8e11786248231fbf',1000010),('98e322203f7a4ed290d09fe03c09fe15',1999999),('9d1aaa27d60140bd96365438aad20286',2000000)]]
        rows[1]['EXIT_STATUS']='1';return handle,rows

    def test_positive_manager_stop_proves_terminal_without_turning_failure_into_success(self):
        handle,rows=self.fixture();proof=journal_terminal(rows,handle,1005,2000)
        self.assertEqual(proof['Result'],'exit-code');self.assertEqual(proof['ExecMainStatus'],'1')
        missing={'LoadState':'not-found','ActiveState':'inactive'}
        self.assertFalse(terminal_proven(missing,handle,3000,{},2000))
        self.assertTrue(terminal_proven(missing,handle,3000,{},2000,rows,1005))
        running={'LoadState':'loaded','ActiveState':'active','SubState':'running','InvocationID':'a'*32,'ExecMainStartTimestampMonotonic':'1000000'}
        self.assertFalse(terminal_proven(running,handle,3000,{},2000,rows,1005))
        compile('PARAMETERS={}\n'+REMOTE,'read_only_journal_bound_archive','exec')

    def test_untrusted_manager_uid_boot_other_invocation_missing_stop_and_clock_rejected(self):
        handle,rows=self.fixture()
        bad=[]
        for k,v in [('_UID','0'),('_COMM','python'),('_EXE','/usr/bin/python'),('_BOOT_ID','c'*32),
                    ('USER_INVOCATION_ID','d'*32),('USER_UNIT','foreign.service'),('__MONOTONIC_TIMESTAMP','3000000')]:
            altered=copy.deepcopy(rows);altered[0][k]=v;bad.append(altered)
        bad.extend([rows[:-1],rows[::-1],rows+rows[:1]])
        for altered in bad:
            with self.assertRaises(ValueError):journal_terminal(altered,handle,1005,2000)

    def test_positive_journal_receipt_releases_only_exact_old_invocation(self):
        handle,rows=self.fixture();proof=journal_terminal(rows,handle,1005,2000)
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);receipt={'run_id':'owned','attempt':1,'lease_epoch':43,'host':'host','unit':handle['unit'],
                'host_boot_id':handle['host_boot_id'],'server_unix':2000,'service_properties':{'LoadState':'not-found'},
                'terminal_proof_kind':'exact_invocation_systemd_journal','terminal_journal':rows,'journal_terminal_proof':proof}
            rp=root/'receipt.json';atomic_json(rp,receipt)
            lease={**{k:receipt[k] for k in ['run_id','attempt','lease_epoch','host','unit']},'identity':handle,'deadline_unix':3000,
                'terminal_receipt_path':'receipt.json','terminal_receipt_sha256':sha256(rp)}
            validate_no_unreconciled_remote(root,{'remote_job':lease})
            receipt['terminal_journal'][-1]['USER_INVOCATION_ID']='f'*32;atomic_json(rp,receipt);lease['terminal_receipt_sha256']=sha256(rp)
            with self.assertRaises(ValueError):validate_no_unreconciled_remote(root,{'remote_job':lease})


if __name__=='__main__':unittest.main()
