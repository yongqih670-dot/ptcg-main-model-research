import hashlib
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ptcg_lab.bounded_io import WriteContext
from ptcg_lab.io import atomic_json,read_json
from ptcg_lab.io_pacing import BytePacer


class BoundedIOTests(unittest.TestCase):
    def test_exact_control_size_reserves_actual_bytes_and_enforces_both_size_directions(self):
        from ptcg_lab.server_workspace import admit
        growth=[]
        def checked(root,n):growth.append(n);return admit(root,n)
        with patch('ptcg_lab.bounded_io.admit',side_effect=checked):
            self.context.atomic_write(self.root/'exact',lambda out:out.write(b'abc'),expected_bytes=3)
        self.assertEqual(growth,[3+64*2**20]);self.assertEqual((self.root/'exact').read_bytes(),b'abc')
        for expected,data in [(2,b'abc'),(4,b'abc')]:
            target=self.root/f'bad-{expected}';target.write_bytes(b'old')
            with self.assertRaises((OSError,ValueError)):self.context.atomic_write(target,lambda out:out.write(data),expected_bytes=expected)
            self.assertEqual(target.read_bytes(),b'old')
        with self.assertRaises(ValueError):self.context.atomic_write(self.root/'too-large-exact',lambda out:None,expected_bytes=128*2**20+1)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.root.chmod(0o700);self.now=[0.];self.deadline=time.time()+120
        uid=os.getuid() if hasattr(os,'getuid') else self.root.stat().st_uid
        atomic_json(self.root/'.ptcg-owner.json',{'project_id':'owned','uid':uid,'root':str(self.root.resolve())})
        self.pacer=BytePacer(4*2**20,self.deadline,clock=lambda:self.now[0],sleep=lambda n:self.now.__setitem__(0,self.now[0]+n))
        self.context=WriteContext(self.root,'owned',self.deadline,pacer=self.pacer)
    def tearDown(self):self.temp.cleanup()
    def test_small_writes_coalesce_to_finite_synced_chunks_without_idle_credit(self):
        data=b'a'*(150*1024)
        def write(stream):
            for offset in range(0,len(data),73):stream.write(data[offset:offset+73])
            self.assertEqual(stream.tell(),len(data))
        receipt=self.context.atomic_write(self.root/'checkpoint.pt',write)
        self.assertEqual((self.root/'checkpoint.pt').read_bytes(),data)
        self.assertEqual(receipt['sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(receipt['synchronous_chunks'],3)
        self.assertLessEqual(receipt['maximum_unsynced_chunk_bytes'],64*1024)
        self.now[0]+=60;before=self.now[0]
        self.context.atomic_write(self.root/'second.pt',lambda stream:stream.write(b'b'*(64*1024)))
        self.assertGreaterEqual(self.now[0]-before,64*1024/(4*2**20))
    def test_partial_serialization_never_replaces_previous_recoverable_checkpoint(self):
        target=self.root/'checkpoint.pt';target.write_bytes(b'previous-commit')
        def interrupted(stream):
            stream.write(b'x'*(64*1024));raise OSError('injected write failure')
        with self.assertRaises(OSError):self.context.atomic_write(target,interrupted)
        self.assertEqual(target.read_bytes(),b'previous-commit')
        partials=list(self.root.glob('checkpoint.pt.*.partial'))
        self.assertEqual(len(partials),1);self.assertEqual(partials[0].stat().st_size,64*1024)
    def test_changed_destination_cannot_be_overwritten_at_publication(self):
        target=self.root/'checkpoint.pt';target.write_bytes(b'previous')
        def changed(stream):
            stream.write(b'candidate');target.write_bytes(b'concurrent-unexpected')
        with self.assertRaisesRegex(RuntimeError,'destination changed'):
            self.context.atomic_write(target,changed)
        self.assertEqual(target.read_bytes(),b'concurrent-unexpected')
    def test_path_escape_deadline_and_envelope_failure_leave_target_unpublished(self):
        with self.assertRaises(ValueError):self.context.atomic_write(self.root.parent/'outside-file',lambda stream:stream.write(b'x'))
        limited=WriteContext(self.root,'owned',self.deadline,max_file_bytes=10,pacer=self.pacer)
        with self.assertRaises(OSError):limited.atomic_write(self.root/'too-large',lambda stream:stream.write(b'x'*11))
        self.assertFalse((self.root/'too-large').exists())
        expired=WriteContext(self.root,'owned',time.time()-1)
        with self.assertRaises(TimeoutError):expired.atomic_write(self.root/'too-late',lambda stream:stream.write(b'x'))
        self.assertFalse((self.root/'too-late').exists())
    def test_disk_pressure_and_immutable_destination_are_rejected(self):
        target=self.root/'same';target.write_bytes(b'old')
        with self.assertRaises(FileExistsError):self.context.atomic_write(target,lambda stream:stream.write(b'new'),replace=False)
        with patch('ptcg_lab.bounded_io.shutil.disk_usage',return_value=type('Disk',(),{'free':20*2**30})()):
            with self.assertRaises(OSError):self.context.atomic_write(self.root/'no-space',lambda stream:stream.write(b'x'))
        self.assertFalse((self.root/'no-space').exists())
    def test_torch_roundtrip_preserves_all_tensor_optimizer_and_rng_contents(self):
        import numpy as np
        import torch
        from ptcg_lab.tensor_io import atomic_torch_save
        value={'model':{'actor':torch.arange(20000,dtype=torch.float32),'blind_value':torch.tensor([1.5])},
               'optimizer':{'state':{7:{'step':torch.tensor(3.),'exp_avg':torch.arange(20000,dtype=torch.float32)}}},
               'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state()},'version':30}
        target=self.root/'checkpoint.pt';receipt=atomic_torch_save(target,value,write_context=self.context)
        restored=torch.load(target,map_location='cpu',weights_only=False)
        self.assertTrue(torch.equal(restored['model']['actor'],value['model']['actor']))
        self.assertTrue(torch.equal(restored['optimizer']['state'][7]['exp_avg'],value['optimizer']['state'][7]['exp_avg']))
        self.assertTrue(torch.equal(restored['rng']['torch'],value['rng']['torch']))
        self.assertTrue(np.array_equal(restored['rng']['numpy'][1],value['rng']['numpy'][1]))
        self.assertEqual(restored['version'],30);self.assertLessEqual(receipt['maximum_unsynced_chunk_bytes'],64*1024)
    def test_owned_json_uses_same_bounded_route_and_retains_original_encoding(self):
        value={'中文':'保留','values':[1,2,3]}
        with patch.dict(os.environ,{'PTCG_PROJECT_ROOT':str(self.root),'PTCG_PROJECT_ID':'owned','PTCG_JOB_DEADLINE_UNIX':str(self.deadline)}):
            atomic_json(self.root/'state.json',value)
        self.assertEqual(read_json(self.root/'state.json'),value)
        self.assertIn('中文',(self.root/'state.json').read_text('utf-8'))
    def test_expired_pacer_cannot_consume_even_after_a_long_idle(self):
        expired=BytePacer(100,time.time()-1,clock=lambda:1000.,sleep=lambda _:None)
        with self.assertRaises(TimeoutError):expired.consume(1)
