import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
import struct
import zlib
from unittest.mock import patch,MagicMock

from ptcg_lab.io import atomic_json,read_json,sha256
from ptcg_lab.server_archive import frame,destination,run,REMOTE,terminal_proven,compressed_chunks,connect,inventory_bound


class ServerArchiveTests(unittest.TestCase):
    def test_measured_file_count_extension_requires_exact_hash_bound_terminal_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'inventory.json'
            rows=[{'path':f'file-{i}.json','bytes':1,'identity':[1,i,1,1]} for i in range(907)]
            proof={'source_run_id':'same-source','actual':{'read_only_metadata_only':True,'owner_uid':1005,'file_count':907,'bytes':907,'rows':rows}}
            atomic_json(path,proof)
            c={'source_run_id':'same-source','maximum_source_files':1024,'source_inventory_evidence':{'path':'inventory.json','sha256':sha256(path)}}
            bound,parameters=inventory_bound(root,c);self.assertEqual(bound,1024);self.assertEqual(parameters['expected_file_count'],907)
            self.assertEqual(inventory_bound(root,{'source_run_id':'legacy'}),(512,{}))
            for change in [{'source_run_id':'other-source'},{'maximum_source_files':2048}]:
                with self.assertRaises(ValueError):inventory_bound(root,{**c,**change})
            atomic_json(path,{})
            with self.assertRaises(ValueError):inventory_bound(root,c)
    def test_only_preinventory_ssh255_close_is_retried_with_a_bounded_count(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);children=[]
            def create(*args,**kw):
                child=MagicMock();children.append(child)
                if len(children)<3:
                    kw['stderr'].write(b'Connection closed by authorized host port15134\n');kw['stderr'].flush()
                    child.stdin.write.side_effect=BrokenPipeError();child.returncode=255
                else:
                    child.stdin.write.side_effect=lambda data:len(data);child.stdout=io.BytesIO(b'{"file_count":0}\n');child.returncode=None
                return child
            with (out/'remote-read-errors.log').open('wb') as errors,patch('ptcg_lab.server_archive.subprocess.Popen',side_effect=create),patch('ptcg_lab.server_archive.ssh_command',return_value=['mock']),patch('ptcg_lab.server_archive.time.sleep') as sleep:
                child,stream,inventory=connect({}, {}, out,errors,time.time()+300)
            self.assertEqual(len(children),3);self.assertEqual(inventory,{'file_count':0});self.assertEqual(sleep.call_count,2)
            self.assertEqual([r['inventory_received'] for r in read_json(out/'archive-connection-attempts.json')],[False,False,True])
    def test_ownership_or_malformed_frame_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);child=MagicMock();child.stdin.write.side_effect=lambda data:len(data)
            child.stdout=io.BytesIO(b'not-json\n');child.returncode=1
            with (out/'remote-read-errors.log').open('wb') as errors,patch('ptcg_lab.server_archive.subprocess.Popen',return_value=child) as launch,patch('ptcg_lab.server_archive.ssh_command',return_value=['mock']):
                with self.assertRaises(ValueError):connect({}, {},out,errors,time.time()+300)
            self.assertEqual(launch.call_count,1)
    def test_collected_unit_requires_prior_identity_and_expired_launch_window(self):
        h={'invocation_id':'a'*32,'start_monotonic':'123'}
        prior={'InvocationID':'a'*32,'ExecMainStartTimestampMonotonic':'123','ActiveState':'active','SubState':'exited'}
        missing={'LoadState':'not-found','ActiveState':'inactive'}
        self.assertFalse(terminal_proven(missing,h,100,prior,150))
        self.assertTrue(terminal_proven(missing,h,100,prior,161))
        self.assertFalse(terminal_proven(missing,h,100,{**prior,'InvocationID':'b'*32},200))
        self.assertFalse(terminal_proven({**prior,'LoadState':'loaded','SubState':'running'},h,100,prior,200))
    def test_frames_and_destination_reject_truncation_and_escape(self):
        with self.assertRaises(ValueError):frame(io.BytesIO(b'{"complete":true}'))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):destination(Path(directory),'../../outside')
        compile('PARAMETERS={}\n'+REMOTE,'reviewed_read_only_endpoint','exec')
    def test_compressed_stream_preserves_bytes_and_rejects_expansion_and_truncation(self):
        raw=b'zero padded trajectory'*10000
        packed=zlib.compress(raw,1)
        wire=struct.pack('>I',len(packed))+packed+struct.pack('>I',0)
        self.assertEqual(b''.join(compressed_chunks(io.BytesIO(wire),len(raw))),raw)
        with self.assertRaises(ValueError):b''.join(compressed_chunks(io.BytesIO(wire),len(raw)-1))
        with self.assertRaises(EOFError):b''.join(compressed_chunks(io.BytesIO(wire[:-2]),len(raw)))
    def exercise(self,corrupt,compressed=False,reuse=False,original=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'runs/source-run-abcdefghijkl';source.mkdir(parents=True)
            output=root/'runs/archive-run-abcdefghijkl';output.mkdir()
            identity={'unit':'ptcg-hybrid-trial-abcdefgh-24.service'}
            atomic_json(root/'research/autonomy/state.json',{})
            atomic_json(source/'remote-terminal.json',{'identity':identity})
            def encoded(x):return json.dumps(x).encode()+b'\n'
            digest=hashlib.sha256(b'abc').hexdigest()
            payload=encoded({'run_id':source.name,'file_count':2,'bytes':3,'identity':identity})
            payload+=encoded({'path':'checkpoint.pt','bytes':3,'file_identity':[1,2,3,4],
                              'codec':'reuse' if reuse or original else 'zlib' if compressed else 'raw'})
            if compressed:
                packed=zlib.compress(b'abc',1);payload+=struct.pack('>I',len(packed))+packed+struct.pack('>I',0)
            elif not reuse and not original:payload+=b'abc'
            payload+=encoded({'path':'checkpoint.pt','bytes':3,'sha256':'0'*64 if corrupt else digest})
            payload+=encoded({'path':'empty.log','bytes':0,'file_identity':[1,3,0,4]})
            payload+=encoded({'path':'empty.log','bytes':0,'sha256':hashlib.sha256(b'').hexdigest()})
            payload+=encoded({'complete':True})
            child=MagicMock();child.stdin.write.side_effect=lambda data:len(data);child.stdout=io.BytesIO(payload)
            child.wait.return_value=0;child.poll.return_value=0
            config={'project_path':str(root),'source_run_id':source.name,'project_id':'test','host':'host','port':22,
                'user':'user','identity_file':'unused-by-mocked-SSH','required_hashes':{'checkpoint.pt':digest},
                'archive_source_sha256':sha256(Path(__file__).resolve().parents[1]/'ptcg_lab/server_archive.py')}
            if reuse:
                prior=root/'runs/prior-archive-abcdefghijkl';(prior/'archive').mkdir(parents=True)
                (prior/'archive/checkpoint.pt').write_bytes(b'abc')
                atomic_json(prior/'inventory.json',{'run_id':source.name,'identity':identity})
                atomic_json(prior/'file-receipts.json',[{'path':'checkpoint.pt','bytes':3,'file_identity':[1,2,3,4],'sha256':digest}])
                config['reuse_archive_run_ids']=[prior.name]
            if original:
                parent=root/'runs/old-original/checkpoint.pt';parent.parent.mkdir(parents=True);parent.write_bytes(b'abc')
                config['reuse_local_sources']=[{'path':'checkpoint.pt','bytes':3,'sha256':digest,'local_path':parent.relative_to(root).as_posix()}]
            with patch.dict(os.environ,{'PTCG_JOB_RUN_ID':output.name,'PTCG_JOB_DEADLINE_UNIX':str(time.time()+600)}), \
                 patch('ptcg_lab.server_archive.subprocess.Popen',return_value=child), \
                 patch('ptcg_lab.server_archive.ssh_command',return_value=['mocked-authorized-ssh']):
                if corrupt:
                    with self.assertRaises(ValueError):run(config,output)
                    self.assertFalse((output/'archive/checkpoint.pt').exists())
                    if not original:self.assertEqual((output/'archive/checkpoint.pt.incoming').read_bytes(),b'abc')
                else:
                    run(config,output)
                    self.assertTrue(read_json(output/'metrics.json')['complete'])
                    self.assertEqual((output/'archive/checkpoint.pt').read_bytes(),b'abc')
                    self.assertEqual((output/'archive/empty.log').read_bytes(),b'')
    def test_binary_and_empty_files_are_durable_and_hash_verified(self):self.exercise(False)
    def test_wrong_hash_keeps_partial_and_never_publishes(self):self.exercise(True)
    def test_lossless_binary_wire_hash_validation(self):self.exercise(False,compressed=True)
    def test_compressed_wrong_hash_never_publishes(self):self.exercise(True,compressed=True)
    def test_durable_prior_objects_reused_only_after_server_identity_and_hash(self):self.exercise(False,reuse=True)
    def test_original_local_bytes_reused_after_fresh_server_identity_and_full_hash(self):self.exercise(False,original=True)
    def test_conflicting_remote_hash_cannot_publish_an_original_local_copy(self):self.exercise(True,original=True)
