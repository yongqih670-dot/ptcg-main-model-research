import io
import json
import unittest
import tempfile,time
from pathlib import Path
from unittest.mock import MagicMock,patch
from ptcg_lab.ssh_stream import read_response,open_readonly_stream,SSHStream
import queue
from ptcg_lab.io import read_json


class CountingRaw(io.RawIOBase):
    def __init__(self,data):self.source=io.BytesIO(data);self.read_calls=0
    def readable(self):return True
    def readinto(self,target):
        self.read_calls+=1
        return self.source.readinto(target)


class SSHFrameTests(unittest.TestCase):
    def fake_channel(self,answers,deadline=1000):
        stream=SSHStream.__new__(SSHStream);stream.deadline=deadline;stream.sequence=0;stream.pending_response=False
        stream.answers=MagicMock();stream.answers.get.side_effect=answers;stream._write=MagicMock();stream.process=MagicMock()
        return stream
    def test_archive_waits_for_same_response_once_without_resending(self):
        stream=self.fake_channel([queue.Empty(),{'sequence':1,'ok':True,'result':{'bytes':7}}])
        with patch('ptcg_lab.ssh_stream.time.time',return_value=900):
            self.assertEqual(stream.request('archive_chunk'),{'bytes':7})
        stream._write.assert_called_once();self.assertEqual(stream.sequence,1)
        self.assertEqual([v.kwargs['timeout'] for v in stream.answers.get.call_args_list],[30,30])
        self.assertFalse(stream.pending_response)
    def test_archive_extra_wait_obeys_original_deadline_and_fatal_timeout_poison(self):
        stream=self.fake_channel([queue.Empty(),queue.Empty()])
        with patch('ptcg_lab.ssh_stream.time.time',side_effect=[900,985]):
            with self.assertRaises(TimeoutError):stream.request('archive_chunk')
        self.assertEqual([v.kwargs['timeout'] for v in stream.answers.get.call_args_list],[30,5])
        with self.assertRaises(RuntimeError):stream.request('status')
        stream._write.assert_called_once();self.assertEqual(stream.sequence,1)
    def test_wrong_sequence_and_nonarchive_timeout_never_retry(self):
        for result,error in [({'sequence':2,'ok':True,'result':{}},ValueError),(queue.Empty(),TimeoutError)]:
            stream=self.fake_channel([result])
            with patch('ptcg_lab.ssh_stream.time.time',return_value=900):
                with self.assertRaises(error):stream.request('status')
            self.assertTrue(stream.pending_response);self.assertEqual(stream.answers.get.call_count,1)
    def test_readonly_preinitialize_close_has_bounded_backoff_and_never_initializes(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'server_environment_agent.py').write_bytes(b'# reviewed readonly definitions')
            calls=[];streams=[]
            def connect(*args,**kwargs):
                stream=MagicMock();streams.append(stream);stream.process.returncode=255
                def start(path):
                    if len(streams)<3:
                        with (out/'ssh-stream-errors.log').open('ab') as e:e.write(b'Connection closed by authorized host port15134\n')
                        raise BrokenPipeError()
                stream.start_agent.side_effect=start
                stream.request.side_effect=lambda op,body=None:(calls.append(op) or {'server_unix':time.time()})
                return stream
            with patch('ptcg_lab.ssh_stream.SSHStream',side_effect=connect),patch('ptcg_lab.ssh_stream.time.sleep') as sleep:
                result=open_readonly_stream({},out,out,time.time()+180)
            self.assertEqual(len(streams),3);self.assertEqual(calls,['load_base','clock']);self.assertEqual(sleep.call_count,2)
            self.assertTrue(all(r['job_initialize_sent'] is False for r in read_json(out/'training-connection-attempts.json')))
            streams[0].close.assert_called_once();streams[1].close.assert_called_once()
    def test_readonly_identity_or_clock_error_never_reconnects(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'server_environment_agent.py').write_bytes(b'# readonly definitions')
            stream=MagicMock();stream.request.side_effect=RuntimeError('Owned root differs')
            with patch('ptcg_lab.ssh_stream.SSHStream',return_value=stream) as launch:
                with self.assertRaises(RuntimeError):open_readonly_stream({},out,out,time.time()+180)
            self.assertEqual(launch.call_count,1);stream.close.assert_called_once()
    def test_large_evidence_is_framed_with_bounded_bulk_pipe_reads(self):
        value={'sequence':1,'ok':True,'result':'x'*(600*1024)}
        raw=CountingRaw(json.dumps(value).encode()+b'\n'+b'{"sequence":2}\n')
        with io.BufferedReader(raw,buffer_size=64*1024) as stream:
            self.assertEqual(read_response(stream),value)
            self.assertEqual(read_response(stream),{'sequence':2})
            self.assertLess(raw.read_calls,20)
    def test_oversized_or_incomplete_frame_is_never_an_accepted_response(self):
        with self.assertRaises(ValueError):read_response(io.BytesIO(b'x'*101+b'\n'),100)
        with self.assertRaises(EOFError):read_response(io.BytesIO(b'{"ok":true}'))
        with self.assertRaises(EOFError):read_response(io.BytesIO())
