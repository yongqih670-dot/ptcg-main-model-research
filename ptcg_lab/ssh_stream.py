"""One SSH process with bounded JSON frames and explicit binary chunk lengths."""
import json
import io
import base64
import hashlib
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time

from .server_boundary_audit import ssh_command
from .io import atomic_json,utcnow


def read_response(stream,limit=2*1024**2):
    line=stream.readline(limit+1)
    if not line:raise EOFError('Persistent SSH stream closed')
    if len(line)>limit:raise ValueError('Oversized SSH response')
    if not line.endswith(b'\n'):raise EOFError('Truncated SSH response frame')
    return json.loads(line)


def open_readonly_stream(config,package,output,deadline):
    """Bounded connection retries end before any job initialize/start request."""
    output=Path(output);error_log=output/'ssh-stream-errors.log';records=[]
    base=(Path(package)/'server_environment_agent.py').read_bytes()
    for attempt in range(1,4):
        offset=error_log.stat().st_size if error_log.exists() else 0
        stream=None
        try:
            stream=SSHStream(config,Path(package)/'server_trial_agent.py',error_log,deadline,defer_bootstrap=True)
            stream.start_agent(Path(package)/'server_trial_agent.py')
            library={'base_source':base64.b64encode(base).decode(),'base_sha256':hashlib.sha256(base).hexdigest()}
            if config.get('server_project_root'):
                library.update(project_root=config['server_project_root'],storage_profile=config['storage_profile'])
            stream.request('load_base',library)
            sent=time.time();clock=stream.request('clock',{'client_unix':sent})
            if time.time()-sent>10 or abs(clock['server_unix']-sent)>5:raise RuntimeError('Bounded clock handshake failed')
            records.append({'at':utcnow(),'connection_attempt':attempt,'readonly_handshake_complete':True,'job_initialize_sent':False})
            atomic_json(output/'training-connection-attempts.json',records);return stream
        except (OSError,EOFError) as exc:
            if stream is not None:stream.close()
            with error_log.open('rb') as errors:errors.seek(offset);detail=errors.read(8192).decode(errors='replace')
            code=stream.process.returncode if stream is not None else None
            records.append({'at':utcnow(),'connection_attempt':attempt,'readonly_handshake_complete':False,
                            'job_initialize_sent':False,'exit_code':code,'error':type(exc).__name__,'stderr':detail})
            atomic_json(output/'training-connection-attempts.json',records)
            if code!=255 or 'Connection closed by ' not in detail or attempt==3 or time.time()>deadline-60:raise
            time.sleep(3*attempt)
        except BaseException:
            if stream is not None:stream.close()
            raise
    raise RuntimeError('Unreachable bounded readonly connection loop')


class SSHStream:
    def __init__(self,config,agent_path,error_log,deadline,defer_bootstrap=False):
        self.deadline=deadline;self.sequence=0;self.answers=queue.Queue(maxsize=4);self.pending_response=False
        bootstrap='import sys,hashlib;_ptcg_code=sys.stdin.buffer.read(int(sys.stdin.buffer.readline()));PTCG_AGENT_SHA256=hashlib.sha256(_ptcg_code).hexdigest();exec(compile(_ptcg_code,"ptcg_owned_agent","exec"))'
        command=ssh_command(config)[:-3]+['/usr/bin/python3','-u','-B','-c',shlex.quote(bootstrap)]
        self.errors=Path(error_log).open('ab')
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.errors,bufsize=0)
        # FileIO.readline() on the unbuffered Windows pipe reads byte by byte.
        # Large evidence frames then starve an otherwise healthy heartbeat.
        self.response_stream=io.BufferedReader(self.process.stdout,buffer_size=64*1024)
        def reader():
            try:
                while True:
                    self.answers.put(read_response(self.response_stream))
            except Exception as exc:self.answers.put(exc)
        self.reader=threading.Thread(target=reader,daemon=True);self.reader.start()
        self.bootstrap_sent=False
        if not defer_bootstrap:self.start_agent(agent_path)
    def start_agent(self,agent_path):
        if self.bootstrap_sent:raise ValueError('SSH agent bootstrap cannot be resent')
        self.bootstrap_sent=True;code=Path(agent_path).read_bytes()
        try:self._write(str(len(code)).encode()+b'\n');self._write(code);self.process.stdin.flush()
        except BaseException:self.close();raise
    def _write(self,data):
        view=memoryview(data)
        while view:
            if time.time()>self.deadline-10:raise TimeoutError('SSH write deadline')
            written=self.process.stdin.write(view)
            if not written:raise EOFError('SSH stdin closed')
            view=view[written:]
    def request(self,operation,body=None,data=None,timeout=30):
        if self.pending_response:raise RuntimeError('SSH channel has an unresolved response; a new request is forbidden')
        self.sequence+=1;message={'sequence':self.sequence,'operation':operation,**(body or {})}
        if data is not None:message['binary_bytes']=len(data)
        encoded=json.dumps(message,separators=(',',':')).encode()+b'\n'
        if len(encoded)>1024**2:raise ValueError('Oversized SSH request')
        self.pending_response=True
        self._write(encoded)
        if data is not None:self._write(data)
        self.process.stdin.flush()
        remaining=min(timeout,self.deadline-time.time()-10)
        if remaining<=0:raise TimeoutError('SSH response deadline')
        try:result=self.answers.get(timeout=remaining)
        except queue.Empty as exc:
            # Wait once for this SAME outstanding immutable read. No resend,
            # new sequence, remote restart, or extension of the job deadline.
            extra=min(30,self.deadline-time.time()-10) if operation=='archive_chunk' else 0
            if extra<=0:raise TimeoutError('SSH response deadline for '+operation+'; retain the same remote lease') from exc
            try:result=self.answers.get(timeout=extra)
            except queue.Empty as final:raise TimeoutError('SSH response deadline for '+operation+'; retain the same remote lease') from final
        if isinstance(result,Exception):raise result
        if result.get('sequence')!=self.sequence:raise ValueError('SSH response sequence differs')
        self.pending_response=False
        if not result.get('ok'):raise RuntimeError('Owned remote operation failed: '+result.get('error','unknown'))
        return result['result']
    def close(self):
        if self.process.poll() is None:
            try:self.process.stdin.close();self.process.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):self.process.terminate()
        if self.process.poll() is None:
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        self.errors.close()
