"""Paced, seekable Python file interface with no native descriptor bypass."""
import hashlib
import os
from pathlib import Path
import time
from .io_pacing import BytePacer
from .server_workspace import owned_root


class PacedReader:
    def __init__(self,stream,pacer,deadline,chunk_bytes=64*1024):
        if not 0<chunk_bytes<=64*1024:raise ValueError('Invalid bounded read chunk')
        self.stream=stream;self.pacer=pacer;self.deadline=deadline;self.chunk_bytes=chunk_bytes
        self.identity=os.fstat(stream.fileno());self.bytes_read=0;self.maximum_read=0;self.calls=0
    def _check(self):
        if time.time()>=self.deadline:raise TimeoutError('Paced read deadline')
        current=os.fstat(self.stream.fileno());old=self.identity
        if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)!=(old.st_dev,old.st_ino,old.st_size,old.st_mtime_ns):
            raise RuntimeError('Read source changed during bounded operation')
    def read(self,count=-1):
        self._check();remaining=max(0,self.identity.st_size-self.tell())
        count=remaining if count is None or count<0 else min(count,remaining)
        if count>128*2**20:raise OSError('Single read result exceeds bounded memory envelope')
        result=bytearray()
        while len(result)<count:
            n=min(self.chunk_bytes,count-len(result));self.pacer.consume(n);self._check()
            block=self.stream.read(n)
            if len(block)!=n:raise EOFError('Immutable bounded read truncated')
            result.extend(block);self.bytes_read+=n;self.calls+=1;self.maximum_read=max(self.maximum_read,n)
        self._check();return bytes(result)
    def readinto(self,target):
        view=memoryview(target).cast('B');total=0
        while total<len(view):
            data=self.read(min(self.chunk_bytes,len(view)-total))
            if not data:break
            view[total:total+len(data)]=data;total+=len(data)
        return total
    def seek(self,offset,whence=os.SEEK_SET):self._check();return self.stream.seek(offset,whence)
    def tell(self):return self.stream.tell()
    def readable(self):return True
    def seekable(self):return True
    def writable(self):return False
    def close(self):self.stream.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def receipt(self):return {'bytes_read':self.bytes_read,'maximum_read_chunk_bytes':self.maximum_read,'read_calls':self.calls,
                              'read_bytes_per_second':self.pacer.rate,'native_descriptor_exposed':False}


class ReadContext:
    def __init__(self,root,project_id,deadline,bytes_per_second=4*2**20,pacer=None):
        self.root=owned_root(root,project_id);self.deadline=deadline
        self.pacer=pacer or BytePacer(bytes_per_second,deadline)
        if not 0<bytes_per_second<=4*2**20 or self.pacer.rate!=bytes_per_second:raise ValueError('Read pacing envelope differs')
    def open(self,path):
        return PacedReader(Path(path).open('rb',buffering=0),self.pacer,self.deadline)
    def sha256(self,path):
        digest=hashlib.sha256()
        with self.open(path) as stream:
            while block:=stream.read(64*1024):digest.update(block)
        return digest.hexdigest()


_environment_read_context=None


def environment_read_context():
    global _environment_read_context
    root=os.environ.get('PTCG_PROJECT_ROOT')
    if not root:return None
    identity=(root,os.environ['PTCG_PROJECT_ID'],float(os.environ['PTCG_JOB_DEADLINE_UNIX']),os.getpid())
    if _environment_read_context is None or _environment_read_context[0]!=identity:
        _environment_read_context=(identity,ReadContext(*identity[:3]))
    return _environment_read_context[1]
