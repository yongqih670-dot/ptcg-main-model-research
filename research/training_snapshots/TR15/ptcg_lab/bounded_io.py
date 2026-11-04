"""Owned atomic writes with paced, synchronous, bounded writeback chunks.

This is an application writer, not a kernel quota or permission to train.
The independent resource guardian and integrated entrypoint acceptance remain
mandatory. Failed temporary files are retained and never published as commits.
"""
import hashlib
import os
from pathlib import Path
import shutil
import time
import uuid

from .io_pacing import BytePacer
from .phase_timing import timed,span
from .server_workspace import owned_root,inside,admit


class BoundedWriter:
    def __init__(self,stream,root,path,pacer,max_bytes,deadline,chunk_bytes=64*1024):
        if type(max_bytes) is not int or not 0<max_bytes<=2*2**30:raise ValueError('Invalid file write envelope')
        if type(chunk_bytes) is not int or not 0<chunk_bytes<=64*1024:raise ValueError('Writeback chunk is too large')
        self.stream=stream;self.root=Path(root);self.path=Path(path);self.pacer=pacer
        self.max_bytes=max_bytes;self.deadline=deadline;self.chunk_bytes=chunk_bytes
        self.buffer=bytearray();self.position=0;self.persisted=0;self.chunks=0;self.max_chunk=0
        self.digest=hashlib.sha256();self.identity=os.fstat(stream.fileno())
    @timed('storage_identity_and_reserve_checks')
    def _check(self):
        if time.time()>=self.deadline:raise TimeoutError('Bounded writer deadline')
        path=inside(self.root,self.path);actual=path.stat();fd=os.fstat(self.stream.fileno())
        if (actual.st_dev,actual.st_ino)!=(self.identity.st_dev,self.identity.st_ino) or (fd.st_dev,fd.st_ino)!=(actual.st_dev,actual.st_ino):
            raise RuntimeError('Owned write target identity changed')
        if hasattr(os,'getuid') and fd.st_uid!=os.getuid():raise RuntimeError('Owned write descriptor UID differs')
        remaining=self.max_bytes-self.persisted
        if shutil.disk_usage(self.root).free-remaining<20*2**30:raise OSError('Bounded writer disk reserve reached')
    def _drain(self):
        if not self.buffer:return
        self._check();count=len(self.buffer)
        with span('storage_pacing_wait'):self.pacer.consume(count)
        self._check()
        view=memoryview(self.buffer)
        try:
            while view:
                with span('storage_write'):n=self.stream.write(view)
                if not n:raise OSError('No progress writing owned checkpoint')
                self.digest.update(view[:n]);self.persisted+=n;view=view[n:]
        finally:del view
        with span('storage_flush_fsync'):
            self.stream.flush();os.fsync(self.stream.fileno())
        self._check()
        self.max_chunk=max(self.max_chunk,count);self.chunks+=1;self.buffer.clear()
    def write(self,value):
        data=memoryview(value).cast('B');count=len(data)
        if self.position+count>self.max_bytes:raise OSError('Bounded file size envelope reached')
        if time.time()>=self.deadline:raise TimeoutError('Bounded writer deadline')
        while data:
            n=min(len(data),self.chunk_bytes-len(self.buffer));self.buffer.extend(data[:n]);data=data[n:]
            self.position+=n
            if len(self.buffer)==self.chunk_bytes:self._drain()
        return count
    def tell(self):return self.position
    def flush(self):self._drain();self._check()
    def writable(self):return True
    def seekable(self):return False
    def readable(self):return False
    def receipt(self):
        if self.buffer or self.persisted!=self.position:raise RuntimeError('Unflushed checkpoint cannot have a receipt')
        return {'bytes':self.persisted,'sha256':self.digest.hexdigest(),'synchronous_chunks':self.chunks,
                'maximum_unsynced_chunk_bytes':self.max_chunk,'configured_chunk_bytes':self.chunk_bytes,
                'write_bytes_per_second':self.pacer.rate}


class WriteContext:
    def __init__(self,root,project_id,deadline,bytes_per_second=4*2**20,max_file_bytes=128*2**20,pacer=None):
        self.root=owned_root(root,project_id);self.project_id=project_id;self.deadline=deadline
        if not 0<bytes_per_second<=4*2**20:raise ValueError('Unaccepted write rate exceeds conservative envelope')
        if type(max_file_bytes) is not int or not 0<max_file_bytes<=2*2**30:raise ValueError('Invalid bounded file size')
        self.max_file_bytes=max_file_bytes;self.pacer=pacer or BytePacer(bytes_per_second,deadline)
        if self.pacer.rate!=bytes_per_second:raise ValueError('Injected pacer rate differs')
    @timed('storage_atomic_publication')
    def atomic_write(self,path,serialize,replace=True,expected_bytes=None):
        path=inside(self.root,path)
        if expected_bytes is not None and (type(expected_bytes) is not int or not 0<expected_bytes<=self.max_file_bytes):
            raise ValueError('Exact serialized size exceeds the unchanged file envelope')
        file_budget=self.max_file_bytes if expected_bytes is None else expected_bytes
        with span('storage_capacity_admission'):
            admit(self.root,file_budget+64*2**20)  # proved file size plus concurrent control logs
        if time.time()>=self.deadline:raise TimeoutError('Atomic write admission deadline')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.parent!=self.root:inside(self.root,path.parent)
        if not replace and path.exists():raise FileExistsError('Immutable output already exists')
        before=path.stat() if path.exists() else None
        temporary=inside(self.root,path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial'))
        with temporary.open('xb',buffering=0) as raw:
            writer=BoundedWriter(raw,self.root,temporary,self.pacer,file_budget,self.deadline)
            serialize(writer);writer.flush();receipt=writer.receipt()
            if expected_bytes is not None and receipt['bytes']!=expected_bytes:
                raise ValueError('Serialized output differs from its exact predeclared byte count')
        # Revalidate immediately before publication; never overwrite a changed
        # or unexpected prior checkpoint and never delete failed partial bytes.
        inside(self.root,path);current=path.stat() if path.exists() else None
        if (before is None)!=(current is None) or (before and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)):
            raise RuntimeError('Checkpoint destination changed during serialization')
        if time.time()>=self.deadline:raise TimeoutError('Checkpoint publication deadline')
        if replace:os.replace(temporary,path)
        else:os.link(temporary,path);temporary.unlink()
        if os.name=='posix':
            descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        return {**receipt,'path':str(path),'atomic_publication':True}


_environment_context=None


def environment_context():
    global _environment_context
    root=os.environ.get('PTCG_PROJECT_ROOT')
    if not root:return None
    identity=(root,os.environ['PTCG_PROJECT_ID'],float(os.environ['PTCG_JOB_DEADLINE_UNIX']),os.getpid())
    if _environment_context is None or _environment_context[0]!=identity:
        _environment_context=(identity,WriteContext(*identity[:3]))
    return _environment_context[1]
