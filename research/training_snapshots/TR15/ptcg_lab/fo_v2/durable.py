"""Private single-writer immutable objects, explicit space reservations and hashes.

All project writers must participate in the same reservation ledger before this
can replace production admission. A filesystem scan is required on restart.
Untracked-writer allowance is explicit; no rate-derived 256MiB cache assumption.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid
from contextlib import contextmanager


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def windows_change_time(path):
    if os.name!='nt':return 0
    import ctypes
    from ctypes import wintypes
    class Basic(ctypes.Structure):
        _fields_=[('creation',ctypes.c_longlong),('access',ctypes.c_longlong),('write',ctypes.c_longlong),('change',ctypes.c_longlong),('attributes',wintypes.DWORD)]
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    api.CreateFileW.restype=wintypes.HANDLE
    api.GetFileInformationByHandleEx.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=api.CreateFileW(str(Path(path).resolve()),0x80,7,None,3,0x00200000,None)
    if handle in (None,ctypes.c_void_p(-1).value):return None
    try:
        basic=Basic()
        if not api.GetFileInformationByHandleEx(handle,0,ctypes.byref(basic),ctypes.sizeof(basic)):return None
        return basic.change
    finally:api.CloseHandle(handle)
def identity(path):
    s=Path(path).stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid,s.st_mode,windows_change_time(path))
def sync_dir(path):
    if os.name=='posix':
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
def full_hash(path):
    h=hashlib.sha256();n=0
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b);n+=len(b)
    return h.hexdigest(),n
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial');data=canonical(value)
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);sync_dir(path.parent)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
@contextmanager
def exclusive(path):
    """OS-held lock; PID reuse or stale text cannot impersonate a live owner."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
def safe(root,relative):
    root=Path(root).resolve();relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:raise ValueError('Unsafe relative object')
    target=root
    for part in relative.parts:
        target=target/part
        if target.is_symlink() or getattr(target,'is_junction',lambda:False)():raise ValueError('Linked object path')
        if target.exists() and hasattr(os,'getuid') and target.stat().st_uid!=os.getuid():raise ValueError('Foreign object owner')
    if root not in target.resolve().parents:raise ValueError('Object outside private root')
    return target


class Capacity:
    """One shared writer ledger; outstanding reservations survive crashes.

    After a crash a fresh whole-project scan replaces stale reservations. Failed
    partial bytes remain counted. They are never assumed to have disappeared.
    """
    def __init__(self,root,hard_bytes,*,retained_bytes=0,free_reserve=20*2**30,
                 metadata_reserve=16*2**20,external_reserve=0,reconcile_seconds=60,
                 disk_free=None,scan=None):
        self.root=Path(root).resolve();self.hard=hard_bytes;self.retained=retained_bytes
        self.free_reserve=free_reserve;self.metadata=metadata_reserve;self.external=external_reserve
        self.disk_free=disk_free or (lambda:shutil.disk_usage(self.root).free)
        self.scan=scan or self._scan;self.interval=reconcile_seconds;self.lock=threading.RLock()
        self.reservations={};self.used=0;self.scans=0;self.reconcile()
    def _scan(self):
        total=0
        for folder,dirs,files in os.walk(self.root,followlinks=False):
            if any((Path(folder)/d).is_symlink() for d in dirs):raise ValueError('Linked writable subtree')
            for name in files:total+=(Path(folder)/name).lstat().st_size
        return total
    def reconcile(self):
        with self.lock:
            self.used=self.scan()+self.retained;self.at=time.monotonic();self.scans+=1
            self._check(0);self._persist()
    def _persist(self):
        atomic_json(self.root/'capacity.json',dict(schema=2,used=self.used,reservations=self.reservations,
            retained=self.retained,hard=self.hard,metadata_reserve=self.metadata,external_reserve=self.external))
    def _check(self,growth):
        outstanding=sum(self.reservations.values())
        if self.used+outstanding+growth+self.metadata+self.external>self.hard:raise OSError('Project capacity exhausted')
        if self.disk_free()-outstanding-growth-self.external<self.free_reserve:raise OSError('Physical free-space reserve')
    def reserve(self,maximum):
        if type(maximum) is not int or maximum<=0:raise ValueError('Invalid maximum write size')
        with self.lock:
            if time.monotonic()-self.at>=self.interval:self.reconcile()
            self._check(maximum);token=uuid.uuid4().hex;self.reservations[token]=maximum;self._persist();return token
    def check(self):
        with self.lock:self._check(0)
    def finish(self,token,actual):
        with self.lock:
            maximum=self.reservations.pop(token)
            if not 0<=actual<=maximum:raise ValueError('Reservation overrun')
            self.used+=actual;self._persist()
    def released(self,count):
        with self.lock:
            if not 0<=count<=self.used:raise ValueError('Invalid release')
            self.used-=count;self._persist()


class HashEvidence:
    """Only process-local evidence; restore/transfer/delete request fresh=True."""
    def __init__(self,root,reader=None):self.root=Path(root);self.cache={};self.bytes_read=0;self.hits=0;self.reader=reader;self.lock=threading.Lock()
    def verify(self,ref,*,fresh=False):
        p=safe(self.root,ref['path']);before=identity(p)
        with self.lock:
            if not fresh and before[-1] is not None and self.cache.get(ref['path'])==(before,ref['sha256'],ref['bytes']):
                self.hits+=1;return ref
        if self.reader:
            h=hashlib.sha256();count=0
            with self.reader.open(p) as stream:
                for data in iter(lambda:stream.read(2**20),b''):h.update(data);count+=len(data)
            digest=h.hexdigest()
        else:digest,count=full_hash(p)
        with self.lock:self.bytes_read+=count
        if identity(p)!=before or (digest,count)!=(ref['sha256'],ref['bytes']):raise ValueError('Object identity/content changed')
        with self.lock:self.cache[ref['path']]=(before,digest,count)
        return ref


class Stream:
    def __init__(self,raw,maximum,capacity,stop,rate,buffer_bytes=2**20):
        self.raw=raw;self.maximum=maximum;self.capacity=capacity;self.stop=stop;self.rate=rate
        self.buffer_bytes=buffer_bytes;self.buffer=bytearray();self.n=0;self.persisted=0
        self.hash=hashlib.sha256();self.started=time.monotonic();self.pacing=0.;self.writes=0
    def tell(self):return self.n
    def writable(self):return True
    def seekable(self):return False
    def write(self,data):
        data=memoryview(data).cast('B');count=len(data)
        if self.n+count>self.maximum:raise OSError('Object exceeds reserved maximum')
        while data:
            n=min(len(data),self.buffer_bytes-len(self.buffer));self.buffer.extend(data[:n]);data=data[n:];self.n+=n
            if len(self.buffer)==self.buffer_bytes:self.flush()
        return count
    def flush(self):
        if not self.buffer:return
        self.stop();self.capacity.check()
        if self.rate:
            due=(self.persisted+len(self.buffer))/self.rate-(time.monotonic()-self.started)
            while due>0:
                tick=min(due,.05);time.sleep(tick);self.pacing+=tick;self.stop();due-=tick
        view=memoryview(self.buffer)
        try:
            while view:
                n=self.raw.write(view)
                if not n:raise OSError('Write made no progress')
                self.hash.update(view[:n]);self.persisted+=n;view=view[n:];self.writes+=1
        finally:del view
        self.buffer.clear()


class Objects:
    def __init__(self,root,capacity,*,stop=lambda:None,rate=4*2**20,fault=lambda stage:None,reader=None):
        if Path(root).is_symlink() or getattr(Path(root),'is_junction',lambda:False)():raise ValueError('Linked store root')
        self.root=Path(root).resolve();self.capacity=capacity;self.stop=stop;self.rate=rate;self.fault=fault
        if rate is not None and rate<=0:raise ValueError('Invalid rate')
        self.reader=reader
        self.evidence=HashEvidence(self.root,reader);self.stats=dict(bytes_written=0,fsyncs=0,pacing_seconds=0.,objects=0,serialization_write_seconds=0.,independent_verify_seconds=0.)
        self.writer_lock=exclusive(self.root/'writer.lock');self.writer_lock.__enter__()
    def close(self):
        if self.writer_lock:self.writer_lock.__exit__(None,None,None);self.writer_lock=None
    def put(self,serialize,maximum,kind='blob'):
        self.stop();token=self.capacity.reserve(maximum);folder=safe(self.root,'objects');folder.mkdir(exist_ok=True)
        tmp=folder/(uuid.uuid4().hex+'.partial');accounted=False;published=0
        try:
            started=time.monotonic()
            with tmp.open('xb',buffering=0) as raw:
                stream=Stream(raw,maximum,self.capacity,self.stop,self.rate)
                serialize(stream);stream.flush();raw.flush();os.fsync(raw.fileno());self.stats['fsyncs']+=1
            self.fault('object_fsynced');self.stop()
            digest=stream.hash.hexdigest();relative=f'objects/{kind}-{digest}.bin';dest=safe(self.root,relative)
            ref=dict(path=relative,sha256=digest,bytes=stream.persisted)
            if dest.exists():
                self.evidence.verify(ref,fresh=True);tmp.unlink();actual=0
            else:
                # One writer owns this namespace. Atomic link publishes without overwrite.
                os.link(tmp,dest);published=stream.persisted;tmp.unlink();sync_dir(folder);actual=stream.persisted
            self.fault('object_published')
            self.capacity.finish(token,actual);accounted=True
            self.stats['serialization_write_seconds']+=time.monotonic()-started
            self.stats['bytes_written']+=stream.persisted;self.stats['pacing_seconds']+=stream.pacing;self.stats['objects']+=1
            # Independent full read is required once, separate from writer SHA.
            began=time.monotonic();self.evidence.verify(ref,fresh=True);self.stats['independent_verify_seconds']+=time.monotonic()-began
            self.fault('object_verified');return ref
        finally:
            if not accounted:self.capacity.finish(token,published+(tmp.stat().st_size if tmp.exists() and not published else 0))
    def json(self,value,maximum=2**20,kind='json'):
        data=canonical(value);return self.put(lambda stream:stream.write(data),maximum,kind)
