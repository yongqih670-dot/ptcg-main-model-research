"""Opt-in engineering primitives. No launcher, budget authority, or science."""
import copy,json,os,re,time,uuid
from pathlib import Path
from .fo_v2.durable import identity,safe,atomic_json,exclusive

FLAGS=('control_cache','journal_cache','outbox_cache','capacity_transaction','progress_coalescing','exit_evidence')
def flags(value):
    if value is None:return dict.fromkeys(FLAGS,False)
    if not isinstance(value,dict) or set(value)-set(FLAGS) or any(type(x) is not bool for x in value.values()):raise ValueError('Unknown v3 engineering switch')
    return {k:value.get(k,False) for k in FLAGS}

def from_config(config,output):
    """A new registered server role is REQUIRED; old TR13 tickets cannot opt in."""
    value=config.get('framework_v3_optimizations')
    if value is None:return None
    from .io import read_json,sha256
    ticket=read_json(Path(output)/'ticket.json')
    if (config.get('engineering_only'),config.get('new_scientific_training'),ticket.get('framework_v3_approved'),ticket.get('engineering_scope'))!=(True,False,True,'framework_efficiency_v3'):
        raise ValueError('V3 cannot borrow a TR13 or closed v2 scientific/engineering ticket')
    if (ticket['run_id'],str(ticket['lease_epoch']))!=(os.environ.get('PTCG_JOB_RUN_ID'),os.environ.get('PTCG_JOB_LEASE_EPOCH')):raise ValueError('V3 job/lease identity differs')
    bound=ticket.get('framework_v3_manifest_sha256')
    if not isinstance(bound,str) or not re.fullmatch('[0-9a-f]{64}',bound) or bound!=config.get('framework_v3_manifest_sha256'):raise ValueError('V3 candidate not registered by sole controller')
    manifest_path=safe(Path(output),'framework-v3-manifest.json')
    if sha256(manifest_path)!=bound:raise ValueError('V3 manifest bytes differ from registered ticket')
    manifest=read_json(manifest_path);package_root=Path(__file__).resolve().parent.parent
    sources=manifest.get('runtime_sources')
    if not sources or not isinstance(sources,dict):raise ValueError('V3 runtime source binding absent')
    for relative,digest in sources.items():
        if sha256(safe(package_root,relative))!=digest:raise ValueError('V3 runtime source bytes differ')
    return flags(value)

class JsonEvidence:
    """Re-stat every read; cache only unchanged, fully owned immutable evidence.

    No TTL suppresses stop/clock changes. Returned values cannot mutate cache.
    A missing file evicts its parse; monotonic sequence validation stays caller's.
    """
    def __init__(self,root,reader,maximum=128*2**10):self.root=Path(root).resolve();self.reader=reader;self.maximum=maximum;self.rows={};self.hits=0;self.reads=0
    def get(self,path,*,optional=False):
        path=Path(path)
        relative=path.relative_to(self.root);safe(self.root,relative)
        try:before=identity(path)
        except FileNotFoundError:
            self.rows.pop(str(relative),None)
            if optional:return None
            raise
        if before[2]>self.maximum:raise ValueError('Cached JSON exceeds bounded envelope')
        key=str(relative);old=self.rows.get(key)
        if before[-1] is not None and old and old[0]==before:self.hits+=1;return copy.deepcopy(old[1])
        value=self.reader(path)
        if identity(path)!=before:raise ValueError('JSON changed during complete read')
        self.reads+=1;self.rows[key]=(before,copy.deepcopy(value));return value

def directory_identity(path):
    if os.name!='nt':return identity(path)
    import ctypes
    from ctypes import wintypes
    stat=Path(path).stat();change=None
    class Basic(ctypes.Structure):
        _fields_=[('creation',ctypes.c_longlong),('access',ctypes.c_longlong),('write',ctypes.c_longlong),('change',ctypes.c_longlong),('attributes',wintypes.DWORD)]
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];api.CreateFileW.restype=wintypes.HANDLE
    api.GetFileInformationByHandleEx.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD];api.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=api.CreateFileW(str(Path(path).resolve()),0x80,7,None,3,0x02200000,None)
    if handle not in (None,ctypes.c_void_p(-1).value):
        try:
            basic=Basic()
            if api.GetFileInformationByHandleEx(handle,0,ctypes.byref(basic),ctypes.sizeof(basic)):change=basic.change
        finally:api.CloseHandle(handle)
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_uid,stat.st_mode,change)

class DirectoryEvidence:
    """Cache only enumeration; re-stat directory AND each row on every query.

    Existing row mutation remains JsonEvidence's responsibility. No timer can
    delay discovery of a new row, missing ACK, stop command or publication error.
    """
    def __init__(self,root):self.root=Path(root).resolve();self.rows={};self.hits=0;self.reads=0
    def paths(self,directory,pattern):
        directory=Path(directory);safe(self.root,directory.relative_to(self.root));key=(str(directory),pattern)
        try:before=directory_identity(directory)
        except FileNotFoundError:self.rows.pop(key,None);return []
        old=self.rows.get(key)
        if before[-1] is not None and old and old[0]==before:self.hits+=1;return list(old[1])
        value=sorted(directory.glob(pattern))
        if directory_identity(directory)!=before:
            # Original glob snapshot remains valid during an append; avoid
            # sealing it, and re-enumerate on the next query. ACKs/errors/stops
            # still use their own fresh individual identity checks.
            self.rows.pop(key,None);self.reads+=1;return value
        self.rows[key]=(before,tuple(value));self.reads+=1;return value

class ProgressPublisher:
    IMMEDIATE_EXCEPT={'collecting','yielding_for_shared_capacity','archive_backpressure_wait'}
    def __init__(self,publish,interval=1.,clock=time.monotonic):
        if not 0<interval<=1:raise ValueError('Progress freshness may not exceed one second')
        self.publish=publish;self.interval=interval;self.clock=clock;self.last=None;self.pending=None;self.emitted=0;self.coalesced=0
    def __call__(self,path,value):
        now=self.clock();self.pending=(path,value)
        if value.get('stage') in self.IMMEDIATE_EXCEPT and self.last is not None and now-self.last<self.interval:
            self.coalesced+=1;return
        self.publish(path,value);self.last=now;self.pending=None;self.emitted+=1
    def flush(self):
        if self.pending:
            path,value=self.pending;self.publish(path,value);self.pending=None;self.last=self.clock();self.emitted+=1

class ExitEvidence:
    """Immutable correlated observations; missing records never prove a cause."""
    ROLES={'controller','receiver','worker','engine'}
    def __init__(self,root,run_id,epoch,correlation,role,*,process_identity):
        if role not in self.ROLES or Path(root).name!=run_id or type(epoch) is not int or epoch<1 or not correlation:raise ValueError('Exit evidence owner/role differs')
        self.root=Path(root);self.run=run_id;self.epoch=epoch;self.correlation=correlation;self.role=role;self.process=process_identity
    def record(self,event,*,observed_cause=None,returncode=None,exitcode_source=None,**fields):
        if returncode is not None and exitcode_source!='Popen.wait':raise ValueError('Exit code requires original process handle wait')
        row=dict(run_id=self.run,lease_epoch=self.epoch,correlation=self.correlation,role=self.role,process=self.process,
                 event=event,at=time.time(),monotonic=time.monotonic(),observed_cause=observed_cause,returncode=returncode,exitcode_source=exitcode_source,**fields)
        directory=self.root/'exit-evidence';directory.mkdir(exist_ok=True);path=directory/(self.role+'-'+uuid.uuid4().hex+'.json');atomic_json(path,row);return path

def correlate(rows):
    if not rows:return dict(initiating_cause='unproven',observations=[])
    owner={(r['run_id'],r['lease_epoch'],r['correlation']) for r in rows}
    if len(owner)!=1:raise ValueError('Exit observations belong to different attempts')
    explicit=[r for r in rows if r.get('observed_cause') in ('explicit_user_stop','owned_supervisor_resource_stop','caught_signal','uncaught_exception')]
    return dict(initiating_cause='unproven' if not explicit else 'explicit_observations_present_causality_not_assumed',
                observations=sorted(rows,key=lambda r:(r['at'],r['role'])),missing_roles=sorted({'controller','receiver','worker'}-{r['role'] for r in rows}))

def capacity_type(base):
    class TransactionCapacity(base):
        """Count our writes exactly; conservatively reserve external growth.

        Other project writers must obey the pre-existing single-heavy/4MiB
        producer +4MiB transfer envelope. Unknown writers require re-admission,
        never an infinite cache. Full scans still occur at most every60s.
        """
        def _scan(self):
            began=time.monotonic();value=super()._scan();elapsed=time.monotonic()-began
            self.external=max(256*2**20,int((60+elapsed)*8*2**20)+64*2**20)
            return value
    return TransactionCapacity
