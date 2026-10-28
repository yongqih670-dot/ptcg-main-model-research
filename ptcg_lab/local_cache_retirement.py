"""Retire only a hash-verified redundant unzipped installer cache on Windows."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
from .io import read_json,atomic_json,utcnow,sha256

CACHE='runs/20261024T113758Z_hybrid-linux-wheelhouse_904030/uv-cache/archive-v0/MXlJE26BKkUgJV5CYSete'
WHEEL='runs/20261024T124736Z_hybrid-digest-assembly_cbadfd/wheelhouse/torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl'


class LocalReadPacer:
    """Bound local verification reads; the server's4MiB guard is unchanged."""
    def __init__(self,rate,deadline):
        if os.name!='nt' or os.environ.get('PTCG_PROJECT_ROOT') or not 0<rate<=16*2**20:
            raise ValueError('Local cache read pacing scope differs')
        self.rate=rate;self.deadline=deadline;self.next=time.monotonic()
    def consume(self,count):
        if count<0:raise ValueError('Negative local read size')
        self.next=max(self.next,time.monotonic())+count/self.rate
        while True:
            if time.time()>=self.deadline:raise TimeoutError('Local cache read deadline')
            remaining=self.next-time.monotonic()
            if remaining<=0:return
            time.sleep(min(.05,remaining))


def windows_security():
    if os.name!='nt':raise RuntimeError('This local retirement is Windows-only')
    from ctypes import wintypes as w
    adv=ctypes.WinDLL('advapi32',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    adv.GetNamedSecurityInfoW.argtypes=[w.LPWSTR,ctypes.c_int,w.DWORD,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)]
    adv.GetNamedSecurityInfoW.restype=w.DWORD
    adv.ConvertSidToStringSidW.argtypes=[ctypes.c_void_p,ctypes.POINTER(w.LPWSTR)];adv.ConvertSidToStringSidW.restype=w.BOOL
    adv.OpenProcessToken.argtypes=[w.HANDLE,w.DWORD,ctypes.POINTER(w.HANDLE)];adv.OpenProcessToken.restype=w.BOOL
    adv.GetTokenInformation.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.POINTER(w.DWORD)];adv.GetTokenInformation.restype=w.BOOL
    kernel.GetCurrentProcess.restype=w.HANDLE;kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[w.HANDLE]
    def text(sid):
        result=w.LPWSTR()
        if not adv.ConvertSidToStringSidW(sid,ctypes.byref(result)):raise ctypes.WinError(ctypes.get_last_error())
        try:return result.value
        finally:kernel.LocalFree(ctypes.cast(result,ctypes.c_void_p))
    token=w.HANDLE()
    if not adv.OpenProcessToken(kernel.GetCurrentProcess(),8,ctypes.byref(token)):raise ctypes.WinError(ctypes.get_last_error())
    try:
        needed=w.DWORD();adv.GetTokenInformation(token,1,None,0,ctypes.byref(needed))
        buffer=ctypes.create_string_buffer(needed.value)
        if not adv.GetTokenInformation(token,1,buffer,len(buffer),ctypes.byref(needed)):raise ctypes.WinError(ctypes.get_last_error())
        current=text(ctypes.cast(buffer,ctypes.POINTER(ctypes.c_void_p)).contents.value)
    finally:kernel.CloseHandle(token)
    def owner(path):
        sid=ctypes.c_void_p();descriptor=ctypes.c_void_p()
        result=adv.GetNamedSecurityInfoW(str(path),1,1,ctypes.byref(sid),None,None,None,ctypes.byref(descriptor))
        if result:raise ctypes.WinError(result)
        try:return text(sid)
        finally:kernel.LocalFree(descriptor)
    return current,owner


def owned(root,path,sid,owner,runs_container_owner_hash=None):
    root=Path(root).absolute();path=Path(path).absolute()
    if '..' in path.parts or path!=root and root not in path.parents:raise ValueError('Retirement path escaped project')
    if os.path.normcase(str(path.resolve()))!=os.path.normcase(str(path)):raise ValueError('Retirement path is not canonical')
    for p in [root,*[root.joinpath(*path.relative_to(root).parts[:n]) for n in range(1,len(path.relative_to(root).parts)+1)]]:
        st=p.lstat()
        if p.is_symlink() or getattr(st,'st_file_attributes',0)&0x400:raise ValueError('Retirement path traverses a reparse point')
        observed=owner(p)
        # The workspace's runs container is created by the Codex sandbox.
        # Only its preregistered OS-token fingerprint may differ. The project,
        # cache, wheel and every actual retirement target must be user-owned.
        if observed!=sid:
            allowed=p==root/'runs' and runs_container_owner_hash and hashlib.sha256(observed.encode()).hexdigest()==runs_container_owner_hash
            if not allowed:raise ValueError('Retirement path OS owner differs')
    return path


def identity(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_mode]


def digest_stream(stream,pacer):
    digest=hashlib.sha256();count=0
    while block:=stream.read(1024*1024):
        pacer.consume(len(block));digest.update(block);count+=len(block)
    return digest.hexdigest(),count


def verify_members(root,cache,wheel,sid,owner,pacer,runs_container_owner_hash=None):
    owned(root,cache,sid,owner,runs_container_owner_hash);owned(root,wheel,sid,owner,runs_container_owner_hash)
    actual={}
    for p in cache.rglob('*'):
        owned(root,p,sid,owner,runs_container_owner_hash)
        if p.is_file():actual[p.relative_to(cache).as_posix()]=p
    verified=[]
    with zipfile.ZipFile(wheel) as archive:
        rows=[r for r in archive.infolist() if not r.is_dir()];names=[r.filename for r in rows]
        if len(names)!=len(set(names)) or len(names)!=len({n.casefold() for n in names}) or set(names)!=set(actual):
            raise ValueError('Cache inventory differs from exact retained wheel members')
        for row in rows:
            path=actual[row.filename]
            if not 0<=row.file_size<=1024**3 or path.stat().st_size!=row.file_size:raise ValueError('Cache/wheel member size differs')
            before=identity(path)
            with path.open('rb') as stream:cached,size=digest_stream(stream,pacer)
            with archive.open(row) as stream:retained,n=digest_stream(stream,pacer)
            if cached!=retained or size!=n or n!=row.file_size or identity(path)!=before:
                raise ValueError('Cache file differs from retained full-content wheel member')
            verified.append({'member':row.filename,'bytes':n,'sha256':cached,'file_identity':before})
    return verified


def run(config,output):
    if config.get('retirement_variant') in ('public_download_prefixes','forensic_replicas','consumed_trajectory_gzip'):
        source_root=Path(__file__).resolve().parents[1]
        for relative,digest in config.get('execution_sources',{}).items():
            path=(source_root/relative).resolve()
            if source_root not in path.parents or sha256(path)!=digest:
                raise ValueError('Pinned local maintenance execution source changed')
    if config.get('retirement_variant')=='public_download_prefixes':
        from .local_fragment_retirement import run as retire_prefixes
        return retire_prefixes(config,output)
    if config.get('retirement_variant')=='forensic_replicas':
        from .local_forensic_retirement import run as retire_replicas
        return retire_replicas(config,output)
    if config.get('retirement_variant')=='consumed_trajectory_gzip':
        from .local_trajectory_archive import run as archive_originals
        return archive_originals(config,output)
    output=Path(output).resolve();root=Path(config['project_path']).absolute()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':raise RuntimeError('Retirement requires controlled local runner')
    if config.get('job_kind')!='local_cache_retirement' or config.get('maximum_attempts')!=1:raise ValueError('Wrong retirement scope')
    if (output/'attempts.json').exists() and read_json(output/'attempts.json'):raise RuntimeError('Original retirement attempt exhausted; reconcile evidence first')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    prior=sum(a['seconds'] for rid in config.get('prior_retirement_run_ids',[]) for a in read_json(root/'runs'/rid/'attempts.json'))
    if abs(prior-config.get('prior_retirement_seconds',0))>.001 or prior+deadline-time.time()>900.5:
        raise ValueError('Retirement repair cannot regain original900s scope')
    pacer=LocalReadPacer(16*2**20,deadline-25)
    sid,owner=windows_security()
    if hashlib.sha256(sid.encode()).hexdigest()!=config['owner_sid_sha256']:raise ValueError('Current OS identity differs from preregistration')
    if config.get('cache_relative')!=CACHE or config.get('wheel_relative')!=WHEEL:raise ValueError('Only the registered redundant install cache can retire')
    allowed_runs_owner=config['runs_container_owner_sid_sha256']
    cache=owned(root,root/CACHE,sid,owner,allowed_runs_owner);wheel=owned(root,root/WHEEL,sid,owner,allowed_runs_owner)
    old=root/'runs/20261024T113758Z_hybrid-linux-wheelhouse_904030'
    if sha256(old/'attempts.json')!=config['source_attempts_sha256'] or any(a['status']!='failed' for a in read_json(old/'attempts.json')):
        raise ValueError('Old install job terminal evidence differs')
    wheel_before=identity(wheel)
    with wheel.open('rb') as stream:wheel_sha,_=digest_stream(stream,pacer)
    if wheel_sha!=config['wheel_sha256'] or identity(wheel)!=wheel_before:raise ValueError('Retained pinned full wheel changed')
    atomic_json(output/'progress.json',{'at':utcnow(),'stage':'verifying_all_cache_members_before_any_unlink'})
    members=verify_members(root,cache,wheel,sid,owner,pacer,allowed_runs_owner)
    if len(members)!=11842 or sum(r['bytes'] for r in members)!=1625917636:raise ValueError('Registered redundant cache inventory changed')
    manifest={'at':utcnow(),'cache_relative':CACHE,'retained_wheel':WHEEL,'retained_wheel_sha256':wheel_sha,
              'OS_owner_sid_sha256':config['owner_sid_sha256'],'files':members,'all_members_verified_before_any_unlink':True}
    atomic_json(output/'verified-retirement-manifest.json',manifest)
    proof_sha=sha256(output/'verified-retirement-manifest.json')
    atomic_json(output/'retirement-intent.json',{'at':utcnow(),'manifest_sha256':proof_sha,'maximum_files':len(members),'no_directory_deletion':True})
    removed=0;released=0
    with (output/'retired-files.jsonl').open('x',encoding='utf-8') as log:
        for row in members:
            if time.time()>deadline-20:raise TimeoutError('Retirement deadline; keep exact partial receipts')
            if identity(wheel)!=wheel_before:raise ValueError('Retained recovery wheel changed before unlink')
            path=owned(root,cache/row['member'],sid,owner,allowed_runs_owner)
            if identity(path)!=row['file_identity']:raise ValueError('Verified cache inode changed before unlink')
            with path.open('rb') as stream:digest,n=digest_stream(stream,pacer)
            if digest!=row['sha256'] or n!=row['bytes'] or identity(path)!=row['file_identity']:raise ValueError('Cache changed before own single-file unlink')
            path.unlink();removed+=1;released+=n
            log.write(json.dumps({'member':row['member'],'bytes':n,'sha256':digest})+'\n')
            if removed%128==0:
                log.flush();os.fsync(log.fileno())
                atomic_json(output/'progress.json',{'at':utcnow(),'stage':'retiring_verified_own_cache_files','files':removed,'bytes':released})
        log.flush();os.fsync(log.fileno())
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,
        'retired_files':removed,'retired_logical_bytes':released,'retained_wheel_sha256':wheel_sha,
        'all_cache_member_bytes_exact_in_retained_wheel':True,'manifest_sha256':proof_sha,
        'original_attempts_logs_and_all_scientific_artifacts_unchanged':True,'no_directory_or_foreign_file_deletion':True,
        'rehydration':'Each retired file is reconstructable as its identical named member of the retained pinned wheel; full original hashes/identities preserved in verified manifest.',
        'promotion_eligible':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();run(read_json(a.config),a.output)
