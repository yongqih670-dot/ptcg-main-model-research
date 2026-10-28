"""Single finite, paced, lossless copy. Never changes the scientific source."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import time
from datetime import datetime,timezone

from .server_workspace import owned_root,inside,logical_bytes,admit
from .storage_profile import OLD,NEW,PROJECT,load,GIB


def utcnow():return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def atomic_json(path,value):
    path=Path(path);inside(NEW,path)
    raw=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n'
    if len(raw)>2*2**20:raise ValueError('Migration control evidence too large')
    temporary=path.with_name(path.name+'.incoming')
    with temporary.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    temporary.replace(path);sync_dir(path.parent)


def inventory(root):
    rows=[]
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in sorted(dirs+files):
            path=Path(directory)/name;s=path.lstat()
            if s.st_uid!=os.getuid():raise ValueError('Source contains a foreign-owned entry: '+str(path.relative_to(root)))
            if not (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode)):
                raise ValueError('Source contains non-file object')
            rows.append({'path':path.relative_to(root).as_posix(),'mode':stat.S_IMODE(s.st_mode),
                'type':'symlink' if stat.S_ISLNK(s.st_mode) else 'directory' if stat.S_ISDIR(s.st_mode) else 'file',
                'identity':[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns],
                'link':os.readlink(path) if stat.S_ISLNK(s.st_mode) else None})
    return sorted(rows,key=lambda r:r['path'])


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def copy(root,source,job,ticket):
    started=time.monotonic();rows=inventory(source);source_bytes=sum(r['identity'][2] for r in rows if r['type']!='directory')
    # Peak includes all old files plus the complete new copy and conservative log/evidence reserve.
    initial=admit(root,source_bytes+128*2**20)
    progress={'started_at':utcnow(),'phase':'copying','source_files':len(rows),'source_logical_bytes':source_bytes,
        'copied_files':0,'copied_unique_bytes':0,'verified_destination_bytes':0,'hardlinks_preserved':0,
        'source_unchanged':False,'complete':False,'initial_capacity':initial}
    atomic_json(job/'copy-progress.json',progress)
    inode_map={};records=[];last_report=0.;next_io=time.monotonic()
    abort=False
    def stop(*_):
        nonlocal abort
        abort=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def check():
        if abort or time.time()>ticket['deadline_unix']-75:raise TimeoutError('Controlled copy stop/deadline')
        if shutil.disk_usage(root).free<20*GIB:raise OSError('Target free reserve reached')
    for row in rows:
        check();rel=row['path'];src=source/rel
        if rel=='.ptcg-owner.json':
            records.append({**row,'source_sha256':sha256(src),'relocation_exception':'new private root marker records actual new root'})
            continue
        dst=root/rel
        if row['type']=='directory':
            if dst.exists():
                inside(root,dst)
                if not dst.is_dir():raise ValueError('Destination collision')
            else:dst.mkdir(mode=row['mode'])
            records.append(row);continue
        if dst.parent!=root:inside(root,dst.parent)
        if dst.exists() or dst.is_symlink():raise FileExistsError('Destination collision: '+rel)
        if row['type']=='symlink':
            # Preserve link text without following it. No external file is read or written.
            os.symlink(row['link'],dst);records.append({**row,'destination_link':os.readlink(dst)})
        else:
            key=tuple(row['identity'][:2]);size=row['identity'][2]
            if key in inode_map:
                original,digest=inode_map[key];os.link(original,dst);progress['hardlinks_preserved']+=1
            else:
                digest=hashlib.sha256();fd=os.open(src,os.O_RDONLY|os.O_NOFOLLOW)
                with os.fdopen(fd,'rb') as incoming,dst.open('xb') as outgoing:
                    before=os.fstat(incoming.fileno())
                    if [before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns]!=row['identity']:
                        raise ValueError('Source identity changed before copy')
                    while block:=incoming.read(1024*1024):
                        check();outgoing.write(block);digest.update(block);progress['copied_unique_bytes']+=len(block)
                        next_io=max(next_io,time.monotonic())+len(block)/(ticket['io_bytes_per_second'])
                        while time.monotonic()<next_io:check();time.sleep(max(0,min(.05,next_io-time.monotonic())))
                    outgoing.flush();os.fsync(outgoing.fileno())
                digest=digest.hexdigest()
                if sha256(dst)!=digest:raise ValueError('Destination full SHA differs')
                progress['verified_destination_bytes']+=size
                os.chmod(dst,row['mode']);os.utime(dst,ns=(row['identity'][3],row['identity'][3]))
                inode_map[key]=(dst,digest)
            current=src.stat()
            if [current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns]!=row['identity']:
                raise ValueError('Source changed after copy')
            records.append({**row,'sha256':digest,'destination_identity':[dst.stat().st_dev,dst.stat().st_ino,size,dst.stat().st_mtime_ns]})
        progress['copied_files']+=1
        if time.monotonic()-last_report>=5:
            check();progress.update(current_path=rel,elapsed_seconds=round(time.monotonic()-started,3))
            atomic_json(job/'copy-progress.json',progress);last_report=time.monotonic()
    if inventory(source)!=rows:raise ValueError('Source tree changed during cold copy')
    for directory,dirs,files in os.walk(root,followlinks=False):sync_dir(directory)
    manifest={'schema':'ptcg_lossless_cold_copy_v1','at':utcnow(),'source_root':str(source),'target_root':str(root),
        'source_logical_bytes':source_bytes,'source_unchanged':True,'files':records,
        'symlinks_preserved_without_following':True,'hardlink_groups_preserved':True,
        'source_marker_sha256':sha256(source/'.ptcg-owner.json'),'target_marker_sha256':sha256(root/'.ptcg-owner.json')}
    raw=json.dumps(manifest,ensure_ascii=False,separators=(',',':')).encode()
    packed=gzip.compress(raw,compresslevel=1,mtime=0);path=job/'full-copy-manifest.json.gz'
    with path.open('xb') as f:f.write(packed);f.flush();os.fsync(f.fileno())
    progress.update(complete=True,phase='cold_copy_fully_verified',source_unchanged=True,
        elapsed_seconds=round(time.monotonic()-started,3),manifest_sha256=sha256(path),
        manifest_uncompressed_sha256=hashlib.sha256(raw).hexdigest(),manifest_bytes=len(packed),
        final_capacity=admit(root,0),training_started=False)
    atomic_json(job/'copy-progress.json',progress);sync_dir(job)
    return progress


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ticket',required=True);args=parser.parse_args()
    ticket=json.loads(Path(args.ticket).read_text());root=owned_root(NEW,PROJECT);source=owned_root(OLD,PROJECT)
    if (ticket['kind']!='storage_migration' or ticket['project_root']!=NEW
            or ticket['run_id']!=os.environ.get('PTCG_JOB_RUN_ID') or ticket['source_root']!=OLD
            or not 0<ticket['io_bytes_per_second']<=16*2**20):raise ValueError('Unexpected controlled copy ticket')
    job=inside(root,ticket['job_directory']);profile=load(root)
    if profile!=ticket['storage_profile']:raise ValueError('Storage profile differs from launch')
    group=Path('/proc/self/cgroup').read_text().strip().split('::',1)[1]
    cg=Path('/sys/fs/cgroup')/group.lstrip('/')
    cpu=(cg/'cpu.max').read_text().split();memory=int((cg/'memory.max').read_text())
    if int(cpu[0])/int(cpu[1])!=ticket['cpu_quota_cores'] or memory!=ticket['memory_max_bytes']:
        raise ValueError('Actual kernel CPU/memory limits differ')
    atomic_json(job/'actual-resource-limits.json',{'at':utcnow(),'cpu.max':cpu,'memory.max':memory,
        'memory.swap.max':(cg/'memory.swap.max').read_text().strip(),'pids.max':(cg/'pids.max').read_text().strip(),
        'cgroup':group,'gpu_visibility':os.environ.get('CUDA_VISIBLE_DEVICES'),'project_profile':profile})
    try:copy(root,source,job,ticket)
    except BaseException as exc:
        atomic_json(job/'copy-failure.json',{'at':utcnow(),'error':type(exc).__name__,'message':str(exc),
            'source_not_modified':True,'target_partial_preserved':True});raise


if __name__=='__main__':main()
