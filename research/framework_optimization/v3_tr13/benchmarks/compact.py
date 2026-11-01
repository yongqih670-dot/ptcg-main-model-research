"""Lossless case retention with dependencies only on pre-frozen original inputs."""
import hashlib,io,os,shutil,tarfile,time
from pathlib import Path
from common import *
class PacedFile:
    def __init__(self,file):
        from ptcg_lab.io_pacing import BytePacer
        self.file=file;self.pacer=BytePacer(4*2**20,float(os.environ['V3_DEADLINE']))
    def __getattr__(self,name):return getattr(self.file,name)
    def write(self,data):guard();self.pacer.consume(len(data));return self.file.write(data)
    def read(self,count=-1):
        data=self.file.read(count);guard();self.pacer.consume(len(data));return data

def compact(label,*,failed_zero_update=False):
    base=(V3/'work'/label).resolve();allowed=(V3/'work').resolve()
    job=job_path(label)
    if base.parent!=allowed or not (job/'local-engineering-owner.json').exists():raise ValueError('Not an explicitly owned new case')
    # Called only after original Popen.wait has returned for each phase.
    receipts=list((V3/'reports').glob(label+'-*-process.json'))
    if not receipts or any((lambda r:r.get('status')!='terminal' or r.get('exitcode_source')!='Popen.wait')(read(p)) for p in receipts):raise ValueError('Live case cannot release originals')
    if failed_zero_update:
        if label not in ('B-baseline','B2-baseline') or (job/'recovery-v2.json').exists():raise ValueError('Only diagnosed pre-optimizer fixture failures with no recovery references')
        if not all(read(p)['returncode']!=0 for p in receipts):raise ValueError('Expected actual failure receipt')
        cold=None
    else:
        if any(read(p)['returncode']!=0 for p in receipts):
            recovery=V3/'reports'/f'{label}-recover-qual-process.json'
            sealed=V3/'reports'/f'{label}-sealed.json'
            diagnosed_zero=(label=='D-baseline-0' and sealed.exists() and read(sealed).get('passed') and read(sealed).get('optimizer_updates')==0 and not read(sealed).get('performance_result_valid'))
            if not diagnosed_zero and (label!='B3-baseline' or not recovery.exists() or read(recovery)['returncode']!=0):raise ValueError('Failed state is retained pending diagnosis')
        suffix='cold.json' if label=='A-real-parent' else ('replay-cold-repair.json' if label=='B3-baseline' else 'replay-cold.json')
        cold=read(V3/'reports'/(label+'-'+suffix))
        if not cold.get('passed') or len(cold['points'])!=2:raise ValueError('Two fresh recovery points required')
    spec=read(V3/'reports/INPUTS.json');catalog=spec['immutable_catalog'];directory=V3/'retained';directory.mkdir(exist_ok=True);bundle=directory/(label+'.tar.gz')
    if bundle.exists():raise ValueError('Retained bundle exists; no overwrite')
    rows=[];began=time.monotonic();checked=set();gz_sources={}
    # Original bytes may already have a complete, verified gzip replica in
    # THIS case. Store that original encoded replica once; decompression is
    # portable and exact, and never refers to another case's new output.
    from ptcg_lab.archive_ack import validate_ack
    from ptcg_lab.archive_compressed import verify_archive
    for receipt in (base/'archive'/job.name/'acks').glob('*.json'):
        ack=read(receipt)
        if ack.get('storage') and ack['sha256'] not in catalog:
            validate_ack(ack,job.name,1,ack['object'],ack['sha256'],ack['bytes'],ack['sequence']);verify_archive(base/'archive',ack)
            gz_sources[ack['sha256']]=('archive/'+ack['archive_relative'])
    for path in sorted(base.rglob('*')):
        guard()
        if path.is_symlink():raise ValueError('Linked case member')
        if not path.is_file():continue
        digest=sha(path);size=path.stat().st_size;row=dict(path=path.relative_to(base).as_posix(),sha256=digest,bytes=size)
        old=catalog.get(digest)
        if old:
            if digest not in checked:
                if sha(ROOT/old['path'])!=digest or (ROOT/old['path']).stat().st_size!=size:raise ValueError('Frozen original dependency changed')
                checked.add(digest)
            row['original_dependency']=old['path']
        elif digest in gz_sources and row['path']!=gz_sources[digest]:row['gzip_dependency']=gz_sources[digest]
        rows.append(row)
    from ptcg_lab.server_workspace import logical_bytes
    actual_project_bytes=logical_bytes(ROOT);stored=[r for r in rows if 'original_dependency' not in r and 'gzip_dependency' not in r]
    # Deflate worst-case expansion, tar headers/padding, and receipt reserve.
    raw_total=sum(r['bytes'] for r in stored)+len(stored)*4096+2**20;archive_growth_upper=raw_total+raw_total//1000+16*2**20
    if actual_project_bytes+archive_growth_upper>120*2**30:raise OSError('Full project + worst-case new archive exceeds unchanged120GiB; originals retained')
    with bundle.open('xb') as raw,tarfile.open(fileobj=PacedFile(raw),mode='w:gz',compresslevel=1,copybufsize=2**20) as out:
        for row in stored:
            path=base/row['path'];info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=path.stat().st_mode & 0o777;info.mtime=path.stat().st_mtime
            with path.open('rb') as member:out.addfile(info,PacedFile(member))
        out.close();raw.flush();os.fsync(raw.fileno())
    expected={r['path']:r for r in rows if 'original_dependency' not in r and 'gzip_dependency' not in r};seen=set()
    # Preload with bounded MiB reads so gzip's tiny header reads do not charge
    # a Windows scheduler quantum per byte. This is an artifact verification
    # buffer, equally applied; it is not a training/inference cache change.
    cached=io.BytesIO()
    with bundle.open('rb') as raw:
        paced=PacedFile(raw)
        while data:=paced.read(2**20):cached.write(data)
    cached.seek(0)
    with tarfile.open(fileobj=cached,mode='r:gz') as source:
        for member in source:
            guard();row=expected.get(member.name)
            if not member.isfile() or member.name in seen or row is None or member.size!=row['bytes']:raise ValueError('Unexpected lossless bundle member')
            h=hashlib.sha256()
            with source.extractfile(member) as stream:
                while data:=stream.read(2**20):h.update(data)
            if h.hexdigest()!=row['sha256']:raise ValueError('Independent lossless archive verification failed')
            seen.add(member.name)
        import gzip
        verified_gzip={}
        for row in rows:
            dependency=row.get('gzip_dependency')
            if not dependency:continue
            if dependency not in expected:raise ValueError('Gzip dependency must be stored inside the same case')
            if dependency not in verified_gzip:
                h=hashlib.sha256();count=0
                with source.extractfile(dependency) as encoded,gzip.GzipFile(fileobj=encoded,mode='rb') as decoded:
                    while data:=decoded.read(2**20):
                        count+=len(data)
                        if count>128*2**20:raise ValueError('Derived original exceeds existing file envelope')
                        h.update(data)
                verified_gzip[dependency]=(h.hexdigest(),count)
            if verified_gzip[dependency]!=(row['sha256'],row['bytes']):raise ValueError('Gzip-derived original bytes differ')
    if seen!=set(expected):raise ValueError('Missing retained new case bytes')
    for row in rows:
        if sha(base/row['path'])!=row['sha256']:raise ValueError('Original changed before complete retention ACK')
    receipt=dict(label=label,bundle_sha256=sha(bundle),bundle_bytes=bundle.stat().st_size,files=rows,full_archive_verified=True,two_recovery_points_cold_verified=cold is not None,pre_optimizer_fixture_failure=failed_zero_update,gzip_derived_originals=sum('gzip_dependency' in r for r in rows),
                 depends_only_on_pre_frozen_original_catalog=True,seconds=time.monotonic()-began,scientific_credit=0,original_path=str(base),released_new_temporary_bytes=False)
    from ptcg_lab.fo_v2.durable import atomic_json as durable_json
    durable_json(directory/(label+'.json'),receipt)
    # Explicit absolute boundary recheck; old runs/v1/v2 are never candidates.
    if base.resolve().parent!=allowed or base.is_symlink():raise ValueError('Release boundary changed')
    shutil.rmtree(base);receipt['released_new_temporary_bytes']=True;receipt['seconds']=time.monotonic()-began;durable_json(directory/(label+'.json'),receipt)
    return receipt

def reconstruct(label,destination):
    destination=Path(destination).resolve();allowed=(V3/'restored').resolve()
    if destination.parent!=allowed or destination.exists():raise ValueError('Fresh isolated restored child required')
    receipt=read(V3/'retained'/f'{label}.json');bundle=V3/'retained'/f'{label}.tar.gz'
    if sha(bundle)!=receipt['bundle_sha256']:raise ValueError('Bundle changed')
    destination.mkdir(parents=True)
    with tarfile.open(bundle,'r:gz') as source:
        members={m.name:m for m in source}
        for row in receipt['files']:
            if 'gzip_dependency' in row:continue
            target=destination/row['path']
            if not target.resolve().is_relative_to(destination):raise ValueError('Manifest path escaped')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as dst:
                if 'original_dependency' in row:
                    with (ROOT/row['original_dependency']).open('rb') as src:shutil.copyfileobj(src,dst,2**20)
                else:
                    with source.extractfile(members[row['path']]) as src:shutil.copyfileobj(src,dst,2**20)
            if sha(target)!=row['sha256']:raise ValueError('Reconstruction differs')
    import gzip
    for row in receipt['files']:
        if 'gzip_dependency' not in row:continue
        source=destination/row['gzip_dependency'];target=destination/row['path']
        if not source.resolve().is_relative_to(destination) or not target.resolve().is_relative_to(destination):raise ValueError('Derived path escaped')
        target.parent.mkdir(parents=True,exist_ok=True)
        count=0
        with gzip.open(source,'rb') as reader,target.open('xb') as writer:
            while data:=reader.read(2**20):
                count+=len(data)
                if count>row['bytes']:raise ValueError('Derived original exceeds exact byte envelope')
                writer.write(data)
        if target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']:raise ValueError('Derived original reconstruction differs')
    return destination
