"""Retire exact, reconstructible public-wheel prefixes from terminal own jobs."""
import os
import hashlib
from pathlib import Path
from .io import atomic_json,read_json,sha256,utcnow
from .server_workspace import inside
from . import systemd_job


def validate_local_plans(project,plans):
    project=Path(project).resolve();verified=set()
    for plan in plans:
        wheel=plan['retained_complete_wheel'];path=(project/wheel['path']).resolve()
        if project not in path.parents or path.is_symlink() or path.stat().st_size!=wheel['bytes']:
            raise ValueError('Retained public wheel escaped or changed size')
        if (str(path),wheel['sha256']) not in verified:
            if sha256(path)!=wheel['sha256']:raise ValueError('Retained full public wheel hash differs')
            verified.add((str(path),wheel['sha256']))
        if not 0<plan['bytes']<wheel['bytes']:raise ValueError('Public prefix length differs')
        digest=hashlib.sha256();left=plan['bytes']
        with path.open('rb') as stream:
            while left:
                data=stream.read(min(left,1024*1024))
                if not data:raise EOFError('Retained public wheel ended early')
                digest.update(data);left-=len(data)
        if digest.hexdigest()!=plan['sha256']:raise ValueError('Public prefix cannot reconstruct from the retained complete wheel')
        proof=plan['terminal_evidence'];terminal=read_json(project/proof['path'])
        if sha256(project/proof['path'])!=proof['sha256'] or terminal!=plan['terminal']:
            raise ValueError('Original public-fragment terminal receipt differs')


def retire(root,job,plans):
    checked=[]
    for plan in plans:
        terminal=plan['terminal'];rid=terminal['run_id'];handle=terminal['identity']
        if '/' in rid or '\\' in rid or not handle['unit'].startswith('ptcg-hybrid-env-'):
            raise ValueError('Public fragment owner is not an original environment job')
        source=inside(root,root/'jobs'/rid)
        ticket=read_json(inside(root,source/'ticket.json'))
        if (ticket['run_id'],ticket['lease_epoch'],ticket['unit'])!=(rid,terminal['lease_epoch'],handle['unit']):
            raise ValueError('Original environment ticket differs')
        if read_json(inside(root,source/'identity.json'))!=handle:
            raise ValueError('Original environment identity differs')
        props=terminal['service_properties']
        if not systemd_job.terminal(props) or props['InvocationID']!=handle['invocation_id'] or props['ExecMainStartTimestampMonotonic']!=handle['start_monotonic']:
            raise ValueError('Original terminal receipt is insufficient')
        current=systemd_job.inspect_service(handle['unit'])
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if current.get('LoadState')!='not-found':
            if boot!=handle['host_boot_id'] or not systemd_job.terminal(current) or systemd_job.identity(current,boot)!=handle:
                raise ValueError('Original environment unit is live or identity changed')
        elif boot==handle['host_boot_id']:
            import time
            if time.time()<terminal['deadline_unix']+60:raise ValueError('Missing unit before old deadline is not terminal proof')
        relative=plan['path']
        if relative!='stage/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl.partial':
            raise ValueError('Only the named obsolete public download prefix can retire')
        path=inside(root,source/relative);st=path.stat()
        ident=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_uid)
        if st.st_size!=plan['bytes'] or not 0<st.st_size<plan['retained_complete_wheel']['bytes'] or sha256(path)!=plan['sha256']:
            raise ValueError('Obsolete prefix differs from retained complete local public wheel')
        checked.append((path,ident,plan))
    if len({str(p) for p,_,_ in checked})!=len(checked):raise ValueError('Duplicate retirement target')
    receipt={'at':utcnow(),'status':'intent','objects':[p for _,_,p in checked],
        'only_exact_owned_terminal_public_download_fragments':True,'full_local_wheel_and_scientific_outputs_preserved':True}
    atomic_json(job/'public-fragment-retirement.json',receipt)
    for path,ident,plan in checked:
        st=inside(root,path).stat()
        if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_uid)!=ident or st.st_uid!=os.getuid():
            raise ValueError('Public prefix identity changed immediately before retirement')
        path.unlink()
    atomic_json(job/'public-fragment-retirement.json',{**receipt,'at':utcnow(),'status':'retired',
        'bytes_released':sum(p['bytes'] for _,_,p in checked)})


def finish_terminal_consumed_retirement(root,job,project_id,spec,proof,acks):
    """Finish the interrupted original ACK-gated retirement safe point only."""
    from .retained_source import resolve
    from .scientific_resume import validate_postcommit_recovery
    from .checkpoint_refs import evict_archived_replicas
    from .archive_ack import validate_ack
    import re
    source,_=resolve(root,project_id,spec)
    validate_postcommit_recovery(source,{'source_postcommit_recovery':proof})
    if len(acks)>102 or sum(a['bytes'] for a in acks)>512*2**20:
        raise ValueError('Terminal consumed replica retirement exceeds the registered original sample scope')
    if len({a['object'] for a in acks})!=len(acks):raise ValueError('Duplicate consumed replica ACK')
    for ack in acks:
        match=re.fullmatch(r'rollouts/v000[01]-s([0-9]{6})\.pt',ack['object'])
        if not match or int(match[1])>=102:raise ValueError('Only the two completed original policy batches may retire')
        validate_ack(ack,source.name,spec['lease_epoch'],ack['object'],ack['sha256'],ack['bytes'],ack['sequence'])
        if read_json(source/'archive-acks'/f"{ack['sequence']:08d}.json")!=ack:
            raise ValueError('Consumed replica lacks the identical original durable server ACK')
    index=sha256(source/'checkpoint-commits/index.json')
    result=evict_archived_replicas(source,source.name,spec['lease_epoch'],acks)
    if sha256(source/'checkpoint-commits/index.json')!=index:
        raise ValueError('Original two published checkpoints changed during terminal safe point')
    atomic_json(job/'original-consumed-cache-retirement.json',{'at':utcnow(),'source_run_id':source.name,
        'objects':result,'source_checkpoint_index_sha256_unchanged':index,
        'bytes_released':sum(a['bytes'] for a,r in zip(acks,result) if r['evicted']),
        'both_full_checkpoints_prefix_and_entire_frozen_history_preserved':True,
        'all_own_raw_original_bytes_unchanged_in_durable_local_archive':True})
