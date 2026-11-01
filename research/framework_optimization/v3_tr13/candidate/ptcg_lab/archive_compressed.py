"""Lossless durable storage for NEW objects; original-byte proof stays in ACK."""
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from .archive_ack import safe_target,validate_ack,ack_body,sync_directory,publication_path,archive_root
from .archive_prefix import canonical_hash,relative_path
from .io import atomic_json,read_json,sha256


def verify_archive(root,ack,collect=False):
    root=archive_root(root)
    validate_ack(ack,ack['run_id'],ack['lease_epoch'],ack['object'],ack['sha256'],ack['bytes'],ack['sequence'])
    target=safe_target(root,ack['archive_relative']);storage=ack.get('storage')
    expected=f"{ack['run_id']}/objects/{ack['object']}"+('.gz' if storage else '')
    if ack['archive_relative']!=expected:raise ValueError('Archive storage path differs')
    if storage:
        if target.stat().st_size!=storage['bytes'] or sha256(target)!=storage['sha256']:raise ValueError('Encoded archive bytes changed')
        stream=gzip.open(target,'rb')
    else:
        if target.stat().st_size!=ack['bytes']:raise ValueError('Raw archive size differs')
        stream=target.open('rb')
    count=0;digest=hashlib.sha256();parts=[]
    with stream:
        while data:=stream.read(1024*1024):
            count+=len(data)
            if count>ack['bytes'] or count>128*2**20:raise ValueError('Archive reconstruction exceeds declared size')
            digest.update(data)
            if collect:parts.append(data)
    if count!=ack['bytes'] or digest.hexdigest()!=ack['sha256']:raise ValueError('Archive original reconstruction differs')
    return b''.join(parts) if collect else True


def receive_compressed_object(root,run_id,lease_epoch,sequence,relative,digest,size,chunks,minimum_free_bytes=20*2**30):
    if type(lease_epoch) is not int or lease_epoch<1 or type(sequence) is not int or sequence<0 or type(size) is not int or not 0<=size<=128*2**20:
        raise ValueError('Compressed archive owner/size envelope differs')
    relative=relative_path(relative);run_id=relative_path(run_id)
    if '/' in run_id:raise ValueError('Run identity escaped archive root')
    root=archive_root(root);target=safe_target(root,f'{run_id}/objects/{relative}.gz');receipt=safe_target(root,f'{run_id}/acks/{sequence:08d}.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    if receipt.exists():
        ack=validate_ack(read_json(receipt),run_id,lease_epoch,relative,digest,size,sequence)
        verify_archive(root,ack);return ack
    if not target.exists():
        if shutil.disk_usage(root).free-size-2**20<minimum_free_bytes:raise OSError('Compressed archive free-space reserve would be violated')
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.incoming-gzip-',delete=False) as dst:
                temporary=Path(dst.name);hasher=hashlib.sha256();received=0
                with gzip.GzipFile(fileobj=dst,mode='wb',filename='',mtime=0,compresslevel=1) as packed:
                    for block in chunks:
                        received+=len(block)
                        if received>size:raise ValueError('Compressed archive input exceeds declared size')
                        hasher.update(block);packed.write(block)
                        if shutil.disk_usage(root).free<minimum_free_bytes:raise OSError('Compressed archive disk reserve reached')
                dst.flush();os.fsync(dst.fileno())
            if received!=size or hasher.hexdigest()!=digest:raise ValueError('Compressed archive transfer raw checksum/size mismatch')
            os.link(publication_path(temporary),publication_path(target));sync_directory(target.parent)
        finally:
            if temporary is not None and temporary.exists():temporary.unlink()
    core={'run_id':run_id,'lease_epoch':lease_epoch,'sequence':sequence,'object':relative,'sha256':digest,'bytes':size,
        'archive_relative':target.relative_to(root).as_posix(),'schema_version':2,
        'storage':{'codec':'gzip','sha256':sha256(target),'bytes':target.stat().st_size}}
    ack={**core,'commit_id':canonical_hash(ack_body(core))}
    # Independently read the published gzip and reconstruct original bytes
    # before writing the durable receipt that can authorize remote eviction.
    verify_archive(root,ack);atomic_json(receipt,ack);sync_directory(receipt.parent)
    return ack


def reuse_compressed_object(root,row,source_root,source_ack):
    """A new lease gets its own ACK while unchanged gzip bytes are hard-linked."""
    if (row['sha256'],row['bytes'])!=(source_ack['sha256'],source_ack['bytes']) or not source_ack.get('storage'):
        raise ValueError('Compressed reuse requires identical full original bytes')
    verify_archive(source_root,source_ack)
    source=safe_target(archive_root(source_root),source_ack['archive_relative'])
    run_id=relative_path(row['run_id']);relative=relative_path(row['object'])
    if '/' in run_id:raise ValueError('Run identity escaped archive root')
    root=archive_root(root);target=safe_target(root,f'{run_id}/objects/{relative}.gz')
    receipt=safe_target(root,f"{run_id}/acks/{row['sequence']:08d}.json")
    if receipt.exists():
        ack=validate_ack(read_json(receipt),run_id,row['lease_epoch'],relative,row['sha256'],row['bytes'],row['sequence'])
        verify_archive(root,ack);return ack
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():os.link(publication_path(source),publication_path(target));sync_directory(target.parent)
    core={'run_id':run_id,'lease_epoch':row['lease_epoch'],'sequence':row['sequence'],'object':relative,
        'sha256':row['sha256'],'bytes':row['bytes'],'archive_relative':target.relative_to(root).as_posix(),
        'schema_version':2,'storage':dict(source_ack['storage'])}
    ack={**core,'commit_id':canonical_hash(ack_body(core))}
    verify_archive(root,ack);atomic_json(receipt,ack);sync_directory(receipt.parent)
    return ack
