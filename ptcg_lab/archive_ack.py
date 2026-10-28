"""Durable local receive and identity-bound ACK; remote eviction is separate."""
import os
from pathlib import Path
import shutil
import tempfile
import re
from .archive_prefix import canonical_hash,relative_path
from .io import atomic_json,read_json,sha256


ACK_FIELDS=('run_id','lease_epoch','sequence','object','sha256','bytes','archive_relative')


def publication_path(path):
    """CreateHardLinkW needs an explicit extended path for long Win32 names."""
    if os.name!='nt':return path
    text=os.path.abspath(path)
    if text.startswith('\\\\?\\'):return text
    if text.startswith('\\\\'):return '\\\\?\\UNC\\'+text[2:]
    return '\\\\?\\'+text


def archive_root(root):
    path=Path(root).resolve()
    return Path(publication_path(path)) if os.name=='nt' else path


def sync_directory(path):
    if os.name=='posix':
        descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)


def ack_body(ack):
    core={key:ack[key] for key in ACK_FIELDS}
    if type(core['bytes']) is not int or core['bytes']<0 or type(core['sequence']) is not int or core['sequence']<0:
        raise ValueError('Archive ACK size/sequence type differs')
    relative_path(core['archive_relative'])
    if ack.get('storage') is not None:
        storage=ack['storage']
        if ack.get('schema_version')!=2 or set(storage)!={'codec','sha256','bytes'} or storage['codec']!='gzip' or not re.fullmatch('[0-9a-f]{64}',storage['sha256']) or type(storage['bytes']) is not int or not 20<=storage['bytes']<=129*2**20:
            raise ValueError('Compressed archive ACK storage proof differs')
        core.update(schema_version=2,storage=dict(storage))
    elif ack.get('schema_version',1)!=1:raise ValueError('Archive ACK schema differs')
    return core


def safe_target(root,relative):
    root=Path(root).resolve();relative=relative_path(relative);current=root
    for part in Path(relative).parts:
        current=current/part
        if current.is_symlink():raise ValueError('Linked archive destination')
    if not current.resolve().is_relative_to(root):raise ValueError('Archive destination escaped')
    return current


def validate_ack(ack,run_id,lease_epoch,relative,digest,size,expected_sequence=None):
    core=ack_body(ack)
    expected=(run_id,lease_epoch,relative_path(relative),digest,size)
    observed=(core['run_id'],core['lease_epoch'],core['object'],core['sha256'],core['bytes'])
    if expected!=observed or canonical_hash(core)!=ack.get('commit_id'):raise ValueError('Archive ACK identity mismatch')
    if expected_sequence is not None and core['sequence']!=expected_sequence:raise ValueError('Archive ACK sequence mismatch')
    return ack


def receive_object(root,run_id,lease_epoch,sequence,relative,digest,size,chunks,minimum_free_bytes=20*2**30):
    if type(lease_epoch) is not int or lease_epoch<1 or type(sequence) is not int or sequence<0:raise ValueError('Invalid archive epoch/sequence')
    if type(size) is not int or size<0:raise ValueError('Invalid archive size')
    relative=relative_path(relative);run_id=relative_path(run_id)
    if '/' in run_id:raise ValueError('Run identity must be one path component')
    root=archive_root(root);target=safe_target(root,run_id+'/objects/'+relative)
    receipt=safe_target(root,run_id+f'/acks/{sequence:08d}.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    if receipt.exists():
        ack=validate_ack(read_json(receipt),run_id,lease_epoch,relative,digest,size,sequence)
        if not target.exists() or target.stat().st_size!=size or sha256(target)!=digest:raise ValueError('Acknowledged archive object changed')
        return ack
    if target.exists():
        if target.stat().st_size!=size or sha256(target)!=digest:raise ValueError('Archive path conflict')
    else:
        if shutil.disk_usage(root).free-size<minimum_free_bytes:raise OSError('Local archive free-space reserve would be violated')
        import hashlib
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.incoming-',delete=False) as dst:
                temporary=Path(dst.name);hasher=hashlib.sha256();received=0
                for block in chunks:
                    received+=len(block)
                    if received>size:raise ValueError('Archive object exceeds declared size')
                    dst.write(block);hasher.update(block)
                    if shutil.disk_usage(root).free<minimum_free_bytes:raise OSError('Local archive disk reserve reached')
                dst.flush();os.fsync(dst.fileno())
            if received!=size or hasher.hexdigest()!=digest:raise ValueError('Archive transfer checksum/size mismatch')
            # Exclusive hardlink publication cannot overwrite earlier evidence.
            os.link(publication_path(temporary),publication_path(target));sync_directory(target.parent)
        finally:
            if temporary is not None and temporary.exists():temporary.unlink()
    core={'run_id':run_id,'lease_epoch':lease_epoch,'sequence':sequence,'object':relative,'sha256':digest,
          'bytes':size,'archive_relative':target.relative_to(root).as_posix()}
    ack={**core,'commit_id':canonical_hash(core)};atomic_json(receipt,ack);sync_directory(receipt.parent)
    return ack


def may_evict(ack,run_id,lease_epoch,relative,digest,size,live_reference_hashes):
    validate_ack(ack,run_id,lease_epoch,relative,digest,size)
    return digest not in set(live_reference_hashes)
