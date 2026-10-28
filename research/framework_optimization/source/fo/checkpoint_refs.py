"""Two immutable recovery commits and ACK-gated remote replica eviction.

The training writer must supply the exact still-pending rollout references.
This module never infers that a completed update makes every shard disposable.
"""
import os
from pathlib import Path
import shutil
import tempfile
from ptcg_lab.archive_ack import safe_target,may_evict
from ptcg_lab.archive_prefix import canonical_hash,relative_path
from ptcg_lab.io import atomic_json,read_json,sha256
from ptcg_lab.bounded_io import environment_context
from ptcg_lab.bounded_read import environment_read_context


def sync_directory(path):
    if os.name=='posix':
        descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)


def checked_reference(root,relative,verified=None):
    relative=relative_path(relative);path=safe_target(root,relative)
    if not path.is_file():raise ValueError('Missing checkpoint reference')
    before=path.stat()
    if hasattr(os,'getuid') and before.st_uid!=os.getuid():raise ValueError('Checkpoint reference UID differs')
    identity=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_uid,before.st_mode)
    key=(str(Path(root).resolve()),relative)
    cached=verified.get(key) if verified is not None else None
    if cached is not None and cached[0]==identity:return dict(cached[1])
    digest=sha256(path);after=path.stat()
    if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_uid,after.st_mode)!=identity:
        raise ValueError('Checkpoint reference changed during full hash')
    value={'path':relative,'sha256':digest,'bytes':after.st_size}
    if verified is not None:verified[key]=(identity,dict(value))
    return value


def _validate_commit(root,commit,verified):
    body={k:v for k,v in commit.items() if k!='commit_id'}
    if canonical_hash(body)!=commit.get('commit_id'):raise ValueError('Checkpoint commit identity changed')
    for reference in [commit['checkpoint'],commit['prefix'],*commit['pending']]:
        actual=checked_reference(root,reference['path'],verified)
        if actual!=reference:raise ValueError('Checkpoint referenced bytes changed')


def read_index(root,run_id,lease_epoch,*,_verified=None):
    root=Path(root).resolve();index=read_json(root/'checkpoint-commits/index.json')
    # A fresh operation always rehashes every unique file. Only duplicate
    # references within this single writer transaction reuse a full hash.
    verified={} if _verified is None else _verified
    if index.get('schema_version')!=1 or index.get('run_id')!=run_id or index.get('lease_epoch')!=lease_epoch:
        raise ValueError('Checkpoint owner/epoch differs')
    commits=index['commits']
    if not 1<=len(commits)<=2:raise ValueError('Expected latest and at most one prior recovery point')
    generations=[c['generation'] for c in commits]
    if generations!=sorted(set(generations)):raise ValueError('Non-monotonic checkpoint generations')
    for commit in commits:_validate_commit(root,commit,verified)
    return index


def _copy_immutable(source,destination,root=None,verified=None,immutable_hardlink=False):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if verified is None or root is None:
            same=sha256(source)==sha256(destination)
        else:
            same=checked_reference(root,source.relative_to(root).as_posix(),verified)['sha256']==checked_reference(root,destination.relative_to(root).as_posix(),verified)['sha256']
        if not same:raise ValueError('Checkpoint generation path conflict')
        return
    if immutable_hardlink:
        from .immutable_link import publish_link
        if publish_link(source,destination,root):return
    context=environment_context()
    if context is not None:
        reader=environment_read_context()
        def serialize(stream):
            with reader.open(source) as src:
                while block:=src.read(64*1024):stream.write(block)
        context.atomic_write(destination,serialize,replace=False)
        return
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,prefix='.checkpoint-',delete=False) as dst,Path(source).open('rb') as src:
            temporary=Path(dst.name);shutil.copyfileobj(src,dst,1024*1024);dst.flush();os.fsync(dst.fileno())
        os.link(temporary,destination);sync_directory(destination.parent)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def commit_checkpoint(root,run_id,lease_epoch,generation,checkpoint_path,prefix_path,pending_paths,cursor,*,_verified=None,immutable_hardlink=False):
    root=Path(root).resolve();index_path=root/'checkpoint-commits/index.json'
    verified={} if _verified is None else _verified
    if type(generation) is not int or generation<0:raise ValueError('Invalid checkpoint generation')
    old=read_index(root,run_id,lease_epoch,_verified=verified) if index_path.exists() else None
    if old:
        latest=old['commits'][-1]
        if generation!=latest['generation']+1:raise ValueError('Checkpoint generation must advance exactly once')
        for field in ('version','trained_decisions','next_series_id'):
            if cursor[field]<latest['cursor'][field]:raise ValueError('Checkpoint cursor regressed')
    for value in cursor.values():
        if type(value) is not int or value<0:raise ValueError('Invalid checkpoint cursor')
    checkpoint_source=checked_reference(root,checkpoint_path,verified);prefix_source=checked_reference(root,prefix_path,verified)
    pending=[checked_reference(root,path,verified) for path in pending_paths]
    if len({p['path'] for p in pending})!=len(pending):raise ValueError('Duplicate pending reference')
    cp_relative=f"checkpoint-commits/objects/cp-{checkpoint_source['sha256']}.pt"
    prefix_relative=f"checkpoint-commits/objects/prefix-{prefix_source['sha256']}.json"
    _copy_immutable(root/checkpoint_path,safe_target(root,cp_relative),root,verified,immutable_hardlink)
    _copy_immutable(root/prefix_path,safe_target(root,prefix_relative),root,verified,immutable_hardlink)
    checkpoint_stored=checked_reference(root,cp_relative,verified);prefix_stored=checked_reference(root,prefix_relative,verified)
    if (checkpoint_stored['sha256'],checkpoint_stored['bytes'])!=(checkpoint_source['sha256'],checkpoint_source['bytes']) or (prefix_stored['sha256'],prefix_stored['bytes'])!=(prefix_source['sha256'],prefix_source['bytes']):
        raise ValueError('Copied recovery bytes differ from verified source')
    body={'generation':generation,'cursor':dict(cursor),'checkpoint':checkpoint_stored,
          'prefix':prefix_stored,'pending':pending}
    commit={**body,'commit_id':canonical_hash(body)}
    commits=([*old['commits'],commit] if old else [commit])[-2:]
    index={'schema_version':1,'run_id':run_id,'lease_epoch':lease_epoch,'commits':commits}
    atomic_json(index_path,index);sync_directory(index_path.parent)
    return commit


def protected_hashes(root,run_id,lease_epoch,active_reference_hashes=(),*,_verified=None):
    index=read_index(root,run_id,lease_epoch,_verified=_verified);protected=set(active_reference_hashes)
    for commit in index['commits']:
        protected.update(r['sha256'] for r in [commit['checkpoint'],commit['prefix'],*commit['pending']])
    return protected


def evict_archived_replica(root,run_id,lease_epoch,relative,ack,active_reference_hashes=()):
    root=Path(root).resolve();relative=relative_path(relative);target=safe_target(root,relative)
    protected=protected_hashes(root,run_id,lease_epoch,active_reference_hashes)
    return _evict_verified(root,run_id,lease_epoch,relative,ack,protected)


def evict_archived_replicas(root,run_id,lease_epoch,acks,active_reference_hashes=(),*,_verified=None):
    """One training-writer safe point; rehash recovery references once per batch."""
    root=Path(root).resolve();index_path=root/'checkpoint-commits/index.json'
    initial=index_path.stat();index_identity=(initial.st_dev,initial.st_ino,initial.st_size,initial.st_mtime_ns)
    protected=protected_hashes(root,run_id,lease_epoch,active_reference_hashes,_verified=_verified)
    result=[]
    for ack in acks:
        current=index_path.stat()
        if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)!=index_identity:
            raise RuntimeError('Recovery index changed during archive eviction safe point')
        relative=relative_path(ack['object'])
        result.append({'object':relative,**_evict_verified(root,run_id,lease_epoch,relative,ack,protected)})
    return result


def _evict_verified(root,run_id,lease_epoch,relative,ack,protected):
    target=safe_target(root,relative)
    if not may_evict(ack,run_id,lease_epoch,relative,ack['sha256'],ack['bytes'],protected):
        return {'evicted':False,'reason':'referenced_by_recovery_or_active_policy'}
    if not target.is_file() or target.stat().st_size!=ack['bytes'] or sha256(target)!=ack['sha256']:
        raise ValueError('Remote replica differs from acknowledged archive')
    identity=target.stat()
    if hasattr(os,'getuid') and identity.st_uid!=os.getuid():raise ValueError('Eviction target UID differs')
    receipt=root/'checkpoint-commits/evictions'/f"{ack['commit_id']}.json"
    atomic_json(receipt,{'status':'intent','object':relative,'archive_commit_id':ack['commit_id'],'sha256':ack['sha256']})
    # This is one identity-verified file inside the caller's own job workspace.
    current=safe_target(root,relative).stat()
    if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)!=(identity.st_dev,identity.st_ino,identity.st_size,identity.st_mtime_ns):
        raise ValueError('Eviction target changed after verification')
    target.unlink();sync_directory(target.parent)
    atomic_json(receipt,{'status':'evicted','object':relative,'archive_commit_id':ack['commit_id'],'sha256':ack['sha256']})
    return {'evicted':True,'archive_commit_id':ack['commit_id']}
