"""Training-writer outbox, bounded reads, durable ACKs and safe backpressure."""
import base64
import hashlib
import os
from pathlib import Path
import re
import time
import zlib

from .archive_ack import safe_target,validate_ack,receive_object
from .archive_prefix import relative_path
from .io import atomic_json,read_json,sha256
from .phase_timing import timed


IMMUTABLE=re.compile(r'(CPU-sampling-reference\.pt|model\.pt|rollouts/v[0-9]{4,}-s[0-9]{6,}\.pt|series/[0-9]{6,}\.json|league/learner-[0-9]{4,}\.pt|checkpoint-commits/objects/(cp-[0-9a-f]{64}\.pt|prefix-[0-9a-f]{64}\.json))')


def identity(path):
    value=path.stat()
    if hasattr(os,'getuid') and value.st_uid!=os.getuid():raise ValueError('Archive source UID differs')
    return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_uid]


class ArchiveOutbox:
    """Only the training writer queues and evicts; the controller only ACKs."""
    def __init__(self,root,run_id,epoch):
        self.root=Path(root).resolve();self.run_id=relative_path(run_id);self.epoch=epoch
        if self.root.name!=run_id or type(epoch) is not int or epoch<1:raise ValueError('Outbox run/lease identity differs')
    def records(self):
        paths=sorted((self.root/'archive-outbox').glob('[0-9]'*8+'.json'))
        if len(paths)>4096:raise RuntimeError('Archive request metadata bound reached')
        result=[];objects=set()
        for number,path in enumerate(paths):
            row=read_json(path)
            if row['run_id']!=self.run_id or row['lease_epoch']!=self.epoch or row['sequence']!=number:
                raise ValueError('Outbox identity or committed sequence differs')
            relative=relative_path(row['object'])
            if not IMMUTABLE.fullmatch(relative) or relative in objects:raise ValueError('Outbox scope or duplicate immutable object')
            objects.add(relative);result.append(row)
        return result
    def queue(self,relative,expected_sha256=None):
        relative=relative_path(relative)
        if not IMMUTABLE.fullmatch(relative):raise ValueError('Only complete immutable training artifacts can be queued')
        path=safe_target(self.root,relative);before=identity(path);rows=self.records()
        if not 0<=before[2]<=128*2**20:raise ValueError('Archive source exceeds bounded file size')
        previous=next((row for row in rows if row['object']==relative),None)
        if previous:
            if previous['file_identity']!=before or expected_sha256 and expected_sha256!=previous['sha256']:
                raise ValueError('Already queued immutable artifact changed')
            return previous
        digest=sha256(path)
        if identity(path)!=before or expected_sha256 and digest!=expected_sha256:raise ValueError('Queued artifact bytes changed')
        if not 0<=before[2]<=128*2**20 or len(rows)>=4096:raise ValueError('Archive object or request bound reached')
        row={'run_id':self.run_id,'lease_epoch':self.epoch,'sequence':len(rows),'object':relative,
            'sha256':digest,'bytes':before[2],'file_identity':before}
        atomic_json(self.root/'archive-outbox'/f"{row['sequence']:08d}.json",row)
        return row
    def queue_series(self,summary):
        if summary['statuses']!=['DONE','DONE']:raise ValueError('Incomplete BO3 cannot enter training outbox')
        trajectory=self.queue(summary['trajectory_path'],summary['trajectory_sha256'])
        record=self.queue(f"series/{summary['series_id']:06d}.json")
        return trajectory,record
    def queue_recovery(self,commit):
        return [self.queue(row['path'],row['sha256']) for row in [commit['checkpoint'],commit['prefix'],*commit['pending']]]
    def _ack(self,row):
        path=self.root/'archive-acks'/f"{row['sequence']:08d}.json"
        if not path.exists():return None
        return validate_ack(read_json(path),self.run_id,self.epoch,row['object'],row['sha256'],row['bytes'],row['sequence'])
    def pending(self,limit=32):
        if type(limit) is not int or not 1<=limit<=128:raise ValueError('Archive polling bound differs')
        rows=[row for row in self.records() if self._ack(row) is None]
        return {'pending_objects':len(rows),'pending_bytes':sum(row['bytes'] for row in rows),'objects':rows[:limit]}
    def backpressured(self,maximum_pending_bytes):
        if type(maximum_pending_bytes) is not int:raise ValueError('Unacknowledged working-set bound differs')
        ceiling=512*2**20
        if maximum_pending_bytes>ceiling:
            from .storage_profile import NEW,load
            if os.environ.get('PTCG_PROJECT_ROOT')==NEW and load(NEW) is not None:ceiling=12*2**30
        if type(maximum_pending_bytes) is not int or not 0<maximum_pending_bytes<=ceiling:raise ValueError('Unacknowledged working-set bound differs')
        status=self.pending(1)
        return status['pending_bytes']>=maximum_pending_bytes,status
    def chunk(self,sequence,offset,count):
        # pending()/ACK still validate the complete ordered outbox. A chunk
        # validates its one immutable row instead of reopening hundreds of
        # unrelated metadata files for every MiB of a checkpoint transfer.
        if type(sequence) is not int or not 0<=sequence<4096:raise ValueError('Unknown archive sequence')
        row=read_json(safe_target(self.root,f'archive-outbox/{sequence:08d}.json'))
        relative=relative_path(row['object'])
        if (row['run_id']!=self.run_id or row['lease_epoch']!=self.epoch or row['sequence']!=sequence
                or not IMMUTABLE.fullmatch(relative) or not re.fullmatch('[0-9a-f]{64}',row['sha256'])
                or type(row['bytes']) is not int or not 0<=row['bytes']<=128*2**20):
            raise ValueError('Archive chunk immutable row identity differs')
        path=safe_target(self.root,relative)
        if type(offset) is not int or not 0<=offset<row['bytes'] or type(count) is not int or not 0<count<=1024*1024:raise ValueError('Archive chunk envelope differs')
        if identity(path)!=row['file_identity']:raise ValueError('Immutable archive source changed before read')
        from .bounded_read import environment_read_context
        context=environment_read_context()
        with context.open(path) if context is not None else path.open('rb') as stream:
            stream.seek(offset);data=stream.read(min(count,row['bytes']-offset))
        if identity(path)!=row['file_identity']:raise ValueError('Immutable archive source changed during read')
        packed=zlib.compress(data,1)
        return {'sequence':sequence,'offset':offset,'raw_bytes':len(data),'raw_sha256':hashlib.sha256(data).hexdigest(),
            'codec':'zlib','packed_sha256':hashlib.sha256(packed).hexdigest(),'base64':base64.b64encode(packed).decode()}
    def accept_ack(self,ack):
        rows=self.records();sequence=ack['sequence']
        if type(sequence) is not int or not 0<=sequence<len(rows):raise ValueError('Unknown archive ACK sequence')
        row=rows[sequence];validate_ack(ack,self.run_id,self.epoch,row['object'],row['sha256'],row['bytes'],sequence)
        expected_archive=f"{self.run_id}/objects/{row['object']}"+('.gz' if ack.get('storage') else '')
        if ack['archive_relative']!=expected_archive:raise ValueError('ACK archive destination identity differs')
        existing=self._ack(row)
        if existing:
            if existing!=ack:raise ValueError('Conflicting replayed ACK')
            return {'acknowledged':True,'replayed':True}
        path=safe_target(self.root,row['object'])
        if identity(path)!=row['file_identity']:raise ValueError('ACK source identity differs')
        atomic_json(self.root/'archive-acks'/f'{sequence:08d}.json',ack)
        return {'acknowledged':True,'replayed':False}
    def evict_consumed(self,archive_state,active_hashes=(),*,_verified=None):
        from .checkpoint_refs import evict_archived_replicas
        archived={row['trajectory_path']:row['trajectory_sha256'] for row in archive_state['archived']}
        archived.update({f"series/{row['series_id']:06d}.json":row['summary_sha256'] for row in archive_state['archived']})
        eligible=[]
        for row in self.records():
            ack=self._ack(row)
            if ack is None or not safe_target(self.root,row['object']).exists():continue
            allowed=archived.get(row['object'])==row['sha256'] or row['object'].startswith('checkpoint-commits/objects/')
            if allowed:eligible.append(ack)
        if not eligible:return []
        return evict_archived_replicas(self.root,self.run_id,self.epoch,eligible,active_hashes,_verified=_verified)


def receive_one(stream,archive_root,row,existing_source=None,progress=None,storage_codec='raw'):
    """A lost ACK retries the same durable object, without a second transfer."""
    if storage_codec=='gzip' and isinstance(existing_source,dict) and existing_source['ack'].get('storage'):
        from .archive_compressed import reuse_compressed_object
        ack=reuse_compressed_object(archive_root,row,existing_source['archive_root'],existing_source['ack'])
        stream.request('archive_ack',{'ack':ack});return ack
    def chunks():
        if existing_source is not None:
            if isinstance(existing_source,dict):
                from .archive_compressed import verify_archive
                import gzip
                old=existing_source['ack'];old_root=existing_source['archive_root']
                if (old['sha256'],old['bytes'])!=(row['sha256'],row['bytes']):
                    raise ValueError('Local compressed source original identity differs')
                verify_archive(old_root,old)
                from .archive_ack import archive_root as canonical_archive_root
                path=safe_target(canonical_archive_root(old_root),old['archive_relative'])
                with gzip.open(path,'rb') if old.get('storage') else path.open('rb') as local:
                    while data:=local.read(1024*1024):yield data
                return
            path=Path(existing_source)
            if path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:raise ValueError('Local original archive source differs')
            with path.open('rb') as local:
                while data:=local.read(1024*1024):yield data
            return
        offset=0
        while offset<row['bytes']:
            result=stream.request('archive_chunk',{'object_sequence':row['sequence'],'offset':offset,'count':1024*1024},timeout=30)
            packed=base64.b64decode(result['base64'],validate=True)
            if result['sequence']!=row['sequence'] or result['offset']!=offset or result['codec']!='zlib' or hashlib.sha256(packed).hexdigest()!=result['packed_sha256']:
                raise ValueError('Archive compressed response identity differs')
            decoder=zlib.decompressobj();data=decoder.decompress(packed,1024*1024+1)
            if not 0<len(data)<=min(1024*1024,row['bytes']-offset) or len(data)!=result['raw_bytes'] or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or hashlib.sha256(data).hexdigest()!=result['raw_sha256']:
                raise ValueError('Archive lossless chunk differs')
            offset+=len(data)
            if progress:progress(row,offset)
            yield data
    if storage_codec=='gzip':
        from .archive_compressed import receive_compressed_object
        receiver=receive_compressed_object
    elif storage_codec=='raw':receiver=receive_object
    else:raise ValueError('Unsupported durable archive storage codec')
    ack=receiver(archive_root,row['run_id'],row['lease_epoch'],row['sequence'],row['object'],row['sha256'],row['bytes'],chunks())
    stream.request('archive_ack',{'ack':ack})
    return ack


def drain_once(stream,archive_root,run_id,epoch,existing_sources=None,progress=None,storage_codec='raw'):
    """Transfer at most one immutable object between controller status polls."""
    pending=stream.request('archive_pending',{'limit':1})
    if not pending['objects']:return {'transferred':False,**pending}
    row=pending['objects'][0]
    if row['run_id']!=run_id or row['lease_epoch']!=epoch:raise ValueError('Live archive server run/epoch differs')
    source=(existing_sources or {}).get(row['sha256'])
    ack=receive_one(stream,archive_root,row,source,progress,storage_codec)
    return {'transferred':True,'ack':ack,'pending_before':pending['pending_objects']}


@timed('archive_receive_compress_and_verify')
def drain_available(stream,archive_root,run_id,epoch,existing_sources=None,progress=None,storage_codec='raw'):
    """Batch quick local-cache ACKs; bound each poll to16 objects or5 seconds."""
    started=time.monotonic();count=0
    while count<16:
        result=drain_once(stream,archive_root,run_id,epoch,existing_sources,progress,storage_codec)
        if not result['transferred']:return {'transferred_objects':count,'pending_objects':0}
        count+=1
        if time.monotonic()-started>=5:break
    pending=stream.request('archive_pending',{'limit':1})
    return {'transferred_objects':count,'pending_objects':pending['pending_objects']}
