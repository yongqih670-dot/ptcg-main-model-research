"""Full recovery set with independent model and sampling-state objects."""
import os
from pathlib import Path
import time
from .durable import atomic_json,read,canonical,safe,full_hash,sync_dir
import hashlib


class Checkpoints:
    def __init__(self,objects,run_id,epoch,*,fault=lambda stage:None):
        self.objects=objects;self.root=objects.root;self.run_id=run_id;self.epoch=epoch;self.fault=fault
        self.path=self.root/'recovery-v2.json';self.model=None
        self.points=[];self.last_commit=None
        if self.path.exists():
            index=read(self.path);self._validate_index(index,fresh=True);self.points=index['points']
            self.model=self.points[-1]['model'];self.last_commit=time.monotonic()
    def _refs(self,point):return [point['model']['object'],point['tail'],*point['references']]+([point['manifest']] if 'manifest' in point else [])
    def _validate_index(self,index,*,fresh=False):
        if index.get('schema') not in (2,3) or (index.get('run_id'),index.get('epoch'))!=(self.run_id,self.epoch):raise ValueError('Recovery owner/schema differs')
        points=index['points']
        if not 1<=len(points)<=2:raise ValueError('Missing recovery points')
        previous=None;seen={}
        for p in points:
            body={k:v for k,v in p.items() if k not in ('commit_id','manifest')}
            if hashlib.sha256(canonical(body)).hexdigest()!=p['commit_id']:raise ValueError('Recovery manifest changed')
            if p['model']['version']!=p['cursor']['version']:raise ValueError('Model/cursor version differs')
            if previous:
                if p['generation']!=previous['generation']+1 or any(p['cursor'][k]<previous['cursor'][k] for k in previous['cursor']):raise ValueError('Recovery cursor/generation regressed')
            for ref in self._refs(p):
                if ref['path'] in seen and seen[ref['path']]!=ref:raise ValueError('Conflicting immutable recovery reference')
                self.objects.evidence.verify(ref,fresh=fresh and ref['path'] not in seen);seen[ref['path']]=ref
            if index['schema']==3:
                if 'manifest' not in p:raise ValueError('Immutable recovery descriptor missing')
                expected=dict(schema=3,run_id=self.run_id,epoch=self.epoch,point={k:v for k,v in p.items() if k!='manifest'})
                if read(safe(self.root,p['manifest']['path']))!=expected:raise ValueError('Immutable descriptor differs from recovery index')
            previous=p
    def publish_model(self,version,payload,*,maximum=128*2**20):
        """Only call at a committed PPO update, never on live mutable tensors."""
        if type(version) is not int or version<0:raise ValueError('Invalid model version')
        if self.model and version<=self.model['version']:raise ValueError('Model version already sealed')
        if set(payload)-{'model','optimizer','precision'} or not {'model','optimizer'}<=set(payload):raise ValueError('Model/optimizer payload differs')
        import torch
        ref=self.objects.put(lambda stream:torch.save(payload,stream),maximum,'model')
        self.model=dict(version=version,object=ref);return self.model
    def commit(self,tail,cursor,references=(),*,maximum=32*2**20):
        if not self.model or cursor['version']!=self.model['version']:raise ValueError('No sealed committed model for this cursor')
        if 'model' in tail or 'optimizer' in tail:raise ValueError('Large model belongs in sealed object')
        if not {'rng','version','trained_decisions','next_series_id'}<=set(tail):raise ValueError('Incomplete sampling state')
        if not isinstance(tail['rng'],dict) or set(tail['rng'])!={'python','numpy','torch','cuda'}:raise ValueError('All RNG streams required')
        if any(tail[k]!=cursor[k] for k in cursor):raise ValueError('Tail/cursor mismatch')
        if self.points and any(cursor[k]<self.points[-1]['cursor'][k] for k in cursor):raise ValueError('Cursor regressed')
        if self.points and cursor['version']==self.points[-1]['cursor']['version'] and cursor['trained_decisions']!=self.points[-1]['cursor']['trained_decisions']:raise ValueError('Count advanced without model update')
        refs=list(references)
        if len({x['path'] for x in refs})!=len(refs):raise ValueError('Duplicate recovery object')
        for ref in refs:self.objects.evidence.verify(ref)
        import torch
        ref=self.objects.put(lambda stream:torch.save(tail,stream),maximum,'tail')
        body=dict(generation=self.points[-1]['generation']+1 if self.points else 0,cursor=dict(cursor),
                  model=self.model,tail=ref,references=refs,committed_unix=time.time())
        point={**body,'commit_id':hashlib.sha256(canonical(body)).hexdigest()}
        points=([*self.points,point])[-2:];sealed=[]
        for p in points:
            if 'manifest' not in p:
                manifest=self.objects.json(dict(schema=3,run_id=self.run_id,epoch=self.epoch,point=p),kind='manifest')
                p={**p,'manifest':manifest}
            sealed.append(p)
        points=sealed;point=points[-1]
        index=dict(schema=3,run_id=self.run_id,epoch=self.epoch,points=points)
        self._validate_index(index);self.fault('before_index');self.objects.stop()
        atomic_json(self.path,index);self.fault('after_index')
        self.points=points;now=time.monotonic();gap=None if self.last_commit is None else now-self.last_commit
        self.last_commit=now
        return dict(point=point,reliable_commit_gap_seconds=gap,two_points=len(points)==2)
    def restore(self,which=-1):
        index=read(self.path);self._validate_index(index,fresh=True);point=index['points'][which]
        import torch
        def load(ref):
            p=safe(self.root,ref['path'])
            if self.objects.reader:
                with self.objects.reader.open(p) as stream:return torch.load(stream,map_location='cpu',weights_only=False)
            return torch.load(p,map_location='cpu',weights_only=False)
        model=load(point['model']['object']);tail=load(point['tail'])
        value={**model,**tail}
        if any(value[k]!=point['cursor'][k] for k in point['cursor']):raise ValueError('Restored cursor differs')
        return value,point
    def protected(self):
        index=read(self.path);self._validate_index(index,fresh=True)
        return {r['sha256'] for p in index['points'] for r in self._refs(p)}
    def archive_references(self):
        index=read(self.path);self._validate_index(index)
        return list({r['path']:r for p in index['points'] for r in self._refs(p)}.values())
    def rebuild_from_manifests(self,references):
        if self.path.exists():raise ValueError('Cannot overwrite a live recovery index')
        points=[]
        for ref in references:
            self.objects.evidence.verify(ref,fresh=True);record=read(safe(self.root,ref['path']))
            if (record.get('schema'),record.get('run_id'),record.get('epoch'))!=(3,self.run_id,self.epoch):raise ValueError('Archived descriptor owner differs')
            points.append({**record['point'],'manifest':ref})
        points=sorted(points,key=lambda p:p['generation'])[-2:]
        index=dict(schema=3,run_id=self.run_id,epoch=self.epoch,points=points);self._validate_index(index,fresh=True)
        atomic_json(self.path,index);self.points=points;self.model=points[-1]['model'];return index
    def delete_acknowledged(self,ref,ack_validator,ack,*,active_hashes):
        """ACK validator must verify original run/lease/sequence/full-content ACK."""
        self.objects.evidence.verify(ref,fresh=True);ack_validator(ack,ref)
        if ref['sha256'] in self.protected()|set(active_hashes):return False
        path=safe(self.root,ref['path']);before=path.stat()
        # The only permitted mutation is this owned single-writer deletion.
        if path.stat()!=before:raise ValueError('Deletion identity changed')
        path.unlink();sync_dir(path.parent);self.objects.capacity.released(ref['bytes']);return True


def split_legacy(value):
    """Values and all RNG/league/archive-tail fields survive without schema loss."""
    model={k:value[k] for k in ('model','optimizer')}
    if 'precision' in value:model['precision']=value['precision']
    return model,{k:v for k,v in value.items() if k not in model}
