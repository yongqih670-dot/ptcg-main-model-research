"""Explicit checkpoint API for a future driver; never monkey-patches imports.

This bridge stages immutable trajectory/prefix/league references and can
materialize their canonical names on restore for the existing journal reader.
Production outbox/RPC admission remains a separate required integration gate.
"""
import os
from pathlib import Path
from .checkpoint import Checkpoints,split_legacy
from .durable import safe,full_hash,sync_dir,identity


class DriverCheckpoints:
    def __init__(self,objects,run_id,epoch):
        self.objects=objects;self.checkpoints=Checkpoints(objects,run_id,epoch);self.reference_map={}
    def stage(self,relative,source,digest,size):
        """Called once per changed complete immutable source; writer owns it."""
        safe(self.objects.root,relative)
        old=self.reference_map.get(relative)
        if old:
            if (old['sha256'],old['bytes'])!=(digest,size):raise ValueError('Canonical immutable source changed')
            self.objects.evidence.verify(old);return old
        source=Path(source)
        before=identity(source)
        if not source.resolve().is_relative_to(self.objects.capacity.root):raise ValueError('Staged input outside admitted project')
        if source.is_symlink() or hasattr(os,'getuid') and source.stat().st_uid!=os.getuid():raise ValueError('Unowned staged input')
        def copy(stream):
            with source.open('rb') as f:
                while data:=f.read(2**20):stream.write(data)
        ref=self.objects.put(copy,max(1,size),'data')
        if (ref['sha256'],ref['bytes'])!=(digest,size) or identity(source)!=before:raise ValueError('Staged source differs')
        self.reference_map[relative]=ref;return ref
    def save(self,value,required_paths):
        if set(required_paths)-set(self.reference_map):raise ValueError('Complete recovery references not staged')
        expected=set()
        if value.get('archive_state'):
            from ptcg_lab.training_checkpoint import pending_references
            expected.update(pending_references(value['archive_state']))
            if value['archive_origin_prefix_sha256'] not in {self.reference_map[n]['sha256'] for n in required_paths}:raise ValueError('Origin prefix not staged')
        if isinstance(value.get('league'),dict):
            for entry in value['league'].get('snapshots',[]):
                expected.add(entry['path'])
                if entry['path'] in self.reference_map and self.reference_map[entry['path']]['sha256']!=entry['sha256']:raise ValueError('League snapshot identity differs')
        if not expected<=set(required_paths):raise ValueError('Complete pending/league reference set omitted')
        model,tail=split_legacy(value);version=value['version']
        if not self.checkpoints.model or version!=self.checkpoints.model['version']:self.checkpoints.publish_model(version,model)
        mapping={name:self.reference_map[name] for name in required_paths}
        tail['v2_reference_map']=mapping
        return self.checkpoints.commit(tail,{k:value[k] for k in ('version','trained_decisions','next_series_id')},list({ref['path']:ref for ref in mapping.values()}.values()))
    def restore(self,which=-1,*,materialize=True):
        value,point=self.checkpoints.restore(which);mapping=value.pop('v2_reference_map')
        if materialize:
            for relative,ref in mapping.items():
                dest=safe(self.objects.root,relative);source=safe(self.objects.root,ref['path'])
                self.objects.evidence.verify(ref,fresh=True)
                if dest.exists():
                    if full_hash(dest)!=(ref['sha256'],ref['bytes']):raise ValueError('Existing canonical restore path conflicts')
                else:
                    token=self.objects.capacity.reserve(ref['bytes']);created=False
                    try:
                        self.objects.stop();dest.parent.mkdir(parents=True,exist_ok=True);os.link(source,dest);created=True;sync_dir(dest.parent)
                    finally:self.objects.capacity.finish(token,ref['bytes'] if created else 0)
        self.reference_map=mapping;return value,point
