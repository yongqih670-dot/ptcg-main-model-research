"""Make opt-in changes only in the isolated verified TR13 package copy."""
from pathlib import Path
import ast,difflib,hashlib,json
V3=Path(__file__).resolve().parents[1]
def change(text,old,new):
    if text.count(old)!=1:raise ValueError('Unexpected baseline integration point: '+old[:100])
    return text.replace(old,new)
def save(name,text):
    ast.parse(text);p=V3/'candidate'/name;p.write_text(text,encoding='utf-8')
def main():
    n='ptcg_lab/framework_v2_training.py';t=(V3/'baseline'/n).read_text(encoding='utf-8')
    t=change(t,'retained_bytes=0, rate=4*2**20):','retained_bytes=0, rate=4*2**20, optimizations=None):')
    t=change(t,'        self.root = Path(output).resolve()',"        from .framework_v3_runtime import flags,JsonEvidence,capacity_type\n        self.optimizations=flags(optimizations)\n        self.journal_cache=JsonEvidence(Path(output),read_json) if self.optimizations['journal_cache'] else None\n        self.root = Path(output).resolve()")
    t=change(t,'self.capacity = ProjectCapacity(capacity_root or self.root, hard_bytes,',"capacity_cls=capacity_type(ProjectCapacity) if self.optimizations['capacity_transaction'] else ProjectCapacity\n        self.capacity = capacity_cls(capacity_root or self.root, hard_bytes,")
    t=change(t,'        self.file_evidence = {}',"        self.file_evidence = {}\n        self.write_admission = None")
    t=change(t,'        self.capacity.reconcile()\n        required = pending_digests',"        if not self.optimizations['capacity_transaction'] or self.write_admission is None or time.monotonic()-self.capacity.at>=self.capacity.interval:self.capacity.reconcile()\n        else:self.capacity.check()\n        required = pending_digests")
    t=change(t,'        self.objects.close()',"        if self.write_admission is not None:self.write_admission.close()\n        self.objects.close()")
    t=change(t,'row = compact_record(read_json(path),ref[\'sha256\'])',"row = compact_record(backend.journal_cache.get(path) if backend.journal_cache else read_json(path),ref['sha256'])")
    t=change(t,'        ack=read_json(path)\n        validate_ack',"        ack=backend.journal_cache.get(path) if backend.journal_cache else read_json(path)\n        validate_ack")
    save(n,t)
    n='ptcg_lab/training_control.py';t=(V3/'baseline'/n).read_text(encoding='utf-8')
    t=change(t,'def __init__(self,output,run_id,lease_epoch,deadline,progress):','def __init__(self,output,run_id,lease_epoch,deadline,progress,*,optimizations=None):')
    t=change(t,'        self.output=Path(output).resolve();self.run_id=run_id;self.epoch=lease_epoch',"        from .framework_v3_runtime import flags,JsonEvidence\n        self.optimizations=flags(optimizations)\n        self._json_evidence=JsonEvidence(output,read_json) if self.optimizations['control_cache'] else None\n        self.output=Path(output).resolve();self.run_id=run_id;self.epoch=lease_epoch")
    t=change(t,'        row=read_json(path)\n        if row.get(\'run_id\')',"        row=self._json_evidence.get(path) if self._json_evidence else read_json(path)\n        if row.get('run_id')")
    t=change(t,"ticket=read_json(self.output/'ticket.json');validate(ticket['shared_capacity'])","ticket=self._json_evidence.get(self.output/'ticket.json') if self._json_evidence else read_json(self.output/'ticket.json');validate(ticket['shared_capacity'])")
    save(n,t)
    n='ptcg_lab/archive_outbox.py';t=(V3/'baseline'/n).read_text(encoding='utf-8')
    t=change(t,'def __init__(self,root,run_id,epoch):','def __init__(self,root,run_id,epoch,*,optimizations=None):')
    t=change(t,'        self._ticket_cache=None',"        from .framework_v3_runtime import flags,JsonEvidence,DirectoryEvidence\n        self.optimizations=flags(optimizations)\n        self._json_evidence=JsonEvidence(self.root,read_json) if self.optimizations['outbox_cache'] else None\n        self._directory_evidence=DirectoryEvidence(self.root) if self.optimizations['outbox_cache'] else None\n        self._ticket_cache=None")
    t=change(t,"        paths=sorted((self.root/'archive-outbox').glob('[0-9]'*8+'.json'))", "        paths=self._directory_evidence.paths(self.root/'archive-outbox','[0-9]'*8+'.json') if self._directory_evidence else sorted((self.root/'archive-outbox').glob('[0-9]'*8+'.json'))")
    t=change(t,"        scope=ticket.get('TR13_runtime_scope',{})", "        if ticket.get('engineering_scope')=='framework_efficiency_v3':\n            from .framework_v3_runtime import from_config\n            from .fo_v2.durable import identity as strong_identity\n            bound=(strong_identity(ticket_path),strong_identity(self.root/'framework-v3-manifest.json'))\n            if getattr(self,'_v3_admission_evidence',None)!=bound:\n                from_config({'engineering_only':True,'new_scientific_training':False,'framework_v3_optimizations':self.optimizations,'framework_v3_manifest_sha256':ticket.get('framework_v3_manifest_sha256')},self.root)\n                self._v3_admission_evidence=bound\n            return (ticket['run_id'],ticket['lease_epoch'])==(self.run_id,self.epoch)\n        scope=ticket.get('TR13_runtime_scope',{})")
    t=change(t,'            row=read_json(path)\n            if row[\'run_id\']',"            row=self._json_evidence.get(path) if self._json_evidence else read_json(path)\n            if row['run_id']")
    t=change(t,'return validate_ack(read_json(path),self.run_id',"return validate_ack(self._json_evidence.get(path) if self._json_evidence else read_json(path),self.run_id")
    # The existing queue_recovery already scans once per transaction. Retain
    # it; only add the same one-scan semantics to the two-object series commit.
    t=change(t,"        trajectory=self.queue(summary['trajectory_path'],summary['trajectory_sha256'])\n        record=self.queue(f\"series/{summary['series_id']:06d}.json\")", "        if self.optimizations['outbox_cache']:\n            rows=self.records();trajectory=self.queue(summary['trajectory_path'],summary['trajectory_sha256'],_records=rows)\n            record=self.queue(f\"series/{summary['series_id']:06d}.json\",_records=rows)\n        else:\n            trajectory=self.queue(summary['trajectory_path'],summary['trajectory_sha256'])\n            record=self.queue(f\"series/{summary['series_id']:06d}.json\")")
    save(n,t)
    diffs=[]
    for p in sorted((V3/'candidate/ptcg_lab').rglob('*.py')):
        name=p.relative_to(V3/'candidate').as_posix();original=V3/'baseline'/name
        before=original.read_text(encoding='utf-8') if original.exists() else '';after=p.read_text(encoding='utf-8')
        if before!=after:diffs.append(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name if original.exists() else '/dev/null',tofile='b/'+name)))
    (V3/'patches/candidate.diff').write_text(''.join(diffs),encoding='utf-8')
    print('Isolated opt-in candidate built; production untouched')
if __name__=='__main__':main()
