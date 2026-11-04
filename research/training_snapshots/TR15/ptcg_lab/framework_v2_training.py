"""Prospective native PPO adapter for the frozen v2 immutable object format.

Partial updates seal a new model revision without advancing scientific version
or decision credit. Model, optimizer, RNG and permutation commit atomically.
"""
import os
from pathlib import Path
import time
import hashlib
import numpy as np
import torch
from .io import atomic_json, read_json, sha256, utcnow
from .fo_v2.durable import Capacity, full_hash, identity, safe, sync_dir
from .fo_v2.transactional import TransactionalObjects
from .fo_v2.bridge import DriverCheckpoints
from .fo_v2.checkpoint import split_legacy
from .fo_v2.io_budget import ReadEnvelope
from .framework_v2_checkpoint_cadence import Cadence, validate_profile
from .training_checkpoint import pending_references


class ProjectCapacity(Capacity):
    def _scan(self):
        # Count owned symlink directory entries without following venv links.
        from .server_workspace import logical_bytes
        return logical_bytes(self.root)


def semantic_digest(value):
    """Stable recursive values hash, independent of Torch serialization storage IDs."""
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            y=x.detach().cpu().contiguous();h.update(b'tensor'+str(y.dtype).encode()+str(tuple(y.shape)).encode())
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(b'ndarray'+x.dtype.str.encode()+str(x.shape).encode()+x.tobytes())
        elif isinstance(x,dict):
            h.update(b'dict'+str(len(x)).encode())
            for k in sorted(x,key=lambda v:(type(v).__name__,repr(v))):add(k);add(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode()+str(len(x)).encode())
            for v in x:add(v)
        else:h.update((type(x).__name__+':'+repr(x)+';').encode())
    add(value);return h.hexdigest()


def pending_digests(state):
    result = {}
    for row in state['live_expected']:
        result[row['trajectory_path']] = row['trajectory_sha256']
        result[f"series/{row['series_id']:06d}.json"] = row['summary_sha256']
    if set(result) != set(pending_references(state)):
        raise ValueError('Pending reference map differs')
    return result


def ensure_two_points(backend,save):
    """Both legacy-pending and extended-tail zero-update inputs need two points."""
    while len(backend.bridge.checkpoints.points)<2:save()


class NativeCheckpoints:
    def __init__(self, output, epoch, profile, *, capacity_root=None, hard_bytes=80*2**30,
                 retained_bytes=0, rate=4*2**20, optimizations=None):
        from .framework_v3_runtime import flags,JsonEvidence,capacity_type
        self.optimizations=flags(optimizations)
        self.journal_cache=JsonEvidence(Path(output),read_json) if self.optimizations['journal_cache'] else None
        self.root = Path(output).resolve()
        self.epoch = epoch
        profile = validate_profile(profile)
        capacity_cls=capacity_type(ProjectCapacity) if self.optimizations['capacity_transaction'] else ProjectCapacity
        self.capacity = capacity_cls(capacity_root or self.root, hard_bytes,
                                 retained_bytes=retained_bytes, external_reserve=256*2**20)
        self.objects = TransactionalObjects(self.root, self.capacity, rate=rate,
                                            reader=ReadEnvelope(rate))
        self.bridge = DriverCheckpoints(self.objects, self.root.name, epoch)
        self.cadence = Cadence(maximum_gap=profile['maximum_reliable_commit_gap_seconds'],
                               commit_reserve=profile['commit_reserve_seconds'])
        self.revision = 0
        self.sealed_state = None
        self.file_evidence = {}
        self.write_admission = None

    def checked(self, relative, digest=None):
        path = self.root/relative
        before = identity(path)
        old = self.file_evidence.get(relative)
        if old is None or old[0] != before:
            found, size = full_hash(path)
            if identity(path) != before:
                raise ValueError('Canonical immutable source changed during full hash')
            old = (before, {'path':relative, 'sha256':found, 'bytes':size})
            self.file_evidence[relative] = old
        if digest is not None and old[1]['sha256'] != digest:
            raise ValueError('Canonical immutable source digest differs')
        return dict(old[1])

    def restore(self, which=-1):
        # Constructor freshly verifies the complete graph. Within this owned
        # reader transaction, unchanged inode/mtime/ctime/UID/mode evidence is
        # reused; changed identities still force a complete independent hash.
        checkpoints=self.bridge.checkpoints
        index=read_json(checkpoints.path);checkpoints._validate_index(index)
        point=index['points'][which]
        def load(ref):
            with self.objects.reader.open(self.root/ref['path']) as stream:
                return torch.load(stream,map_location='cpu',weights_only=False)
        value={**load(point['model']['object']),**load(point['tail'])}
        if any(value[k]!=point['cursor'][k] for k in point['cursor']):raise ValueError('Restored cursor differs')
        if point['model'].get('state_revision')!=value.get('model_state_revision'):
            raise ValueError('Partial model revision differs from its committed tail')
        mapping=value.pop('v2_reference_map')
        for relative,ref in mapping.items():
            dest=safe(self.root,relative)
            if dest.exists():self.checked(relative,ref['sha256'])
            else:
                token=self.capacity.reserve(ref['bytes']);created=False
                try:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    os.link(safe(self.root,ref['path']),dest);created=True;sync_dir(dest.parent)
                finally:self.capacity.finish(token,ref['bytes'] if created else 0)
        self.bridge.reference_map=mapping
        self.revision = value.get('model_state_revision', 0)
        self.sealed_state = (value['version'], self.revision)
        atomic_json(self.root/'league.json', value['league'])
        return value, point

    def save(self, value, archive_spec, league, *, model_revision):
        started = time.monotonic()
        if not self.optimizations['capacity_transaction'] or self.write_admission is None or time.monotonic()-self.capacity.at>=self.capacity.interval:self.capacity.reconcile()
        else:self.capacity.check()
        required = pending_digests(value['archive_state'])
        required[archive_spec['prefix_path']] = archive_spec['prefix_sha256']
        for row in league['snapshots']:
            required[row['path']] = row['sha256']
        for name, digest in required.items():
            ref = self.checked(name, digest)
            self.bridge.stage(name, self.root/name, digest, ref['bytes'])
        value = dict(value, league=league, model_state_revision=model_revision)
        value['state_semantic_sha256']=semantic_digest(value)
        model, tail = split_legacy(value)
        key = (value['version'], model_revision)
        if self.sealed_state != key:
            ref = self.objects.put(lambda stream:torch.save(model, stream), 128*2**20, 'model')
            self.bridge.checkpoints.model = {'version':value['version'], 'state_revision':model_revision, 'object':ref}
            self.sealed_state = key
        mapping = {name:self.bridge.reference_map[name] for name in required}
        tail['v2_reference_map'] = mapping
        cursor = {k:value[k] for k in ('version','trained_decisions','next_series_id')}
        result = self.bridge.checkpoints.commit(tail, cursor, list({r['path']:r for r in mapping.values()}.values()))
        assessment = self.cadence.committed()
        self.cadence.observed_commit_seconds = time.monotonic()-started
        event = {k:v for k,v in assessment.items()}
        event.update(at=utcnow(),generation=result['point']['generation'],model_revision=model_revision,
                     commit_seconds=self.cadence.observed_commit_seconds)
        events = self.root/'framework-v2-cadence.jsonl'
        with events.open('a', encoding='utf-8') as stream:
            import json
            stream.write(json.dumps(event)+'\n');stream.flush();os.fsync(stream.fileno())
        if assessment.get('within_maximum_gap') is False:
            atomic_json(self.root/'framework-v2-cadence-failure.json', event)
            raise RuntimeError('Reliable checkpoint completion gap exceeded the authorized maximum')
        point = result['point']
        return {'generation':point['generation'], 'cursor':cursor,
                'checkpoint':point['model']['object'],
                'prefix':self.checked(archive_spec['prefix_path'], archive_spec['prefix_sha256']),
                'pending':[self.checked(name, digest) for name,digest in pending_digests(value['archive_state']).items()],
                'framework_v2':point,
                'archive_references':self.bridge.checkpoints.archive_references()}

    def close(self):
        if self.write_admission is not None:self.write_admission.close()
        self.objects.close()


def advance_native_state(backend, previous, origin_prefix, archive_spec, cursor, run_id, epoch):
    """Same compact history semantics with identity-bound immutable evidence."""
    from .archive_prefix import compact_record
    from .archive_state import advance_state
    from .archive_ack import validate_ack
    records = []
    for path in sorted((backend.root/'series').glob('[0-9]'*6+'.json')):
        ref = backend.checked(path.relative_to(backend.root).as_posix())
        row = compact_record(backend.journal_cache.get(path) if backend.journal_cache else read_json(path),ref['sha256'])
        backend.checked(row['trajectory_path'],row['trajectory_sha256'])
        records.append(row)
    acks = {}
    for path in sorted((backend.root/'archive-acks').glob('*.json')):
        ack=backend.journal_cache.get(path) if backend.journal_cache else read_json(path)
        validate_ack(ack,run_id,epoch,ack['object'],ack['sha256'],ack['bytes'])
        if ack['object'] in acks and acks[ack['object']] != ack:
            raise ValueError('Conflicting archive ACK')
        acks[ack['object']] = ack
    return advance_state(previous,origin_prefix,archive_spec['prefix_sha256'],cursor,records,acks,run_id,epoch)
