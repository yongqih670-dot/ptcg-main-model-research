"""Local fixed-fixture archive transport with real compression and full ACKs.

The production remote RPC does not yet accept this bundle. This implementation
is not a substitute for its route/identity admission.
"""
import gzip
import os
from pathlib import Path
import tarfile
import tempfile
import time
from .durable import safe,identity,full_hash
from ptcg_lab.archive_ack import validate_ack
from ptcg_lab.archive_compressed import receive_compressed_object


def validate(ack,row):
    return validate_ack(ack,row['run_id'],row['lease_epoch'],row['object'],row['sha256'],row['bytes'],row['sequence'])


class LocalBundleTransport:
    def __init__(self,objects,archive_root,*,minimum_free_bytes=20*2**30,fault=lambda stage:None):
        self.objects=objects;self.archive_root=Path(archive_root);self.archive_root.mkdir(parents=True,exist_ok=True)
        self.minimum_free=minimum_free_bytes;self.fault=fault
        self.stats=dict(batches=0,objects=0,source_bytes=0,packed_bytes=0,transfer_seconds=0.)
    def __call__(self,rows,stop):
        started=time.monotonic();self.fault('before_transfer');stop()
        maximum=int(sum(r['bytes'] for r in rows)*1.02)+2**20
        token=self.objects.capacity.reserve(maximum);directory=self.objects.root/'bundles';directory.mkdir(exist_ok=True)
        name=None
        try:
            with tempfile.NamedTemporaryFile(dir=directory,suffix='.tar.gz',delete=False) as raw:
                name=Path(raw.name)
                with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed,mode='w|') as archive:
                        for row in rows:
                            stop();ref=dict(path=row['object'],sha256=row['sha256'],bytes=row['bytes'])
                            self.objects.evidence.verify(ref,fresh=True);path=safe(self.objects.root,row['object']);before=identity(path)
                            info=tarfile.TarInfo(row['object']);info.size=row['bytes'];info.mtime=0;info.mode=0o600
                            with path.open('rb') as source:archive.addfile(info,source)
                            if identity(path)!=before:raise ValueError('Source changed while packing')
                raw.flush();os.fsync(raw.fileno())
            if name.stat().st_size>maximum:raise OSError('Bundle reservation overrun')
            packed=name.stat().st_size;self.fault('bundle_complete');stop();acks=[]
            with tarfile.open(name,'r:gz') as archive:
                members=archive.getmembers()
                if [m.name for m in members]!=[r['object'] for r in rows] or any(not m.isfile() or m.size!=r['bytes'] for m,r in zip(members,rows)):raise ValueError('Bundle object set differs')
                for member,row in zip(members,rows):
                    self.fault('before_receive');stop()
                    with archive.extractfile(member) as source:
                        def blocks():
                            while data:=source.read(2**20):stop();yield data
                        ack=receive_compressed_object(self.archive_root,row['run_id'],row['lease_epoch'],row['sequence'],
                            row['object'],row['sha256'],row['bytes'],blocks(),self.minimum_free)
                    validate(ack,row);acks.append(ack)
            self.stats['batches']+=1;self.stats['objects']+=len(rows);self.stats['source_bytes']+=sum(r['bytes'] for r in rows)
            self.stats['packed_bytes']+=packed;self.stats['transfer_seconds']+=time.monotonic()-started
            self.fault('after_receive_before_ACK');return acks
        finally:
            if name is not None and name.exists():name.unlink()
            self.objects.capacity.finish(token,0)


class LocalObjectTransport:
    """Same production compressed receiver, one original object per transfer."""
    def __init__(self,root,archive_root,minimum_free_bytes=20*2**30):
        self.root=Path(root);self.archive_root=Path(archive_root);self.minimum_free=minimum_free_bytes
        self.stats=dict(objects=0,source_bytes=0,transfer_seconds=0.)
    def __call__(self,rows,stop):
        began=time.monotonic();acks=[]
        for row in rows:
            path=safe(self.root,row['object']);stop()
            with path.open('rb') as source:
                def blocks():
                    while data:=source.read(2**20):stop();yield data
                ack=receive_compressed_object(self.archive_root,row['run_id'],row['lease_epoch'],row['sequence'],
                    row['object'],row['sha256'],row['bytes'],blocks(),self.minimum_free)
            validate(ack,row);acks.append(ack);self.stats['objects']+=1;self.stats['source_bytes']+=row['bytes']
        self.stats['transfer_seconds']+=time.monotonic()-began;return acks
