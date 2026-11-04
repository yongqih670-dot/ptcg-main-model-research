"""Lossless byte deltas for immutable league history; no tensor conversion."""
import hashlib
import os
from pathlib import Path
import time
import zlib

import numpy as np
from .io import atomic_json,read_json,sha256,utcnow

MAX_FILE_BYTES=128*2**20


def byte_delta(raw,previous):
    size=max(len(raw),len(previous))
    if size>MAX_FILE_BYTES:raise ValueError('History member exceeds bound')
    left=np.zeros(size,dtype=np.uint8);right=np.zeros(size,dtype=np.uint8)
    left[:len(raw)]=np.frombuffer(raw,dtype=np.uint8)
    right[:len(previous)]=np.frombuffer(previous,dtype=np.uint8)
    return np.bitwise_xor(left,right).tobytes()


def decode(packed,previous,record):
    if record['previous_sha256']!=(hashlib.sha256(previous).hexdigest() if previous else None):
        raise ValueError('History delta parent changed')
    size=record['delta_bytes']
    if not 0<=record['raw_bytes']<=size<=MAX_FILE_BYTES:raise ValueError('History delta size invalid')
    if hashlib.sha256(packed).hexdigest()!=record['packed_sha256']:raise ValueError('History packed bytes changed')
    decoder=zlib.decompressobj();delta=decoder.decompress(packed,size+1)
    if len(delta)!=size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('History compressed size or stream invalid')
    raw=byte_delta(delta,previous)[:record['raw_bytes']]
    if hashlib.sha256(raw).hexdigest()!=record['raw_sha256']:raise ValueError('Reconstructed history hash differs')
    return raw


def encode(raw,previous):
    delta=byte_delta(raw,previous);packed=zlib.compress(delta,1)
    record={'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),
        'previous_sha256':hashlib.sha256(previous).hexdigest() if previous else None,
        'delta_bytes':len(delta),'packed_bytes':len(packed),'packed_sha256':hashlib.sha256(packed).hexdigest()}
    if decode(packed,previous,record)!=raw:raise ValueError('History local byte reconstruction differs')
    return packed,record


def run(config,output):
    output=Path(output);root=Path(config['project_path']).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('History pack requires controlled runner')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])-15
    origin=(root/'runs'/config['origin_run_id']).resolve()
    if origin.parent!=root/'runs':raise ValueError('History origin escaped project')
    manifest=origin/'league.json'
    if sha256(manifest)!=config['league_manifest_sha256']:raise ValueError('Frozen history manifest changed')
    league=read_json(manifest);entries=league['snapshots']
    if len(entries)!=config['expected_history_members']:raise ValueError('Complete history member count differs')
    destination=output/'history-delta';destination.mkdir()
    previous=b'';records=[];raw_total=0;packed_total=0;started=time.monotonic()
    for number,entry in enumerate(entries):
        if time.time()>deadline:raise TimeoutError('Finite history delta pack deadline')
        relative=Path(entry['path'].replace('\\','/'));path=(origin/relative).resolve()
        if not path.is_relative_to(origin/'league') or path.is_symlink():raise ValueError('History member escaped frozen league')
        before=path.stat()
        if before.st_size>MAX_FILE_BYTES:raise ValueError('History member exceeds bounded input')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Original immutable league member changed')
        packed,record=encode(raw,previous)
        name=f'history-delta/{number:04d}.zlib';target=output/name
        with target.open('xb') as stream:stream.write(packed);stream.flush();os.fsync(stream.fileno())
        if sha256(target)!=record['packed_sha256']:raise ValueError('Durable compressed member changed')
        # Re-read the published compressed bytes and verify the external original hash.
        reconstructed=decode(target.read_bytes(),previous,record)
        if hashlib.sha256(reconstructed).hexdigest()!=entry['sha256']:raise ValueError('Published delta cannot reconstruct original member')
        after=path.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('Original league member changed during read')
        records.append({**record,'packed_path':name,'original':entry});raw_total+=len(raw);packed_total+=len(packed)
        atomic_json(output/'progress.json',{'at':utcnow(),'members_verified':len(records),'members_total':len(entries),
            'raw_bytes':raw_total,'packed_bytes':packed_total,'original_files_untouched':True})
        previous=raw
    capsule={'schema_version':1,'codec':'xor-previous-file-zlib-level1','origin_run_id':origin.name,
        'league_manifest_sha256':config['league_manifest_sha256'],'members':records,'raw_bytes':raw_total,'packed_bytes':packed_total}
    atomic_json(output/'history-delta-manifest.json',capsule)
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'promotion_eligible':False,
        'original_files_untouched':True,'all_original_members_reconstructed_exactly':True,'members':len(records),
        'raw_bytes':raw_total,'packed_bytes':packed_total,'ratio':packed_total/raw_total,
        'manifest_sha256':sha256(output/'history-delta-manifest.json'),'seconds':time.monotonic()-started,
        'linux_reconstruction_verified':False,'not_a_reduction_in_history_or_sampling_support':True})
