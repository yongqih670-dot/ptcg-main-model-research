"""Finish a fully verified, partially committed public-cache transaction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from .io import read_json, atomic_json, sha256, utcnow
from .local_cache_retirement import CACHE, WHEEL, windows_security, owned, identity, LocalReadPacer, digest_stream


def remaining_members(manifest, receipts):
    rows = {r['member']: r for r in manifest['files']}
    if len(rows) != len(manifest['files']):
        raise ValueError('Duplicate original proof member')
    seen = set()
    for r in receipts:
        if r['member'] in seen or r['member'] not in rows:
            raise ValueError('Unknown or duplicated prior unlink')
        original = rows[r['member']]
        if any(r[k] != original[k] for k in ['bytes', 'sha256']):
            raise ValueError('Prior unlink receipt differs from complete proof')
        seen.add(r['member'])
    return [r for r in manifest['files'] if r['member'] not in seen]


def run(config, output):
    root = Path(config['project_path']).absolute(); output = Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID') != output.name or output.parent != root/'runs':
        raise RuntimeError('Controlled local runner required')
    deadline = float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if deadline-time.time() > 600.5 or config['maximum_attempts'] != 1:
        raise ValueError('New finalization repair is bounded to one600s job')
    prior = root/'runs'/config['verified_transaction_run_id']
    for name, digest in config['prior_evidence_sha256'].items():
        if sha256(prior/name) != digest: raise ValueError('Frozen partial transaction changed')
    manifest = read_json(prior/'verified-retirement-manifest.json')
    if (not manifest['all_members_verified_before_any_unlink'] or manifest['cache_relative'] != CACHE
            or manifest['retained_wheel'] != WHEEL or manifest['retained_wheel_sha256'] != config['wheel_sha256']):
        raise ValueError('Original whole-wheel/member verification missing')
    receipts = [json.loads(line) for line in (prior/'retired-files.jsonl').read_text(encoding='utf-8').splitlines()]
    remaining = remaining_members(manifest, receipts)
    sid, owner = windows_security(); allowed = config['runs_container_owner_sid_sha256']
    if hashlib.sha256(sid.encode()).hexdigest() != config['owner_sid_sha256']:
        raise ValueError('Actual project OS owner differs')
    cache = owned(root, root/CACHE, sid, owner, allowed)
    wheel = owned(root, root/WHEEL, sid, owner, allowed); wheel_before = identity(wheel)
    pacer = LocalReadPacer(16*2**20, deadline-20)
    with wheel.open('rb') as stream: digest, _ = digest_stream(stream, pacer)
    if digest != config['wheel_sha256'] or identity(wheel) != wheel_before:
        raise ValueError('Retained complete recovery wheel differs')
    actual = {p.relative_to(cache).as_posix() for p in cache.rglob('*') if p.is_file()}
    if actual != {r['member'] for r in remaining}: raise ValueError('Remaining inventory differs from committed proof')
    # Validate every remaining owner/path/inode before committing another unlink.
    for r in remaining:
        path = owned(root, cache/r['member'], sid, owner, allowed)
        if identity(path) != r['file_identity']: raise ValueError('Remaining verified identity changed')
    atomic_json(output/'retirement-intent.json', {'at': utcnow(), 'prior_run': prior.name,
        'prior_proof_sha256': sha256(prior/'verified-retirement-manifest.json'),
        'files': len(remaining), 'bytes': sum(r['bytes'] for r in remaining), 'no_directory_deletion': True})
    count = 0; released = 0
    with (output/'retired-files.jsonl').open('x', encoding='utf-8') as log:
        for r in remaining:
            path = owned(root, cache/r['member'], sid, owner, allowed)
            if identity(path) != r['file_identity'] or identity(wheel) != wheel_before:
                raise ValueError('Cache or recovery file changed before unlink')
            with path.open('rb') as stream: digest, size = digest_stream(stream, pacer)
            if digest != r['sha256'] or size != r['bytes'] or identity(path) != r['file_identity']:
                raise ValueError('Full remaining member content differs')
            path.unlink(); count += 1; released += size
            log.write(json.dumps({k:r[k] for k in ['member','bytes','sha256']})+'\n')
            log.flush(); os.fsync(log.fileno())
            if count%64 == 0:
                atomic_json(output/'progress.json', {'at':utcnow(),'files':count,'bytes':released,'stage':'finalizing_verified_cache'})
    atomic_json(output/'metrics.json', {'at':utcnow(),'complete':True,'engineering_only':True,
        'retired_files':count,'retired_logical_bytes':released,'prior_partial_files':len(receipts),
        'retained_wheel_sha256':config['wheel_sha256'],'prior_transaction_remains_failed':True,
        'all_scientific_data_and_prior_evidence_retained':True,'promotion_eligible':False})


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(read_json(a.config),a.output)
