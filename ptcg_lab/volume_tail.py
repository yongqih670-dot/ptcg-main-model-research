"""Exact complete post-checkpoint journal handoff; never invent a new state."""
from pathlib import Path
from .io import read_json,sha256

def validate_spec(spec,source_id,point):
    if spec is None:return
    if spec['source_run_id']!=source_id or spec['checkpoint_sha256']!=point['checkpoint']['sha256']:
        raise ValueError('Tail is bound to another full checkpoint')
    refs=spec['references'];ids=spec['series_ids'];cursor=point['cursor']
    if not ids or ids!=sorted(set(ids)) or len(ids)>100 or min(ids)<cursor['next_series_id']:
        raise ValueError('Invalid complete post-checkpoint BO3 identities')
    names={r['path'] for r in refs};expected={p for sid in ids for p in (f"rollouts/v{cursor['version']:04d}-s{sid:06d}.pt",f'series/{sid:06d}.json')}
    if names!=expected or len(refs)!=len(expected) or spec['pending_decisions']<=0:
        raise ValueError('Complete tail must bind every trajectory and summary')
    if spec['journal_next_series_id']!=max(ids)+1 or not spec['journal_next_series_id']<=spec['next_series_id_after_interruption_exclusion']<=spec['journal_next_series_id']+1:
        raise ValueError('Tail/interruption cursor differs')

def verify_journal(directory,saved,archive_spec,digest,spec):
    from .rl_resume import resume_journal
    from .archive_ack import safe_target
    directory=Path(directory)
    for ref in spec['references']:
        path=safe_target(directory,ref['path'])
        if path.stat().st_size!=ref['bytes'] or sha256(path)!=ref['sha256']:raise ValueError('Complete tail bytes changed')
    journal=resume_journal(directory,saved,archive_spec,digest)
    if len(journal['decisions'])!=spec['pending_decisions'] or journal['next_series_id']!=spec['journal_next_series_id']:
        raise ValueError('Transferred journal lost or duplicated complete tail')
    if journal['rng_after'] is None or journal['cpu_sampler_rng_after'] is None:raise ValueError('Latest complete BO3 RNG missing')
    next_id=journal['next_series_id'];inflight=directory/'inflight.json'
    if spec.get('inflight_sha256'):
        if sha256(inflight)!=spec['inflight_sha256']:raise ValueError('Original interrupted record changed')
        interrupted=read_json(inflight)['series_id']
        if (directory/f'series/{interrupted:06d}.json').exists():raise ValueError('Interrupted ID is actually complete')
        next_id=max(next_id,interrupted+1)
    if next_id!=spec['next_series_id_after_interruption_exclusion']:raise ValueError('Interrupted next cursor differs')
    return journal
