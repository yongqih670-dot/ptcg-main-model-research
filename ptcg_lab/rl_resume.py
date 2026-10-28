"""Shared PPO resume journal for full archives and compact server worksets.

Only complete, hash-verified BO3 records enter statistics. A rolling archive
state never makes an old policy's unacknowledged shard on-policy again.
This module authorizes neither training nor remote replica eviction.
"""
from pathlib import Path
import torch
from .archive_ack import safe_target,validate_ack
from .archive_prefix import compact_record,load_prefix,relative_path
from .archive_state import initial_state,validate_state,restore_live_state,advance_state
from .io import atomic_json,read_json,sha256
from .rl_journal import committed_fixed_results
from .tensor_io import load_torch
from .checkpoint_refs import checked_reference
from .phase_timing import timed


def resume_journal(output,saved,archive_spec=None,checkpoint_digest=None):
    output=Path(output);version=saved['version'];cursor={k:saved[k] for k in ['version','trained_decisions','next_series_id']}
    if saved.get('archive_state') and not archive_spec:raise ValueError('Compact checkpoint requires its archive origin')
    prefix=None;state=None
    if archive_spec:
        prefix=load_prefix(safe_target(output,relative_path(archive_spec['prefix_path'])),archive_spec['prefix_sha256'])
        state=saved.get('archive_state')
        if state is None:
            if checkpoint_digest!=archive_spec['initial_checkpoint_sha256'] or checkpoint_digest!=prefix['checkpoint']['sha256']:
                raise ValueError('Initial compact checkpoint does not match its frozen origin')
            state=initial_state(prefix,archive_spec['prefix_sha256'])
        else:
            if saved.get('archive_origin_prefix_sha256')!=archive_spec['prefix_sha256']:
                raise ValueError('Checkpoint and compact origin differ')
            validate_state(state,prefix,archive_spec['prefix_sha256'])
        if state['checkpoint_cursor']!=cursor:raise ValueError('Compact checkpoint and rolling cursor differ')
    decisions=[];shards=[];last_rng=None;last_cpu_sampler_rng=None;next_id=cursor['next_series_id']
    for path in sorted((output/'rollouts').glob(f'v{version:04d}-s*.pt')):
        data=load_torch(path,map_location='cpu',weights_only=False)
        expected=f"v{version:04d}-s{data['series_id']:06d}.pt"
        if path.name!=expected or data['policy_version']!=version:raise ValueError('Rollout filename/version identity differs')
        summary=output/'series'/f"{data['series_id']:06d}.json"
        if not summary.exists():
            if data.get('summary',{}).get('statuses')!=['DONE','DONE']:raise ValueError('Uncommitted incomplete BO3 cannot be recovered')
            atomic_json(summary,{**data['summary'],'trajectory_sha256':sha256(path),'trajectory_path':path.relative_to(output).as_posix()})
        row=compact_record(read_json(summary),sha256(summary))
        if (row['series_id']!=data['series_id'] or row['policy_version']!=version or row['trajectory_sha256']!=sha256(path)
                or row['trajectory_path']!=path.relative_to(output).as_posix()):
            raise ValueError('Resume summary and trajectory differ')
        for decision in data['decisions']:
            if decision['series_id']!=data['series_id'] or int(decision['policy_version'])!=version:
                raise ValueError('Mixed decision series/policy in resume shard')
        decisions.extend(data['decisions']);shards.append(path.relative_to(output).as_posix())
        next_id=max(next_id,data['series_id']+1);last_rng=data['rng_after']
        if data.get('cpu_sampler_rng_after') is not None:last_cpu_sampler_rng=data['cpu_sampler_rng_after']
    if state is not None:
        restored=restore_live_state(output,state,prefix,archive_spec['prefix_sha256'])
        if restored['on_policy_shards']!=shards:raise ValueError('Compact restore changed ordered on-policy update inputs')
        stats=restored['fixed_statistics'];next_id=max(next_id,restored['next_series_id'])
    else:stats=committed_fixed_results(output)
    return {'decisions':decisions,'shards':shards,'next_series_id':next_id,'rng_after':last_rng,'cpu_sampler_rng_after':last_cpu_sampler_rng,
            'fixed_statistics':stats,'archive_state':state,'origin_prefix':prefix}


@timed('checkpoint_journal_validation')
def advance_resume_state(output,previous,origin_prefix,archive_spec,cursor,run_id,lease_epoch,*,_verified=None):
    if type(lease_epoch) is not int or lease_epoch<1:raise ValueError('Compact save requires a fenced server lease')
    output=Path(output);records=[];archived={row['series_id']:row for row in previous['archived']}
    for path in sorted((output/'series').glob('[0-9][0-9][0-9][0-9][0-9][0-9].json')):
        row=compact_record(read_json(path),sha256(path))
        if row['series_id']!=int(path.stem):raise ValueError('Changed complete BO3 identity')
        old=archived.get(row['series_id']);trajectory=output/row['trajectory_path']
        if old is not None and old!=row:raise ValueError('Archived complete BO3 summary changed')
        # An interruption between individually ACK-gated replica removals may
        # leave the archived summary after its trajectory was already removed.
        # Its exact compact record remains part of both local archive/history.
        digest=checked_reference(output,row['trajectory_path'],_verified)['sha256'] if trajectory.exists() else None
        if (digest is None and old is None) or digest is not None and digest!=row['trajectory_sha256']:
            raise ValueError('Changed complete BO3 cannot enter compact checkpoint')
        records.append(row)
    acks={}
    for path in sorted((output/'archive-acks').glob('*.json')):
        ack=read_json(path)
        validate_ack(ack,run_id,lease_epoch,ack['object'],ack['sha256'],ack['bytes'])
        old=acks.get(ack['object'])
        if old is not None and old!=ack:raise ValueError('Conflicting archive acknowledgements')
        acks[ack['object']]=ack
    return advance_state(previous,origin_prefix,archive_spec['prefix_sha256'],cursor,records,acks,run_id,lease_epoch)
