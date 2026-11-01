"""Compact, externally hash-bound consumed-trajectory history for PPO restore.

This module does not delete shards or authorize training. The original run and
the full archived trajectories remain untouched on the controller host.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
from .io import read_json,sha256


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def relative_path(value):
    value=str(value).replace(chr(92),'/');path=PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or ':' in value or str(path)!=value:
        raise ValueError('Unsafe journal path')
    return value


def compact_record(row,summary_hash):
    sid=row['series_id'];seat=row['learner_seat'];version=int(row['policy_version'])
    if type(sid) is not int or sid<0 or version<0 or type(seat) is not int or seat not in (0,1):raise ValueError('Invalid series identity')
    if row['statuses']!=['DONE','DONE'] or len(row['rewards'])!=2:raise ValueError('Incomplete BO3')
    if any(x not in (-1,0,1) for x in row['rewards']):raise ValueError('Invalid terminal reward')
    path=relative_path(row['trajectory_path'])
    if not path.startswith('rollouts/'):raise ValueError('Trajectory outside rollout journal')
    return {'series_id':sid,'policy_version':version,'learner_seat':seat,'rewards':row['rewards'],
            'assignment':row['assignment'],'trajectory_path':path,
            'trajectory_sha256':row['trajectory_sha256'],'summary_sha256':summary_hash}


def fixed_statistics(records):
    stats={};seen=set()
    for row in records:
        if row['series_id'] in seen:raise ValueError('Duplicate BO3 in archive history')
        seen.add(row['series_id'])
        assignment=row['assignment']
        if assignment['fixed']:
            entry=stats.setdefault(assignment['opponent_id'],{'n':0,'losses':0.})
            reward=row['rewards'][row['learner_seat']];entry['n']+=1
            entry['losses']+=1 if reward<0 else .5 if reward==0 else 0
    return stats


def validate_prefix(prefix):
    if prefix.get('schema_version')!=1:raise ValueError('Unsupported archive prefix')
    records=prefix['consumed'];tail=prefix['tail'];version=prefix['checkpoint']['version']
    ids=[r['series_id'] for r in records+tail]
    if len(ids)!=len(set(ids)):raise ValueError('Prefix/tail duplicate BO3')
    if any(r['policy_version']>=version for r in records):raise ValueError('Unconsumed data in archived prefix')
    if any(r['policy_version']!=version for r in tail):raise ValueError('Mixed policy versions in tail')
    if ids and max(ids)>=prefix['checkpoint']['next_series_id']:raise ValueError('Series cursor precedes committed history')
    if fixed_statistics(records)!=prefix['consumed_fixed_statistics']:raise ValueError('Archived statistics mismatch')
    if fixed_statistics(records+tail)!=prefix['full_fixed_statistics']:raise ValueError('Prefix/tail statistics mismatch')
    if canonical_hash(records)!=prefix['consumed_records_root']:raise ValueError('Archived record root mismatch')
    for row in records+tail:relative_path(row['trajectory_path'])
    return prefix


def load_prefix(path,expected_sha256):
    if not expected_sha256 or sha256(path)!=expected_sha256:raise ValueError('Archive prefix checksum mismatch')
    return validate_prefix(read_json(path))


def build_prefix(project,run,inventory,inventory_sha256,checkpoint_sha256):
    project=Path(project).resolve();run=Path(run).resolve();run_prefix=run.relative_to(project).as_posix()
    if sha256(inventory)!=inventory_sha256:raise ValueError('Local archive inventory changed')
    files=read_json(inventory)['files']
    if sha256(run/'checkpoint.pt')!=checkpoint_sha256:raise ValueError('Origin checkpoint changed')
    import torch
    cp=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    records=[]
    for path in sorted((run/'series').glob('[0-9][0-9][0-9][0-9][0-9][0-9].json')):
        key=path.relative_to(project).as_posix();summary_hash=sha256(path)
        if summary_hash!=files[key]['sha256']:raise ValueError('Archived summary changed')
        row=compact_record(read_json(path),summary_hash)
        if row['series_id']!=int(path.stem):raise ValueError('Series filename identity mismatch')
        shard_key=run_prefix+'/'+row['trajectory_path']
        if files[shard_key]['sha256']!=row['trajectory_sha256']:raise ValueError('Trajectory/archive identity mismatch')
        if not (project/shard_key).is_file():raise ValueError('Original trajectory no longer archived')
        records.append(row)
    consumed=[r for r in records if r['policy_version']<cp['version']]
    tail=[r for r in records if r['policy_version']>=cp['version']]
    # The unconsumed tail will actually be used by PPO; verify its bytes now.
    for row in tail:
        if sha256(run/row['trajectory_path'])!=row['trajectory_sha256']:raise ValueError('Unconsumed trajectory changed')
    prefix={'schema_version':1,'origin_run_id':run.name,'archive_inventory_sha256':inventory_sha256,
            'origin_config_sha256':sha256(run/'config.json'),'origin_source_manifest_sha256':sha256(run/'source-manifest.json'),
            'checkpoint':{'sha256':checkpoint_sha256,'version':cp['version'],'trained_decisions':cp['trained_decisions'],
                          'next_series_id':cp['next_series_id']},
            'consumed':consumed,'tail':tail,'consumed_records_root':canonical_hash(consumed),
            'consumed_fixed_statistics':fixed_statistics(consumed),'full_fixed_statistics':fixed_statistics(records),
            'past_trajectory_bytes_verified_by':'previous full local inventory; summaries and live tail reverified during prefix creation',
            'original_data_deleted':False}
    if prefix['full_fixed_statistics']!=read_json(run/'league.json')['fixed_training_results']:
        raise ValueError('Archive-prefix statistics differ from original league')
    validate_prefix(prefix)
    return prefix


def restored_statistics(output,prefix,live_series_paths=None):
    """Reconstruct with a compact prefix and actual still-live trajectory bytes."""
    output=Path(output);validate_prefix(prefix);records=list(prefix['consumed']);seen={r['series_id'] for r in records}
    expected_tail={r['series_id']:r for r in prefix['tail']};found=set()
    paths=live_series_paths if live_series_paths is not None else (output/'series').glob('[0-9][0-9][0-9][0-9][0-9][0-9].json')
    for path in sorted(paths):
        path=Path(path)
        if not path.resolve().is_relative_to((output/'series').resolve()):raise ValueError('Live summary outside journal')
        row=compact_record(read_json(path),sha256(path));sid=row['series_id']
        if sid in seen:raise ValueError('Consumed prefix duplicated in live journal')
        if sid in expected_tail and row!=expected_tail[sid]:raise ValueError('Restore tail identity changed')
        if row['policy_version']<prefix['checkpoint']['version']:raise ValueError('Old data outside archived prefix')
        if sha256(output/row['trajectory_path'])!=row['trajectory_sha256']:raise ValueError('Live trajectory changed')
        records.append(row);seen.add(sid);found.add(sid)
    if not set(expected_tail)<=found:raise ValueError('Missing unconsumed trajectory')
    return fixed_statistics(records)
