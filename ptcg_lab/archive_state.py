"""Rolling compact history, allowing archival ACKs to lag PPO updates safely."""
from pathlib import Path
from .archive_ack import validate_ack
from .archive_prefix import canonical_hash,compact_record,fixed_statistics,validate_prefix
from .io import read_json,sha256


def _seal(body):return {**body,'state_id':canonical_hash(body)}


def initial_state(origin_prefix,origin_prefix_sha256):
    validate_prefix(origin_prefix)
    body={'schema_version':2,'origin_prefix_sha256':origin_prefix_sha256,
          'archive_cut_version':origin_prefix['checkpoint']['version'],
          'checkpoint_cursor':{k:origin_prefix['checkpoint'][k] for k in ['version','trained_decisions','next_series_id']},
          'archived':list(origin_prefix['consumed']),'live_expected':list(origin_prefix['tail']),
          'fixed_statistics':origin_prefix['full_fixed_statistics'],'parent_state_id':None}
    return _seal(body)


def validate_state(state,origin_prefix,origin_prefix_sha256):
    validate_prefix(origin_prefix)
    if state.get('schema_version')!=2 or state.get('origin_prefix_sha256')!=origin_prefix_sha256:
        raise ValueError('Rolling archive origin mismatch')
    if canonical_hash({k:v for k,v in state.items() if k!='state_id'})!=state.get('state_id'):
        raise ValueError('Rolling archive state changed')
    records=state['archived']+state['live_expected'];indexed={}
    for row in records:
        if compact_record(dict(row,statuses=['DONE','DONE']),row['summary_sha256'])!=row:
            raise ValueError('Invalid compact BO3 record')
        if row['series_id'] in indexed:raise ValueError('Duplicate rolling BO3')
        indexed[row['series_id']]=row
    for old in origin_prefix['consumed']+origin_prefix['tail']:
        if indexed.get(old['series_id'])!=old:raise ValueError('Origin history was lost or changed')
    cut=state['archive_cut_version'];cursor=state['checkpoint_cursor']
    if not origin_prefix['checkpoint']['version']<=cut<=cursor['version']:
        raise ValueError('Invalid archived cut/model version')
    if any(r['policy_version']>=cut for r in state['archived']):raise ValueError('Unconsumed record classified archived')
    if any(not cut<=r['policy_version']<=cursor['version'] for r in state['live_expected']):raise ValueError('Invalid live policy version')
    if records and max(r['series_id'] for r in records)>=cursor['next_series_id']:raise ValueError('Rolling series cursor regressed')
    if fixed_statistics(records)!=state['fixed_statistics']:raise ValueError('Rolling opponent statistics changed')
    return state


def advance_state(state,origin_prefix,origin_prefix_sha256,cursor,new_records,acks,run_id,lease_epoch):
    validate_state(state,origin_prefix,origin_prefix_sha256)
    for key in ['version','trained_decisions','next_series_id']:
        if type(cursor[key]) is not int or cursor[key]<state['checkpoint_cursor'][key]:raise ValueError('Model checkpoint cursor regressed')
    if (cursor['version']==state['checkpoint_cursor']['version'] and
            cursor['trained_decisions']!=state['checkpoint_cursor']['trained_decisions']):
        raise ValueError('Training count changed without a committed model update')
    indexed={r['series_id']:r for r in state['archived']+state['live_expected']}
    for row in new_records:
        if compact_record(dict(row,statuses=['DONE','DONE']),row['summary_sha256'])!=row:raise ValueError('Invalid new record')
        previous=indexed.get(row['series_id'])
        if previous is not None and previous!=row:raise ValueError('Conflicting committed BO3 identity')
        indexed[row['series_id']]=row
    origin_ids={r['series_id'] for r in origin_prefix['consumed']+origin_prefix['tail']}
    already_archived={r['series_id'] for r in state['archived']}
    def is_durable(row):
        if row['series_id'] in origin_ids or row['series_id'] in already_archived:return True
        relative_summary=f"series/{row['series_id']:06d}.json"
        for name,digest in [(row['trajectory_path'],row['trajectory_sha256']),(relative_summary,row['summary_sha256'])]:
            ack=acks.get(name)
            if ack is None:return False
            validate_ack(ack,run_id,lease_epoch,name,digest,ack['bytes'])
        return True
    records=sorted(indexed.values(),key=lambda r:r['series_id']);cut=cursor['version']
    for row in records:
        if row['policy_version']<cut and not is_durable(row):cut=min(cut,row['policy_version'])
    if cut<state['archive_cut_version']:raise ValueError('Archived history cut regressed')
    body={'schema_version':2,'origin_prefix_sha256':origin_prefix_sha256,'archive_cut_version':cut,
          'checkpoint_cursor':dict(cursor),'archived':[r for r in records if r['policy_version']<cut],
          'live_expected':[r for r in records if r['policy_version']>=cut],
          'fixed_statistics':fixed_statistics(records),'parent_state_id':state['state_id']}
    return validate_state(_seal(body),origin_prefix,origin_prefix_sha256)


def restore_live_state(output,state,origin_prefix,origin_prefix_sha256):
    """Validate live bytes and return only current-model on-policy inputs."""
    output=Path(output);validate_state(state,origin_prefix,origin_prefix_sha256)
    archived={r['series_id']:r for r in state['archived']};expected={r['series_id']:r for r in state['live_expected']}
    live={};cursor=state['checkpoint_cursor']
    for path in sorted((output/'series').glob('[0-9][0-9][0-9][0-9][0-9][0-9].json')):
        row=compact_record(read_json(path),sha256(path));sid=row['series_id']
        if sid!=int(path.stem):raise ValueError('Live summary identity mismatch')
        if sid in archived:
            if row!=archived[sid]:raise ValueError('Archived replica summary changed')
            continue
        if sid in expected and row!=expected[sid]:raise ValueError('Expected live record changed')
        if not state['archive_cut_version']<=row['policy_version']<=cursor['version']:raise ValueError('Live model version differs')
        if sha256(output/row['trajectory_path'])!=row['trajectory_sha256']:raise ValueError('Live update bytes changed')
        live[sid]=row
    if not set(expected)<=set(live):raise ValueError('Required live archive tail missing')
    records=[*archived.values(),*live.values()]
    return {'fixed_statistics':fixed_statistics(records),
            'on_policy_shards':[r['trajectory_path'] for r in sorted(live.values(),key=lambda r:r['series_id']) if r['policy_version']==cursor['version']],
            'journal_reference_paths':[r['trajectory_path'] for r in live.values()],
            'next_series_id':max([cursor['next_series_id'],*[r['series_id']+1 for r in live.values()]])}
