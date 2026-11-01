"""Authoritative two-point commits; a hot checkpoint alone is never a commit."""
from pathlib import Path
from .archive_ack import safe_target
from .archive_prefix import load_prefix
from .archive_state import validate_state
from .checkpoint_refs import commit_checkpoint,read_index
from .io import sha256
from .tensor_io import atomic_torch_save,load_torch
from .phase_timing import timed


CURSOR=('version','trained_decisions','next_series_id')


def pending_references(state):
    paths=[]
    for row in state['live_expected']:
        paths.extend([row['trajectory_path'],f"series/{row['series_id']:06d}.json"])
    if len(paths)!=len(set(paths)):raise ValueError('Duplicate live recovery reference')
    return paths


def load_training_checkpoint(output,archive_spec=None,lease_epoch=None):
    output=Path(output);path=output/'checkpoint.pt';recovery=None;effective=archive_spec
    index_path=output/'checkpoint-commits/index.json'
    if index_path.exists():
        if archive_spec is None or type(lease_epoch) is not int:raise ValueError('Recovery index requires its compact/fenced context')
        index=read_index(output,output.name,lease_epoch);recovery=index['commits'][-1]
        path=safe_target(output,recovery['checkpoint']['path'])
        if recovery['prefix']['sha256']!=archive_spec['prefix_sha256']:raise ValueError('Recovery index archive origin differs')
        effective={**archive_spec,'prefix_path':recovery['prefix']['path']}
    if not path.exists():return {'saved':None,'digest':None,'archive_spec':effective,'recovery':None}
    saved=load_torch(path,map_location='cpu',weights_only=False);digest=sha256(path)
    if recovery and {k:saved[k] for k in CURSOR}!=recovery['cursor']:raise ValueError('Checkpoint payload and recovery cursor differ')
    if recovery:
        state=saved.get('archive_state')
        if state is None:
            from .archive_state import initial_state
            prefix=load_prefix(safe_target(output,effective['prefix_path']),effective['prefix_sha256'])
            state=initial_state(prefix,effective['prefix_sha256'])
        expected={}
        for row in state['live_expected']:
            expected[row['trajectory_path']]=row['trajectory_sha256']
            expected[f"series/{row['series_id']:06d}.json"]=row['summary_sha256']
        observed={row['path']:row['sha256'] for row in recovery['pending']}
        if len(observed)!=len(recovery['pending']) or observed!=expected:
            raise ValueError('Recovery commit omitted or changed a payload live-tail reference')
    if not recovery and archive_spec and saved.get('archive_state'):
        raise ValueError('Hot compact checkpoint has no published recovery index')
    return {'saved':saved,'digest':digest,'archive_spec':effective,'recovery':recovery}


def initialize_recovery(output,saved,archive_spec,state,lease_epoch):
    output=Path(output)
    if (output/'checkpoint-commits/index.json').exists():return read_index(output,output.name,lease_epoch)['commits'][-1]
    prefix=load_prefix(safe_target(output,archive_spec['prefix_path']),archive_spec['prefix_sha256'])
    validate_state(state,prefix,archive_spec['prefix_sha256'])
    if sha256(output/'checkpoint.pt')!=archive_spec['initial_checkpoint_sha256'] or archive_spec['initial_checkpoint_sha256']!=prefix['checkpoint']['sha256']:
        raise ValueError('Cannot bootstrap recovery from another checkpoint')
    cursor={k:saved[k] for k in CURSOR}
    if state['checkpoint_cursor']!=cursor:raise ValueError('Bootstrap cursor differs')
    return commit_checkpoint(output,output.name,lease_epoch,0,'checkpoint.pt',archive_spec['prefix_path'],pending_references(state),cursor)


@timed('checkpoint_commit')
def publish_training_checkpoint(output,value,archive_spec=None,lease_epoch=None,*,_verified=None):
    output=Path(output);state=value.get('archive_state')
    if archive_spec is None:
        if state is not None:raise ValueError('Compact payload requires recovery metadata')
        return {'write':atomic_torch_save(output/'checkpoint.pt',value),'recovery':None}
    verified={} if _verified is None else _verified
    index=read_index(output,output.name,lease_epoch,_verified=verified)
    prefix=load_prefix(safe_target(output,archive_spec['prefix_path']),archive_spec['prefix_sha256'])
    validate_state(state,prefix,archive_spec['prefix_sha256']);cursor={k:value[k] for k in CURSOR}
    if state['checkpoint_cursor']!=cursor or value.get('archive_origin_prefix_sha256')!=archive_spec['prefix_sha256']:
        raise ValueError('Checkpoint state/cursor/origin mismatch')
    receipt=atomic_torch_save(output/'checkpoint.pt',value)
    commit=commit_checkpoint(output,output.name,lease_epoch,index['commits'][-1]['generation']+1,
                            'checkpoint.pt',archive_spec['prefix_path'],pending_references(state),cursor,_verified=verified)
    return {'write':receipt,'recovery':commit}
