"""A local wrapper exiting does not prove its independent server job stopped."""
from pathlib import Path
import re
from .io import read_json,sha256
from .systemd_job import identity,terminal


def validate_no_unreconciled_remote(root,state):
    lease=state.get('remote_job')
    if not lease:return
    relative=lease.get('terminal_receipt_path');digest=lease.get('terminal_receipt_sha256')
    if not relative or not digest:raise RuntimeError('Unreconciled remote job; verify its service before starting computation')
    root=Path(root).resolve();path=(root/relative).resolve()
    if root not in path.parents or sha256(path)!=digest:raise RuntimeError('Remote terminal receipt changed')
    receipt=read_json(path)
    for key in ['run_id','attempt','lease_epoch','host','unit']:
        if receipt.get(key)!=lease.get(key):raise RuntimeError('Remote terminal receipt identity differs')
    handle=lease.get('identity');properties=receipt.get('service_properties',{})
    if not handle or receipt.get('host_boot_id')!=handle.get('host_boot_id'):
        raise RuntimeError('Remote job termination identity is not established')
    if properties.get('LoadState')=='not-found':
        if receipt.get('terminal_proof_kind')=='exact_invocation_systemd_journal':
            from .terminal_journal import journal_terminal
            proof=journal_terminal(receipt['terminal_journal'],handle,1005,receipt['server_unix'])
            if proof!=receipt.get('journal_terminal_proof'):raise RuntimeError('Original terminal journal proof changed')
            return
        # Before the deadline, a still-running bootstrap might create the unit.
        if receipt.get('server_unix',0)<lease['deadline_unix']+60:
            raise RuntimeError('Missing service before deadline does not prove termination')
    elif identity(properties,receipt['host_boot_id'])!=handle or not terminal(properties):
        raise RuntimeError('Same remote invocation has not terminated')


def sync_remote_intent(root,state,run_id):
    """The runner alone writes state; its child atomically publishes its intent."""
    root=Path(root).resolve();directory=(root/'runs'/run_id).resolve()
    if directory.parent!=root/'runs':raise ValueError('Remote intent run is outside project')
    path=directory/'remote-job.json'
    if not path.exists():return
    lease=read_json(path)
    if lease.get('run_id')!=run_id or lease.get('host')!='TRAINING_HOST':raise ValueError('Remote intent target differs')
    if not re.fullmatch(r'ptcg-hybrid-[a-z0-9-]{8,80}\.service',lease.get('unit','')):raise ValueError('Unowned remote unit')
    if type(lease.get('attempt')) is not int or lease['attempt']<1:raise ValueError('Invalid remote attempt')
    if type(lease.get('lease_epoch')) is not int or lease['lease_epoch']<1:raise ValueError('Invalid remote lease epoch')
    previous=state.get('remote_job')
    if previous and previous.get('unit')!=lease['unit']:
        validate_no_unreconciled_remote(root,state)
        if lease['lease_epoch']<=previous['lease_epoch']:raise ValueError('Stale remote lease epoch')
        state.setdefault('remote_job_history',[]).append(previous)
    elif previous:
        if any(previous.get(k)!=lease.get(k) for k in ['run_id','attempt','lease_epoch','host','deadline_unix']):
            raise ValueError('Remote lease identity changed')
        if previous.get('identity') and previous['identity']!=lease.get('identity'):
            raise ValueError('Remote invocation identity changed')
    state['remote_job']=lease
    state['remote_lease_epoch']=max(state.get('remote_lease_epoch',0),lease['lease_epoch'])
