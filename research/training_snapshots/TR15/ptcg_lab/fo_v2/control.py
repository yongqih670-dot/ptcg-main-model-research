"""Typed outcomes and a single persistent, revision-fenced lifecycle journal."""
import copy
import hashlib
from pathlib import Path
from .durable import atomic_json,read,canonical,exclusive

INTEGRITY={'integrity_failure','checkpoint_hash_mismatch','restore_failed'}
CHECKPOINT_STAGES={'checkpointed_archive_backpressure':'checkpointed_backpressure',
    'checkpointed_at_soft_deadline':'deadline_checkpointed',
    'checkpointed_before_update_at_soft_deadline':'deadline_checkpointed',
    'partial_update_abandoned_at_deadline':'deadline_checkpointed',
    'checkpointed_for_control':'user_stopped',
    'partial_update_abandoned_for_control':'user_stopped'}


def outcome(metrics,props,*,identity_verified,checkpoint_verified,user_stopped=False):
    if not identity_verified:return 'unreconciled'
    if not metrics:return 'unreconciled'
    p=metrics.get('last_training_progress',{});stage=p.get('stage','')
    if stage in INTEGRITY or metrics.get('integrity_failure'):return 'integrity_failure'
    if any(props.get(k)!=v for k,v in {'LoadState':'loaded','Result':'success','ExecMainStatus':'0','ControlGroup':''}.items()):return 'unreconciled'
    target=metrics.get('actual_config',{}).get('target_decisions')
    trained=p.get('unique_PPO_trained_decisions',metrics.get('unique_PPO_trained_decisions'))
    if not checkpoint_verified:return 'integrity_failure'
    if user_stopped:return 'user_stopped'
    if metrics.get('complete') is True and type(trained) is int and type(target) is int and trained>=target:return 'goal_complete'
    if stage in ('checkpointed_for_control','partial_update_abandoned_for_control'):return 'unreconciled'
    if stage in CHECKPOINT_STAGES:return CHECKPOINT_STAGES[stage]
    return 'unreconciled'


class Lifecycle:
    """One controller owns this journal. Events require expected_revision.

    A launch intent without a bound terminal identity is unreconciled forever
    until the existing service is queried. No automatic second launch exists.
    External finalization steps must be idempotent on (run_id, step) keys.
    """
    def __init__(self,path,run_id):
        self.path=Path(path);self.run_id=run_id
        with exclusive(self.path.with_suffix('.lock')):
            if not self.path.exists():atomic_json(self.path,dict(schema=2,run_id=run_id,revision=0,phase='new',stop=False,identity=None,events={},steps={},charges={}))
        if self.state()['run_id']!=run_id:raise ValueError('Foreign lifecycle journal')
    def state(self):return read(self.path)
    def event(self,event_id,kind,payload,revision,*,fault=lambda stage:None):
        with exclusive(self.path.with_suffix('.lock')):
            return self._event(event_id,kind,payload,revision,fault=fault)
    def _event(self,event_id,kind,payload,revision,*,fault):
        s=self.state();digest=hashlib.sha256(canonical(dict(kind=kind,payload=payload))).hexdigest()
        if event_id in s['events']:
            if s['events'][event_id]!=digest:raise ValueError('Conflicting event replay')
            return s
        if revision!=s['revision']:raise ValueError('Stale controller event revision')
        if kind=='stop':s['stop']=True
        elif kind=='launch_intent':
            if s['stop'] or s['phase']!='new':raise ValueError('Launch forbidden or already claimed')
            s['phase']='unreconciled';s['identity']=payload
        elif kind=='bind_identity':
            if s['phase']!='unreconciled' or not s['identity'].get('pending_start') or payload['intent']!=s['identity']:raise ValueError('No matching unresolved original launch intent')
            service=payload['service']
            for key in ('run_id','lease_epoch','unit_name','manifest_sha256','token'):
                if service.get(key)!=s['identity'].get(key):raise ValueError('Existing remote service differs from launch intent')
            if not payload.get('identity_verified') or not service.get('pid') or not service.get('created'):raise ValueError('Actual service identity not verified')
            s['identity']=service
        elif kind=='terminal':
            if s['identity']!=payload['identity'] or not payload['identity_verified']:raise ValueError('Terminal identity differs (including PID creation/token)')
            s['phase']=payload['outcome'];s['terminal']=payload
        elif kind=='charge':
            key=payload['attempt_identity'];old=s['charges'].get(key)
            if old is not None and old!=payload:raise ValueError('Conflicting attempt accounting')
            s['charges'][key]=payload
        else:raise ValueError('Unknown lifecycle event')
        s['revision']+=1;s['events'][event_id]=digest;fault('before_journal');atomic_json(self.path,s);fault('after_journal');return s
    def finalize(self,steps):
        """A failed archival step never suppresses remaining log/accounting work."""
        errors={}
        for name,fn in steps.items():
            with exclusive(self.path.with_suffix('.lock')):
                s=self.state()
                if s['steps'].get(name)=='done':continue
                s['steps'][name]='intent';s['revision']+=1;atomic_json(self.path,s)
            try:fn(f'{self.run_id}/{name}')
            except Exception as exc:errors[name]=repr(exc);continue
            with exclusive(self.path.with_suffix('.lock')):
                s=self.state();s['steps'][name]='done';s['revision']+=1;atomic_json(self.path,s)
        return errors


def future_policy(ticket,*,user_stop=False,identity_valid=True,integrity_valid=True,
                  actual_resource_shortage=False,goal_complete=False,no_progress=False):
    """Old tickets retain old semantics. Only explicit schema2 applies overlay."""
    if ticket.get('efficiency_policy_schema')!=2:return dict(mode='legacy',apply_overlay=False)
    reasons=[name for name,flag in [('user_stop',user_stop),('identity_invalid',not identity_valid),
        ('integrity_invalid',not integrity_valid),('actual_resource_shortage',actual_resource_shortage),
        ('goal_complete',goal_complete),('no_progress_requires_diagnosis',no_progress)] if flag]
    return dict(mode='prospective',apply_overlay=True,stop_reasons=reasons,
                administrative_time_and_attempt_limits='record_and_alert',historical_costs_preserved=True)


def gpu_pressure(*,gpu_index,visible_count,free_bytes,own_peak_bytes,own_allocated_bytes,
                 margin_bytes=512*2**20,other_compute_pids=(),oom=False,identity_valid=True,
                 severe_pressure=False):
    if gpu_index not in (4,5,6,7) or visible_count!=1 or not identity_valid:return 'stop_identity_or_scope'
    if oom or severe_pressure:return 'checkpoint_reduce_load'
    required=max(0,own_peak_bytes-own_allocated_bytes)+margin_bytes
    if free_bytes<required:return 'checkpoint_reduce_load'
    return 'continue'  # Other PIDs alone neither pause nor prove no contention.
