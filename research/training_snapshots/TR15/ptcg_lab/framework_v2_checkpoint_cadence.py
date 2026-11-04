"""Prospective framework cadence. Frozen 600-second implementations stay intact.

This is a reusable scheduler/validator, not an integrated training driver.
The caller must return from save only after a durable commit completes.
"""
import copy
import math
import time
from pathlib import Path
from .io import read_json, sha256

SCHEMA='framework_v2_reliable_checkpoint_v1'
DEFAULT_MAXIMUM=800
DEFAULT_RESERVE=60

def number(value, *, positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (value<=0 if positive else value<0):
        raise ValueError('Invalid checkpoint timing value')
    return value

def validate_profile(profile):
    if profile.get('schema')!=SCHEMA or profile.get('semantics')!='completed_commit_to_completed_commit':
        raise ValueError('Checkpoint completion semantics required')
    maximum=number(profile['maximum_reliable_commit_gap_seconds'],positive=True)
    reserve=number(profile['commit_reserve_seconds'],positive=True)
    number(profile['trigger_after_seconds'])
    if reserve>=maximum or profile['trigger_after_seconds']!=maximum-reserve:
        raise ValueError('Checkpoint trigger must reserve commit time')
    return dict(profile)

def load_profile(root):
    root=Path(root).resolve();policy=read_json(root/'research/autonomy/policy.json')
    active=policy['execution']['framework_v2_checkpoint_cadence'];reference=active['configuration']
    path=(root/reference['path']).resolve()
    if not path.is_relative_to(root) or path.is_symlink() or sha256(path)!=reference['sha256']:
        raise ValueError('Bound checkpoint cadence configuration changed')
    profile=validate_profile(read_json(path))
    for key in ('maximum_reliable_commit_gap_seconds','commit_reserve_seconds','trigger_after_seconds'):
        if active[key]!=profile[key]:raise ValueError('Policy and runtime cadence file differ')
    return profile

def bind_runtime_config(root,config):
    """Explicit future candidate config; does not grant a lease or start a job."""
    result=copy.deepcopy(config);spec=result.get('framework_v2_acceptance',{})
    if spec.get('kind') not in ('checkpoint_difference','fixed_replay_comparison') or not result.get('engineering_only'):
        raise ValueError('This cadence overlay is limited to framework candidates')
    spec['checkpoint_cadence']=load_profile(root)
    result['framework_v2_acceptance']=spec
    return result

def validate_runtime_config(root,config):
    profile=validate_profile(config['framework_v2_acceptance']['checkpoint_cadence'])
    if profile!=load_profile(root):raise ValueError('Runtime and effective cadence policy differ')
    return profile

def assess_gap(gap,maximum_gap_seconds=DEFAULT_MAXIMUM):
    number(gap);number(maximum_gap_seconds,positive=True)
    return dict(reliable_commit_gap_seconds=gap,maximum_reliable_commit_gap_seconds=maximum_gap_seconds,
        within_maximum_gap=gap<=maximum_gap_seconds,
        # Compatibility is literal: this historical predicate never becomes <=800.
        within_600_seconds=gap<=600)

def assess_runtime_gap(config,gap):
    profile=config.get('framework_v2_acceptance',{}).get('checkpoint_cadence')
    maximum=validate_profile(profile)['maximum_reliable_commit_gap_seconds'] if profile else 600
    return assess_gap(gap,maximum)

class Cadence:
    def __init__(self,*,maximum_gap=DEFAULT_MAXIMUM,commit_reserve=DEFAULT_RESERVE,clock=time.monotonic):
        number(maximum_gap,positive=True);number(commit_reserve,positive=True)
        if commit_reserve>=maximum_gap:raise ValueError('Invalid cadence reserve')
        self.maximum=maximum_gap;self.reserve=commit_reserve;self.clock=clock
        self.last=None;self.gaps=[];self.observed_commit_seconds=0

    @classmethod
    def from_runtime_config(cls,config,*,clock=time.monotonic):
        p=validate_profile(config['framework_v2_acceptance']['checkpoint_cadence'])
        return cls(maximum_gap=p['maximum_reliable_commit_gap_seconds'],commit_reserve=p['commit_reserve_seconds'],clock=clock)

    def committed(self):
        now=number(self.clock());gap=None if self.last is None else number(now-self.last)
        self.last=now
        if gap is None:return dict(reliable_commit_gap_seconds=None,maximum_reliable_commit_gap_seconds=self.maximum,
            within_maximum_gap=None,within_600_seconds=None,initial_reliable_commit=True)
        self.gaps.append(gap);return assess_gap(gap,self.maximum)

    def decision(self,next_operation_seconds=0,commit_seconds_estimate=0):
        number(next_operation_seconds);number(commit_seconds_estimate)
        reserve=max(self.reserve,self.observed_commit_seconds,commit_seconds_estimate)
        if next_operation_seconds+reserve>self.maximum:return 'operation_exceeds_safe_window'
        if self.last is None:return 'initial_checkpoint_required'
        elapsed=number(number(self.clock())-self.last)
        due=elapsed>=self.maximum-reserve or elapsed+next_operation_seconds>self.maximum-reserve
        return 'checkpoint_before_operation' if due else 'continue'

    def must_checkpoint(self,next_operation_seconds=0,commit_seconds_estimate=0):
        return self.decision(next_operation_seconds,commit_seconds_estimate)!='continue'

    def run_checkpoint(self,save):
        started=number(self.clock());value=save()  # failure does not advance the reliable anchor
        duration=number(number(self.clock())-started);result=self.committed()
        self.observed_commit_seconds=duration
        return dict(result,commit_duration_seconds=duration,saved=value)

    def before_operation(self,save,*,next_operation_seconds=0,commit_seconds_estimate=0):
        action=self.decision(next_operation_seconds,commit_seconds_estimate)
        if action=='operation_exceeds_safe_window':raise ValueError('Operation cannot fit the checkpoint window with commit reserve')
        if action=='continue':return None
        result=self.run_checkpoint(save)
        if self.decision(next_operation_seconds,commit_seconds_estimate)!='continue':
            raise ValueError('Operation is unsafe even after a reliable commit')
        return result

    def interrupt_inflight(self,series_id,completed_pending_ids):
        if not self.must_checkpoint():return None
        if series_id in set(completed_pending_ids):raise ValueError('Complete BO3 cannot be excluded')
        return dict(excluded_unfinished_BO3=series_id,next_series_id=series_id+1,
            reason='series_boundary_required_before_reliable_checkpoint',optimized_decisions_credit=0)
