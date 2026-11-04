"""Direct-human-authorized independent 100k stage, without reopening TR12."""
from pathlib import Path
import hashlib
import json
from .io import read_json, sha256

AUTH={'path': 'research/autonomy/authorizations/TR14-volume-100000-local200-20261031.json', 'sha256': 'b944ae19f40e22a0261268c3df37e60bff14f03e5bf002b70c826ccbc8b0a4f9'}
BASE=206957
TARGET=306957


def scoped(config):return config.get('forward_phase')=='TR14'


def validate(root,config,state=None):
    if not scoped(config):return None
    root=Path(root);state=state or read_json(root/'research/autonomy/state.json')
    if sha256(root/AUTH['path'])!=AUTH['sha256'] or config.get('TR14_authorization')!=AUTH:
        raise ValueError('TR14 direct human authorization differs')
    phase=state.get('forward_phases',{}).get('TR14')
    if not phase or phase.get('authority')!=AUTH or phase['status'] not in ('preparing','active'):
        raise ValueError('TR14 phase is absent or closed')
    if phase.get('user_stop') or state.get('current_user_pause') or (root/'research/autonomy/STOP').exists():
        raise ValueError('Current user stop/pause applies to TR14')
    if phase.get('no_further_training') and config.get('job_kind')=='remote_train_ppo' and config.get('engineering_only') is False:
        raise ValueError('TR14 fixed endpoint reached; further training is prohibited')
    recovery=phase.get('recovery_required_from_positive_updates')
    if recovery and recovery.get('status')!='verified' and config.get('engineering_only') is False:
        raise ValueError('Positive-update failure requires latest complete recovery; the v15 source must not restart')
    if config.get('engineering_only') is False:
        evaluation=phase.get('evaluation_protocol')
        if not evaluation or config.get('evaluation_protocol')!=evaluation or sha256(root/evaluation['path'])!=evaluation['sha256']:
            raise ValueError('Frozen endpoint evaluation inputs must be bound before scientific work')
    protocol=phase['protocol']
    if sha256(root/protocol['path'])!=protocol['sha256']:
        raise ValueError('TR14 frozen scientific protocol changed')
    registry=read_json(root/'research/autonomy/TR14-approved-configurations.json')
    digest=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest not in registry['configs'] or registry['authority']!=AUTH:
        raise ValueError('TR14 configuration was not registered')
    workers=config.get('workers');maximum_workers=8 if config.get('job_kind')=='evaluation' else 1
    if not 1<=config.get('maximum_attempts',0)<=3 or type(workers) is not int or not 1<=workers<=maximum_workers:
        raise ValueError('TR14 attempt/worker boundary differs')
    return phase


def validate_stage(root,config,policy,state,research,final,*,new=True,run_id=None):
    phase=validate(root,config,state)
    if phase is None:return False
    if final:raise ValueError('TR14 does not authorize final pool or submission')
    if research and config.get('job_kind') not in ('remote_train_ppo','evaluation'):
        raise ValueError('TR14 research role differs from the frozen protocol')
    if not research and config.get('job_kind')!='compact_audit':
        raise ValueError('TR14 engineering role differs from the frozen protocol')
    field='engineering_run_ids' if not research else 'training_run_ids' if config['job_kind']=='remote_train_ppo' else 'evaluation_run_ids'
    if research != (config.get('engineering_only') is False):raise ValueError('TR14 registered role differs')
    if new and len(phase[field])>=phase['raw_limits'][field]:raise ValueError('TR14 role raw allowance exhausted')
    if new and not research and config.get('job_kind')=='remote_train_ppo' and config.get('parent_acceptance_only'):
        prior=[rid for rid in phase['engineering_run_ids']
               if read_json(Path(root)/'runs'/rid/'config.json').get('parent_acceptance_only')]
        if len(prior)>=3:raise ValueError('TR14 zero-update startup engineering attempts exhausted')
    if not new and run_id not in phase[field]:raise ValueError('TR14 resume outside registered raw')
    b=policy['budget'];used=state['budget_used']
    if research and new and used['executed_research_experiments']>=b['executed_research_experiments_max']-6:
        raise ValueError('TR14 preserves the final research raw reserve')
    return True


def register(root,config,policy,state,research,final,run_id):
    if not validate_stage(root,config,policy,state,research,final):return False
    phase=state['forward_phases']['TR14']
    field='engineering_run_ids' if not research else 'training_run_ids' if config['job_kind']=='remote_train_ppo' else 'evaluation_run_ids'
    phase[field].append(run_id)
    if research:
        phase['status']='active';state['research_stage']['regular_raw_runs']+=1
    return True


def limits(root,config,kind,policy=None,state=None):
    phase=validate(root,config,state)
    if phase is None:return None
    state=state or read_json(Path(root)/'research/autonomy/state.json')
    used=state['budget_used'];base=phase['baseline_budget_used']
    gpu=used['gpu_job_wall_hours']-base['gpu_job_wall_hours']
    cpu=used['cpu_job_wall_hours']-base['cpu_job_wall_hours']
    if min(gpu,cpu)<-1e-9:raise ValueError('TR14 historical charges regressed')
    remaining=min((phase['GPU_hours_max']-gpu if kind=='gpu' else phase['CPU_hours_max']-cpu)*3600,
                  (phase['combined_hours_max']-gpu-cpu)*3600)
    if kind=='cpu':remaining-=300  # Finite executor preparation/analysis wall-time charge reserve.
    cap=config.get('training_wall_seconds',21600)
    if cap>(1800 if config.get('engineering_only') else 28800 if kind=='gpu' else 21600):
        raise ValueError('TR14 raw execution limit exceeds registered role')
    return {'job_seconds_max':cap,'hypothesis_seconds_max':phase['combined_hours_max']*3600,
            'phase_remaining_seconds':max(0,remaining)}


def validate_shared(root,config,state):
    phase=validate(root,config,state)
    if not phase:return False
    if config.get('check_variant')!='tr14_native':raise ValueError('TR14 requires its native v2 route')
    profile=config['server_resource_profile']
    from .resource_profile import limits as resource_limits
    resource_limits(profile)
    if profile['cpu_quota_cores']>4 or profile['memory_max_bytes']>62*2**30:
        raise ValueError('TR14 exceeds original per-GPU resource share')
    if config.get('gpu_index') not in (4,5,6,7) or config.get('sampling_device')!='cpu':
        raise ValueError('TR14 GPU/sampler scope differs')
    if phase.get('resource_authority'):
        from .shared_capacity import validate as validate_capacity,AUTH as resource_AUTH
        if phase['resource_authority']!=resource_AUTH or sha256(Path(root)/resource_AUTH['path'])!=resource_AUTH['sha256']:
            raise ValueError('TR14 direct capacity authority differs')
        validate_capacity(config.get('shared_capacity'))
    if config.get('acceptance_reuse'):
        from .tr14_acceptance_reuse import validate_local as validate_reuse
        if config.get('engineering_only') or config.get('GPU_cursor_difference_required'):
            raise ValueError('Initial differences must finish before reuse')
        validate_reuse(root,config['acceptance_reuse'])
    from .tr14_prepare import validate_local
    validate_local(root,config,state=state)
    if config.get('engineering_only') is False:
        if phase.get('require_exact_restored_GPU_state') and not config.get('require_exact_restored_GPU_state'):
            raise ValueError('Latest full-state CUDA restore must be checked before new scientific work')
        reference=phase.get('native_GPU_acceptance')
        if not reference or sha256(Path(root)/reference['path'])!=reference['sha256']:
            raise ValueError('TR14 native/GPU difference acceptance is required before science')
        proof=read_json(Path(root)/reference['path'])
        if not proof.get('complete') or not proof.get('parent_acceptance_passed'):
            raise ValueError('TR14 native/GPU acceptance did not pass')
    return True


def validate_remote(scope,mode,identity):
    if scope.get('resource_authorization'):
        from .shared_capacity import AUTH as resource_AUTH
        if scope['resource_authorization']!=resource_AUTH:raise ValueError('TR14 runtime resource authority differs')
    if mode!='derived_scientific' or identity!='TR14' or scope.get('authorization')!=AUTH:
        raise ValueError('TR14 runtime authority/mode differs')
    if scope.get('absolute_unique_target')!=TARGET or scope.get('CPU_quota_cores')!=4 or scope.get('memory_max_bytes')!=48*2**30:
        raise ValueError('TR14 runtime target/resources differ')
    if scope.get('maximum_reliable_commit_gap_seconds')!=800 or scope.get('commit_reserve_seconds')!=60:
        raise ValueError('TR14 checkpoint completion scope differs')
    if type(scope.get('execution_limit_seconds')) not in (int,float) or not 0<scope['execution_limit_seconds']<=28800:
        raise ValueError('TR14 finite execution window differs')
    return scope['execution_limit_seconds']
