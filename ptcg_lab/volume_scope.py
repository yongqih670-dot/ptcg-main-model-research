"""Hash-bound TR12 phase budgets; historical tickets retain their own limits."""
from pathlib import Path
from .io import read_json,sha256

AUTH_PATH='research/autonomy/authorizations/post-TR11-320768-completion-budget-20261028.json'
AUTH_SHA='971709f6510a8c83527dd42175a9459e12861c00544f489067765dc6d2412ff4'
AUTH={'path':AUTH_PATH,'sha256':AUTH_SHA}
PARENT='20261027T160313Z_tr11-authorized-second-posix-fixed_1eae55'
PARENT_CP='4da07a52c3ef931f9dcea5413a4ad8c884e8e8a908ce3a7c7894dd6871a8bd40'
BASE=49173
TARGET=369941


def scoped(config):
    return (config.get('forward_phase')=='TR12' or config.get('hypothesis_id')=='TR12'
        or config.get('budget_hypothesis_id')=='TR12' or 'TR12_authorization' in config)


def validate(root,config,policy=None,state=None):
    if not scoped(config):return None
    root=Path(root)
    if (config.get('forward_phase')!='TR12' or config.get('TR12_authorization')!=AUTH
            or config.get('budget_hypothesis_id',config.get('hypothesis_id'))!='TR12'):
        raise ValueError('TR12 forward allowance cannot change another hypothesis or omit its authority')
    if sha256(root/AUTH_PATH)!=AUTH_SHA:raise ValueError('TR12 user authorization changed')
    policy=read_json(root/'research/autonomy/policy.json') if policy is None else policy
    state=read_json(root/'research/autonomy/state.json') if state is None else state
    if policy.get('forward_authorizations',{}).get('TR12_volume')!=AUTH:
        raise ValueError('TR12 phase allowance is not activated')
    phase=state.get('forward_phases',{}).get('TR12')
    if not phase or phase['authority']!=AUTH or phase['status'] not in ('preparing','active'):
        raise ValueError('TR12 phase is absent or closed')
    if type(config.get('maximum_attempts')) is not int or not 1<=config['maximum_attempts']<=3:
        raise ValueError('TR12 single raw attempt cap exceeded')
    return phase


def limits(root,config,kind,policy=None,state=None):
    phase=validate(root,config,policy,state)
    if phase is None:return None
    state=read_json(Path(root)/'research/autonomy/state.json') if state is None else state
    used=state['budget_used'];base=phase['baseline_budget_used']
    gpu=used['gpu_job_wall_hours']-base['gpu_job_wall_hours']
    cpu=used['cpu_job_wall_hours']-base['cpu_job_wall_hours']
    if min(gpu,cpu)<-1e-9:raise ValueError('Historical budget counters decreased')
    job=config.get('job_kind','evaluation')
    cap=28800 if job=='remote_train_ppo' and config.get('engineering_only') is False else 21600 if job=='evaluation' and config.get('hypothesis_id')=='TR12' else 1800
    remaining=min((84-gpu if kind=='gpu' else 12-cpu)*3600,(96-gpu-cpu)*3600)
    return {'job_seconds_max':cap,'hypothesis_seconds_max':96*3600,'phase_remaining_seconds':max(0,remaining)}


def validate_raws(root,config,policy,state,research,new=True):
    phase=validate(root,config,policy,state)
    if phase is None:return
    if not research:return
    if config['hypothesis_id']!='TR12':raise ValueError('Only one new research hypothesis was authorized')
    if config.get('job_kind') not in ('remote_train_ppo','evaluation'):raise ValueError('Unknown TR12 research role')
    field='training_run_ids' if config['job_kind']=='remote_train_ppo' else 'evaluation_run_ids'
    cap=16 if field=='training_run_ids' else 4
    all_runs=phase['training_run_ids']+phase['evaluation_run_ids']
    if len(set(all_runs))!=len(all_runs):raise ValueError('Duplicate phase research accounting')
    if new and (len(all_runs)>=20 or len(phase[field])>=cap):raise ValueError('TR12 phase research raw allowance exhausted')


def register(config,state,research,run_id):
    if not scoped(config):return
    phase=state['forward_phases']['TR12']
    field=('training_run_ids' if config['job_kind']=='remote_train_ppo' else 'evaluation_run_ids') if research else 'engineering_run_ids'
    if any(run_id in phase[k] for k in ('training_run_ids','evaluation_run_ids','engineering_run_ids')):
        raise ValueError('A phase job cannot be registered twice')
    phase[field].append(run_id)
    if research:phase['status']='active'


def validate_remote(scope,mode,identity):
    if (mode,identity,scope.get('authorization_sha256'),scope.get('seconds_max'),scope.get('CPU_quota_cores')) != (
            'derived_scientific','TR12',AUTH_SHA,28800,3):
        raise ValueError('TR12 runtime ticket exceeds its exact forward scope')
    if scope.get('absolute_unique_target')!=TARGET or scope.get('phase_GPU_hours_max')!=84 or scope.get('phase_CPU_hours_max')!=12:
        raise ValueError('TR12 target/phase budget scope differs')
    return 28800
