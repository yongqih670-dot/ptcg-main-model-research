"""Exact forward-only TR11 runtime allowance from the latest human request."""
from pathlib import Path
import re
from .io import read_json,sha256

SECONDS=30000


def attempt_limit(root,config):
    ref=config.get('TR11_attempt_authorization')
    if ref is None:return 1
    root=Path(root).resolve();path=(root/ref['path']).resolve()
    if root not in path.parents or sha256(path)!=ref['sha256']:raise ValueError('Attempt authorization changed')
    authority=read_json(path);policy=read_json(root/'research/autonomy/policy.json')
    if (policy.get('forward_authorizations',{}).get('TR11_attempt3_raw42')!=ref
            or authority.get('TR11_training_attempts_max')!=3 or authority.get('TR11_research_raw_max')!=4
            or authority.get('project_research_raw_max')!=42):
        raise ValueError('Explicit active attempt3/raw4/project42 authority required')
    if config.get('TR11_original_remote_deadline_unix')!=authority['original_absolute_remote_deadline_unix']:
        raise ValueError('TR11 original absolute deadline cannot be extended')
    if (config.get('hypothesis_id'),config.get('hypothesis_max_training_attempts'))!=('TR11',3):
        raise ValueError('Additional attempts belong only to current TR11')
    return 3


def local_limit(root,config):
    ref=config.get('TR11_runtime_authorization')
    if ref is None:return None
    maximum=attempt_limit(root,config);index=config.get('training_attempt_index')
    if type(index) is not int or not 1<=index<=maximum:raise ValueError('TR11 training attempt cap exhausted')
    if (config.get('hypothesis_id'),config.get('check_variant'),config.get('job_kind'),
            config.get('engineering_only'),config.get('maximum_attempts'),
            config.get('training_wall_seconds'))!=('TR11','continuous_mainline','remote_train_ppo',False,1,SECONDS):
        raise ValueError('Extended runtime belongs only to the unstarted soleTR11 scientific attempt')
    root=Path(root).resolve();path=(root/ref['path']).resolve()
    if root not in path.parents or sha256(path)!=ref['sha256']:raise ValueError('Runtime authorization changed')
    a=read_json(path)
    if (not a['start_existing_TR11_after_all_gates'] or a['TR11_training_wall_seconds_max']!=SECONDS
            or a['TR11_CPU_quota_cores']!=3 or a['new_scientific_attempts_added']!=0):
        raise ValueError('Human runtime scope differs')
    if config['server_resource_profile']['cpu_quota_cores']!=3:raise ValueError('TR11 must receive the authorized3CPU quota')
    return SECONDS


def validate_remote_scope(scope,mode,hypothesis):
    if (mode,hypothesis,scope.get('seconds_max'),scope.get('CPU_quota_cores'))!=('derived_scientific','TR11',SECONDS,3):
        raise ValueError('Extended ticket cannot change another experiment or resource scope')
    if not re.fullmatch('[0-9a-f]{64}',scope.get('authorization_sha256','')):
        raise ValueError('Hash-bound human runtime scope required')
    return SECONDS
