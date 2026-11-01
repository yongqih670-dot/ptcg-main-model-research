"""Forward50GiB archive cadence; all ACK/reference deletion rules stay intact."""
import math

PROFILE={'schema':'HDD2_50GiB_archive_cadence_v1','transfer_interval_seconds':600,
    'transfer_batch_trigger_bytes':2**30,'pending_warning_bytes':8*2**30,
    'maximum_pending_bytes':12*2**30,'checkpoint_interval_seconds':600,
    'minimum_reliable_points':2,'mandatory_per_update_ACK_barrier':False}


def validate(profile):
    if profile!=PROFILE:raise ValueError('Unreviewed50GiB archive cadence')
    return profile


def capacity_signals(capacity,storage_profile=None):
    if not isinstance(capacity,dict) or 'workspace_bytes' not in capacity:
        return {'capacity_observation_available':False}
    used=capacity['workspace_bytes'];growth=capacity.get('forecast_growth_bytes',0);free=capacity['free_bytes']
    hard=50*2**30;warning=40*2**30;reserve=20*2**30
    if storage_profile is not None:
        from .storage_profile import validate as validate_storage
        validate_storage(storage_profile,storage_profile['root'])
        hard=storage_profile['hard_max_bytes'];warning=storage_profile['warning_bytes'];reserve=storage_profile['free_reserve_bytes']
        if capacity.get('hard_max_bytes')!=hard:raise ValueError('Observed capacity and authorized storage boundary differ')
    return {'capacity_observation_available':True,'workspace_warning':used+growth>=warning,
        'free_space_warning':free-growth<reserve+5*2**30,'hard_max_bytes':hard,'warning_bytes':warning,
        'free_reserve_bytes':reserve}


def transfer_due(profile,pending_bytes,seconds_since_last_transfer,terminal=False,workspace_warning=False):
    validate(profile)
    if type(pending_bytes) is not int or pending_bytes<0:raise ValueError('Invalid pending archive size')
    if not isinstance(seconds_since_last_transfer,(int,float)) or not math.isfinite(seconds_since_last_transfer) or seconds_since_last_transfer<0:
        raise ValueError('Invalid archive cadence clock')
    return bool(pending_bytes and (terminal or workspace_warning or pending_bytes>=profile['transfer_batch_trigger_bytes']
        or seconds_since_last_transfer>=profile['transfer_interval_seconds']))


def apply(config,manifest,root):
    """Explicit future profile only; migration/CUDA/route admission remains outside."""
    profile=manifest.get('storage_cadence')
    if profile is None:return config
    from .storage_profile import NEW,load
    if str(root)!=NEW or load(root) is None:raise ValueError('New cadence requires authorized privateHDD2 storage')
    validate(profile)
    if not manifest.get('actual_HDD2_runtime_acceptance_sha256'):raise ValueError('ActualHDD2runtime acceptance binding required')
    return {**config,'checkpoint_interval_seconds':600,'online_archive':{'maximum_pending_bytes':12*2**30},
        'consumed_reference_barrier_required':False,'retire_consumed_with_second_checkpoint':True,
        'storage_cadence':dict(profile)}
