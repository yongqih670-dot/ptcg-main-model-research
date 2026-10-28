"""Explicit future per-GPU CPU/RAM caps; legacy jobs retain their own limits."""
import math
import os
from pathlib import Path
import re
import subprocess

GIB=2**30


def limits(profile):
    if not isinstance(profile,dict) or profile.get('schema')!='per_GPU_CPU_RAM_share_v1':
        raise ValueError('Unrecognized server CPU/RAM resource profile')
    for name in ['authorization_sha256','machine_evidence_sha256']:
        if not re.fullmatch('[0-9a-f]{64}',profile.get(name,'')):raise ValueError('Resource authority/evidence hash missing')
    machine=profile['observed_machine'];cpu=machine['logical_CPU'];gpus=machine['GPU_count'];memory=machine['RAM_total_bytes']
    if any(type(v) is not int or v<=0 for v in [cpu,gpus,memory]):raise ValueError('Invalid measured machine capacity')
    requested_cpu=profile['cpu_quota_cores'];requested_memory=profile['memory_max_bytes']
    cpu_max=min(4,cpu/gpus);memory_max=min(62*GIB,(memory//gpus//GIB)*GIB)
    if type(requested_cpu) not in (int,float) or not math.isfinite(requested_cpu) or not 0<requested_cpu<=cpu_max:
        raise ValueError('CPU quota exceeds authorized observed perGPU share')
    if type(requested_memory) is not int or not 0<requested_memory<=memory_max:
        raise ValueError('Memory quota exceeds authorized observed perGPU share')
    return requested_cpu,requested_memory


def ticket_limits(ticket):
    profile=ticket.get('server_resource_profile')
    if profile is None:return None
    cpu,memory=limits(profile)
    if ticket.get('cpu_quota_cores')!=cpu or ticket.get('memory_max_bytes')!=memory:
        raise ValueError('Service ticket and explicit resource profile differ')
    return cpu,memory


def verify_actual_machine(profile):
    """Read-only target observation before any service starts."""
    cpu,memory=limits(profile)
    logical=min(os.cpu_count() or 1,len(os.sched_getaffinity(0)))
    total=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))*1024
    rows=subprocess.run(['nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,check=True,timeout=10).stdout.splitlines()
    gpus=len(rows)
    if not gpus or cpu>logical/gpus or memory>(total//gpus//GIB)*GIB:
        raise ValueError('Actual machine no longer supports this perGPU CPU/RAM share')
    return {'logical_CPU':logical,'GPU_count':gpus,'RAM_total_bytes':total,
            'requested_cpu_quota_cores':cpu,'requested_memory_max_bytes':memory,'actual_per_GPU_share_verified':True}


def validate_local_authority(root,profile,state):
    """Require the activated, hash-bound human supplement and real observation."""
    from .io import read_json,sha256
    root=Path(root).resolve();policy=read_json(root/'research/autonomy/policy.json')
    if policy.get('status')!='active' or not policy.get('activation',{}).get('executable'):
        raise ValueError('PerGPU resource policy is not active')
    gate=policy['execution']['hybrid']['shared_server_safety'];spec=gate.get('per_GPU_resource_authorization')
    if not spec or state.get('server_resource_profile_authorization')!=spec:
        raise ValueError('New perGPU resource allowance has not been forward-activated')
    path=(root/spec['path']).resolve()
    if root not in path.parents or sha256(path)!=spec['sha256'] or profile['authorization_sha256']!=spec['sha256']:
        raise ValueError('Latest human resource authority changed')
    authority=read_json(path);evidence=authority['actual_machine_evidence'];probe=(root/evidence['path']).resolve()
    if root not in probe.parents or sha256(probe)!=evidence['sha256'] or profile['machine_evidence_sha256']!=evidence['sha256']:
        raise ValueError('Actual perGPU machine observation changed')
    actual=read_json(probe)['actual']
    expected={'logical_CPU':min(actual['logical_cpu_total'],actual['logical_cpu_affinity']),
              'GPU_count':actual['gpu_count'],'RAM_total_bytes':actual['memory']['MemTotal']}
    if profile['observed_machine']!=expected:raise ValueError('Explicit resource profile changed measured capacity')
    cpu,memory=limits(profile);ceiling=authority['authorized_future_ceiling']
    if cpu>ceiling['logical_CPU_quota'] or memory>ceiling['memory_max_GiB']*GIB:
        raise ValueError('Requested profile exceeds the direct human resource allowance')
    return cpu,memory


def bind_manifest(root,config,manifest):
    profile=config.get('server_resource_profile')
    if profile is None:return manifest
    from .io import read_json
    validate_local_authority(root,profile,read_json(Path(root)/'research/autonomy/state.json'))
    return {**manifest,'server_resource_profile':profile}
