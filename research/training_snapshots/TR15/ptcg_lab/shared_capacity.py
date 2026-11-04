"""Opt-in capacity sharing under the direct TR13 resource authorization.

Only the named owned service may change its runtime CPU quota. Foreign PID
presence alone never blocks this route; telemetry, memory and ownership gates do.
"""
import os
from pathlib import Path
import subprocess
import time
from .io import read_json,atomic_json,utcnow

AUTH={'path':'research/autonomy/authorizations/TR13-shared-capacity-continue-20261030.json',
      'sha256':'dd43651ae29a2147d1016dc4c6099644ca7c7d0a782175689473217585ce0b4a'}
PROFILE={'schema':'TR13_capacity_sharing_v1','authority':AUTH,'GPU_target_MiB':10240,
         'GPU_minimum_budget_MiB':4096,'GPU_free_reserve_MiB':2048,
         'host_available_RAM_minimum_bytes':8*2**30,'CPU_quota_choices':[1,2,4],
         'foreign_GPU_compute_duty_fraction_target':.25}


def validate(profile):
    if profile!=PROFILE:raise ValueError('Shared-capacity profile is not the directly authorized bounded profile')
    return profile


def choose_capacity_gpu(rows,profile):
    """Called only after the usual idle-card preference found none."""
    validate(profile);eligible={}
    for line in rows:
        row=[x.strip() for x in line.split(',')]
        if len(row)!=5:raise ValueError('GPU capacity observation differs')
        index=int(row[0]);free=int(row[3])
        if index in (4,5,6,7) and free>=profile['GPU_minimum_budget_MiB']+profile['GPU_free_reserve_MiB']:
            eligible[index]=[row[0],row[1],row[2],row[4]]
    return next((eligible[i] for i in [7,6,5,4] if i in eligible),None)


def decide(gpu,host,profile,maximum_CPU=4):
    validate(profile)
    if gpu.get('monitoring_unavailable') or 'foreign_compute_pids_observed' not in gpu:
        return {'command':'pause','reason':'GPU capacity telemetry unavailable','CPU_quota_cores':1}
    if not {'memory_free_MiB','memory_total_MiB','own_service_process_gpu_memory_mib'}<=set(gpu):
        return {'command':'pause','reason':'GPU memory capacity observation incomplete','CPU_quota_cores':1}
    if not {'available_RAM_bytes','logical_CPU','load_average_1m'}<=set(host):
        return {'command':'pause','reason':'Host capacity observation incomplete','CPU_quota_cores':1,'CPU_resource_blocked':True}
    idle=max(0,host['logical_CPU']-host['load_average_1m'])
    cpu=4 if idle>=6 else 2 if idle>=3 else 1
    cpu=max(x for x in profile['CPU_quota_choices'] if x<=min(cpu,maximum_CPU))
    own=gpu['own_service_process_gpu_memory_mib'];free=gpu['memory_free_MiB']
    if min(own,free)<0 or gpu['memory_total_MiB']<=0:raise ValueError('Invalid GPU capacity observation')
    budget=min(profile['GPU_target_MiB'],own+free-profile['GPU_free_reserve_MiB'])
    reason=None
    if budget<profile['GPU_minimum_budget_MiB']:reason='GPU memory below the bounded runnable capacity'
    if host['available_RAM_bytes']<profile['host_available_RAM_minimum_bytes']:reason='Host available RAM below bounded reserve'
    return {'command':'pause' if reason else 'run','reason':reason or 'Measured capacity sufficient; foreign GPU processes permitted',
        'CPU_quota_cores':cpu,'torch_threads':min(2,cpu),'GPU_budget_MiB':max(0,budget),
        'CPU_resource_blocked':host['available_RAM_bytes']<profile['host_available_RAM_minimum_bytes'],
        'GPU_total_MiB':gpu['memory_total_MiB'],'GPU_co_tenant_mode':bool(gpu['foreign_compute_pids_observed']),
        'foreign_compute_pids_observed':gpu['foreign_compute_pids_observed'],
        'foreign_GPU_compute_duty_fraction_target':profile['foreign_GPU_compute_duty_fraction_target'],
        'authority':AUTH,'capacity_evidence':{'GPU_free_MiB':free,'GPU_own_MiB':own,**host}}


def host_snapshot():
    values=Path('/proc/meminfo').read_text().splitlines()
    available=int(next(line.split()[1] for line in values if line.startswith('MemAvailable:')))*1024
    return {'available_RAM_bytes':available,'logical_CPU':min(os.cpu_count() or 1,len(os.sched_getaffinity(0))),
            'load_average_1m':os.getloadavg()[0]}


class OwnCPUQuota:
    def __init__(self,ticket,job):
        self.ticket=ticket;self.job=Path(job);self.maximum=ticket['cpu_quota_cores'];self.current=self.maximum
        validate(ticket['shared_capacity'])
    def set(self,cores):
        if cores not in (1,2,4) or cores>self.maximum:raise ValueError('CPU quota exceeds original ticket ceiling')
        if cores==self.current:return
        from .systemd_job import inspect_service,identity,user_manager_environment
        expected=read_json(self.job/'identity.json');boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        properties=inspect_service(self.ticket['unit'])
        if identity(properties,boot)!=expected:raise ValueError('Own service identity changed before CPU adaptation')
        group=Path('/proc/self/cgroup').read_text().strip().split(':',2)[2]
        if group.split('/')[-1]!=self.ticket['unit']:raise ValueError('CPU adaptation is outside the exact owned service')
        subprocess.run(['systemctl','--user','set-property','--runtime',self.ticket['unit'],f'CPUQuota={cores*100}%'],
                       check=True,capture_output=True,text=True,timeout=10,env=user_manager_environment())
        actual=(Path('/sys/fs/cgroup')/group.lstrip('/')/'cpu.max').read_text().split()
        if actual[0]=='max' or abs(int(actual[0])/int(actual[1])-cores)>1e-6:
            raise ValueError('Own CPU quota adaptation did not take effect')
        self.current=cores
        path=self.job/'shared-capacity-CPU-changes.jsonl'
        import json
        with path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'at':utcnow(),'CPU_quota_cores':cores,'unit':self.ticket['unit'],'identity':expected})+'\n')
    def probe(self):
        previous=self.current;observed=[]
        for cores in [2,1,previous]:
            self.set(cores);observed.append(self.current)
        atomic_json(self.job/'shared-capacity-CPU-acceptance.json',{'at':utcnow(),'complete':True,
            'only_exact_owned_service_modified':True,'observed_CPU_quotas':observed,'maximum_original_ticket_CPU':self.maximum})


def publish(output,run_id,epoch,gpu,profile,quota):
    host=host_snapshot();decision=decide(gpu,host,profile,quota.maximum)
    quota.set(decision['CPU_quota_cores'])
    path=Path(output)/'resource-training-control.json';old=read_json(path) if path.exists() else None
    if old and (old['run_id'],old['lease_epoch'])!=(run_id,epoch):raise ValueError('Resource-control identity differs')
    row={'run_id':run_id,'lease_epoch':epoch,'sequence':old['sequence']+1 if old else 0,'at':utcnow(),**decision}
    if decision.get('GPU_budget_MiB'):
        other=max(0,gpu['own_service_process_gpu_memory_mib']-gpu.get('child_GPU_memory_MiB',0))
        row['GPU_allocator_budget_MiB']=max(1024,decision['GPU_budget_MiB']-other-512)
    atomic_json(path,row);return row
