"""A service's main process guards its own private worker session.

Freezing only the worker process group keeps this guardian runnable. Any
guardian failure exits the service unsuccessfully so systemd's KillMode cleans
up only the same service's remaining processes. No global configuration writes.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .io import atomic_json,read_json,utcnow
from .server_workspace import owned_root,inside,admit
from .unit_io_guard import IOBudget
from .host_block_io import HostBlockCounters


def process_record(pid):
    base=Path('/proc')/str(pid);fields=(base/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'uid':base.stat().st_uid,'start_ticks':fields[19],
            'process_group':int(fields[2]),'session':int(fields[3]),'state':fields[0]}


class WorkerSession:
    def __init__(self,pid,control_group):
        self.root=process_record(pid);self.group=control_group;self.members={};self.retained={}
        if self.root['process_group']!=pid or self.root['session']!=pid or self.root['uid']!=os.getuid():
            raise RuntimeError('Worker does not own a private session')
    def sample(self):
        cg=Path('/sys/fs/cgroup')/self.group.lstrip('/')
        members=[];pids={os.getpid()}
        for listing in set(cg.rglob('cgroup.procs')):
            pids.update(int(value) for value in listing.read_text().split())
        # Include unreaped zombies, which need not appear in cgroup.procs.
        pending=list(pids)
        while pending:
            parent=pending.pop()
            for listing in (Path('/proc')/str(parent)/'task').glob('*/children'):
                for value in listing.read_text().split():
                    child=int(value)
                    if child not in pids:pids.add(child);pending.append(child)
        totals=[0,0];raw=[]
        for pid in sorted(pids,key=lambda value:value==os.getpid()):
            record=process_record(pid)
            if record['uid']!=self.root['uid']:raise RuntimeError('Foreign UID in owned resource subtree')
            if pid!=os.getpid():
                if record['process_group']!=self.root['pid'] or record['session']!=self.root['pid']:
                    raise RuntimeError('Unexpected process outside private worker session')
                if record['state'] not in ('T','t','Z'):raise RuntimeError('I/O snapshot is not quiescent')
                if pid in self.members and self.members[pid]['start_ticks']!=record['start_ticks']:raise RuntimeError('Worker PID identity changed')
                self.members[pid]=record;members.append(record)
            try:
                values={k:int(v) for k,v in (line.split(':',1) for line in (Path('/proc')/str(pid)/'io').read_text().splitlines())}
            except PermissionError:
                if record['state']!='Z':raise
                # Zombie /proc I/O is inaccessible on this host. This auxiliary
                # early pacer retains its last prefix; mandatory whole-host
                # kernel block accounting independently covers all disk I/O.
                raw.append({'identity':record,'counters':None,'coverage':'kernel_block_io_required'})
                continue
            for key in ['rchar','wchar','read_bytes','write_bytes']:
                if key not in values or values[key]<0:raise RuntimeError('Incomplete kernel I/O snapshot')
            key=(pid,record['start_ticks'])
            self.retained[key]=(max(values['rchar'],values['read_bytes']),max(values['wchar'],values['write_bytes']))
            raw.append({'identity':record,'counters':values})
        if not members:raise RuntimeError('Worker accounting disappeared before completion barrier')
        # Retained prefixes and waited-child transfer deliberately overcount.
        totals=[sum(v[i] for v in self.retained.values()) for i in (0,1)]
        return {'read_bytes':totals[0],'write_bytes':totals[1],'members':members,'raw_process_counters':raw}
    def signal(self,value):
        current=process_record(self.root['pid'])
        if any(current[k]!=self.root[k] for k in ['pid','uid','start_ticks','process_group','session']):
            raise RuntimeError('Refusing signal to changed worker identity')
        # This exact private session was created above; no names/PID patterns.
        os.killpg(self.root['pid'],value)
    def quiesce(self):
        self.signal(signal.SIGSTOP);until=time.monotonic()+.4
        cg=Path('/sys/fs/cgroup')/self.group.lstrip('/')
        while True:
            try:
                pids={int(value) for listing in cg.rglob('cgroup.procs') for value in listing.read_text().split()}
                states=[process_record(pid) for pid in pids if pid!=os.getpid()]
                if states and all(row['state'] in ('T','t','Z') for row in states):return self.sample()
            except FileNotFoundError:
                # No data is accepted from the raced snapshot. The whole private
                # session stays frozen while the stable kernel tree is re-read.
                pass
            if time.monotonic()>until:raise RuntimeError('Unable to obtain a quiescent owned I/O snapshot')
            time.sleep(.005)


def run(ticket_path):
    ticket=read_json(ticket_path);root=owned_root(ticket['project_root'],ticket['project_id']);job=inside(root,ticket['job_directory'])
    if os.environ.get('PTCG_JOB_RUN_ID')!=ticket['run_id']:raise RuntimeError('Guardian ticket differs')
    profile=ticket['resource_profile'];deadline=ticket['deadline_unix'];stop=False;child=None;session=None;frozen=False
    events=[];samples=[];reason=None;success=False;began=time.monotonic();last_report=began;last_disk=began
    def on_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,on_stop);signal.signal(signal.SIGINT,on_stop)
    membership=Path('/proc/self/cgroup').read_text().strip().splitlines()
    group=next(line[3:] for line in membership if line.startswith('0::'))
    cg=Path('/sys/fs/cgroup')/group.lstrip('/')
    kernel={name:(cg/name).read_text().strip() for name in ['cpu.max','memory.max','memory.swap.max']}
    quota,period=kernel['cpu.max'].split()
    if quota=='max' or int(quota)/int(period)>ticket['cpu_quota_cores']+.001:raise RuntimeError('CPU quota not enforced')
    if int(kernel['memory.max'])!=ticket['memory_max_bytes'] or kernel['memory.swap.max']!='0':raise RuntimeError('Memory kernel limits differ')
    admit(root,64*2**20)
    device=str(os.major(root.stat().st_dev))+':'+str(os.minor(root.stat().st_dev))
    physical=HostBlockCounters(device);physical_first=physical.sample()
    atomic_json(job/'guardian-start.json',{'at':utcnow(),'pid':os.getpid(),'kernel':kernel,'control_group':group,'profile':profile,'kernel_block_io':physical_first})
    budget=IOBudget(profile['read_bps'],profile['write_bps'],burst_bytes=profile['burst_bytes'],
                    stop_debt_bytes=profile['stop_debt_bytes'],max_gap_seconds=profile['max_gap_seconds'])
    block_budget=IOBudget(profile['read_bps'],profile['write_bps'],burst_bytes=profile['burst_bytes'],
                    stop_debt_bytes=profile['stop_debt_bytes'],max_gap_seconds=profile['max_gap_seconds'])
    block_budget.observe(time.monotonic(),0,0)
    try:
        # Reparent any orphaned descendants here so their waited I/O remains
        # inside the same observed subtree. This changes only this process.
        libc=ctypes.CDLL(None,use_errno=True)
        if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'Cannot become owned-child subreaper')
        command=[sys.executable,'-B',str(job/'control/payload_worker.py'),'--ticket',str(ticket_path)]
        child=subprocess.Popen(command,cwd=job/'control',start_new_session=True)
        session=WorkerSession(child.pid,group)
        first=session.quiesce();decision=budget.observe(time.monotonic(),first['read_bytes'],first['write_bytes'])
        if decision['action']=='stop':raise RuntimeError('Initial I/O coverage failed')
        if decision['action']=='freeze':
            frozen=True
            events.append({'at_monotonic':time.monotonic(),'action':'freeze','worker_identity':session.root,'members':first['members']})
        else:session.signal(signal.SIGCONT)
        atomic_json(job/'guardian-ready.json',{'at':utcnow(),'worker_identity':session.root})
        while child.poll() is None:
            now=time.monotonic()
            if stop or time.time()>deadline-15:reason='service_stop_or_deadline';break
            if ticket['audit_profile']=='guardian_fault' and (job/'payload-started.json').exists() and now-began>1:
                raise RuntimeError('Injected guardian failure after arming')
            pulse=job/'control-heartbeat.json'
            if not pulse.exists() or time.time()-pulse.stat().st_mtime>3:
                reason='control_heartbeat_lost';break
            disk=job/'disk-refusal.json'
            if disk.exists():reason='disk_admission_refused';break
            if now-last_disk>=1:admit(root,0);last_disk=now
            sample=session.quiesce();samples.append(sample)
            if len(samples)>256:samples.pop(0)
            block=physical.sample();sample['kernel_block_io']=block
            decisions=[budget.observe(now,sample['read_bytes'],sample['write_bytes']),
                       block_budget.observe(now,block['read_bytes'],block['write_bytes'])]
            sample['at_monotonic']=now
            sample['budget_decisions']={'auxiliary_process':decisions[0],'kernel_block':decisions[1],
                    'auxiliary_tokens':list(budget.tokens),'kernel_tokens':list(block_budget.tokens)}
            stopped=next((d for d in decisions if d['action']=='stop'),None)
            if stopped:reason=stopped['reason'];break
            zombies=[row['identity'] for row in sample['raw_process_counters'] if row['identity']['state']=='Z']
            if zombies and not (job/'kernel-zombie-covered.json').exists():
                proof={'action':'kernel_zombie_coverage','at_monotonic':now,'members':zombies,'kernel_block_io':block}
                events.append(proof);atomic_json(job/'kernel-zombie-covered.json',proof)
            should_freeze=budget.frozen or block_budget.frozen
            if should_freeze and not frozen:
                frozen=True
                events.append({'at_monotonic':now,'action':'freeze','worker_identity':session.root,'members':sample['members']})
            elif not should_freeze and frozen:
                session.signal(signal.SIGCONT);frozen=False
                events.append({'at_monotonic':now,'action':'thaw','worker_identity':session.root})
            elif not should_freeze:session.signal(signal.SIGCONT)
            exit_ready=job/'payload-exit-ready.json'
            if (ticket['audit_profile']=='normal_completion' and exit_ready.exists() and not frozen
                    and min(budget.tokens+block_budget.tokens)>=0 and not (job/'guardian-exit-ack.json').exists()):
                ready=read_json(exit_ready)
                if ready.get('pid')!=session.root['pid'] or ready.get('run_id')!=ticket['run_id']:
                    raise RuntimeError('Completion barrier identity differs')
                atomic_json(job/'guardian-exit-ack.json',{'at':utcnow(),'worker_identity':session.root,'final_io':sample})
            if now-last_report>.5:
                atomic_json(job/'guardian-progress.json',{'at':utcnow(),'io':sample,'frozen':frozen,'event_count':len(events)})
                last_report=now
            time.sleep(.02)
        if (reason is None and ticket['audit_profile']=='normal_completion' and child.returncode==0
                and (job/'guardian-exit-ack.json').exists()):
            remaining=[int(v) for v in (cg/'cgroup.procs').read_text().split() if int(v)!=os.getpid()]
            if remaining:raise RuntimeError('Unexpected descendants after successful payload exit')
            reason='payload_completed_after_exit_barrier';success=True
        if reason is None:reason='payload_exited_before_expected_fault'
    except Exception as exc:
        reason='guardian_failure:'+type(exc).__name__+':'+str(exc)
    finally:
        atomic_json(job/'guardian-result.json',{'at':utcnow(),'reason':reason,'worker_identity':session.root if session else None,
                    'events':events,'recent_samples':samples,'kernel':kernel,'profile':profile,
                    'seconds':time.monotonic()-began,'guardian_pid':os.getpid(),'clean_completion':success})
        if child and child.poll() is None:
            # quiesce() can fail after sending SIGSTOP but before the budget
            # records a freeze. Always resume this verified private session
            # before asking its root to checkpoint and exit.
            if session:
                try:session.signal(signal.SIGCONT)
                except (OSError,RuntimeError):pass
            current=process_record(child.pid)
            if session and any(current[k]!=session.root[k] for k in ['pid','uid','start_ticks','session']):
                raise RuntimeError('Refusing cleanup of changed worker')
            child.terminate()
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
    if success:return
    # Nonzero is intentional on faults: KillMode closes any descendants.
    raise RuntimeError('Resource audit ended: '+str(reason))


def main():
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);run(p.parse_args().ticket)
