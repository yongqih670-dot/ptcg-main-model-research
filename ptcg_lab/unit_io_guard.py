"""Conservative per-unit I/O pacing candidate; runtime acceptance is mandatory.

This is not a kernel io.max controller. It reads only this owned unit's process
counters and requests identity-checked freeze/thaw or stop through its manager.
"""
from pathlib import Path
import os
import time


class IOBudget:
    def __init__(self,read_bps,write_bps,burst_bytes=8*2**20,stop_debt_bytes=32*2**20,max_gap_seconds=.5):
        if not 0<read_bps<=16*2**20 or not 0<write_bps<=16*2**20:raise ValueError('I/O rate exceeds conservative profile')
        if not 0<burst_bytes<=16*2**20 or not burst_bytes<=stop_debt_bytes<=64*2**20:raise ValueError('Invalid I/O burst boundary')
        self.rates=(read_bps,write_bps);self.burst=burst_bytes;self.debt=stop_debt_bytes;self.max_gap=max_gap_seconds
        self.tokens=[float(burst_bytes),float(burst_bytes)];self.last=None;self.counts=(0,0);self.frozen=False
    def observe(self,now,read_bytes,write_bytes):
        if self.last is None:self.last=now
        elapsed=now-self.last
        if elapsed<0 or elapsed>self.max_gap:return {'action':'stop','reason':'monitor_time_gap'}
        current=(read_bytes,write_bytes)
        if any(type(value) is not int or value<old for value,old in zip(current,self.counts)):
            return {'action':'stop','reason':'io_counter_regressed_or_invalid'}
        for i,(value,old) in enumerate(zip(current,self.counts)):
            self.tokens[i]=min(self.burst,self.tokens[i]+elapsed*self.rates[i])-(value-old)
        self.last=now;self.counts=current
        if any(value < -self.debt for value in self.tokens):return {'action':'stop','reason':'io_burst_exceeds_stop_boundary'}
        if not self.frozen and min(self.tokens)<0:self.frozen=True;return {'action':'freeze','reason':'io_rate_debt'}
        if self.frozen and min(self.tokens)>=self.burst/2:self.frozen=False;return {'action':'thaw','reason':'io_budget_recovered'}
        return {'action':'hold' if self.frozen else 'run','tokens':list(self.tokens)}


def parse_io(text):
    values={name:int(value.strip()) for name,value in (line.split(':',1) for line in text.splitlines())}
    for name in ['rchar','wchar','read_bytes','write_bytes']:
        if name not in values or values[name]<0:raise ValueError('Incomplete process I/O accounting')
    return max(values['rchar'],values['read_bytes']),max(values['wchar'],values['write_bytes'])


class UnitCounters:
    def __init__(self,control_group,uid,proc_root='/proc',cgroup_root='/sys/fs/cgroup'):
        if not control_group.startswith('/') or '..' in control_group.split('/'):raise ValueError('Invalid owned cgroup')
        self.group=control_group;self.uid=uid;self.proc=Path(proc_root);self.cg=Path(cgroup_root)/control_group.lstrip('/')
        self.known={};self.retained={};self.high_water=[0,0]
    def sample(self):
        pids=[int(value) for value in (self.cg/'cgroup.procs').read_text().split()]
        if not pids:raise RuntimeError('No live process accounting; reconcile service state')
        sampled=[]
        for pid in pids:
            process=self.proc/str(pid)
            try:
                if process.stat().st_uid!=self.uid:raise RuntimeError('Non-owned process in job cgroup')
                state=process.joinpath('stat').read_text().rsplit(')',1)[1].split();start=state[19]
                membership=process.joinpath('cgroup').read_text()
                if '0::'+self.group not in membership.splitlines():raise RuntimeError('Process left owned job cgroup')
                if pid in self.known and self.known[pid]!=start:raise RuntimeError('PID identity reused during accounting')
                self.known[pid]=start
                read_bytes,write_bytes=parse_io(process.joinpath('io').read_text())
                key=(pid,start);old=self.retained.get(key,(0,0))
                self.retained[key]=(max(old[0],read_bytes),max(old[1],write_bytes))
                sampled.append({'pid':pid,'start_ticks':start})
            except FileNotFoundError:
                # Re-sample; a never-readable task cannot be declared covered.
                raise RuntimeError('Process accounting raced an exit; suspend and reconcile')
        # Retaining departed tasks can double-count waited-for child I/O. This
        # deliberately errs toward throttling, never crediting vanished bytes.
        sums=[sum(v[i] for v in self.retained.values()) for i in (0,1)]
        self.high_water=[max(old,new) for old,new in zip(self.high_water,sums)]
        return {'read_bytes':self.high_water[0],'write_bytes':self.high_water[1],'processes':sampled,
                'kernel_frozen':dict(line.split() for line in (self.cg/'cgroup.events').read_text().splitlines()).get('frozen')}


class IOGuard:
    def __init__(self,handle,control_group,read_bps,write_bps,report_callback):
        self.handle=handle;self.counters=UnitCounters(control_group,os.getuid());self.budget=IOBudget(read_bps,write_bps)
        self.report=report_callback;self.reason=None;self.freeze_count=0;self.thaw_count=0
    def step(self,now=None):
        from .systemd_job import freeze_verified,stop_verified
        try:
            sample=self.counters.sample();decision=self.budget.observe(time.monotonic() if now is None else now,
                                        sample['read_bytes'],sample['write_bytes'])
            if decision['action']=='freeze':
                result=freeze_verified(self.handle,True);self.freeze_count+=1;self.report({'decision':decision,'sample':sample,'action_proof':result})
            elif decision['action']=='thaw':
                result=freeze_verified(self.handle,False);self.thaw_count+=1;self.report({'decision':decision,'sample':sample,'action_proof':result})
            elif decision['action']=='stop':
                self.reason=decision['reason'];result=stop_verified(self.handle);self.report({'decision':decision,'sample':sample,'terminal_properties':result})
                return False
            return True
        except Exception as exc:
            self.reason='guard_unverified:'+type(exc).__name__
            # Failure never creates permission to continue an unmonitored unit.
            result=stop_verified(self.handle);self.report({'decision':{'action':'stop','reason':self.reason},'terminal_properties':result})
            return False
