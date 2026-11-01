"""Early same-job pause proof, before any scientific optimizer update."""
import os
from pathlib import Path
import time
from .io import atomic_json,read_json,sha256,utcnow


def begin(job,ticket):
    path=Path(job)/'manual-training-control.json'
    if path.exists():raise ValueError('Early cooperative probe cannot overwrite a manual command')
    atomic_json(path,{'run_id':ticket['run_id'],'lease_epoch':ticket['lease_epoch'],'sequence':0,
        'command':'pause','reason':'Predeclared early cooperative pause before optimizer steps'})


class EarlyPauseProbe:
    def __init__(self,job,ticket,child_pid):
        self.job=Path(job);self.ticket=ticket;self.pid=child_pid;self.observed=None;self.released=False;self.complete=False
    def observe(self,kernel_frozen):
        paths=sorted((self.job/'control-events').glob('[0-9]'*6+'.json'))
        if len(paths)>256:raise ValueError('Cooperative event envelope exceeded')
        events=[(path,read_json(path)) for path in paths]
        for _,row in events:
            if (row['run_id'],row['lease_epoch'])!=(self.ticket['run_id'],self.ticket['lease_epoch']):raise ValueError('Cooperative event identity changed')
        own=[(p,r) for p,r in events if r['process_id']==self.pid]
        pause=next(((p,r) for p,r in own if r['stage']=='paused_for_resource'),None)
        if pause is None:return
        if kernel_frozen!='0':raise ValueError('Cooperative pause unexpectedly froze OS process clocks')
        if self.observed is None:self.observed=time.monotonic()
        if not self.released and time.monotonic()-self.observed>=5:
            path=self.job/'manual-training-control.json';row=read_json(path)
            if row['sequence']!=0 or row['command']!='pause':raise ValueError('Early probe cannot clear a later manual pause')
            atomic_json(path,{**row,'sequence':1,'command':'run','reason':'Early cooperative probe released after observed>=5s; original deadline unchanged'})
            self.released=True
        resume=next(((p,r) for p,r in own if r['stage']=='resumed_after_resource_pause' and r['sequence']>pause[1]['sequence']),None)
        if self.released and resume is not None:
            self.complete=True
            atomic_json(self.job/'cooperative-pause-acceptance.json',{'at':utcnow(),'complete':True,
                'run_id':self.ticket['run_id'],'lease_epoch':self.ticket['lease_epoch'],'child_pid':self.pid,
                'supervisor_pid':os.getpid(),'kernel_frozen':'0','same_supervisor_remained_active':True,
                'pause':pause[1],'pause_event_sha256':sha256(pause[0]),'resume':resume[1],'resume_event_sha256':sha256(resume[0]),
                'original_deadline_unix':self.ticket['deadline_unix'],'no_deadline_or_attempt_reset':True,
                'observed_pause_before_any_optimizer_update':True})
