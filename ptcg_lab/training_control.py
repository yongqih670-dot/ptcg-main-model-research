"""Cooperative GPU yield between operations; no process-clock freezing."""
import os
from pathlib import Path
import time
from .io import read_json,atomic_json,utcnow
from .phase_timing import timed,span


class ControlledTrainingStop(RuntimeError):
    pass


def publish_resource_control(output,run_id,lease_epoch,gpu):
    output=Path(output).resolve()
    if output.name!=run_id or type(lease_epoch) is not int or lease_epoch<1:raise ValueError('Resource control identity differs')
    path=output/'resource-training-control.json';old=read_json(path) if path.exists() else None
    if old and (old.get('run_id')!=run_id or old.get('lease_epoch')!=lease_epoch):raise ValueError('Prior resource control belongs to another lease')
    reason=gpu.get('monitoring_unavailable')
    if not reason and 'foreign_compute_pids_observed' not in gpu:reason='GPU observation incomplete'
    if not reason and gpu['foreign_compute_pids_observed']:reason='External GPU compute process observed'
    row={'run_id':run_id,'lease_epoch':lease_epoch,'sequence':old['sequence']+1 if old else 0,
        'command':'pause' if reason else 'run','reason':str(reason or 'GPU telemetry valid; no external compute process')[:512],'at':utcnow()}
    atomic_json(path,row);return row


class TrainingControl:
    def __init__(self,output,run_id,lease_epoch,deadline,progress):
        self.output=Path(output).resolve();self.run_id=run_id;self.epoch=lease_epoch
        if self.output.name!=run_id or type(lease_epoch) is not int or lease_epoch<1:raise ValueError('Training control identity differs')
        self.deadline=deadline;self.progress=progress;self.seen={}
        self.wait_recorder=None
        settings=self.output/'active-clock-settings.json'
        if settings.exists():
            from .active_clock import read_small,validate_scope,WaitRecorder
            row=read_small(settings)
            if (row['run_id'],row['lease_epoch'])!=(run_id,lease_epoch):raise ValueError('Active clock settings identity differs')
            validate_scope(row['scope']);self.wait_recorder=WaitRecorder(self.output,run_id,lease_epoch)
    def event(self,stage,**fields):
        folder=self.output/'control-events'
        if folder.is_symlink():raise ValueError('Linked cooperative event directory')
        paths=sorted(folder.glob('[0-9]'*6+'.json'))
        if len(paths)>=(16384 if self.wait_recorder else 256):raise ControlledTrainingStop('Bounded cooperative event journal full')
        path=folder/f'{len(paths):06d}.json'
        if path.exists():raise ValueError('Cooperative event publication conflict')
        atomic_json(path,{'at':utcnow(),'run_id':self.run_id,'lease_epoch':self.epoch,
            'process_id':os.getpid(),'sequence':len(paths),'stage':stage,'deadline_unix':self.deadline,**fields})
    def request(self,name):
        path=self.output/name
        if path.is_symlink():raise ValueError('Linked training control request')
        if not path.exists():return None
        st=path.stat()
        if st.st_size>4096 or hasattr(os,'getuid') and st.st_uid!=os.getuid():raise ValueError('Unowned/oversized training control request')
        row=read_json(path)
        if row.get('run_id')!=self.run_id or row.get('lease_epoch')!=self.epoch or row.get('command') not in ('run','pause','stop'):
            raise ValueError('Foreign/stale training control request')
        seq=row.get('sequence')
        if type(seq) is not int or not 0<=seq<2**63 or not isinstance(row.get('reason'),str) or len(row['reason'])>512:
            raise ValueError('Training control request envelope differs')
        old=self.seen.get(name)
        if old and (seq<old['sequence'] or seq==old['sequence'] and row!=old):raise ValueError('Training control request regressed/conflicted')
        self.seen[name]=row;return row
    @timed('shared_resource_gate')
    def wait(self,margin=45,*,gpu_required=True):
        if type(gpu_required) is not bool:raise ValueError('GPU requirement must be explicit boolean')
        paused=False;started=None;last=0
        credit_from=None
        try:
            while True:
                now=time.monotonic()
                if credit_from is not None:
                    self.wait_recorder.add(now-credit_from);credit_from=None
                named=[(name,self.request(name)) for name in ['manual-training-control.json','resource-training-control.json']]
                rows=[x for _,x in named if x]
                if any(x['command']=='stop' for x in rows):raise ControlledTrainingStop('Verified experiment stop request')
                reasons=[x['reason'] for name,x in named if x and x['command']=='pause'
                         and (gpu_required or name=='manual-training-control.json')]
                if self.wait_recorder:
                    from .active_clock import remaining_from_record
                    if remaining_from_record(self.output,self.run_id,self.epoch)<=margin:
                        raise ControlledTrainingStop('Active execution allowance reached checkpoint reserve')
                if not gpu_required and time.time()>self.deadline-margin:
                    raise ControlledTrainingStop('CPU work reached unchanged checkpoint deadline')
                if not reasons:
                    if paused:
                        elapsed=time.monotonic()-started;self.event('resumed_after_resource_pause',resource_pause_seconds=elapsed)
                        self.progress('resumed_after_resource_pause',resource_pause_seconds=elapsed)
                    return
                if time.time()>self.deadline-margin:raise ControlledTrainingStop('Resource pause reached unchanged checkpoint deadline')
                now=time.monotonic()
                if not paused:
                    paused=True;started=now;self.event('paused_for_resource',reasons=reasons)
                if now-last>=10 or last==0:
                    self.progress('paused_for_resource',resource_pause_reasons=reasons,resource_pause_seconds=now-started);last=now
                manual=any(name=='manual-training-control.json' and x and x['command']=='pause' for name,x in named)
                if self.wait_recorder and gpu_required and not manual:credit_from=time.monotonic()
                with span('shared_resource_wait'):time.sleep(.25)
        finally:
            if self.wait_recorder and paused:
                self.wait_recorder.add(time.monotonic()-credit_from if credit_from is not None else 0,force=True)
