"""TR12 wall-minus-proven-wait clock; fixed safety ceiling and fail-closed notify.

Only the service main process notifies systemd. Child processes may report
their own cooperative resource wait, but never extend deadlines themselves.
"""
import math
import os
from pathlib import Path
import socket
import threading
import time
from .io import atomic_json,read_json,sha256

AUTH={'path':'research/autonomy/authorizations/TR12-shared-wait-exemption-20261028.json',
      'sha256':'1f4a7e4c79782ec85001a43e268957e9e57969f2be50cbd329d17047dd8627a4'}
SPEC={'schema':1,'authorization':AUTH,'hard_safety_window_seconds':86400,
      'basis':'gross_wall_minus_verified_shared_resource_wait'}

def validate_spec(value):
    if value!=SPEC:raise ValueError('Unreviewed active clock specification')

def validate_local(root,config,state=None):
    value=config.get('active_clock')
    if value is None:return False
    validate_spec(value)
    if config.get('forward_phase')!='TR12' or config.get('check_variant')!='volume_extension':
        raise ValueError('Wait exemption is scoped to TR12 full-state segments')
    if sha256(Path(root)/AUTH['path'])!=AUTH['sha256']:raise ValueError('Wait authorization changed')
    if state is not None and state.get('forward_phases',{}).get('TR12',{}).get('active_clock_authorization')!=AUTH:
        raise ValueError('Active clock not activated')
    return True

def finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Clock requires finite numbers')
    return float(value)

def validate_scope(scope):
    validate_spec(scope['spec'])
    limit=finite(scope['execution_limit_seconds']);window=finite(scope['hard_window_seconds'])
    origin=finite(scope['job_started_unix']);deadline=finite(scope['hard_deadline_unix'])
    if not 0<limit<=28800 or not limit<=window<=86400 or abs(deadline-origin-window)>.01:
        raise ValueError('Active/wall scope inconsistent')
    wait=finite(scope.get('startup_wait_seconds',0))
    if not 0<=wait<=window:raise ValueError('Invalid startup wait')
    return limit

def process_ticks(pid):
    return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]

def notify(message):
    address=os.environ.get('NOTIFY_SOCKET')
    if not address:raise RuntimeError('Active clock requires the service notify socket')
    if address.startswith('@'):address='\0'+address[1:]
    with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as channel:
        channel.connect(address);channel.sendall(message.encode())

def read_small(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>16384 or (hasattr(os,'getuid') and path.stat().st_uid!=os.getuid()):
        raise ValueError('Clock record owner/envelope differs')
    return read_json(path)

class WaitRecorder:
    def __init__(self,job,run_id,epoch):
        self.job=Path(job);self.run=run_id;self.epoch=epoch
        self.pid=os.getpid();self.ticks=process_ticks(self.pid)
        self.path=self.job/'resource-wait'/(str(self.pid)+'-'+self.ticks+'.json')
        self.total=0.;self.last_publish=0.
    def add(self,seconds,force=False):
        seconds=finite(seconds)
        if seconds<0:raise ValueError('Wait clock regressed')
        self.total+=seconds;now=time.monotonic()
        if force or now-self.last_publish>=5:
            atomic_json(self.path,{'run_id':self.run,'lease_epoch':self.epoch,'pid':self.pid,
                'start_ticks':self.ticks,'verified_wait_seconds':self.total,'observed_monotonic':now})
            self.last_publish=now

def remaining_from_record(job,run_id,epoch):
    path=Path(job)/'active-clock-server.json'
    if not path.exists():raise RuntimeError('Active clock health record missing')
    row=read_small(path)
    if row['run_id']!=run_id or row['lease_epoch']!=epoch or row.get('failure'):
        raise RuntimeError('Active clock identity or health differs')
    age=time.monotonic()-finite(row['observed_monotonic'])
    if not 0<=age<=30:raise RuntimeError('Active clock health record stale')
    # Unobserved intervals are charged conservatively until proven waiting.
    return finite(row['remaining_execution_seconds'])-age

class ServiceClock:
    def __init__(self,job,run_id,epoch,scope,*,reserve=50,notifier=notify):
        self.job=Path(job);self.run=run_id;self.epoch=epoch;self.scope=scope
        self.limit=validate_scope(scope);self.reserve=reserve;self.notifier=notifier
        self.origin=time.monotonic()-(time.time()-scope['job_started_unix'])
        self.initial_wait=scope.get('startup_wait_seconds',0.)
        self.children={};self.lock=threading.Lock();self.done=threading.Event();self.failure=None
        self.last_notified=-1.;self.thread=None
    def register(self,pid,ticks):
        if process_ticks(pid)!=ticks:raise ValueError('Child identity changed before clock registration')
        with self.lock:
            key=(pid,str(ticks))
            if key in self.children:raise ValueError('Clock child registered twice')
            self.children[key]=0.
    def sample(self):
        records=[]
        with self.lock:
            for (pid,ticks),previous in list(self.children.items()):
                path=self.job/'resource-wait'/(str(pid)+'-'+ticks+'.json')
                if not path.exists():continue
                row=read_small(path)
                if (row['run_id'],row['lease_epoch'],row['pid'],str(row['start_ticks']))!=(self.run,self.epoch,pid,ticks):
                    raise ValueError('Foreign wait record')
                credit=finite(row['verified_wait_seconds'])
                if credit<previous or row['observed_monotonic']>time.monotonic()+.01:raise ValueError('Wait record regressed/future')
                self.children[(pid,ticks)]=credit
                records.append({'path':path.relative_to(self.job).as_posix(),'sha256':sha256(path),
                                'verified_wait_seconds':credit})
            credit=self.initial_wait+sum(self.children.values())
        now=time.monotonic();gross=now-self.origin
        if not 0<=credit<=gross+.01:raise ValueError('Wait exceeds elapsed job time')
        remaining=min(self.limit-(gross-credit),self.scope['hard_deadline_unix']-time.time())
        row={'run_id':self.run,'lease_epoch':self.epoch,'observed_monotonic':now,
            'gross_seconds':gross,'verified_wait_seconds':credit,'counted_execution_seconds':gross-credit,
            'remaining_execution_seconds':remaining,'hard_deadline_unix':self.scope['hard_deadline_unix']}
        row['wait_records']=records
        atomic_json(self.job/'active-clock-server.json',row)
        # Extensions are earned only by newly observed wait. If this thread or
        # its parent dies, the last finite systemd timeout is still enforced.
        if credit>self.last_notified and remaining>self.reserve:
            self.notifier('EXTEND_TIMEOUT_USEC='+str(int((remaining-self.reserve)*1e6)))
            self.last_notified=credit
        return row
    def start(self):
        self.sample();self.notifier('READY=1')
        def loop():
            while not self.done.wait(1):
                try:self.sample()
                except Exception as exc:
                    self.failure=exc
                    atomic_json(self.job/'active-clock-server.json',{'run_id':self.run,'lease_epoch':self.epoch,
                        'failure':type(exc).__name__+': '+str(exc),'observed_monotonic':time.monotonic()})
                    return
        self.thread=threading.Thread(target=loop,name='bounded-active-clock',daemon=True);self.thread.start()
    def check(self):
        if self.failure:raise RuntimeError('Active clock failed') from self.failure
    def close(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=5)
        self.check();return self.sample()

def local_credit(output,run_id,attempt,maximum_elapsed):
    path=Path(output)/f'active-clock-local-{attempt:02d}.json'
    if not path.exists():return 0.
    row=read_small(path)
    lease=read_small(Path(output)/'remote-job.json')
    if (row['run_id'],row['attempt'],row['lease_epoch'])!=(run_id,attempt,lease['lease_epoch']) or lease['attempt']!=attempt:
        raise ValueError('Foreign local clock record')
    value=finite(row['verified_wait_seconds'])
    if not 0<=value<=maximum_elapsed+.05:raise ValueError('Local wait exceeds current attempt')
    return min(value,maximum_elapsed)
