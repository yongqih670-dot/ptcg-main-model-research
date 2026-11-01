"""Opt-in bounded aggregate host timing. No profiler, RNG or CUDA operations.

Exclusive categories partition one process/thread wall clock. Inclusive spans
are explanatory only; processes and nested spans must never be summed.
"""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import os,threading,time

SPEC={'schema':1,'publish_interval_seconds':60,'clock':'monotonic','CUDA_synchronize':False,'profiler':False}
_timer=None

class Timer:
    def __init__(self,clock=time.monotonic):
        self.clock=clock;self.start=self.last=clock();self.owner=threading.get_ident()
        self.base='initialization_restore';self.stack=[];self.exclusive={};self.inclusive={};self.calls={}
        self.other_thread_calls=0;self.metadata={};self.last_publish=self.start;self.path=None;self.publishing=False
    def tick(self):
        now=self.clock();key=self.stack[-1][0] if self.stack else self.base
        self.exclusive[key]=self.exclusive.get(key,0.)+now-self.last;self.last=now
        return now
    @contextmanager
    def span(self,name):
        if threading.get_ident()!=self.owner or self.publishing:
            self.other_thread_calls+=int(threading.get_ident()!=self.owner);yield;return
        now=self.tick();self.stack.append((name,now));self.calls[name]=self.calls.get(name,0)+1
        try:yield
        finally:
            now=self.tick();key,began=self.stack.pop()
            self.inclusive[key]=self.inclusive.get(key,0.)+now-began
    def report(self):
        now=self.tick();inclusive=dict(self.inclusive)
        for key,began in self.stack:inclusive[key]=inclusive.get(key,0.)+now-began
        wall=now-self.start
        return {'schema':1,'pid':os.getpid(),'scope':'single_process_owner_thread_host_elapsed',
            'wall_seconds':wall,'exclusive_seconds':dict(self.exclusive),'inclusive_seconds_nonadditive':inclusive,
            'exclusive_partition_residual_seconds':wall-sum(self.exclusive.values()),'calls':dict(self.calls),
            'other_thread_calls_unattributed':self.other_thread_calls,'CUDA_kernel_timing_claimed':False,
            'cross_process_intervals_overlap_do_not_sum':True,'metadata':dict(self.metadata)}

def configure(output,spec,lane):
    global _timer
    if spec is None:return
    if spec!=SPEC:raise ValueError('Unreviewed phase timing options')
    if _timer is not None:raise RuntimeError('One timer per process')
    _timer=Timer();_timer.path=Path(output)/('phase-timing-'+lane+'.json')

@contextmanager
def span(name):
    if _timer is None:yield;return
    with _timer.span(name):yield

def timed(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args,**kwargs):
            if _timer is None:return fn(*args,**kwargs)
            with _timer.span(name):return fn(*args,**kwargs)
        return wrapped
    return decorate

def base(name):
    if _timer is not None:_timer.tick();_timer.base=name

def publish(force=False,**metadata):
    if _timer is None:return
    _timer.metadata.update(metadata)
    if not force and _timer.clock()-_timer.last_publish<60:return
    report=_timer.report();_timer.publishing=True
    try:
        from .io import atomic_json
        atomic_json(_timer.path,report)
    finally:_timer.publishing=False
    # Output overhead is explicitly assigned, not hidden in another phase.
    now=_timer.clock();_timer.exclusive['timing_publication']=_timer.exclusive.get('timing_publication',0.)+now-_timer.last
    _timer.last=_timer.last_publish=now
