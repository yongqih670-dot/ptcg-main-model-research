"""Explicit role hooks for a separately admitted engineering driver.

No signal handler does I/O. EOF observation is not asserted to be its cause.
No retry, disconnect timeout or owned-process stop behavior is changed.
"""
import os,time
from .framework_v3_runtime import ExitEvidence

class RoleObserver:
    def __init__(self,evidence):self.evidence=evidence;self.failure=None;self.ended=False;self.count=0;self.last_operation=None;self.emit('role_started')
    def emit(self,event,**kw):
        try:return self.evidence.record(event,**kw)
        except Exception as error:self.failure=repr(error);return None
    def failed(self,error):
        self.emit('role_exception',observed_cause='peer_stream_closed' if isinstance(error,(EOFError,BrokenPipeError)) else 'uncaught_exception',
                  exception_type=type(error).__name__,exception=str(error)[:1024],last_operation=self.last_operation,completed_calls=self.count)
        self.ended=True
    def finish(self):
        if not self.ended:self.emit('role_function_returned',completed_calls=self.count,last_operation=self.last_operation);self.ended=True
        if self.failure:raise RuntimeError('Persistent exit evidence failed: '+self.failure)

class ObservedStream:
    """Proxy the real receiver/controller channel without altering its stop rules."""
    def __init__(self,stream,controller,receiver):self.stream=stream;self.controller=controller;self.receiver=receiver
    def __getattr__(self,name):return getattr(self.stream,name)
    def request(self,operation,*a,**kw):
        role=self.receiver if operation in ('archive_pending','archive_chunk','archive_ack') else self.controller
        role.last_operation=operation
        try:result=self.stream.request(operation,*a,**kw)
        except BaseException as error:role.failed(error);raise
        role.count+=1;return result
    def close(self):
        try:result=self.stream.close()
        except BaseException as error:
            self.controller.failed(error)
            try:self.receiver.finish()
            except BaseException:pass
            raise
        failures=[]
        for role in (self.controller,self.receiver):
            try:role.finish()
            except BaseException as error:failures.append(error)
        if failures:raise failures[0]
        return result

def evidence(root,run_id,epoch,correlation,role):
    import psutil
    proc=psutil.Process();return RoleObserver(ExitEvidence(root,run_id,epoch,correlation,role,process_identity={'pid':os.getpid(),'created':proc.create_time()}))

class ObservedCall:
    def __init__(self,observer):self.observer=observer
    def __enter__(self):return self.observer
    def __exit__(self,kind,error,tb):
        if error is not None:self.observer.failed(error)
        else:self.observer.finish()
        return False
