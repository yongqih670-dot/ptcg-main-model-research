"""Housekeeping gates outside official agent.act timers; official run clock stays live."""
import time


class MaintenanceClosed(RuntimeError):
    pass


def install_step_windows(env,emit,read,diagnostic):
    sequence=0
    def window():
        nonlocal sequence
        emit({'event':'maintenance_window','protocol':'step_boundary_v1','sequence':sequence})
        began=time.monotonic();request=read()
        diagnostic['maintenance_seconds']+=time.monotonic()-began
        if request.get('command')=='close':raise MaintenanceClosed('Owned engine closed between actions')
        if request!={'command':'continue','sequence':sequence}:raise ValueError('Maintenance window acknowledgement differs')
        sequence+=1;diagnostic['maintenance_windows']=sequence
    original=env.step
    def step(actions,logs=None):
        result=original(actions,logs)
        if not env.done:window()
        return result
    env.step=step
    window() # Before the first official agent invocation, too.
