"""Opt-in bounded Python call profiling; no fabricated CUDA activity metrics."""
from contextlib import contextmanager
import sys
import threading
import time

PHASES={
    ('rl_core','load_full_actor'):'initialize.model',
    ('training_checkpoint','load_training_checkpoint'):'restore.checkpoint',
    ('rl_resume','resume_journal'):'restore.journal',
    ('rl_collector','collect_series'):'collect.total',
    ('rl_collector','read'):'engine.IPC_read_wait',
    ('rl_collector','send'):'engine.IPC_write',
    ('rl_core','choose_decision'):'policy.decision',
    ('rl_core','make'):'features.encode',
    ('token_features','make'):'features.encode',
    ('rl_core','evaluate_decisions'):'PPO.evaluate',
    ('rl_core','ppo_step'):'PPO.total',
    ('ppo','ppo_step'):'PPO.total',
    ('model','forward_eval'):'policy.forward_submission',
    ('__init__','backward'):'PPO.backward_submission',
    ('adam','step'):'PPO.optimizer_submission',
    ('sampling_parity','before_optimizer'):'validation.CPU_GPU',
    ('scientific_resume','audit_runtime'):'validation.restore',
    ('training_checkpoint','publish_training_checkpoint'):'checkpoint.publish',
    ('tensor_io','atomic_torch_save'):'checkpoint.serialize_write',
    ('io','sha256'):'IO.hash',
    ('archive_barrier','retire_before_sampling'):'archive.barrier',
    ('archive_outbox','pending'):'archive.pending_scan',
    ('archive_outbox','backpressured'):'archive.backpressure_check',
    ('archive_outbox','receive_one'):'archive.transfer_ACK',
    ('training_control','wait'):'resource.guard_wait',
}


@contextmanager
def profile_window(trace, seconds=30):
    if not 0<seconds<=60:raise ValueError('Profiler window must be <=60 seconds')
    if sys.getprofile() is not None:raise ValueError('Existing profiler must not be replaced')
    deadline=time.perf_counter()+seconds;active={}
    def profile(frame,event,arg):
        now=time.perf_counter()
        if event=='call' and now<deadline:
            module=frame.f_globals.get('__name__','').rsplit('.',1)[-1]
            phase=PHASES.get((module,frame.f_code.co_name))
            if phase:active[id(frame)]=(phase,now)
        elif event=='return' and id(frame) in active:
            phase,start=active.pop(id(frame))
            with trace.lock:
                if len(trace.spans)<trace.maximum_spans:
                    trace.spans.append(dict(phase=phase,start=start-trace.start,end=now-trace.start,
                        thread=threading.get_ident(),python_profile=True,cuda_synchronized=False))
                else:trace.dropped+=1
    sys.setprofile(profile)
    try:yield
    finally:sys.setprofile(None)
