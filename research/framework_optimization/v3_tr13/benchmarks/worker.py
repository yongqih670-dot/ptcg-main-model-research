"""Finite, disposable local fixed-input acceptance; never starts an engine."""
import argparse,os,sys,time
from pathlib import Path
from common import *

def zero(label,variant):
    import torch
    job,spec=prepare_job(label);setup_environment(job)
    backend,_=native_backend(job,variant)
    try:
        parent=legacy_parent(spec);league=read(job/'league.json')
        archive={'prefix_path':'prefix.json','prefix_sha256':sha(job/'prefix.json')}
        for _ in range(3):
            guard();backend.save(parent,archive,league,model_revision=0)
            write(job/'progress.json',dict(stage='zero_checkpoint',at=time.time()))
        expected=expected_value(parent,league);torch.save(expected,job/'expected.pt')
        return dict(passed=True,expected_sha256=sha(job/'expected.pt'),points=read(job/'recovery-v2.json')['points'],scientific_decisions=0)
    finally:backend.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['zero','cold','qual','recover-qual','interrupt','resume','perf','replay-cold','replay-cold-repair','det-reference','det-native','seal-failed']);p.add_argument('label');p.add_argument('--variant',choices=['baseline','candidate'],default='candidate');a=p.parse_args()
    sys.path.insert(0,str(V3/a.variant));import psutil,torch
    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[-3:]);torch.set_num_threads(2);admitted_worker()
    if a.mode=='perf':torch.use_deterministic_algorithms(True)
    start=time.monotonic()
    if a.mode=='zero':result=zero(a.label,a.variant)
    elif a.mode=='seal-failed':
        from seal_failure import seal
        result=seal(a.label);write(V3/'reports'/f'{a.label}-final.json',result)
    elif a.mode in ('cold','replay-cold','replay-cold-repair'):
        job=job_path(a.label);warm=read(V3/'reports'/f'{a.label}-zero.json') if a.mode=='cold' else read(V3/'reports'/f'{a.label}-final.json')
        if sha(job/'expected.pt')!=warm['expected_sha256']:raise ValueError('Expected state changed')
        result=cold_points(job,a.variant)
    else:
        from trace import Trace,install
        from replay import execute
        tracer=Trace();install(tracer)
        if a.mode.startswith('det-'):
            from seam_repair import execute as deterministic
            result=deterministic(a.label,a.variant,tracer,native=a.mode=='det-native')
        else:result=execute(a.label,a.variant,tracer,count=48 if a.mode=='perf' else 2,control_calls=a.mode=='perf',interrupt=a.mode=='interrupt',restore=a.mode in ('resume','recover-qual'),capture_step=2 if a.mode in ('qual','recover-qual') else 1)
        result['trace']=tracer.report()
        if not result['interrupted']:write(V3/'reports'/f'{a.label}-final.json',result)
    result.update(mode=a.mode,label=a.label,variant=a.variant,seconds=time.monotonic()-start)
    write(V3/'reports'/f'{a.label}-{a.mode}.json',result);print(__import__('json').dumps(result,default=str))
if __name__=='__main__':main()
