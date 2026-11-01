"""Finite server CPU zero-update smoke entry; requires original controller ticket.

This does not run a native engine, an optimizer, or a stopped scientific loop.
Native/GPU fault acceptance remains an explicit separate controller gate.
"""
import argparse,os,time
from pathlib import Path
import torch
from .fo_v2.durable import safe
from .framework_v3_runtime import from_config
from .framework_v3_driver import components
from .framework_v2_training import semantic_digest
from .io import read_json,sha256,atomic_json
from .framework_v3_exit import evidence,ObservedCall

def run(config,output):
    output=Path(output).resolve();from_config(config,output)
    if config.get('mode')!='framework_v3_zero_update' or config.get('scientific_credit')!=0:raise ValueError('Finite zero-update mode required')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if not 0<deadline-time.time()<=1800:raise ValueError('Original finite1800s engineering hard endpoint required')
    if config.get('torch_threads')!=2:raise ValueError('Initial CPU envelope differs')
    torch.set_num_threads(2);epoch=int(os.environ['PTCG_JOB_LEASE_EPOCH'])
    parent=safe(output,config['parent_input']['path'])
    if sha256(parent)!=config['parent_input']['sha256']:raise ValueError('Registered full parent input changed')
    observer=evidence(output,output.name,epoch,config['correlation'],'worker')
    with ObservedCall(observer):
        from .fo_v2.io_budget import ReadEnvelope
        def stop():
            if time.time()>=deadline-60:raise TimeoutError('Original finite engineering endpoint reserve')
        with ReadEnvelope(4*2**20,stop).open(parent) as stream:value=torch.load(stream,map_location='cpu',weights_only=False)
        league=read_json(output/'league.json')
        expected=dict(value,league=league,model_state_revision=value.get('model_state_revision',0));expected.pop('state_semantic_sha256',None);expected=semantic_digest(expected)
        kw=dict(capacity_root=Path(os.environ['PTCG_PROJECT_ROOT']),hard_bytes=config['project_hard_bytes'],progress=lambda *a,**k:None)
        backend,control,outbox,publisher=components(config,output,epoch,config['checkpoint_profile'],deadline,**kw)
        try:
            for _ in range(3):
                control.wait(60,gpu_required=False)
                if deadline-time.time()<60:raise TimeoutError('Do not consume reserved commit time')
                commit=backend.save(value,config['archive'],league,model_revision=value.get('model_state_revision',0));outbox.queue_recovery(commit)
        finally:backend.close()
        # New object cache and graph validation; a separate controlled process
        # cold restore and full ACK barrier are still required by acceptance A.
        backend,_,_,_=components(config,output,epoch,config['checkpoint_profile'],deadline,**kw)
        try:
            points=[]
            for index in (0,1):
                restored,point=backend.restore(index);semantic=restored.pop('state_semantic_sha256')
                if semantic!=expected or semantic_digest(restored)!=expected:raise ValueError('Full state differs on fresh object-cache restore')
                points.append(point['generation'])
        finally:backend.close()
        atomic_json(output/'framework-v3-zero-update.json',dict(passed=True,points=points,full_semantic_sha256=expected,optimizer_steps=0,scientific_credit=0,
                    fresh_object_cache=True,separate_process_cold_restore=False,full_ACK_barrier_pending=True,native_GPU_fault_gate_pending=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();run(read_json(args.config),args.output)
if __name__=='__main__':main()
