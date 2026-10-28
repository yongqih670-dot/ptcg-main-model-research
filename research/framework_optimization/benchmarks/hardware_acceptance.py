"""Controller-only GPU replay payload. Does not create tickets or launch jobs.

The current dispatcher has no admission for this new payload. The sole research
controller must bind and admit it through its controlled engineering path first.
"""
import argparse
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from common import WORK
import numpy as np
import torch
from ptcg_lab.io import read_json,sha256,atomic_json
from ptcg_lab.server_workspace import owned_root,inside
from ptcg_lab.training_control import TrainingControl
from ptcg_lab.rl_core import evaluate_decisions as baseline,ppo_step as baseline_step,optimizer_for,UNUSED,rng_state,restore_rng
from fo.packed import evaluate_decisions
from fo.precision import Precision,parity
from fo.ppo import ppo_step
from fo.archive_queue import freeze_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--ticket',type=Path,required=True);a=p.parse_args()
    t=read_json(a.ticket)
    if sys.platform!='linux':raise RuntimeError('Server-only controlled acceptance')
    root=owned_root(t['project_root'],t['project_id']);job=inside(root,t['job_directory'])
    if root!=Path('SERVER_PROJECT_ROOT'):raise ValueError('Writable project root differs')
    if os.environ.get('PTCG_JOB_RUN_ID')!=t['run_id'] or job.name!=t['run_id']:
        raise ValueError('Controlled job identity differs')
    if int(os.environ.get('PTCG_JOB_LEASE_EPOCH','-1'))!=t['lease_epoch']:
        raise ValueError('Controlled lease differs')
    manifest_path=job/'training-manifest.raw.json'
    if sha256(manifest_path)!=t['manifest_sha256']:raise ValueError('Input manifest changed')
    m=read_json(manifest_path);spec=m.get('framework_optimization_acceptance')
    if not spec or spec.get('engineering_only') is not True or t.get('framework_optimization_approved') is not True:
        raise ValueError('Payload lacks explicit controller engineering admission')
    deadline=min(t['deadline_unix'],float(os.environ['PTCG_JOB_DEADLINE_UNIX']))
    if not 60<deadline-time.time()<=1800 or not 30<=spec['measurement_seconds']<=600:
        raise ValueError('Engineering deadline/window differs')
    if t['gpu_index'] not in (4,5,6,7) or os.environ.get('CUDA_VISIBLE_DEVICES')!=t['gpu_uuid'] or torch.cuda.device_count()!=1:
        raise ValueError('Exactly one authorized visible GPU required')
    output=job/'framework-optimization';output.mkdir(exist_ok=True)
    control=TrainingControl(job,t['run_id'],t['lease_epoch'],deadline,
        lambda stage,**kw:atomic_json(output/'progress.json',dict(stage=stage,**kw)))
    def guard():control.wait(60)
    guard();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    ref=inside(root,spec['fixture_path']);bundle=inside(root,spec['bundle_path'])
    if sha256(ref)!=spec['fixture_sha256']:raise ValueError('Fixture changed')
    value=torch.load(ref,map_location='cpu',weights_only=False)
    if not value.get('not_training_data'):raise ValueError('Engineering fixture flag missing')
    from ptcg_lab.rl_core import RLFeatures
    features=RLFeatures(bundle)
    from ptcg.rl.model import PTCGTransformer,arch_from_state_dict
    def new_model():
        guard();model=PTCGTransformer(**arch_from_state_dict(value['model'],heads=8),critic='blind')
        model.load_state_dict(value['model']);model.eval()
        for n,p in model.named_parameters():
            if n.startswith(UNUSED):p.requires_grad_(False)
        return model.cuda()
    d=value['fixture'];model=new_model();results={}
    with torch.no_grad():
        ref_out=baseline(model,d,features.b.STOP)
        cpu_reference=(value['fixture_logp'].cuda(),value['fixture_values'].cuda(),ref_out[2])
        if not parity(cpu_reference,ref_out)['passed']:raise ValueError('CPU/GPU FP32 reference failed')
    stop_time=min(time.time()+spec['measurement_seconds'],deadline-90)
    for mode in spec.get('modes',['fp32','bf16','fp16']):
        guard()
        try:precision=Precision(mode,'cuda')
        except ValueError as exc:
            results[mode]={'status':'unsupported','reason':str(exc)};continue
        fn=functools.partial(evaluate_decisions,precision=precision)
        try:
            with torch.no_grad():candidate=fn(model,d,features.b.STOP)
            comparison=parity(ref_out,candidate)
            if not comparison['passed']:
                results[mode]={'status':'rejected_numerical_gate','parity':comparison};continue
            # Real rows, deterministic engineering targets, disposable model.
            # This is one acceptance update; zero scientific volume credit.
            baseline_model=new_model();candidate_model=new_model()
            oa=optimizer_for(baseline_model);ob=optimizer_for(candidate_model)
            adv=np.linspace(-1,1,len(d),dtype=np.float32)
            ret=np.asarray([x['old_value'] for x in d],np.float32)+.1*adv
            settings=spec['ppo'];kw=dict(clip=settings['clip'],entropy_coef=settings['entropy_coefficient'],
                value_coef=settings['value_loss_coefficient'],max_grad_norm=settings['max_grad_norm'],
                target_kl=settings['target_kl'],normalize_advantages=False)
            for optimizer in (oa,ob):
                optimizer.param_groups[0]['lr']=settings['actor_learning_rate']
                optimizer.param_groups[1]['lr']=settings['value_head_learning_rate']
            ma=baseline_step(baseline_model,oa,d,adv,ret,features.b.STOP,**kw)
            mb=ppo_step(candidate_model,ob,d,adv,ret,features.b.STOP,evaluator=fn,precision=precision,**kw)
            if ma['skipped_for_kl'] or mb['skipped_for_kl']:
                results[mode]={'status':'update_not_verified_KL_skip','baseline':ma,'candidate':mb,'parity':comparison}
                continue
            grad=max(float((x.grad-y.grad).abs().max()) for x,y in zip(baseline_model.parameters(),candidate_model.parameters()) if x.grad is not None)
            parameter=max(float((x.detach()-y.detach()).abs().max()) for x,y in zip(baseline_model.parameters(),candidate_model.parameters()))
            accepted=(ma['skipped_for_kl']==mb['skipped_for_kl'] and abs(ma['loss']-mb['loss'])<=1e-4
                      and abs(ma['clip_fraction']-mb['clip_fraction'])==0 and grad<=2e-4 and parameter<=2e-5)
            if not accepted:
                results[mode]={'status':'rejected_update_gate','parity':comparison,'baseline':ma,'candidate':mb,'gradient_error':grad,'parameter_error':parameter};continue
            saved=freeze_state(dict(model=candidate_model.state_dict(),optimizer=ob.state_dict(),rng=rng_state(),precision=precision.state_dict()))
            snapshot=output/f'{mode}-engineering-state.pt';torch.save(saved,snapshot)
            restored=torch.load(snapshot,map_location='cpu',weights_only=False)
            candidate_model.load_state_dict(restored['model']);ob.load_state_dict(restored['optimizer']);precision.load_state_dict(restored['precision'])
            from ptcg_lab.bounded_checkpoint_audit import exact_state
            exact_state(freeze_state(candidate_model.state_dict()),restored['model'])
            exact_state(freeze_state(ob.state_dict()),restored['optimizer'])
            restore_rng(restored['rng'])
            # Actual RNG draws are compared across two restorations.
            def draws():
                import random
                return (random.random(),np.random.rand(4),torch.rand(4),torch.rand(4,device='cuda').cpu())
            draws1=draws();restore_rng(restored['rng']);draws2=draws();restore_rng(restored['rng'])
            if not (draws1[0]==draws2[0] and np.array_equal(draws1[1],draws2[1]) and all(torch.equal(x,y) for x,y in zip(draws1[2:],draws2[2:]))):
                raise ValueError('RNG restoration failed')
            del baseline_model,candidate_model,oa,ob
            samples={'baseline':[],'candidate':[]};torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(3):guard();fn(model,d,features.b.STOP)
                for repeat in range(9):
                    # AMP speed is compared with packed FP32 to isolate precision.
                    comparison_fn=baseline if mode=='fp32' else evaluate_decisions
                    for name,function in ( [('baseline',comparison_fn),('candidate',fn)] if repeat%2==0 else [('candidate',fn),('baseline',comparison_fn)] ):
                        guard()
                        if time.time()>=stop_time:raise TimeoutError('Bounded acceptance measurement window')
                        torch.cuda.synchronize();begin=time.perf_counter();function(model,d,features.b.STOP);torch.cuda.synchronize()
                        samples[name].append(time.perf_counter()-begin)
            results[mode]={'status':'replay_passed_not_training_admitted','parity':comparison,'baseline':ma,'candidate':mb,
                'gradient_error':grad,'parameter_error':parameter,'checkpoint_sha256':sha256(snapshot),'rng_roundtrip':True,
                'samples_seconds':samples,'median_seconds':{k:statistics.median(v) for k,v in samples.items()},
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            results[mode]['timing_baseline']='original_FP32' if mode=='fp32' else 'packed_FP32'
        except (ValueError,RuntimeError) as exc:
            results[mode]={'status':'rejected_error','reason':str(exc)}
        atomic_json(output/'replay-results.json',{'modes':results,'new_scientific_decisions':0,'end_to_end_speedup':None})
    atomic_json(output/'replay-results.json',{'modes':results,'new_scientific_decisions':0,'end_to_end_speedup':None})


if __name__=='__main__':main()
