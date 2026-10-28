"""Controller-only native sampling/publication acceptance; no optimizer updates."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from common import WORK
import torch
from ptcg_lab.io import read_json,sha256,atomic_json
from ptcg_lab.server_workspace import owned_root,inside
from ptcg_lab.training_control import TrainingControl
from ptcg_lab.tensor_io import load_torch,atomic_torch_save
from ptcg_lab.bounded_checkpoint_audit import exact_state
from ptcg_lab.rl_core import RLFeatures,rng_state
from ptcg_lab.rl_collector import Engine
from ptcg_lab.public_agent import PublicRuleAgent
from fo.coordinator import Coordinator
from fo.archive_queue import freeze_state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ticket',type=Path,required=True)
    parser.add_argument('--kind',choices=['native','publication'],required=True);a=parser.parse_args()
    t=read_json(a.ticket);root=owned_root(t['project_root'],t['project_id']);job=inside(root,t['job_directory'])
    if sys.platform!='linux' or root!=Path('SERVER_PROJECT_ROOT'):
        raise ValueError('Controlled Linux HDD2 job required')
    if os.environ.get('PTCG_JOB_RUN_ID')!=t['run_id'] or job.name!=t['run_id'] or int(os.environ.get('PTCG_JOB_LEASE_EPOCH','-1'))!=t['lease_epoch']:
        raise ValueError('Job/lease differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.device_count()!=0:
        raise ValueError('CPU job must hide CUDA')
    if sha256(job/'training-manifest.raw.json')!=t['manifest_sha256']:raise ValueError('Manifest changed')
    m=read_json(job/'training-manifest.raw.json');s=m.get('framework_optimization_acceptance')
    if not s or not s.get('engineering_only') or t.get('framework_optimization_approved') is not True or s['kind']!=a.kind:
        raise ValueError('Explicit engineering admission missing')
    deadline=min(t['deadline_unix'],float(os.environ['PTCG_JOB_DEADLINE_UNIX']))
    if not 60<deadline-time.time()<=1800:raise ValueError('Engineering deadline differs')
    output=job/'framework-optimization';output.mkdir(exist_ok=True)
    control=TrainingControl(job,t['run_id'],t['lease_epoch'],deadline,
        lambda stage,**kw:atomic_json(output/'progress.json',dict(stage=stage,**kw)))
    def guard():control.wait(60,gpu_required=False)
    guard()
    if a.kind=='publication':
        # Run file publication and safety tests on the actual filesystem.
        # Their reports stay in this engineering job, not archived source inputs.
        from bench_publication import main as publication
        from ptcg_lab.bounded_io import environment_context
        from ptcg_lab.bounded_read import environment_read_context
        if environment_context() is None or environment_read_context() is None:
            raise ValueError('Controlled paced read/write contexts required')
        publication(output/'publication-results.json',output)
        return
    if not 30<=s['window_seconds']<=300:raise ValueError('Native window differs')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    path=inside(root,s['source_checkpoint_path'])
    if sha256(path)!=s['source_checkpoint_sha256']:raise ValueError('Parent checkpoint differs')
    saved=load_torch(path,map_location='cpu',weights_only=False)
    for key in ('model','optimizer','rng','version','next_series_id'):
        if key not in saved:raise ValueError('Full parent field absent: '+key)
    bundle=inside(root,s['bundle_path']);source=inside(root,s['source_path'])
    features=RLFeatures(bundle)
    from ptcg.rl.model import PTCGTransformer,arch_from_state_dict
    actor=PTCGTransformer(**arch_from_state_dict(saved['model'],heads=8),critic='blind')
    actor.load_state_dict(saved['model']);actor.eval()
    for parameter in actor.parameters():parameter.requires_grad_(False)
    deck=list(map(int,(bundle/'deck.csv').read_text(encoding='utf-8-sig').split()))
    opponent_spec=copy.deepcopy(s['fixed_opponent']);folder=inside(root,opponent_spec['path']);opponent_spec['path']=str(folder)
    for name,digest in opponent_spec['files'].items():
        if sha256(inside(folder,folder/name))!=digest:raise ValueError('Fixed opponent changed')
    reports=[]
    for workers in (1,2,3):
        guard();c=Coordinator(workers=workers,version=saved['version'],wave_seconds=60)
        start=time.monotonic();complete=decisions=forward_calls=0;wave=0;failure=None
        while time.monotonic()-start<s['window_seconds']:
            guard()
            if time.time()>=deadline-90:break
            def factory(sid,seat):
                opponent=PublicRuleAgent(opponent_spec,1-seat,s['seed']+sid)
                try:engine=Engine(root,source,output/f'engine-{workers}-{sid}.log',s['engine_manifest_sha256'])
                except BaseException:opponent.close();raise
                return engine,opponent,{'bucket':'fixed_diverse','opponent_id':opponent_spec['id'],'fixed':True}
            result=None;failure=None
            try:result=c.collect_wave(actor,features,deck,factory,guard)
            except (TimeoutError,ValueError,RuntimeError) as exc:failure=repr(exc)
            # Persist before any pending acknowledgement; retain latest two
            # small coordinator checkpoints and full inherited state by reference.
            state={'coordinator':c.state_dict(),'global_rng':rng_state(),
                'source_checkpoint':{'path':str(path),'sha256':s['source_checkpoint_sha256']},
                'fixed_opponent':opponent_spec,'scientific_volume_credit':0}
            destination=output/f'coordinator-{workers}-{wave:04d}.pt'
            atomic_torch_save(destination,freeze_state(state));loaded=load_torch(destination,map_location='cpu',weights_only=False)
            exact_state(freeze_state(state),loaded)
            restored=Coordinator(workers=workers);restored.load_state_dict(loaded['coordinator'])
            exact_state(c.state_dict(),restored.state_dict())
            n=sum(len(t['decisions']) for x in c.pending for t in x['trajectories'])
            decisions+=n;complete+=len(c.pending)
            if result:forward_calls+=result['forward_calls']
            # Never delete test checkpoints before an outer full archive ACK.
            # Bound the number of waves and artifacts instead.
            c.acknowledge_pending([x['series_id'] for x in c.pending]);wave+=1
            if failure or wave>=8:break
        elapsed=time.monotonic()-start
        reports.append({'workers':workers,'seconds_including_startup_save_restore':elapsed,'complete_BO3':complete,
            'sampled_decisions':decisions,'decisions_per_second':decisions/elapsed,'forward_calls':forward_calls,
            'wave_limit':8,'failure':failure,'new_actual_unique_optimized_decisions':0,
            'native_seed_exposed':False,'not_end_to_end_PPO_throughput':True})
        atomic_json(output/'native-results.json',{'results':reports,'full_source_checkpoint_sha256':sha256(path)})
    if sha256(path)!=s['source_checkpoint_sha256']:raise ValueError('Read-only source changed')


if __name__=='__main__':main()
