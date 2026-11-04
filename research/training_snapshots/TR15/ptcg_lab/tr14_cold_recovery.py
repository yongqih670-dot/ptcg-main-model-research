"""Fresh server process restores both v2 points and checks full semantic values."""
import argparse
import os
from pathlib import Path
import torch
from .io import read_json,atomic_json,utcnow
from .framework_v2_training import NativeCheckpoints,semantic_digest
from .server_workspace import owned_root,inside
from .tr14_scope import validate_remote
from .rl_core import load_full_actor,optimizer_for,restore_rng,rng_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);a=p.parse_args()
    ticket=read_json(a.ticket);root=owned_root(ticket['project_root'],ticket['project_id'])
    job=inside(root,ticket['job_directory'])
    validate_remote(ticket['TR13_runtime_scope'],ticket['mode'],ticket['new_diagnostic_identity'])
    if (os.environ.get('PTCG_JOB_RUN_ID'),os.environ.get('PTCG_JOB_LEASE_EPOCH'))!=(job.name,str(ticket['lease_epoch'])):
        raise ValueError('Cold recovery run/lease differs')
    config=read_json(job/'payload-config.json');torch.set_num_threads(2)
    reuse=config.get('acceptance_reuse')
    from .training_control import TrainingControl
    control=TrainingControl(job,job.name,ticket['lease_epoch'],float(os.environ['PTCG_JOB_DEADLINE_UNIX']),lambda *a,**k:None)
    control.wait(90)
    backend=NativeCheckpoints(job,ticket['lease_epoch'],config['framework_v2_checkpoint_cadence'],capacity_root=root,
                              retained_bytes=config.get('readonly_retained_bytes',0))
    results=[]
    try:
        if len(backend.bridge.checkpoints.points)!=2:raise ValueError('Two reliable recovery points required')
        model,features,_=load_full_actor(job/'control'/config['bundle_path'],config['parent_model_sha256'],'cuda')
        settings=config['ppo'];optimizer=optimizer_for(model,settings['actor_learning_rate'],settings['value_head_learning_rate'])
        for which in ((-1,) if reuse else (0,1)):
            control.wait(90)
            value,point=backend.restore(which)
            expected=value.pop('state_semantic_sha256')
            if semantic_digest(value)!=expected:raise ValueError('Cold full state semantic hash differs')
            if reuse:
                from .tr14_acceptance_reuse import check_state
                check_state(value,reuse)
            model.load_state_dict(value['model']);optimizer.load_state_dict(value['optimizer']);restore_rng(value['rng'])
            if semantic_digest(model.state_dict())!=semantic_digest(value['model']):raise ValueError('CUDA model values differ')
            if semantic_digest(optimizer.state_dict())!=semantic_digest(value['optimizer']):raise ValueError('CUDA optimizer values differ')
            if semantic_digest(rng_state())!=semantic_digest(value['rng']):raise ValueError('Restored all RNG streams differ')
            results.append({'generation':point['generation'],'cursor':point['cursor'],
                'unique_PPO_trained_decisions':value['unique_PPO_trained_decisions'],
                'partial_PPO_preserved':value.get('partial_ppo') is not None,
                'full_CPU_and_GPU_values_RNG_exact':True,'semantic_sha256':expected})
        report={'at':utcnow(),'complete':True,'fresh_process':True,'optimizer_steps_executed':0,
            'no_scientific_decision_credit':True,'two_recovery_points_retained':True}
        if reuse:report.update(checked_points=results,latest_only=True,reused_acceptance=reuse['reference'])
        else:report['two_points']=results
        atomic_json(job/'framework-v2-cold-recovery.json',report)
    finally:backend.close()


if __name__=='__main__':main()
