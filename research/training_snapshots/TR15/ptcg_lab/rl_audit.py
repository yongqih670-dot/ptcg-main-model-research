"""Bounded TR00 inheritance, blind PPO, real BO3 and restart audit."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .io import atomic_json,read_json,sha256,utcnow
from .public_agent import PublicRuleAgent
from .replay_data import extract_episode,actor_observation
from .rl_core import load_full_actor,choose_decision,evaluate_decisions,merged_scores,gae,optimizer_for,ppo_step,checkpoint,restore_rng
from .rl_collector import Engine,collect_series
from .token_features import forced_selection


def tensor_digest(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit_execution_profile(device):
    # Public opponents may reset process-wide Torch threading on import.
    # Normalize immediately before either side of a strict restart comparison.
    if device=='cpu':
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return {'threads':torch.get_num_threads(),'deterministic':torch.are_deterministic_algorithms_enabled()}


def run(config,output):
    output=Path(output);source=Path(__file__).resolve().parents[1];project=Path(config['project_path'])
    torch.set_num_threads(1 if config.get('audit_device')=='cpu' else 2)
    if config.get('audit_device')=='cpu':torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.monotonic();device=config.get('audit_device','cuda')
    if device not in ('cpu','cuda'):raise ValueError('Unknown audit device')
    if device=='cpu' and (os.name!='posix' or not os.environ.get('PTCG_PROJECT_ROOT') or not os.environ.get('PTCG_JOB_RUN_ID')):
        raise RuntimeError('CPU PPO audit is restricted to the controlled server training-check service')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA runtime unavailable')
    model,features,inheritance=load_full_actor(source/config['bundle_path'],config['parent_model_sha256'],device)
    optimizer=optimizer_for(model);cp=output/'checkpoint.pt'
    def progress(stage,**values):
        atomic_json(output/'progress.json',{'at':utcnow(),'stage':stage,'seconds':time.monotonic()-started,**values})
    if cp.exists():
        from .tensor_io import load_torch
        saved=load_torch(cp,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        profile=audit_execution_profile(device)
        if saved.get('execution_profile') is not None and saved['execution_profile']!=profile:
            raise AssertionError('Checkpoint execution profile differs before restored update')
        progress('restart_compare')
        update=ppo_step(model,optimizer,saved['decisions'],saved['advantages'],saved['returns'],features.b.STOP)
        expected=read_json(output/'expected-resume.json');actual=tensor_digest(model)
        if actual!=expected['model_sha256']:
            diagnostic={'at':utcnow(),'expected_model_sha256':expected['model_sha256'],'actual_model_sha256':actual,
                'cpu_threads':torch.get_num_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
            reference_path=output/'expected-next-model.pt'
            if reference_path.exists():
                reference=load_torch(reference_path,map_location='cpu',weights_only=True)
                diagnostic['different_tensors']=[{'name':name,'maximum_absolute_difference':float((value.detach().cpu().float()-reference[name].float()).abs().max())}
                    for name,value in model.state_dict().items() if not torch.equal(value.detach().cpu(),reference[name])]
            atomic_json(output/'restart-difference.json',diagnostic)
            raise AssertionError('Restored update differs from uninterrupted update')
        report=read_json(output/'audit.json')
        report.update(recovery={'checkpoint_sha256':sha256(cp),'exact_next_update_match':True,'resumed_update':update},
            complete=True,passed=True,stage='complete',finished_at=utcnow())
        atomic_json(output/'audit.json',report);atomic_json(output/'metrics.json',report)
        progress('complete');return
    progress('initial_parity')
    # Compare identical inputs on CPU against the original frozen adapter.
    spec=config['opponent'];parent=PublicRuleAgent(spec,0,config['seed']);parent.load(1)
    inherited_cpu,_,_=load_full_actor(source/config['bundle_path'],config['parent_model_sha256'],'cpu')
    episode=project/config['replay_path']
    if sha256(episode)!=config['replay_sha256']:raise ValueError('Audit replay changed')
    with gzip.open(episode,'rt',encoding='utf-8') as f:records,_=extract_episode(json.load(f),'tr00-inheritance-only')
    trackers={};forced={};maxdiff=0.;mask_checks=0;action_checks=0;gpu_maxdiff=0.;sample=None
    for r in records:
        obs=r['observation'];key=(r['seat'],r['round'])
        if key not in trackers:trackers[key]=features.Tracker(r['own_deck']);forced[key]=0
        tracker=trackers[key];tracker.update(obs)
        if forced_selection(obs['select']):forced[key]+=1;continue
        if len(obs['select']['option'])>features.b.STOP:continue
        rows,mapping=features.make(obs,tracker,forced[key],[])
        raw=np.zeros(features.b.OBS_SIZE,np.float32)
        features.encode(raw,obs,[],obs['select']['minCount']==0,forced[key],tracker=tracker)
        features.field(raw,'global_f')[features.b.GlobF.ACTOR_SEAT]=0
        np.testing.assert_array_equal(raw,rows[0]);mask_checks+=1
        with torch.no_grad():
            x=torch.from_numpy(rows);a=parent.module.SA.policy.forward_policy(x);b=inherited_cpu.forward_policy(x)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            maxdiff=max(maxdiff,float((a-b).abs().max()))
            c=model.forward_policy(x.to(device)).cpu()
            torch.testing.assert_close(a,c,rtol=1e-4,atol=2e-4)
            gpu_maxdiff=max(gpu_maxdiff,float((a-c).abs().max()))
        original_action=parent.module.SA._decompose(obs,tracker,forced[key])
        new_action,_=choose_decision(inherited_cpu,features,obs,tracker,forced[key],greedy=True)
        if original_action!=new_action:raise AssertionError('Parent greedy action mismatch')
        action_checks+=1;forced[key]=0;sample=(copy.deepcopy(obs),copy.deepcopy(tracker),r['own_deck'])
        if action_checks>=config['parity_decisions']:break
    if action_checks<config['parity_decisions']:raise AssertionError('Insufficient replay coverage')
    parent.close();del parent,inherited_cpu
    progress('blindness_and_conditional_actions',parity_decisions=action_checks)
    obs,tracker,deck=sample
    rows,mapping=features.make(obs,tracker,0,[])
    if config.get('write_actor_fixture'):
        from .tensor_io import atomic_torch_save
        with torch.no_grad():fixture_scores,fixture_values=model.forward_eval(torch.from_numpy(rows).to(device))
        atomic_torch_save(output/'actor-parity-fixture.pt',{'rows':rows,
            'scores':fixture_scores.cpu(),'values':fixture_values.cpu(),'parent_model_sha256':config['parent_model_sha256']})
    poisoned=copy.deepcopy(obs);poisoned.update(visualize={'opponent_hand':'secret'},search_begin_input='secret',future_draw=[1,2],reward=999)
    np.testing.assert_array_equal(rows,features.make(poisoned,tracker,0,[])[0])
    x=torch.from_numpy(rows).to(device);dirty=x.clone();a,z=features.b.OFFSETS['oracle'];dirty[:,a:z]=123
    with torch.no_grad():
        clean_output=model.forward_eval(x);dirty_output=model.forward_eval(dirty)
        for clean,poison in zip(clean_output,dirty_output):torch.testing.assert_close(clean,poison,rtol=0,atol=0)
    # Synthetic overflow must include every candidate and one legal STOP.
    large=copy.deepcopy(obs);large['select'].update(option=[{'type':0,'number':i} for i in range(140)],minCount=2,maxCount=4)
    rows,mapping=features.make(large,tracker,0,[2,133]);masks=np.stack([features.field(row,'opt_mask') for row in rows])
    scores=merged_scores(torch.tensor(masks),mapping,140,features.b.STOP)
    assert set(torch.where(scores==1)[0].tolist())==set(range(141))-{2,133}
    for lo,hi in [(0,0),(0,1),(1,1),(2,4),(4,4)]:
        large['select'].update(minCount=lo,maxCount=hi)
        action,decision=choose_decision(model,features,large,tracker,0)
        assert lo<=len(action)<=hi and len(action)==len(set(action))
        if decision:
            with torch.no_grad():logp,_,_=evaluate_decisions(model,[decision],features.b.STOP)
            sequence_error=abs(float(logp[0])-decision['old_logp'])
            with torch.no_grad():
                sequential=sum(float(evaluate_decisions(model,[{'steps':[step]}],features.b.STOP)[0][0]) for step in decision['steps'])
            diagnostic={'min':lo,'max':hi,'sampled':decision['old_logp'],
                    'batched':float(logp[0]),'sequential_replay':sequential,'abs_difference':sequence_error,
                    'cuda_tf32':torch.backends.cuda.matmul.allow_tf32,'steps':len(decision['steps'])}
            atomic_json(output/f'conditional-numeric-{lo}-{hi}.json',diagnostic)
            if abs(sequential-decision['old_logp'])>1e-10 or sequence_error>1e-4:
                raise AssertionError('Conditional logp discrepancy: '+str(sequence_error))
    toy=[{'old_value':0.,'reward':r,'terminal':t} for r,t in [(0,False),(1,True),(0,False),(-1,True)]]
    np.testing.assert_allclose(gae(toy)[0],[.95,1,-.95,-1],rtol=1e-6)
    np.testing.assert_allclose(gae([{'old_value':.25,'reward':0.,'terminal':False}],bootstrap=.5)[0],[.25])
    progress('real_official_bo3')
    atomic_json(output/'pre-collector-checks.json',{'at':utcnow(),'inheritance':inheritance,
        'parity_decisions':action_checks,'mask_checks':mask_checks,'cpu_logit_max_abs_diff':maxdiff,
        'gpu_logit_max_abs_diff':gpu_maxdiff,'blind_actor_critic_poison_test':'passed',
        'conditional_and_gae_checks':'passed'})
    deck=list(map(int,(source/config['bundle_path']/'deck.csv').read_text(encoding='utf-8-sig').split()))
    engine=Engine(project,source,output/'engine.log',config['engine_manifest_sha256'])
    all_decisions=[];series=[]
    try:
        for sid in range(config['series_count']):
            opponent=PublicRuleAgent(spec,1-sid%2,config['seed']+sid)
            trajectories,summary=collect_series(engine,model,features,deck,opponent,sid%2,sid,'TR00-parent',
                progress=lambda counts:progress('real_official_bo3',series=sid,**counts))
            summary['round_rewards']=[t['reward'] for t in trajectories]
            for trajectory in trajectories:
                if trajectory['decisions']:
                    assert trajectory['decisions'][-1]['terminal']
                    assert all(not d['terminal'] and d['reward']==0 for d in trajectory['decisions'][:-1])
                    all_decisions+=trajectory['decisions']
            series.append(summary);atomic_json(output/'series-audit.json',series)
            checkpoint(cp.with_name('collector-checkpoint.pt'),model,optimizer,decisions=all_decisions,series=series,stage='collecting')
    finally:engine.close()
    if len(all_decisions)<8:raise AssertionError('Insufficient real learner trajectory')
    progress('real_trajectory_ppo',decisions=len(all_decisions))
    # Complete rounds for GAE, then a fixed small minibatch for engineering gradients.
    advantages,returns=gae(all_decisions)
    indices=np.linspace(0,len(all_decisions)-1,min(16,len(all_decisions)),dtype=int)
    batch=[all_decisions[i] for i in indices];ad=advantages[indices];ret=returns[indices]
    with torch.no_grad():lp,_,_=evaluate_decisions(model,batch,features.b.STOP)
    error=max(abs(float(v)-d['old_logp']) for v,d in zip(lp,batch))
    if error>1e-4:raise AssertionError('On-policy old/new log probability mismatch')
    static_before=model.card_id_emb.weight.detach().clone()
    audit_execution_profile(device)
    update=ppo_step(model,optimizer,batch,ad,ret,features.b.STOP)
    static_delta=float((model.card_id_emb.weight-static_before).abs().max().detach())
    if static_delta<=0:raise AssertionError('Learned static embeddings did not update')
    assert not hasattr(model,'_frozen_tables')
    peak=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else 0.
    if peak>10:raise AssertionError('VRAM budget exceeded')
    report={'complete':False,'passed':False,'stage':'checkpointed_pending_restart','inheritance':inheritance,
        'audit_device':device,'cross_device_gpu_parity_verified':device=='cuda',
        'parity':{'input_and_mask_checks':mask_checks,'greedy_action_checks':action_checks,'cpu_logit_max_abs_diff':maxdiff,'gpu_logit_max_abs_diff':gpu_maxdiff},
        'blindness':{'forbidden_observation_fields_ignored':True,'oracle_poison_actor_and_value_equal':True,'oracle_aux_gradients_absent':True},
        'conditional':{'overflow_options':140,'unique_stop':True,'forced_and_optional_and_required_counts':'passed','sequence_logp_replay':'passed'},
        'trajectory':{'real_official_bo3':len(series),'learner_seats':[s['learner_seat'] for s in series],
            'complete_rounds':sum(s['rounds'] for s in series),'learner_decisions':len(all_decisions),'gae_terminal_and_bootstrap':'passed',
            'old_new_logp_max_abs_diff':error,'inference_seconds':sum(s['learner_inference_seconds'] for s in series)},
        'ppo_update':update,'static_embedding_max_change':static_delta,'peak_vram_gib':peak,
        'input_hashes':{'replay':sha256(episode),'engine_manifest':config['engine_manifest_sha256'],'deck':sha256(source/config['bundle_path']/'deck.csv')},
        'scope':'Engineering correctness only. Two short BO3 are not strength or promotion evidence.',
        'seconds_before_restart':time.monotonic()-started,'at':utcnow()}
    profile=audit_execution_profile(device)
    checkpoint(cp,model,optimizer,decisions=batch,advantages=ad,returns=ret,stage='before_second_update',execution_profile=profile)
    audit_execution_profile(device)
    ppo_step(model,optimizer,batch,ad,ret,features.b.STOP)
    if device=='cpu':
        from .tensor_io import atomic_torch_save
        atomic_torch_save(output/'expected-next-model.pt',model.state_dict())
    atomic_json(output/'expected-resume.json',{'model_sha256':tensor_digest(model),'checkpoint_sha256':sha256(cp)})
    atomic_json(output/'audit.json',report);atomic_json(output/'metrics.json',report)
    progress('checkpointed_pending_hard_stop_and_restart')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(read_json(a.config),a.output)


if __name__=='__main__':main()
