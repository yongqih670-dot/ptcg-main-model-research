"""Deterministic CPU next-step proof from ONE actual committed partial state.

The input is a fixed closed engineering checkpoint, not a scientific restart.
Both branches continue the original PPO and leave scientific credit deferred.
"""
import copy,json,os,time
from common import *

def source():
    job=job_path('B3-candidate');points={}
    for path in (job/'objects').glob('manifest-*.bin'):
        point=read(path)['point'];point['manifest']=dict(path=path.relative_to(job).as_posix(),sha256=sha(path),bytes=path.stat().st_size);points[point['generation']]=point
    return job,points

def materialize(job,origin,points):
    from ptcg_lab.io_pacing import BytePacer
    pacer=BytePacer(4*2**20,float(os.environ['V3_DEADLINE']));refs={}
    for point in points.values():
        for ref in [point['model']['object'],point['tail'],point['manifest'],*point['references']]:refs[ref['path']]=ref
    for name,ref in refs.items():
        guard();src=origin/name;dst=job/name
        if sha(src)!=ref['sha256']:raise ValueError('Fixed partial input object changed')
        dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as reader,dst.open('xb') as writer:
            while data:=reader.read(2**20):guard();pacer.consume(len(data));writer.write(data)
            writer.flush();os.fsync(writer.fileno())
        if sha(dst)!=ref['sha256']:raise ValueError('New isolated partial input copy differs')
    index=read(origin/'recovery-v2.json');index['points']=[points[0],points[1]];write(job/'recovery-v2.json',index)

def execute(label,variant,tracer,*,native):
    import torch,numpy as np
    from ptcg_lab.framework_v2_training import semantic_digest
    from ptcg_lab.rl_core import load_full_actor,optimizer_for,gae,ppo_step
    import ptcg_lab.ppo_transaction as transaction
    from ptcg_lab.tensor_io import atomic_torch_save
    from ptcg_lab.ppo_accounting import update_decisions
    torch.use_deterministic_algorithms(True)
    origin,points=source();spec=read(V3/'reports/INPUTS.json');job=job_path(label)
    if job.exists():raise ValueError('New deterministic proof namespace required')
    job.mkdir(parents=True);write(job.parent/'.ptcg-owner.json',dict(project_id='framework-v3-isolated-engineering',uid=os.getuid() if hasattr(os,'getuid') else 0,root=str(job.parent.resolve())))
    write(job/'local-engineering-owner.json',dict(run_id=job.name,lease_epoch=1,parent_token=os.environ['V3_TOKEN'],scope='disposable fixed replay; no scientific authority'));setup_environment(job)
    point=points[1]
    parts=[]
    for ref in (point['model']['object'],point['tail']):
        if sha(origin/ref['path'])!=ref['sha256']:raise ValueError('Original partial object binding differs')
        parts.append(torch.load(origin/ref['path'],map_location='cpu',weights_only=False))
    value={**parts[0],**parts[1]};value.pop('v2_reference_map');producer=read(origin/'expected-tails.json')['1'];expected_tail=torch.load(origin/producer['path'],map_location='cpu',weights_only=False)
    if sha(origin/producer['path'])!=producer['sha256'] or value['state_semantic_sha256']!=expected_tail['state_semantic_sha256']:raise ValueError('Input not bound to its original producer expectation')
    check={k:v for k,v in value.items() if k!='state_semantic_sha256'}
    if semantic_digest(check)!=value['state_semantic_sha256']:raise ValueError('Input complete logical value differs from pre-serialization producer digest')
    strict_state(expected_tail,{k:v for k,v in value.items() if k not in ('model','optimizer')})
    backend=None
    if native:
        materialize(job,origin,{0:points[0],1:points[1]});backend,_=native_backend(job,variant);restored,_=backend.restore();strict_state(value,restored);value=restored
    decisions=[]
    for row in spec['trajectories'][:2]:
        path=ROOT/row['trajectory']['path']
        if sha(path)!=row['trajectory']['sha256']:raise ValueError('Frozen whole BO3 changed')
        decisions.extend(torch.load(path,map_location='cpu',weights_only=False)['decisions'])
    settings=spec['ppo'];bundle=ROOT/spec['current_bundle'];model,features,_=load_full_actor(bundle,sha(bundle/'model.pt'),reset_value=False)
    opt=optimizer_for(model,settings['actor_learning_rate'],settings['value_head_learning_rate']);model.load_state_dict(value['model'],strict=True);opt.load_state_dict(value['optimizer']);cpu_rng_restore(value['rng'])
    partial=copy.deepcopy(value['partial_ppo']);transaction.validate(partial,len(decisions),settings['epochs']);adv,ret=gae(decisions,settings['gamma'],settings['gae_lambda']);adv=(adv-adv.mean())/(adv.std()+1e-8)
    if not np.array_equal(adv,partial['advantages']) or not np.array_equal(ret,partial['returns']):raise ValueError('Original full GAE differs')
    next_state=None;executed=0
    try:
        while not partial['finished']:
            guard();chosen=transaction.next_batch(partial,settings['epochs'])
            metrics=ppo_step(model,opt,[decisions[i] for i in chosen],partial['advantages'][chosen],partial['returns'][chosen],features.b.STOP,
                clip=settings['clip'],entropy_coef=settings['entropy_coefficient'],value_coef=settings['value_loss_coefficient'],max_grad_norm=settings['max_grad_norm'],target_kl=settings['target_kl'],normalize_advantages=False,
                critic_updates_shared_trunk=settings.get('critic_updates_shared_trunk',True),record_gradient_components=False)
            transaction.acknowledge(partial,chosen,metrics,settings['epochs'],0);executed+=int(not metrics['skipped_for_kl'])
            if next_state is None:next_state=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=full_rng(value),partial=copy.deepcopy(partial),chosen=chosen.copy())
        final=dict(value,model=model.state_dict(),optimizer=opt.state_dict(),rng=full_rng(value),partial_ppo=partial,model_state_revision=value['model_state_revision']+executed);final.pop('state_semantic_sha256',None)
        expected=expected_value(final,final['league'],final['model_state_revision'])
        if backend:
            archive=dict(prefix_path='prefix.json',prefix_sha256=sha(job/'prefix.json'))
            for _ in range(2):backend.save(final,archive,final['league'],model_revision=final['model_state_revision'])
        atomic_torch_save(job/'expected.pt',expected,replace=False);atomic_torch_save(job/'next-step.pt',next_state,replace=False)
        credit=update_decisions(dict(decisions=len(decisions),steps=partial['logs']))
        return dict(interrupted=False,complete_BO3=2,decisions=len(decisions),first_epoch_credit=credit,positive_optimizer_steps=partial['positive_optimizer_steps'],newly_executed_optimizer_steps=executed,
            expected_sha256=sha(job/'expected.pt'),next_step_sha256=sha(job/'next-step.pt'),scientific_credit=0,CPU_deterministic_algorithms=True,input_model_revision=1,input_producer_semantic_sha256=value['state_semantic_sha256'],
            full_input_state_bound_to_original_pre_serialization_producer=True,Native_restore=native,complete_update_ready_for_commit_scientific_counters_unchanged=True)
    finally:
        if backend:backend.close()
