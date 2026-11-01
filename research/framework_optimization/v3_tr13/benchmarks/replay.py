"""Actual TR13 components on immutable whole-BO3 tapes, CPU engineering only."""
import copy,importlib.util,os,re,sys,time
from pathlib import Path
from common import *

def observer_module():
    # Common observation adapter is identical on both arms, not a baseline hot
    # path optimization. It imports the selected arm's unchanged durable I/O.
    for name in ('framework_v3_runtime','framework_v3_exit'):
        key='ptcg_lab.'+name
        if key not in sys.modules:
            spec=importlib.util.spec_from_file_location(key,V3/'candidate/ptcg_lab'/f'{name}.py');module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
    return sys.modules['ptcg_lab.framework_v3_exit']

def stage_tape(row,job,admission=None):
    from ptcg_lab.io_pacing import BytePacer
    pace=BytePacer(4*2**20,float(os.environ['V3_DEADLINE']))
    for key,name in [('trajectory',row['summary_value']['trajectory_path']),('summary',f"series/{row['series_id']:06d}.json")]:
        target=job/name;source=ROOT/row[key]['path'];digest=row[key]['sha256']
        if target.exists():
            if sha(target)!=digest:raise ValueError('Initial immutable file differs')
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        import contextlib
        with admission.publication(target,row[key]['bytes']) if admission is not None else contextlib.nullcontext():
            with source.open('rb') as src,target.open('xb') as dst:
                while data:=src.read(2**20):guard();pace.consume(len(data));dst.write(data)
                dst.flush();os.fsync(dst.fileno())
        if sha(target)!=digest:raise ValueError('Staged trajectory/summary changed')

class Replay:
    def __init__(self,job,spec,variant,tracer,*,restore=False):
        import torch
        from ptcg_lab.io import atomic_json
        from ptcg_lab.training_control import TrainingControl
        from ptcg_lab.shared_capacity import AUTH,PROFILE
        from ptcg_lab.archive_outbox import ArchiveOutbox
        from ptcg_lab.rl_core import load_full_actor,optimizer_for
        from ptcg_lab.framework_v2_training import advance_native_state
        self.job=job;self.spec=spec;self.variant=variant;self.trace=tracer;setup_environment(job)
        self.backend,self.flags=native_backend(job,variant);self.parent=legacy_parent(spec);self.settings=spec['ppo'];self.league=read(job/'league.json');self.archive={'prefix_path':'prefix.json','prefix_sha256':sha(job/'prefix.json')}
        if restore:
            owner_path=job/'local-engineering-owner.json';owner=read(owner_path)
            if owner['scope']!='disposable fixed replay; no scientific authority':raise ValueError('No local recovery authority')
            if owner['parent_token']!=os.environ['V3_TOKEN']:
                # Fresh Native constructor has independently verified the full
                # recovery graph. Retain old local admission bytes, then bind
                # the closed disposable case to this finite parent lock.
                backup=job/'previous-local-engineering-owner.json'
                if backup.exists():raise ValueError('Recovery admission already replaced; inspect evidence')
                backup.write_bytes(owner_path.read_bytes());owner['parent_token']=os.environ['V3_TOKEN'];write(owner_path,owner)
        self.origin=read(job/'prefix.json');self.state=dict(self.parent,league=self.league,model_state_revision=0);self.decisions=[];self.shards=[];self.revision=0
        bundle=ROOT/spec['current_bundle']
        if sha(bundle/'ptcg/rl/model.py')!=spec['actor_source_sha256']:raise ValueError('Actor source changed')
        self.model,self.features,_=load_full_actor(bundle,sha(bundle/'model.pt'),reset_value=False)
        self.optimizer=optimizer_for(self.model,self.settings['actor_learning_rate'],self.settings['value_head_learning_rate'])
        self.epoch=1
        kw={'optimizations':self.flags} if variant=='candidate' else {}
        # Local fixture admission, equally applied: it is not a TR13 ticket.
        from local_adapter import local_outbox
        self.outbox=local_outbox(job,1,optimizations=self.flags if variant=='candidate' else None,prefix_sha256=self.archive['prefix_sha256'])
        atomic_json(job/'ticket.json',dict(shared_capacity=PROFILE,run_id=job.name,lease_epoch=1,gpu_uuid=None,engineering_only=True,new_scientific_training=False))
        atomic_json(job/'resource-training-control.json',dict(run_id=job.name,lease_epoch=1,sequence=0,command='run',reason='fixed sufficient local CPU capacity',authority=AUTH,torch_threads=2,GPU_co_tenant_mode=False))
        self.control=TrainingControl(job,job.name,1,float(os.environ['V3_DEADLINE']),lambda *a,**k:None,**kw)
        self.publish=atomic_json
        if variant=='candidate' and self.flags['progress_coalescing']:
            from ptcg_lab.framework_v3_runtime import ProgressPublisher
            self.publish=ProgressPublisher(atomic_json)
        obs=observer_module();correlation='fixed-'+job.parent.name
        self.worker=obs.evidence(job,job.name,1,correlation,'worker');self.controller=obs.evidence(job,job.name,1,correlation,'controller');self.receiver=obs.evidence(job,job.name,1,correlation,'receiver')
        outer=self
        from local_adapter import install_archive_publication_adapter
        install_archive_publication_adapter()
        class LocalStream:
            def request(this,operation,payload,*,timeout=None):
                if timeout is not None and not 0<timeout<=30:raise ValueError('Original bounded local transfer timeout envelope differs')
                guard()
                if operation=='archive_ack':return outer.outbox.accept_ack(payload['ack'])
                if operation=='archive_chunk':return outer.outbox.chunk(payload['object_sequence'],payload['offset'],payload['count'])
                raise ValueError('Unexpected fixture operation')
            def close(this):return None
        self.stream=obs.ObservedStream(LocalStream(),self.controller,self.receiver)
        self.catalog={k:ROOT/v['path'] for k,v in spec['immutable_catalog'].items()}
        self.archive_root=job.parent/'archive';self.partial=None;self.next_step=None
        if restore:
            self.state,_=self.backend.restore();self.league=self.state['league'];self.revision=self.state['model_state_revision'];self.partial=self.state['partial_ppo']
            self.shards=list(read(job/'replay-cursor.json')['shards']) if (job/'replay-cursor.json').exists() else [r['trajectory_path'] for r in sorted(self.state['archive_state']['live_expected'],key=lambda r:r['series_id'])]
            from ptcg_lab.fo_v2.io_budget import ReadEnvelope
            for name in self.shards:
                with ReadEnvelope(4*2**20,guard).open(job/name) as f:self.decisions.extend(torch.load(f,map_location='cpu',weights_only=False)['decisions'])
        self.model.load_state_dict(self.state['model'],strict=True);self.optimizer.load_state_dict(self.state['optimizer']);cpu_rng_restore(self.state['rng'])
        self.sampler_rng=self.state['cpu_sampler_rng'];self.cursor=self.state['next_series_id']

    def drain(self):
        from ptcg_lab.archive_outbox import receive_one
        with self.trace.span('local_archive_receive_verify_and_ACK'):
            for row in self.outbox.records():
                if self.outbox._ack(row) is None:
                    ack=receive_one(self.stream,self.archive_root,row,self.catalog.get(row['sha256']),storage_codec='gzip')
                    # Match actual TR13 drain_once: reuse verified compressed
                    # bytes within THIS case, always mint a new fenced ACK.
                    self.catalog[row['sha256']]={'archive_root':self.archive_root,'ack':ack}
            if self.outbox.pending()['pending_objects']:raise ValueError('Missing complete archive ACK')

    def progress(self,stage,**extra):
        guard();self.publish(self.job/'progress.json',dict(stage=stage,at=time.time(),cursor=self.cursor,**extra))

    def save(self):
        from ptcg_lab.framework_v2_training import advance_native_state
        cursor={k:self.state[k] for k in ('version','trained_decisions')};cursor['next_series_id']=self.cursor
        archive_state=advance_native_state(self.backend,self.state['archive_state'],self.origin,self.archive,cursor,self.job.name,1)
        value=dict(self.state,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),rng=full_rng(self.parent),cpu_sampler_rng=self.sampler_rng,
                   next_series_id=self.cursor,archive_state=archive_state,partial_ppo=self.partial,model_state_revision=self.revision)
        value.pop('state_semantic_sha256',None);self.state=value
        commit=self.backend.save(value,self.archive,self.league,model_revision=self.revision)
        from ptcg_lab.tensor_io import atomic_torch_save
        expected=expected_value(value,self.league,self.revision);tail={k:v for k,v in expected.items() if k not in ('model','optimizer')}
        name=f"expected-tail-{commit['generation']:04d}.pt";atomic_torch_save(self.job/name,tail,replace=False)
        index=read(self.job/'expected-tails.json') if (self.job/'expected-tails.json').exists() else {};index[str(commit['generation'])]=dict(path=name,sha256=sha(self.job/name));write(self.job/'expected-tails.json',index)
        self.outbox.queue_recovery(commit);self.drain();self.progress('checkpoint_complete',generation=commit['generation'])
        write(self.job/'replay-cursor.json',dict(shards=self.shards));return commit

    def ingest(self,count,*,control_calls):
        import torch
        from ptcg_lab.fo_v2.io_budget import ReadEnvelope
        scheduled=set(read(V3/'reports/PERFORMANCE_PROTOCOL.json').get('ingest_checkpoint_BO3',range(8,count+1,8))) if control_calls else set(range(8,count+1,8))
        for index,row in enumerate(self.spec['trajectories'][:count]):
            for n in range(row['control_calls'] if control_calls else 0):
                self.control.wait(gpu_required=False)
                if (n+1)%50==0:self.progress('collecting',engine_decisions=n+1)
            with self.trace.span('fixed_trajectory_copy_and_load'):
                stage_tape(row,self.job,getattr(self.backend,'write_admission',None));name=row['summary_value']['trajectory_path']
                with ReadEnvelope(4*2**20,guard).open(self.job/name) as f:tape=torch.load(f,map_location='cpu',weights_only=False)
            if tape['series_id']!=row['series_id'] or len(tape['decisions'])!=row['summary_value']['learner_decisions']:raise ValueError('Whole BO3 input differs')
            self.decisions.extend(tape['decisions']);self.shards.append(name);self.cursor=max(self.cursor,row['series_id']+1);cpu_rng_restore(tape['rng_after']);self.sampler_rng=tape.get('cpu_sampler_rng_after',self.sampler_rng)
            self.outbox.queue_series(row['summary_value']);self.drain()
            # Equal deterministic startup cadence on both arms. The invalid
            # first protocol omitted initial archive drain from its schedule.
            if index+1 in scheduled:self.save()
        self.save()

    def optimize(self,*,interrupt=False,capture_step=1):
        import numpy as np,torch
        import ptcg_lab.ppo_transaction as transaction
        from ptcg_lab.rl_core import gae,ppo_step
        from ptcg_lab.ppo_accounting import update_decisions
        from ptcg_lab.tensor_io import atomic_torch_save
        settings=self.settings
        if self.partial is None:
            advantages,returns=gae(self.decisions,settings['gamma'],settings['gae_lambda']);advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
            self.partial=transaction.begin(self.decisions,advantages,returns,settings['minibatch_initial'])
        transaction.validate(self.partial,len(self.decisions),settings['epochs'])
        steps_this_process=0
        while not self.partial['finished']:
            guard();chosen=transaction.next_batch(self.partial,settings['epochs'])
            with self.trace.span('PPO_host_optimizer_step'):
                metrics=ppo_step(self.model,self.optimizer,[self.decisions[i] for i in chosen],self.partial['advantages'][chosen],self.partial['returns'][chosen],self.features.b.STOP,
                    clip=settings['clip'],entropy_coef=settings['entropy_coefficient'],value_coef=settings['value_loss_coefficient'],max_grad_norm=settings['max_grad_norm'],target_kl=settings['target_kl'],normalize_advantages=False,
                    critic_updates_shared_trunk=settings.get('critic_updates_shared_trunk',True),record_gradient_components=self.partial['epoch']==0 and self.partial['offset']==0)
            transaction.acknowledge(self.partial,chosen,metrics,settings['epochs'],0)
            if not metrics['skipped_for_kl']:self.revision+=1
            steps_this_process+=1
            if steps_this_process==capture_step:
                self.next_step=dict(model=copy.deepcopy(self.model.state_dict()),optimizer=copy.deepcopy(self.optimizer.state_dict()),rng=full_rng(self.parent),partial=copy.deepcopy(self.partial),chosen=chosen.copy())
            first=not metrics['skipped_for_kl'] and self.partial['positive_optimizer_steps']==1
            if first:
                self.save()
                if interrupt:
                    if hasattr(self.publish,'flush'):self.publish.flush()
                    return dict(interrupted=True,partial_offset=self.partial['offset'],positive_steps=self.partial['positive_optimizer_steps'])
            self.progress('optimizer_step_complete',revision=self.revision)
        row=dict(version=self.state['version']+1,decisions=len(self.decisions),series_ids=[r['series_id'] for r in self.spec['trajectories'][:len(self.shards)]],terminal_rounds=sum(d['terminal'] for d in self.decisions),minibatch=self.partial['minibatch'],steps=self.partial['logs'],
                 actor_lr=settings['actor_learning_rate'],value_lr=settings['value_head_learning_rate'],at=self.spec['logical_update_timestamp'])
        row['unique_PPO_trained_decisions']=update_decisions(row);row['inherited_pending_first_epoch_coverage']=0;row['new_sampling_first_epoch_coverage']=row['unique_PPO_trained_decisions'];row['inherited_pending_not_covered']=0;row['positive_optimizer_steps']=self.partial['positive_optimizer_steps']
        self.state=dict(self.state,version=row['version'],trained_decisions=self.state['trained_decisions']+len(self.decisions),unique_PPO_trained_decisions=self.state['unique_PPO_trained_decisions']+row['unique_PPO_trained_decisions'],updates=[*self.state['updates'],row],inherited_pending_remaining=0)
        self.partial=None;self.shards=[];self.decisions=[]
        name=f"league/learner-{self.state['version']:04d}.pt";atomic_torch_save(self.job/name,{k:v.detach().cpu() for k,v in self.model.state_dict().items()},replace=False)
        self.league['snapshots'].append(dict(id=f"learner-{self.state['version']:04d}",version=self.state['version'],path=name,sha256=sha(self.job/name)))
        self.save();self.save()
        expected=expected_value(self.state,self.league,self.revision);atomic_torch_save(self.job/'expected.pt',expected,replace=False);atomic_torch_save(self.job/'next-step.pt',self.next_step,replace=False)
        return dict(interrupted=False,complete_BO3=len(row['series_ids']),decisions=row['decisions'],first_epoch_credit=row['unique_PPO_trained_decisions'],positive_optimizer_steps=self.revision,
                    expected_sha256=sha(self.job/'expected.pt'),next_step_sha256=sha(self.job/'next-step.pt'),scientific_credit=0)

    def close(self,error=None):
        if error:self.worker.failed(error)
        else:self.worker.finish()
        self.stream.close();self.backend.close()

def execute(label,variant,tracer,*,count,control_calls,interrupt=False,restore=False,capture_step=1):
    if restore:job=job_path(label);spec=read(V3/'reports/INPUTS.json')
    else:job,spec=prepare_job(label)
    replay=Replay(job,spec,variant,tracer,restore=restore)
    try:
        if not restore:replay.ingest(count,control_calls=control_calls)
        result=replay.optimize(interrupt=interrupt,capture_step=capture_step)
        result['native_objects_stats']=dict(replay.backend.objects.stats)
        result['capacity_scans']=replay.backend.capacity.scans
        result['registered_write_accounting']=getattr(replay.backend,'write_admission',None) is not None
        result['cache_evidence']={name:dict(hits=getattr(value,'hits',0),reads=getattr(value,'reads',0)) for name,value in [('control',getattr(replay.control,'_json_evidence',None)),('queue',getattr(replay.outbox,'_json_evidence',None)),('journal',getattr(replay.backend,'journal_cache',None))]}
        result['progress_coalesced']=getattr(replay.publish,'coalesced',0)
    except BaseException as error:replay.close(error);raise
    else:replay.close();return result
