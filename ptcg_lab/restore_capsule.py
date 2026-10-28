"""Lossless TR04 transport and real Linux compact-recovery checks; no PPO."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import random
import time
import zlib

import numpy as np
import torch
from .archive_prefix import load_prefix,relative_path
from .bounded_checkpoint_audit import exact_state
from .io import atomic_json,read_json,sha256,utcnow
from .rl_core import gae,load_full_actor,optimizer_for,evaluate_decisions
from .rl_resume import resume_journal
from .tensor_io import load_torch


def raw_digest(data):return hashlib.sha256(data).hexdigest()


def reference(decisions,settings):
    advantages,returns=gae(decisions,settings['gamma'],settings['gae_lambda'])
    return {'decisions':len(decisions),'advantages_sha256':raw_digest(advantages.tobytes()),
        'returns_sha256':raw_digest(returns.tobytes()),
        'old_logp_sha256':raw_digest(np.asarray([x['old_logp'] for x in decisions],dtype='<f8').tobytes()),
        'ordered_series_ids':list(dict.fromkeys(x['series_id'] for x in decisions))}


def unpack(packed,record):
    size=record['raw_bytes']
    if type(size) is not int or not 0<=size<=128*2**20:raise ValueError('Restore component size exceeds bound')
    if len(packed)!=record['packed_bytes'] or raw_digest(packed)!=record['packed_sha256']:raise ValueError('Restore packed identity differs')
    decoder=zlib.decompressobj();raw=decoder.decompress(packed,size+1)
    if len(raw)!=size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or raw_digest(raw)!=record['raw_sha256']:
        raise ValueError('Restore component cannot reconstruct original bytes')
    return raw


def restore_capacity(capsule):
    history=capsule['history']['members'];components=capsule['components']
    if len(history)!=31 or capsule['expected_update_input']['decisions']!=5430:
        raise ValueError('Full original TR04 restore scope missing')
    paths=[relative_path(row['path']) for row in components]
    if len(paths)!=len(set(paths)) or set(paths)-{'checkpoint.pt','archive-prefix.json','league.json'}!={
            x for x in paths if x.startswith(('rollouts/','series/'))}:
        raise ValueError('Restore component scope differs')
    if sum(x.startswith('rollouts/') for x in paths)!=35 or sum(x.startswith('series/') for x in paths)!=35:
        raise ValueError('Full original TR04 live tail missing')
    if not {'checkpoint.pt','archive-prefix.json','league.json'}.issubset(paths):raise ValueError('Full restore metadata missing')
    raw=sum(row['raw_bytes'] for row in components)+sum(row['raw_bytes'] for row in history)
    if raw!=capsule['expanded_bytes'] or sum(row['raw_bytes'] for row in history)!=capsule['history']['raw_bytes']:
        raise ValueError('Expanded restore size differs from complete member sum')
    return raw+384*2**20


def prepare(config,output):
    output=Path(output);root=Path(config['project_path']).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Restore packing requires controlled runner')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])-20
    origin=root/'runs'/config['origin_run_id'];history_root=root/'runs'/config['history_pack_run_id']
    if origin.resolve().parent!=root/'runs' or history_root.resolve().parent!=root/'runs':raise ValueError('Restore source escaped project')
    if sha256(origin/'checkpoint.pt')!=config['origin_checkpoint_sha256']:raise ValueError('Original full checkpoint changed')
    if sha256(history_root/'history-delta-manifest.json')!=config['history_manifest_sha256']:raise ValueError('Complete history pack changed')
    prefix_path=root/config['prefix_path'];prefix=load_prefix(prefix_path,config['prefix_sha256'])
    if prefix['checkpoint']['sha256']!=config['origin_checkpoint_sha256'] or sha256(origin/'config.json')!=prefix['origin_config_sha256']:
        raise ValueError('Original configuration/checkpoint and compact prefix differ')
    history=read_json(history_root/'history-delta-manifest.json')
    league=read_json(origin/'league.json')
    if sha256(origin/'league.json')!=history['league_manifest_sha256'] or [x['original'] for x in history['members']]!=league['snapshots']:
        raise ValueError('Full original history membership differs')
    files=[]
    for row in history['members']:
        packed=history_root/row['packed_path']
        if sha256(packed)!=row['packed_sha256']:raise ValueError('History transport bytes changed')
        files.append({'sha256':row['packed_sha256'],'bytes':row['packed_bytes'],'local_path':packed.relative_to(root).as_posix()})
    raw_components=[(origin/'checkpoint.pt','checkpoint.pt',config['origin_checkpoint_sha256']),
        (prefix_path,'archive-prefix.json',config['prefix_sha256']),
        (origin/'league.json','league.json',history['league_manifest_sha256'])]
    decisions=[]
    for row in prefix['tail']:
        raw_components.extend([(origin/row['trajectory_path'],row['trajectory_path'],row['trajectory_sha256']),
            (origin/'series'/f"{row['series_id']:06d}.json",f"series/{row['series_id']:06d}.json",row['summary_sha256'])])
        data=torch.load(origin/row['trajectory_path'],map_location='cpu',weights_only=False)
        decisions.extend(data['decisions'])
    components=[];(output/'components').mkdir()
    for number,(path,name,expected) in enumerate(raw_components):
        if time.time()>deadline:raise TimeoutError('Finite capsule preparation deadline')
        raw=path.read_bytes()
        if raw_digest(raw)!=expected:raise ValueError('Original recovery component changed')
        packed=zlib.compress(raw,1);entry={'path':relative_path(name),'raw_bytes':len(raw),'raw_sha256':expected,
            'packed_bytes':len(packed),'packed_sha256':raw_digest(packed)}
        target=output/'components'/f'{number:03d}.zlib'
        with target.open('xb') as stream:stream.write(packed);stream.flush();os.fsync(stream.fileno())
        if unpack(target.read_bytes(),entry)!=raw:raise ValueError('Published capsule bytes differ')
        files.append({'sha256':entry['packed_sha256'],'bytes':len(packed),'local_path':target.relative_to(root).as_posix()})
        components.append(entry)
        atomic_json(output/'progress.json',{'at':utcnow(),'components_verified':len(components),'components_total':len(raw_components)})
    settings=read_json(origin/'config.json')['ppo'];reference_input=reference(decisions,settings)
    if reference_input['decisions']!=5430 or len(prefix['tail'])!=35 or len(history['members'])!=31:raise ValueError('Original TR04 recovery scope differs')
    raw_bytes=sum(x['raw_bytes'] for x in components)+history['raw_bytes']
    capsule={'schema_version':1,'origin_run_id':origin.name,'origin_checkpoint_sha256':config['origin_checkpoint_sha256'],
        'prefix_sha256':config['prefix_sha256'],'history':history,'components':components,'files':files,
        'expanded_bytes':raw_bytes,'packed_bytes':sum(x['bytes'] for x in files),'settings':settings,
        'checkpoint_cursor':prefix['checkpoint'],'expected_update_input':reference_input,
        'original_TR04_attempts_and_budget_unchanged':True,'cuda_rng_application_verified':False}
    restore_capacity(capsule)
    atomic_json(output/'restore-capsule.json',capsule)
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'promotion_eligible':False,
        'origin_files_untouched':True,'full_history_members':31,'tail_shards':35,'pending_decisions':5430,
        'packed_bytes':capsule['packed_bytes'],'expanded_bytes':raw_bytes,'component_files':len(files),
        'manifest_sha256':sha256(output/'restore-capsule.json'),'linux_restore_verified':False})


def read_packed(job,digest):
    from .bounded_read import environment_read_context
    reader=environment_read_context()
    with reader.open(job/'stage'/(digest+'.blob')) as stream:return stream.read()


def original_saved(job,capsule):
    record=next(x for x in capsule['components'] if x['path']=='checkpoint.pt')
    return torch.load(io.BytesIO(unpack(read_packed(job,record['packed_sha256']),record)),map_location='cpu',weights_only=False)


def verify(ticket_path,fresh=False,existing_source=None):
    from .server_workspace import owned_root,inside
    from .bounded_io import environment_context
    from .history_delta import decode
    from .training_checkpoint import initialize_recovery,publish_training_checkpoint,load_training_checkpoint
    from .checkpoint_refs import read_index
    ticket=read_json(ticket_path);root=owned_root(ticket['project_root'],ticket['project_id']);job=inside(root,ticket['job_directory']);report_job=job
    if os.environ.get('PTCG_JOB_RUN_ID')!=job.name or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU restore identity or GPU isolation differs')
    manifest=read_json(job/'training-manifest.raw.json');epoch=ticket['lease_epoch'];source_run_id=None
    if existing_source is not None:
        if not fresh:raise ValueError('Existing restore source is read-only and fresh-process-only')
        from . import systemd_job
        source_run_id=relative_path(existing_source['run_id'])
        if '/' in source_run_id:raise ValueError('Existing restore source run identity escaped')
        source=inside(root,root/'jobs'/source_run_id)
        if sha256(source/'training-manifest.raw.json')!=existing_source['manifest_sha256'] or read_json(source/'identity.json')!=existing_source['identity']:
            raise ValueError('Existing restore source identity changed')
        previous=read_json(source/'ticket.json')
        if (previous['run_id'],previous['unit'],previous['lease_epoch'],previous['project_id'])!=(source_run_id,existing_source['identity']['unit'],existing_source['lease_epoch'],ticket['project_id']):
            raise ValueError('Existing restore ticket and authoritative lease differ')
        properties=systemd_job.inspect_service(previous['unit'])
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if boot!=existing_source['identity']['host_boot_id']:raise ValueError('Existing restore source boot identity changed')
        if properties.get('LoadState')=='not-found':
            if time.time()<previous['deadline_unix']+60:raise ValueError('Missing prior service before original deadline cannot prove termination')
        elif not systemd_job.terminal(properties) or systemd_job.identity(properties,boot)!=existing_source['identity']:
            raise ValueError('Existing restore source still active or invocation changed')
        job=source;manifest=read_json(job/'training-manifest.raw.json');epoch=previous['lease_epoch']
    capsule=manifest['restore_capsule']
    context=environment_context();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if existing_source is not None:
        metadata=next(row for row in manifest['file_map'] if row['path']=='ptcg_lab/public_cg/metadata.json')
        from .bounded_read import environment_read_context
        with environment_read_context().open(job/'control'/metadata['path']) as stream:data=stream.read()
        if len(data)!=metadata['bytes'] or raw_digest(data)!=metadata['sha256']:raise ValueError('Original public-only card metadata changed')
        target=report_job/'control'/metadata['path']
        context.atomic_write(target,lambda stream:stream.write(data),replace=False)
    spec={'prefix_path':'archive-prefix.json','prefix_sha256':capsule['prefix_sha256'],'initial_checkpoint_sha256':capsule['origin_checkpoint_sha256']}
    if not fresh:
        previous=b''
        for row in capsule['history']['members']:
            raw=decode(read_packed(job,row['packed_sha256']),previous,row)
            target=inside(root,job/relative_path(row['original']['path']))
            receipt=context.atomic_write(target,lambda stream:stream.write(raw),replace=False)
            if receipt['sha256']!=row['original']['sha256']:raise ValueError('Linux history reconstruction differs')
            previous=raw
        for row in capsule['components']:
            raw=unpack(read_packed(job,row['packed_sha256']),row)
            receipt=context.atomic_write(inside(root,job/relative_path(row['path'])),lambda stream:stream.write(raw),replace=False)
            if receipt['sha256']!=row['raw_sha256']:raise ValueError('Linux original component reconstruction differs')
        saved=load_torch(job/'checkpoint.pt',map_location='cpu',weights_only=False);digest=sha256(job/'checkpoint.pt')
        plan=resume_journal(job,saved,spec,digest)
    else:
        expected_rollouts={row['path'] for row in capsule['components'] if row['path'].startswith('rollouts/')}
        actual_rollouts={path.relative_to(job).as_posix() for path in (job/'rollouts').glob('*.pt')}
        if expected_rollouts!=actual_rollouts:raise ValueError('Read-only restore source live-tail membership differs')
        for row in capsule['components']:
            if row['path'].startswith('series/') and not (job/row['path']).is_file():
                raise ValueError('Read-only restore source summary missing; refuse journal repair')
        loaded=load_training_checkpoint(job,spec,epoch);saved=loaded['saved'];digest=loaded['digest'];spec=loaded['archive_spec']
        if saved.get('restore_audit_marker')!='published-generation-one':raise ValueError('Unpublished hot checkpoint became authoritative')
        if sha256(job/'checkpoint.pt')==digest:raise ValueError('Unpublished hot checkpoint case has not been exercised')
        plan=resume_journal(job,saved,spec,digest)
    if reference(plan['decisions'],capsule['settings'])!=capsule['expected_update_input']:raise ValueError('Linux pending PPO update inputs differ')
    prefix=load_prefix(job/'archive-prefix.json',capsule['prefix_sha256'])
    if plan['fixed_statistics']!=prefix['full_fixed_statistics'] or plan['next_series_id']!=capsule['checkpoint_cursor']['next_series_id']:
        raise ValueError('Linux compact league statistics or cursor differs')
    original=original_saved(job,capsule)
    equality=exact_state({k:original[k] for k in original},{k:saved[k] for k in original})
    if len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97:raise ValueError('Linux full checkpoint coverage differs')
    bundle=job/'control/agents/frozen/ppo-zero-anchor'
    model,features,_=load_full_actor(bundle,manifest['training_config']['parent_model_sha256'],'cpu')
    model.load_state_dict(saved['model'],strict=True);optimizer=optimizer_for(model);optimizer.load_state_dict(saved['optimizer'])
    if any(not torch.equal(v,saved['model'][k]) for k,v in model.state_dict().items()):raise ValueError('Full restored Linux model differs')
    exact_state(optimizer.state_dict(),saved['optimizer'])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with torch.no_grad():logp,values,_=evaluate_decisions(model,plan['decisions'][:48],features.b.STOP)
    old=np.asarray([x['old_logp'] for x in plan['decisions'][:48]])
    logp_error=float(np.max(np.abs(logp.cpu().numpy()-old)))
    if not torch.isfinite(logp).all() or not torch.isfinite(values).all() or logp_error>1e-4:raise ValueError('Restored CPU conditional log-prob parity failed')
    # Match the production loader: a later committed pending shard owns the
    # sampling RNG after its BO3; the older checkpoint RNG remains preserved.
    rng=plan['rng_after'] if plan['rng_after'] is not None else saved['rng']
    def draws():
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
        return random.random(),np.random.randint(0,1000000,8),torch.rand(8)
    a=draws();b=draws()
    if a[0]!=b[0] or not np.array_equal(a[1],b[1]) or not torch.equal(a[2],b[2]):raise ValueError('Linux CPU RNG restore differs')
    if not fresh:
        initial=initialize_recovery(job,saved,spec,plan['archive_state'],epoch)
        value={**saved,'archive_state':plan['archive_state'],'archive_origin_prefix_sha256':capsule['prefix_sha256'],
            'restore_audit_marker':'published-generation-one'}
        publish_training_checkpoint(job,value,spec,epoch)
        # Exercise an actual failed index publication on the owned Linux files.
        from unittest.mock import patch
        from . import checkpoint_refs
        real=checkpoint_refs.atomic_json
        def fail_index(path,data):
            if Path(path)==job/'checkpoint-commits/index.json':raise OSError('injected-unpublished-index')
            return real(path,data)
        with patch.object(checkpoint_refs,'atomic_json',side_effect=fail_index):
            try:publish_training_checkpoint(job,{**value,'restore_audit_marker':'unpublished-hot-must-not-win'},spec,epoch)
            except OSError as exc:
                if str(exc)!='injected-unpublished-index':raise
            else:raise RuntimeError('Failed-index injection did not execute')
        atomic_json(report_job/'restore-audit.json',{'at':utcnow(),'stage':'published-one-and-uncommitted-hot-ready','fresh_process_verified':False,
            'initial_commit_id':initial['commit_id'],'exact_original_core_state':equality,'conditional_logp_max_error':logp_error})
    else:
        index=read_index(job,job.name,epoch)
        if [x['generation'] for x in index['commits']]!=[0,1]:raise ValueError('Expected both retained recovery points')
        protected={x['path'] for point in index['commits'] for x in point['pending']}
        expected={name for row in prefix['tail'] for name in [row['trajectory_path'],f"series/{row['series_id']:06d}.json"]}
        if protected!=expected:raise ValueError('Recovery points lost pending rollout or complete summary')
        # Every decoded historical member remains present and byte-exact. A
        # read-only continuation never recopies or changes the prior workspace.
        for row in capsule['history']['members']:
            if sha256(job/row['original']['path'])!=row['original']['sha256']:raise ValueError('Decoded full history member changed')
        atomic_json(report_job/'restore-audit.json',{'at':utcnow(),'complete':True,'linux_full_compact_restore_verified':True,
            'fresh_process_verified':True,'full_model_tensors':127,'optimizer_states':97,'history_members':31,
            'pending_shards':35,'pending_decisions':len(plan['decisions']),'original_checkpoint_sha256':capsule['origin_checkpoint_sha256'],
            'original_core_state_exact':equality,'cpu_rng_application_verified':True,'cuda_rng_bytes_preserved':len(rng['cuda'])==1,
            'applied_rng_source':'last_pending_shard' if plan['rng_after'] is not None else 'checkpoint',
            'cuda_rng_application_verified':False,'pending_update_input_exact':True,'fixed_statistics_and_cursor_exact':True,
            'conditional_logp_checks':48,'conditional_logp_max_error':logp_error,'conditional_logp_tolerance':1e-4,
            'two_published_recovery_points_verified':True,'unpublished_hot_checkpoint_ignored':True,
            'all_pending_shards_and_summaries_pinned':True,'training_updates_executed':0,'promotion_eligible':False,
            'existing_source_run_id':source_run_id,'existing_source_lease_epoch':epoch if source_run_id else None,
            'existing_source_files_untouched':source_run_id is not None,
            'long_unattended_training_admitted':False,'live_archive_ack_integrated':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--output');parser.add_argument('--ticket');parser.add_argument('--fresh',action='store_true');parser.add_argument('--existing-source-json');args=parser.parse_args()
    if args.ticket:verify(args.ticket,args.fresh,read_json(args.existing_source_json) if args.existing_source_json else None)
    else:prepare(read_json(args.config),args.output)
