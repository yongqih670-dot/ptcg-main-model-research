"""Prepare a distinct fresh H002 experiment; retire only fully archived replicas."""
import copy
from pathlib import Path
from .archive_prefix import canonical_hash
from .io import atomic_json,read_json,sha256


def identity(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_uid)


def retire_caches(root,job,project_id,plans):
    from .retained_source import resolve
    from .server_workspace import inside
    checked=[];seen=set()
    for plan in plans:
        source,manifest=resolve(root,project_id,plan['source'])
        for row in plan['objects']:
            relative=row['path'];key=(source.name,relative)
            if key in seen:raise ValueError('Duplicate cache retirement target')
            seen.add(key)
            allowed=relative=='checkpoint.pt' or relative.startswith(('rollouts/','league/','checkpoint-commits/objects/')) and relative.endswith('.pt')
            if manifest.get('mode')=='journal_io_check' and manifest.get('no_training_updates') is True:
                allowed=allowed or relative.startswith('serialization-probe/') and relative.endswith('.pt')
            if relative.startswith('stage/'):
                capsule=manifest.get('restore_capsule')
                original=next((r for r in capsule['components'] if r['path']=='checkpoint.pt'),None) if capsule else None
                allowed=original is not None and relative=='stage/'+original['packed_sha256']+'.blob' and row['sha256']==original['packed_sha256']
            if not allowed:raise ValueError('Only explicit terminal tensor replicas or original transport cache may retire')
            path=inside(root,source/relative);st=path.stat()
            if st.st_size!=row['bytes'] or sha256(path)!=row['sha256']:raise ValueError('Remote cache differs from durable complete local archive')
            if row.get('file_identity') and list(identity(st)[:4])!=row['file_identity']:raise ValueError('Archived remote cache inode changed')
            checked.append((path,identity(st),{'source_run_id':source.name,**row}))
    atomic_json(job/'terminal-cache-retirement.json',{'status':'intent','objects':[r for _,_,r in checked],
        'scientific_parent_published_checkpoints_and_full31history_excluded':True,
        'local_full_archives_unchanged':True,'terminal_sources_will_never_be_resumed':True})
    for path,expected,row in checked:
        path=inside(root,path)
        if identity(path.stat())!=expected:raise ValueError('Cache identity changed before own single-file retirement')
        path.unlink()
    atomic_json(job/'terminal-cache-retirement.json',{'status':'retired','objects':[r for _,_,r in checked],
        'bytes_released':sum(r['bytes'] for _,_,r in checked),'local_full_archives_and_scientific_parent_untouched':True})


def validate_single_change(config,reference):
    """Bind each fresh hypothesis to its distinct registered scientific change."""
    comparable=copy.deepcopy(config);reference=copy.deepcopy(reference)
    name=config['hypothesis_id']
    if name=='TR05':
        flag=comparable['ppo'].pop('critic_updates_shared_trunk',None)
        if flag is not False:raise ValueError('TR05 requires head-only critic gradient')
    elif name=='TR06':
        if comparable['ppo'].get('entropy_coefficient')!=.001 or reference['ppo'].get('entropy_coefficient')!=.01:
            raise ValueError('TR06 requires only entropy0.01 to0.001')
        comparable['ppo']['entropy_coefficient']=.01
    elif name=='TR07':
        if comparable['ppo'].get('value_loss_coefficient')!=.1 or reference['ppo'].get('value_loss_coefficient')!=.5:
            raise ValueError('TR07 requires only value loss coefficient0.5 to0.1')
        comparable['ppo']['value_loss_coefficient']=.5
    elif name=='TR08':
        if comparable['ppo'].get('actor_learning_rate')!=3e-6 or reference['ppo'].get('actor_learning_rate')!=1e-5:
            raise ValueError('TR08 requires only actor learning rate1e-5 to3e-6')
        comparable['ppo']['actor_learning_rate']=1e-5
    elif name=='TR09':
        if comparable['ppo'].get('epochs')!=4 or reference['ppo'].get('epochs')!=2:
            raise ValueError('TR09 requires only PPO epochs2 to4')
        comparable['ppo']['epochs']=2
    elif name=='TR10':
        if comparable['ppo'].get('gae_lambda')!=1. or reference['ppo'].get('gae_lambda')!=.95:
            raise ValueError('TR10 requires only GAE lambda0.95 to1.0')
        comparable['ppo']['gae_lambda']=.95
    else:raise ValueError('Unreviewed fresh scientific hypothesis')
    for value in [comparable,reference]:
        for key in ['hypothesis_id','primary_change','success_criteria']:value.pop(key,None)
    if comparable!=reference or config['target_decisions']!=32768:
        raise ValueError('Fresh scientific configuration changed outside its one variable')
    return name


def prepare(root,job,ticket,manifest,engine_hash):
    from .server_workspace import admit
    config=copy.deepcopy(manifest['training_config']);reference=copy.deepcopy(manifest['matched_training_reference'])
    if canonical_hash(reference)!=manifest['matched_reference_data_sha256']:raise ValueError('Matched TR02 reference changed')
    hypothesis=validate_single_change(config,reference)
    if config['parent_model_sha256']!='959b220daf6777f45ad5d6aa14c35d3274bf3958ed502c7551679453e5f4aaea':
        raise ValueError('Fresh experiment cannot use unpromoted or partial TR04 weights')
    retire_caches(root,job,ticket['project_id'],manifest['terminal_cache_retirement'])
    admit(root,manifest['post_retirement_forecast_bytes'])
    config.update(project_path=str(root),engine_manifest_sha256=engine_hash,
        source_manifest_sha256=sha256(job/'source-hashes.json'),fresh_compact_bootstrap=True,
        online_archive={'maximum_pending_bytes':512*2**20},checkpoint_interval_seconds=600,
        retire_consumed_with_second_checkpoint=True,cooperative_resource_control=True)
    if hypothesis in ('TR06','TR07','TR08','TR09','TR10'):config['checkpoint_transaction_hash_reuse']=True
    if hypothesis in ('TR08','TR09','TR10'):
        if manifest.get('sampling_device')!='cpu' or not manifest.get('CPU_sampler_acceptance_sha256'):
            raise ValueError('Fresh CPU sampling requires recorded actual server acceptance')
        config.update(sampling_device='cpu',CPU_GPU_batch_parity_required=True)
    if hypothesis in ('TR09','TR10'):
        if not manifest.get('ACK_lag_barrier_acceptance_sha256') or manifest.get('consumed_reference_barrier_required') is not True:
            raise ValueError('CPU PPO requires actual ACK barrier acceptance before new rollouts')
        config['consumed_reference_barrier_required']=True
    for spec in config['league']['fixed_opponents']:
        spec['files']={key.replace('\\','/'):value for key,value in spec['files'].items()}
    atomic_json(job/'payload-config.json',config)
    atomic_json(job/'fresh-experiment-inheritance.json',{'hypothesis_id':hypothesis,'initial_parent_sha256':config['parent_model_sha256'],
        'initial_trained_decisions':0,'initial_policy_version':0,'target_decisions':32768,
        'matched_reference_data_sha256':manifest['matched_reference_data_sha256'],
        'single_primary_variable':{'TR05':'critic_updates_shared_trunk=False','TR06':'entropy_coefficient=0.001',
                                   'TR07':'value_loss_coefficient=0.1','TR08':'actor_learning_rate=3e-6','TR09':'epochs=4','TR10':'gae_lambda=1.0'}[hypothesis],
        'no_TR04_or_TR05_partial_checkpoint_or_optimizer_inherited':True,
        'full_H002_actor_and_reset_blind_value_last_layer':True})
    return config
