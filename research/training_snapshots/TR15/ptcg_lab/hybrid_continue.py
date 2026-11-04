"""Prepare a NEW scientific continuation from immutable published TR04 inputs."""
import os
from .io import atomic_json,read_json,sha256


def validate_original_attempt_scope(project,config,deadline):
    import time
    from pathlib import Path
    original=Path(project)/'runs'/config['original_training_run_id']
    attempts=read_json(original/'attempts.json')
    if len(attempts)!=1 or abs(attempts[0]['seconds']-7160.515)>.01:
        raise ValueError('Original TR04 attempt expenditure changed')
    if config['original_attempt_number']!=2 or deadline-time.time()>3600.5:
        raise ValueError('TR04 only original second attempt<=3600s remains')
    if config['hypothesis_id']!='TR04' or config['maximum_attempts']!=1:
        raise ValueError('Derived continuation cannot regain scientific attempts')
    if attempts[0]['seconds']+deadline-time.time()>10800.5:raise ValueError('Original total training scope exceeded')


def prepare(root,job,ticket,manifest,engine_hash):
    from .retained_source import resolve,checkpoint_reference
    from .server_workspace import inside,admit
    from .checkpoint_refs import _copy_immutable
    source,original=resolve(root,ticket['project_id'],manifest['existing_restore_source'])
    checkpoint_reference(source,manifest['existing_restore_source'])
    capsule=original['restore_capsule'];point=manifest['continuation_point']
    if point not in read_json(source/'checkpoint-commits/index.json')['commits']:raise ValueError('Continuation point no longer retained')
    if point['checkpoint']['sha256']!=capsule['origin_checkpoint_sha256']:raise ValueError('Scientific bootstrap must inherit original full checkpoint')
    if len(point['pending'])!=70:raise ValueError('Full35 pending BO3 pairs required')
    # Only redundant transport blobs are retired. Expanded31 history weights,
    # checkpoint objects, live trajectories and every original local file stay.
    plan=manifest.get('redundant_transport_cache',[])
    expected={x['packed_sha256']:x['packed_bytes'] for x in capsule['history']['members']}
    if {x['sha256']:x['bytes'] for x in plan}!=expected:raise ValueError('Cache retirement must match exactly31 original history transports')
    verified=[]
    for row in plan:
        path=inside(root,source/'stage'/(row['sha256']+'.blob'));st=path.stat()
        if st.st_size!=row['bytes'] or sha256(path)!=row['sha256']:raise ValueError('Retirable transport differs from durable local original')
        verified.append((row,path,(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_uid)))
    atomic_json(job/'transport-cache-retirement.json',{'status':'intent','source_run_id':source.name,
        'objects':plan,'expanded_history_and_all_training_evidence_retained':True})
    for row,path,identity in verified:
        st=inside(root,path).stat()
        if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_uid)!=identity:raise ValueError('Own cache identity changed; refuse unlink')
        path.unlink()
    atomic_json(job/'transport-cache-retirement.json',{'status':'retired','source_run_id':source.name,
        'objects':plan,'bytes_released':sum(x['bytes'] for x in plan),'local_originals_and_expanded_history_preserved':True})
    retired=[]
    for spec in manifest.get('retired_fixture_checkpoint_caches',[]):
        fixture,old=resolve(root,ticket['project_id'],spec)
        if old.get('mode')!='live_archive_check':raise ValueError('Only terminal synthetic protocol-fixture copies can be retired')
        from .archive_outbox import ArchiveOutbox
        if ArchiveOutbox(fixture,fixture.name,spec['lease_epoch']).pending()['pending_objects']:
            raise ValueError('Terminal fixture still lacks durable archive ACKs')
        ref=old['archive_fixture']['source_point']['checkpoint']
        if ref['sha256']!=capsule['origin_checkpoint_sha256']:raise ValueError('Fixture does not contain identical original checkpoint')
        path=inside(root,fixture/ref['path']);st=path.stat()
        if st.st_size!=ref['bytes'] or sha256(path)!=ref['sha256']:raise ValueError('Fixture redundant checkpoint bytes differ')
        row={'run_id':fixture.name,'reference':ref,'local_original_sha256':capsule['origin_checkpoint_sha256'],
            'terminal_identity':spec['identity'],'never_used_for_training_or_resumed_again':True}
        atomic_json(job/'fixture-cache-retirement.json',{'status':'intent','retired':retired,'next':row})
        current=path.stat()
        if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_uid)!=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_uid):
            raise ValueError('Own terminal fixture file identity changed before retirement')
        path.unlink();retired.append(row)
    atomic_json(job/'fixture-cache-retirement.json',{'status':'retired','objects':retired,
        'reason':'Identical original full checkpoint is durably retained locally and in scientific source; protocol fixture lease ended, all ACKs durable. Preserve all logs, indexes, receipts and scientific recovery points.'})
    admit(root,manifest['post_cache_retirement_forecast_bytes'])
    for ref,relative in [(point['checkpoint'],'checkpoint.pt'),(point['prefix'],'archive-prefix.json'),
                         *[(r,r['path']) for r in point['pending']]]:
        src=inside(root,source/ref['path'])
        if src.stat().st_size!=ref['bytes'] or sha256(src)!=ref['sha256']:raise ValueError('Inherited raw object changed')
        _copy_immutable(src,inside(root,job/relative))
    league=read_json(source/'league.json')
    members={x['path']:x['sha256'] for x in league['snapshots']}
    if len(members)!=31 or league['snapshots']!=[x['original'] for x in capsule['history']['members']]:
        raise ValueError('Full original frozen31-member history required')
    atomic_json(job/'league.json',league)
    config=manifest['training_config']
    if config['hypothesis_id']!='TR04' or config['target_decisions']!=262144 or config['seed']!=230101 or config['ppo']['gae_lambda']!=.99 or config['ppo']['rollout_complete_learner_decisions']!=8192:
        raise ValueError('Original scientific parameters changed')
    config.update(project_path=str(root),engine_manifest_sha256=engine_hash,checkpoint_interval_seconds=600,
        retire_consumed_with_second_checkpoint=True,
        archive_restore={'prefix_path':'archive-prefix.json','prefix_sha256':capsule['prefix_sha256'],
                         'initial_checkpoint_sha256':capsule['origin_checkpoint_sha256']},
        online_archive={'maximum_pending_bytes':384*2**20})
    config['league']['retained_history']={'root':str(source),'members':members}
    for spec in config['league']['fixed_opponents']:spec['files']={k.replace('\\','/'):v for k,v in spec['files'].items()}
    atomic_json(job/'payload-config.json',config)
    atomic_json(job/'continuation-inheritance.json',{'original_run_id':capsule['origin_run_id'],
        'checkpoint_sha256':capsule['origin_checkpoint_sha256'],'cursor':point['cursor'],
        'pending_pairs':35,'pending_decisions':5430,'all31_history_retained_readonly':True,
        'original_scientific_attempt_number':2,'remaining_attempt_seconds_max':3600,
        'no_engineering_weights_substituted':True,'history_distribution_unchanged':True})
    return config
