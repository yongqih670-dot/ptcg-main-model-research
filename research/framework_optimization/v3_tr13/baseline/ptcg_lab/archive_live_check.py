"""Bounded real-server archive acceptance using an immutable original BO3."""
import argparse
import hashlib
from pathlib import Path
import shutil
import time

from .io import atomic_json,read_json,sha256,utcnow


def server(ticket_path):
    from .server_workspace import owned_root,inside,admit
    from .retained_source import resolve,checkpoint_reference
    from .checkpoint_refs import _copy_immutable,commit_checkpoint
    from .archive_outbox import ArchiveOutbox
    from .archive_prefix import compact_record
    ticket=read_json(ticket_path);root=owned_root(ticket['project_root'],ticket['project_id'])
    job=inside(root,ticket['job_directory']);manifest=read_json(job/'training-manifest.raw.json')
    if manifest.get('ack_lag_barrier_check'):
        from .fresh_experiment import retire_caches
        retire_caches(root,job,ticket['project_id'],manifest['terminal_cache_retirement'])
        admit(root,manifest['post_retirement_forecast_bytes'])
    source,_=resolve(root,ticket['project_id'],manifest['existing_restore_source'])
    checkpoint_reference(source,manifest['existing_restore_source'])
    fixture=manifest['archive_fixture'];point=fixture['source_point'];pair=fixture['pair']
    retained=read_json(source/'checkpoint-commits/index.json')['commits']
    if point not in retained:raise ValueError('Fixture published source point changed')
    for ref in [point['checkpoint'],point['prefix'],*pair]:
        src=inside(root,source/ref['path'])
        if src.stat().st_size!=ref['bytes'] or sha256(src)!=ref['sha256']:raise ValueError('Fixture source bytes changed')
        target=inside(root,job/ref['path']);_copy_immutable(src,target)
    cp=point['checkpoint']['path'];prefix=point['prefix']['path'];paths=[r['path'] for r in pair]
    cursor=point['cursor'];box=ArchiveOutbox(job,job.name,ticket['lease_epoch'])
    commit_checkpoint(job,job.name,ticket['lease_epoch'],0,cp,prefix,paths,cursor)
    summary=read_json(job/paths[1]);box.queue_series(summary)
    for path in [cp,prefix]:box.queue(path)
    if manifest.get('ack_lag_barrier_check'):
        from .archive_barrier_check import server_case
        return server_case(root,job,ticket,source,point,pair,summary,box)
    quota_refused=False
    try:admit(root,10*2**30)
    except OSError:quota_refused=True
    if not quota_refused:raise ValueError('Server quota admission accepted impossible extra10GiB')
    started=time.monotonic();samples=0;maximum=0
    while True:
        blocked,status=box.backpressured(1);maximum=max(maximum,status['pending_bytes'])
        if blocked:samples+=1
        atomic_json(job/'progress.json',{'at':utcnow(),'stage':'archive_backpressure_wait',
            'pending_objects':status['pending_objects'],'pending_bytes':status['pending_bytes'],
            'generation_paused':blocked,'blocked_samples':samples,'new_BO3_generated':0})
        if not blocked:break
        if time.time()>ticket['deadline_unix']-160:raise TimeoutError('Archive ACK wait exceeded bounded acceptance deadline')
        admit(root,0);time.sleep(1)
    wait_seconds=time.monotonic()-started
    archived={'archived':[compact_record(summary,sha256(job/paths[1]))]}
    first=box.evict_consumed(archived)
    commit_checkpoint(job,job.name,ticket['lease_epoch'],1,cp,prefix,[],cursor)
    previous=box.evict_consumed(archived)
    if any(x['evicted'] for x in first+previous):raise ValueError('One of two retained points lost a referenced object')
    commit_checkpoint(job,job.name,ticket['lease_epoch'],2,cp,prefix,[],cursor)
    retired=box.evict_consumed(archived)
    deleted={x['object'] for x in retired if x['evicted']}
    if deleted!=set(paths) or any((job/p).exists() for p in paths):raise ValueError('ACK-gated own replica retirement differs')
    if samples<1 or wait_seconds<15:raise ValueError('Actual15s receiver withholding/backpressure interval not observed')
    atomic_json(job/'archive-live-audit.json',{'at':utcnow(),'passed':True,'engineering_only':True,
        'source_run_id':source.name,'source_checkpoint_sha256':point['checkpoint']['sha256'],
        'real_complete_BO3_series_id':summary['series_id'],'new_training_updates':0,
        'backpressure_samples':samples,'wait_seconds':wait_seconds,
        'maximum_unacknowledged_bytes':maximum,'server_quota_admission_refused':quota_refused,
        'latest_point_protected':first,'previous_point_protected':previous,'retired_own_replicas':retired,
        'source_files_untouched':all((source/p).exists() for p in paths),
        'full_model_recovery_proof_reused_separately':True,'not_a_new_training_or_strength_result':True})


class FaultStream:
    def __init__(self,stream):self.stream=stream;self.reads=0;self.corrupt=False;self.lose=False
    def request(self,op,body=None,**kwargs):
        if op=='archive_ack' and self.lose:
            self.lose=False;raise TimeoutError('injected ACK loss before remote delivery')
        result=self.stream.request(op,body,**kwargs)
        if op=='archive_chunk':
            self.reads+=1
            if self.corrupt:result={**result,'raw_sha256':'0'*64}
        return result


def controller_faults(stream,archive_root,run_id,epoch,progress,startup_timeout_seconds=120):
    from .archive_outbox import receive_one
    from .archive_compressed import receive_compressed_object,verify_archive
    from .archive_ack import ack_body
    from .archive_prefix import canonical_hash
    if not 0<startup_timeout_seconds<=480:raise ValueError('Readonly preparation wait exceeded its bounded scope')
    deadline=time.monotonic()+startup_timeout_seconds;row=None
    while time.monotonic()<deadline:
        status=stream.request('archive_pending',{'limit':1})
        observation=stream.request('status')
        if observation.get('terminal'):raise RuntimeError('Archive fixture terminated before backpressure handshake')
        if status['objects'] and observation.get('progress.json',{}).get('stage')=='archive_backpressure_wait':
            row=status['objects'][0];break
        progress('waiting_live_archive_fixture');time.sleep(2)
    if row is None:raise TimeoutError('Real archive fixture did not queue')
    if row['run_id']!=run_id or row['lease_epoch']!=epoch or not row['object'].startswith('rollouts/'):
        raise ValueError('Expected first real complete BO3 trajectory')
    progress('intentional_ACK_withholding_15_seconds');time.sleep(15)
    archive_root=Path(archive_root);archive_root.mkdir(parents=True,exist_ok=True)
    refused=False
    try:receive_compressed_object(archive_root,run_id,epoch,row['sequence'],row['object'],row['sha256'],row['bytes'],(),
        minimum_free_bytes=shutil.disk_usage(archive_root).free+2**30)
    except OSError:refused=True
    if not refused:raise ValueError('Local reserve injection did not refuse archive')
    probe=FaultStream(stream);probe.corrupt=True;corrupt_refused=False
    try:receive_one(probe,archive_root,row,storage_codec='gzip')
    except ValueError:corrupt_refused=True
    if not corrupt_refused:raise ValueError('Damaged real wire response accepted')
    if stream.request('archive_pending',{'limit':1})['objects'][0]!=row:raise ValueError('Failed archive ACKed original object')
    probe.corrupt=False;probe.lose=True;lost=False
    try:receive_one(probe,archive_root,row,storage_codec='gzip')
    except TimeoutError:lost=True
    if not lost:raise ValueError('ACK-loss injection not exercised')
    before=probe.reads;ack=receive_one(probe,archive_root,row,storage_codec='gzip')
    if probe.reads!=before:raise ValueError('Lost ACK retry redownloaded durable object')
    replay=stream.request('archive_ack',{'ack':ack})
    if replay.get('replayed') is not True:raise ValueError('Identical ACK replay was not idempotent')
    rejected=[]
    for field,value in [('lease_epoch',epoch+1),('archive_relative',ack['archive_relative']+'-conflict')]:
        bad={**ack,field:value};bad['commit_id']=canonical_hash(ack_body(bad))
        try:stream.request('archive_ack',{'ack':bad})
        except RuntimeError:rejected.append(field)
    if len(rejected)!=2:raise ValueError('Stale/conflicting ACK not rejected')
    verify_archive(archive_root,ack)
    return {'at':utcnow(),'passed':True,'real_wire_chunks_read':probe.reads,'raw_bytes':row['bytes'],
        'raw_sha256':row['sha256'],'storage':ack['storage'],'durable_ack_commit':ack['commit_id'],
        'ACK_withholding_seconds':15,'damaged_response_rejected_without_ACK':True,
        'local_space_reserve_refused_without_ACK':True,'lost_ACK_durable_replay_without_redownload':True,
        'duplicate_ACK_idempotent':True,'stale_epoch_and_conflicting_destination_rejected':rejected,
        'fault_methods':'15s withholding; local stricter space threshold; altered one wire response hash; suppress one ACK before delivery. No physical disconnect or disk filling claimed.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ticket',required=True);args=parser.parse_args();server(args.ticket)
