"""Actual production-barrier reference test using an untouched full payload."""
import os,random,time
import numpy as np
import torch
from .io import read_json,atomic_json,sha256,utcnow
from .server_io_check import same
from .checkpoint_refs import commit_checkpoint
from .archive_prefix import compact_record
from .archive_barrier import retire_before_sampling
from .tensor_io import load_torch
from .staged_environment import kernel_snapshot
from .server_workspace import admit


def server_case(root,job,ticket,source,point,pair,summary,box):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.device_count()!=0:raise ValueError('ACK barrier check must hide CUDA')
    torch.set_num_threads(2);cp=point['checkpoint']['path'];prefix=point['prefix']['path'];paths=[r['path'] for r in pair];cursor=point['cursor']
    value=load_torch(job/cp,map_location='cpu',weights_only=False)
    if len(value['model'])!=127 or len(value['optimizer']['state'])!=97:raise ValueError('Actual full source payload required')
    before_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone());before_source=sha256(source/'checkpoint-commits/index.json')
    from .bounded_io import environment_context
    rejected=[]
    for expected in [1,3]:
        target=job/f'exact-size-negative-{expected}.bin'
        try:environment_context().atomic_write(target,lambda out:out.write(b'ab'),expected_bytes=expected)
        except (OSError,ValueError):rejected.append(expected)
        if target.exists():raise ValueError('Incorrect exact-size declaration was published')
    if rejected!=[1,3]:raise ValueError('Actual exact-size writer negative checks failed')
    commit_checkpoint(job,job.name,ticket['lease_epoch'],1,cp,prefix,paths,cursor)
    archived={'archived':[compact_record(summary,sha256(job/paths[1]))]};observations=[];generation=1
    quota_refused=False
    from .storage_profile import ticket_ceiling
    rejected_growth=ticket_ceiling(ticket)
    try:admit(root,rejected_growth)
    except OSError:quota_refused=True
    if not quota_refused:raise ValueError('Impossible quota growth accepted')
    def save():
        nonlocal generation
        before=box.evict_consumed(archived);generation+=1
        commit_checkpoint(job,job.name,ticket['lease_epoch'],generation,cp,prefix,[],cursor)
        after=box.evict_consumed(archived)
        observations.append({'generation':generation,'before_rotation':before,'after_rotation':after})
    def progress(stage,**kwargs):
        atomic_json(job/'progress.json',{'at':utcnow(),'stage':'archive_backpressure_wait' if stage=='waiting_consumed_archive_ACK' else stage,
            'production_barrier_stage':stage,'new_BO3_generated':0,**kwargs})
    result=retire_before_sampling(box,pair,cursor,save,float(ticket['deadline_unix'])-80,lambda:60,progress)
    if result['ACK_wait_seconds']<15 or result['ACK_wait_samples']<1 or result['post_ACK_writer_rotations']!=2:
        raise ValueError('Actual delayed ACK and both rotations must be observed')
    if len(observations)!=2 or any(x['evicted'] for x in observations[0]['before_rotation']+observations[0]['after_rotation']):
        raise ValueError('A still-pinned original was removed')
    if {x['object'] for x in observations[1]['after_rotation'] if x['evicted']}!=set(paths):raise ValueError('Consumed replicas retired before both points cleared')
    if not same(before_rng,(random.getstate(),np.random.get_state(),torch.get_rng_state())):raise ValueError('Barrier changed Python/numpy/CPU RNG')
    if sha256(job/cp)!=point['checkpoint']['sha256'] or sha256(source/'checkpoint-commits/index.json')!=before_source:
        raise ValueError('Own full payload or protected original source changed')
    if any(sha256(source/r['path'])!=r['sha256'] for r in [point['checkpoint'],point['prefix'],*pair]):raise ValueError('Read-only original fixture changed')
    report={'at':utcnow(),'passed':True,'engineering_only':True,'actual_ACK_lag_barrier_passed':True,
        'production_barrier':result,'actual_retained_rotations':observations,'full_payload_model_tensors':127,'optimizer_states':97,
        'full_payload_bytes_unchanged':True,'Python_numpy_CPU_RNG_unchanged':True,'GPU_visible_devices':0,'new_training_updates':0,
        'new_training_samples':0,'reference_fixture_only_not_a_resumable_scientific_checkpoint':True,
        'source_run_id':source.name,'source_checkpoint_sha256':point['checkpoint']['sha256'],'source_files_untouched':True,
        'backpressure_samples':result['ACK_wait_samples'],'wait_seconds':result['ACK_wait_seconds'],
        'latest_point_protected':observations[0]['before_rotation'],'previous_point_protected':observations[0]['after_rotation'],
        'retired_own_replicas':observations[1]['after_rotation'],'server_quota_admission_refused':quota_refused,
        'quota_rejected_growth_bytes':rejected_growth,
        'kernel':kernel_snapshot(ticket),'final_capacity':admit(root,0)}
    report['actual_under_and_over_declared_size_refused']=rejected
    atomic_json(job/'archive-live-audit.json',report)
