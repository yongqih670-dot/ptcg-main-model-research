"""Actual CPU-only journal capacity and immutable checkpoint I/O acceptance."""
import argparse
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .io import read_json,atomic_json,sha256,utcnow
from .server_workspace import owned_root,inside,admit
from .retained_source import resolve,checkpoint_reference
from .checkpoint_refs import read_index,protected_hashes,commit_checkpoint
from .training_checkpoint import publish_training_checkpoint,load_training_checkpoint
from .rl_resume import advance_resume_state
from .bounded_read import environment_read_context
from .resource_journal import capacity_profile,encode,decode,append,MAX_RECORD_BYTES
from .staged_environment import kernel_snapshot
from .io_pacing import BytePacer
from .tensor_io import load_torch


def same(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and a.layout==b.layout and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def run(ticket_path):
    ticket=read_json(ticket_path);root=owned_root(ticket['project_root'],ticket['project_id']);job=inside(root,ticket['job_directory'])
    if os.environ.get('PTCG_JOB_RUN_ID')!=job.name or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.device_count()!=0:
        raise ValueError('CPU-only acceptance identity differs')
    torch.set_num_threads(2);manifest=read_json(job/'training-manifest.raw.json')
    if manifest['mode']!='journal_io_check' or not manifest.get('no_training_updates'):raise ValueError('No learning is admitted by this check')
    source,original=resolve(root,ticket['project_id'],manifest['existing_restore_source'])
    point=checkpoint_reference(source,manifest['existing_restore_source'])
    source_index_sha=sha256(source/'checkpoint-commits/index.json');rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    started=time.monotonic();deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    def stage(name,**kw):
        if time.time()>deadline-30:raise TimeoutError('CPU component acceptance deadline')
        atomic_json(job/'progress.json',{'at':utcnow(),'stage':name,'seconds':time.monotonic()-started,**kw})
    stage('actual_journal_capacity')
    probe=inside(root,job/'journal-capacity-probe');probe.mkdir()
    profile=capacity_profile();kernel=kernel_snapshot(ticket)
    sample={'capacity_test_only':True,'record':0,'kernel':kernel,'padding':''}
    sample['padding']='x'*max(0,MAX_RECORD_BYTES-64-len(encode(sample)))
    pacer=BytePacer(4*2**20,deadline-30)
    for n in range(profile['planned_record_count_upper']):
        row=dict(sample,record=n);pacer.consume(len(encode(row)));append(root,probe,row)
    journal=probe/'resource-samples.jsonl';records=[decode(line) for line in journal.read_text().splitlines()]
    if len(records)!=724 or [r['record'] for r in records]!=list(range(724)) or any(r['kernel']!=kernel for r in records) or not 512*1024<journal.stat().st_size<profile['maximum_journal_bytes']:
        raise ValueError('Actual full planned-record capacity did not pass')
    journal_result={'records_written_and_read_back':len(records),'actual_bytes':journal.stat().st_size,
        'sha256':sha256(journal),'profile':profile,'record_payloads_losslessly_reconstructed':True,
        'capacity_test_elapsed_seconds':time.monotonic()-started,'not_a_two_hour_elapsed_training_claim':True}
    stage('actual_retained_checkpoint_read_comparison')
    context=environment_read_context();original_hash=context.sha256;counter={'calls':0,'bytes':0}
    def counted(path):
        counter['calls']+=1;counter['bytes']+=Path(path).stat().st_size
        return original_hash(path)
    context.sha256=counted;measurements={};indexes={};pinsets={}
    try:
        for name in ['uncached','one_transaction_cache']:
            verified={} if name=='one_transaction_cache' else None
            before=dict(counter);t=time.monotonic()
            indexes[name]=read_index(source,source.name,manifest['existing_restore_source']['lease_epoch'],_verified=verified)
            pinsets[name]=protected_hashes(source,source.name,manifest['existing_restore_source']['lease_epoch'],_verified=verified)
            measurements[name]={'seconds':time.monotonic()-t,'full_hash_calls':counter['calls']-before['calls'],
                'full_hash_input_bytes':counter['bytes']-before['bytes']}
            stage('retained_read_comparison_'+name,measurement=measurements[name])
    finally:context.sha256=original_hash
    if indexes['uncached']!=indexes['one_transaction_cache'] or pinsets['uncached']!=pinsets['one_transaction_cache'] or measurements['one_transaction_cache']['full_hash_input_bytes']>=measurements['uncached']['full_hash_input_bytes']:
        raise ValueError('Cache changed actual index/protection or did not reduce duplicate reads')
    stage('actual_full_payload_serialization_parity')
    saved=load_torch(inside(root,source/point['checkpoint']['path']),map_location='cpu',weights_only=False)
    if len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97:raise ValueError('Full real trained payload required')
    if saved['archive_state']['live_expected']:raise ValueError('Serialization probe requires the published fully consumed state')
    case=inside(root,job/'serialization-probe');case.mkdir();objects=case/'checkpoint-commits/objects';objects.mkdir(parents=True)
    for ref,targets in [(point['checkpoint'],[case/'checkpoint.pt',objects/('cp-'+point['checkpoint']['sha256']+'.pt')]),
                        (point['prefix'],[case/'archive-prefix.json',objects/('prefix-'+point['prefix']['sha256']+'.json')])]:
        src=inside(root,source/ref['path'])
        if sha256(src)!=ref['sha256']:raise ValueError('Terminal source changed before own immutable links')
        for target in targets:os.link(src,inside(root,target))
    prefix=read_json(case/'archive-prefix.json')
    spec={'prefix_path':'archive-prefix.json','prefix_sha256':point['prefix']['sha256'],
          'initial_checkpoint_sha256':prefix['checkpoint']['sha256']}
    commit_checkpoint(case,case.name,ticket['lease_epoch'],0,'checkpoint.pt','archive-prefix.json',[],point['cursor'])
    commits=[]
    for name in ['uncached','one_transaction_cache']:
        verification={} if name=='one_transaction_cache' else None
        state=advance_resume_state(case,saved['archive_state'],prefix,spec,point['cursor'],case.name,ticket['lease_epoch'],_verified=verification)
        value={**saved,'archive_state':state};t=time.monotonic()
        result=publish_training_checkpoint(case,value,spec,ticket['lease_epoch'],_verified=verification)
        loaded=load_training_checkpoint(case,spec,ticket['lease_epoch'])
        if not same(value,loaded['saved']):raise ValueError('Full real model/optimizer/RNG payload changed')
        commits.append({'role':name,'checkpoint':result['recovery']['checkpoint'],'seconds':time.monotonic()-t})
        stage('full_payload_'+name+'_verified',checkpoint_sha256=result['recovery']['checkpoint']['sha256'])
    if commits[0]['checkpoint']!=commits[1]['checkpoint']:raise ValueError('Cached serialization differs from the identical uncached CPU payload')
    stage('actual_changed_file_cache_invalidation')
    negative=inside(root,job/'negative-probe');negative.mkdir()
    for name,data in [('cp.bin',b'cp'),('prefix.bin',b'prefix'),('pending.bin',b'original')]:
        (negative/name).write_bytes(data)
    commit_checkpoint(negative,negative.name,ticket['lease_epoch'],0,'cp.bin','prefix.bin',['pending.bin'],
        {'version':0,'trained_decisions':0,'next_series_id':0})
    verification={};read_index(negative,negative.name,ticket['lease_epoch'],_verified=verification)
    old=(negative/'pending.bin').stat();(negative/'pending.bin').write_bytes(b'changed!')
    os.utime(negative/'pending.bin',ns=(old.st_atime_ns,old.st_mtime_ns+10000000))
    rejected=False
    try:read_index(negative,negative.name,ticket['lease_epoch'],_verified=verification)
    except ValueError:rejected=True
    if not rejected:raise ValueError('Actual changed own file bypassed cache invalidation')
    if not same(rng,(random.getstate(),np.random.get_state(),torch.get_rng_state())):raise ValueError('Component checks changed sampling RNG')
    if sha256(source/'checkpoint-commits/index.json')!=source_index_sha or sha256(source/point['checkpoint']['path'])!=point['checkpoint']['sha256']:
        raise ValueError('Read-only scientific source changed')
    result={'at':utcnow(),'passed':True,'complete':True,'engineering_only':True,'journal_capacity':journal_result,
        'actual_source_run_id':source.name,'source_checkpoint_sha256':point['checkpoint']['sha256'],
        'source_index_sha256_unchanged':source_index_sha,'retained_reference_read_measurements':measurements,
        'retained_indexes_and_protected_hashes_equal':True,'CPU_serialization_commits':commits,
        'full127_model_97_optimizer_and_RNG_payloads_exact':True,'Python_numpy_CPU_RNG_unchanged':True,
        'changed_file_rejected_within_same_cache':rejected,'GPU_visible_devices':0,'optimizer_steps':0,
        'training_samples_collected':0,'source_files_unchanged':True,'kernel':kernel_snapshot(ticket),
        'final_capacity':admit(root,0),'seconds':time.monotonic()-started,'promotion_eligible':False}
    atomic_json(job/'journal-io-audit.json',result);stage('journal_io_check_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ticket',required=True);run(parser.parse_args().ticket)
