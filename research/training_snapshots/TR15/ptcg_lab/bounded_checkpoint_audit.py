"""Local, controlled exact-state and compact-input audit; never trains a model."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .archive_prefix import load_prefix
from .bounded_io import WriteContext
from .io import atomic_json,read_json,sha256,utcnow
from .rl_resume import resume_journal
from .tensor_io import atomic_torch_save


def exact_state(left,right):
    counts={'torch_tensors':0,'numpy_arrays':0,'scalar_leaves':0,'tensor_bytes':0}
    forward={};reverse={}
    def check(a,b,path):
        if type(a) is not type(b):raise ValueError('Restored state type differs at '+path)
        if isinstance(a,torch.Tensor):
            if (a.device.type,b.device.type)!=('cpu','cpu') or a.dtype!=b.dtype or a.shape!=b.shape or a.stride()!=b.stride() or a.storage_offset()!=b.storage_offset() or not torch.equal(a,b):
                raise ValueError('Restored tensor differs at '+path)
            sa=a.untyped_storage();sb=b.untyped_storage()
            if sa.nbytes():
                pa=sa.data_ptr();pb=sb.data_ptr()
                if forward.get(pa,pb)!=pb or reverse.get(pb,pa)!=pa:raise ValueError('Restored storage alias differs at '+path)
                forward[pa]=pb;reverse[pb]=pa
            counts['torch_tensors']+=1;counts['tensor_bytes']+=a.numel()*a.element_size()
        elif isinstance(a,np.ndarray):
            if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b):raise ValueError('Restored NumPy state differs at '+path)
            counts['numpy_arrays']+=1
        elif isinstance(a,dict):
            if list(a)!=list(b):raise ValueError('Restored state keys/order differ at '+path)
            for k in a:check(a[k],b[k],path+'/'+str(k))
            if hasattr(a,'_metadata'):
                if not hasattr(b,'_metadata'):raise ValueError('Model metadata missing')
                check(a._metadata,b._metadata,path+'/_metadata')
        elif isinstance(a,(tuple,list)):
            if len(a)!=len(b):raise ValueError('Restored sequence length differs at '+path)
            for i,(x,y) in enumerate(zip(a,b)):check(x,y,path+'/'+str(i))
        else:
            if a!=b:raise ValueError('Restored scalar differs at '+path)
            counts['scalar_leaves']+=1
    check(left,right,'checkpoint');counts['distinct_nonempty_tensor_storages']=len(forward)
    return counts


def run(config,output):
    output=Path(output).resolve();project=Path(config['project_path']).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Checkpoint audit requires controlled runner')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])-20;torch.set_num_threads(1)
    origin=project/'runs'/config['origin_run_id'];source_cp=origin/'checkpoint.pt';prefix_path=project/config['prefix_path']
    if sha256(source_cp)!=config['origin_checkpoint_sha256']:raise ValueError('Frozen inherited checkpoint changed')
    prefix=load_prefix(prefix_path,config['prefix_sha256']);saved=torch.load(source_cp,map_location='cpu',weights_only=False)
    for k in ['version','trained_decisions','next_series_id']:
        if saved[k]!=prefix['checkpoint'][k]:raise ValueError('Real inherited cursor and prefix differ')
    if len(saved['model'])!=config['expected_model_tensors'] or len(saved['optimizer']['state'])!=config['expected_optimizer_states']:
        raise ValueError('Full inherited model/optimizer coverage differs')
    work=output/'owned-workspace';work.mkdir(mode=0o700)
    uid=os.getuid() if hasattr(os,'getuid') else work.stat().st_uid
    atomic_json(work/'.ptcg-owner.json',{'project_id':output.name,'uid':uid,'root':str(work)})
    context=WriteContext(work,output.name,deadline);copies=[]
    def copy(source,relative,expected):
        if time.time()>=deadline:raise TimeoutError('Local migration audit deadline')
        def serialize(stream):
            with Path(source).open('rb') as src:
                while block:=src.read(64*1024):stream.write(block)
        receipt=context.atomic_write(work/relative,serialize,replace=False)
        if receipt['sha256']!=expected:raise ValueError('Bounded copy checksum differs')
        copies.append({'path':relative,**receipt});return receipt
    copy(source_cp,'checkpoint.pt',config['origin_checkpoint_sha256'])
    copy(prefix_path,'archive-prefix.json',config['prefix_sha256'])
    for index,row in enumerate(prefix['tail']):
        copy(origin/row['trajectory_path'],row['trajectory_path'],row['trajectory_sha256'])
        relative=f"series/{row['series_id']:06d}.json";copy(origin/relative,relative,row['summary_sha256'])
        atomic_json(output/'progress.json',{'at':utcnow(),'stage':'copying_bounded_live_tail','tail_records_done':index+1,'tail_records_total':len(prefix['tail'])})
    started=time.monotonic();encoded=atomic_torch_save(work/'checkpoint-reencoded.pt',saved,write_context=context,replace=False)
    encoding_seconds=time.monotonic()-started
    if encoding_seconds+.01<encoded['bytes']/(4*2**20):raise ValueError('Observed checkpoint write bypassed its pacing floor')
    restored=torch.load(work/'checkpoint-reencoded.pt',map_location='cpu',weights_only=False)
    equality=exact_state(saved,restored)
    spec={'prefix_path':'archive-prefix.json','prefix_sha256':config['prefix_sha256'],'initial_checkpoint_sha256':config['origin_checkpoint_sha256']}
    plan=resume_journal(work,saved,spec,sha256(work/'checkpoint.pt'))
    if plan['fixed_statistics']!=read_json(origin/'league.json')['fixed_training_results']:raise ValueError('Compact PPO startup lost league history')
    if len(plan['decisions'])!=config['expected_pending_decisions'] or len(plan['shards'])!=len(prefix['tail']):raise ValueError('PPO startup update input differs')
    if plan['next_series_id']!=saved['next_series_id']:raise ValueError('Compact PPO startup cursor differs')
    # Test interruption on the new audit copy, never on the original checkpoint.
    committed_hash=sha256(work/'checkpoint-reencoded.pt')
    def interrupted(stream):
        stream.write(b'uncommitted'*(64*1024//11+1));stream.flush()
        raise OSError('injected-local-checkpoint-interruption')
    try:context.atomic_write(work/'checkpoint-reencoded.pt',interrupted)
    except OSError as exc:
        if str(exc)!='injected-local-checkpoint-interruption':raise
    else:raise RuntimeError('Failure injection did not execute')
    if sha256(work/'checkpoint-reencoded.pt')!=committed_hash:raise ValueError('Interrupted write replaced committed checkpoint')
    originals={source_cp:config['origin_checkpoint_sha256'],prefix_path:config['prefix_sha256']}
    if any(sha256(path)!=digest for path,digest in originals.items()):raise ValueError('Original evidence modified by migration audit')
    atomic_json(output/'copy-receipts.json',copies)
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'training_started':False,
        'server_training_started':False,'gpu_used':False,'simulator_started':False,'gradient_updates':0,
        'origin_checkpoint_sha256':config['origin_checkpoint_sha256'],'full_actor_tensors':len(saved['model']),
        'optimizer_states':len(saved['optimizer']['state']),'entire_checkpoint_exact_state_comparison':equality,
        'model_optimizer_python_numpy_torch_cuda_rng_preserved':True,'reencoded_checkpoint':encoded,
        'reencoding_wall_seconds':encoding_seconds,'compact_consumed_records_not_copied':len(prefix['consumed']),
        'live_tail_records':len(plan['shards']),'pending_decisions':len(plan['decisions']),
        'checkpoint_cursor':{k:saved[k] for k in ['version','trained_decisions','next_series_id']},
        'startup_loader_used_by_train_ppo':True,'fixed_opponent_statistics_match':True,
        'interrupted_write_preserved_prior_commit':True,'source_evidence_unchanged':True,
        'copy_receipts_sha256':sha256(output/'copy-receipts.json'),'linux_restore_verified':False,
        'server_resource_acceptance':False,'archive_ack_streaming_integrated':False,
        'two_recovery_point_runtime_integrated':False,'production_install_or_training_admitted':False,'promotion_eligible':False})
