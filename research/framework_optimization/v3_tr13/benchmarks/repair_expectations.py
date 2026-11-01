"""Rebuild archive lineage expectations from original inputs, never restored tails."""
import os,sys,time
from common import *
def main():
    sys.path.insert(0,str(V3/'baseline'));import torch,psutil
    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[-3:]);torch.set_num_threads(2)
    from ptcg_lab.archive_prefix import compact_record
    from ptcg_lab.archive_state import advance_state
    from ptcg_lab.framework_v2_training import semantic_digest
    job=job_path('B3-baseline');spec=read(V3/'reports/INPUTS.json');os.environ['V3_DEADLINE']=str(time.time()+1800)
    parent=torch.load(ROOT/spec['parent']['path'],map_location='cpu',weights_only=False);final=torch.load(job/'expected.pt',map_location='cpu',weights_only=False)
    records=[]
    for row in spec['trajectories'][:2]:
        path=ROOT/row['summary']['path']
        if sha(path)!=row['summary']['sha256']:raise ValueError('Original summary bytes changed')
        records.append(compact_record(read(path),row['summary']['sha256']))
    acks={}
    from ptcg_lab.archive_ack import validate_ack
    from ptcg_lab.archive_compressed import verify_archive
    for p in (job/'archive-acks').glob('*.json'):
        ack=read(p);validate_ack(ack,job.name,1,ack['object'],ack['sha256'],ack['bytes']);acks[ack['object']]=ack
    for row in records:
        for name in (row['trajectory_path'],f"series/{row['series_id']:06d}.json"):verify_archive(job.parent/'archive',acks[name])
    previous=parent['archive_state'];origin=read(job/'prefix.json');digest=sha(job/'prefix.json');index={};states={}
    for generation in range(4):
        cursor={k:(parent[k] if generation<2 else final[k]) for k in ('version','trained_decisions')};cursor['next_series_id']=474
        previous=advance_state(previous,origin,digest,cursor,records,acks,job.name,1);states[generation]=previous
        if generation>=2:
            expected=dict(final,archive_state=previous);expected.pop('state_semantic_sha256',None);expected['state_semantic_sha256']=semantic_digest(expected)
            name=f'expected-tail-verified-{generation:04d}.pt';target=job/name
            if target.exists():raise ValueError('Do not overwrite producer evidence')
            torch.save({k:v for k,v in expected.items() if k not in ('model','optimizer')},target);index[str(generation)]=dict(path=name,sha256=sha(target))
    strict_state(states[3],final['archive_state'])
    write(job/'expected-tails.json',index)
    repair=V3/'repairs/per-generation-expectation-01';repair.mkdir(exist_ok=False)
    write(repair/'diagnosis.json',dict(at=time.time(),cause='Independent consecutive saves advance parent_state_id; fixture incorrectly compared both points to final producer state',
        failed_receipt='reports/B3-baseline-replay-cold-process.json',new_actor_optimizer_steps=0,numerical_or_storage_alias_tolerance_relaxed=False,
        reference_method='Replay4 original archive_state transitions from original full parent, original verified complete BO3 records and own fully verified canonical ACKs; final independently reconstructed state strictly equals original pre-restore producer expectation',
        states={str(k):v['state_id'] for k,v in states.items()},full_point_expectation_sha256s=index))
    print('Per-generation archive expectations independently reconstructed; completed PPO unchanged')
if __name__=='__main__':main()
