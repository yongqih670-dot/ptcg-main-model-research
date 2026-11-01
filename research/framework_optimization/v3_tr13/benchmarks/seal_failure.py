"""Seal an already-terminal zero-update failed engineering attempt, not retry it."""
from common import *

def seal(label):
    import torch
    from ptcg_lab.framework_v2_training import semantic_digest,pending_digests
    from ptcg_lab.tensor_io import atomic_torch_save
    from ptcg_lab.archive_outbox import receive_one
    from ptcg_lab.archive_compressed import verify_archive
    from local_adapter import local_outbox,install_archive_publication_adapter
    diagnosis=read(V3/'repairs/D-cadence-01/DIAGNOSIS.json')
    original=read(V3/'reports'/f'{label}-perf-process.json')
    if label!=diagnosis['invalid_case'] or original['status']!='terminal' or original['returncode']!=1 or original['exitcode_source']!='Popen.wait':raise ValueError('Only identified terminal invalid attempt')
    job=job_path(label);setup_environment(job);backend,_=native_backend(job,'baseline');spec=read(V3/'reports/INPUTS.json')
    old_owner=job/'local-engineering-owner.json';backup=job/'pre-seal-local-engineering-owner.json'
    if backup.exists():raise ValueError('Existing sealing attempt retained')
    backup.write_bytes(old_owner.read_bytes());owner=read(old_owner);owner['parent_token']=os.environ['V3_TOKEN'];write(old_owner,owner)
    install_archive_publication_adapter();outbox=local_outbox(job,1,prefix_sha256=sha(job/'prefix.json'))
    archive_root=job.parent/'archive';catalog={k:ROOT/v['path'] for k,v in spec['immutable_catalog'].items()}
    for p in (archive_root/job.name/'acks').glob('*.json'):
        ack=read(p);verify_archive(archive_root,ack);catalog[ack['sha256']]={'archive_root':archive_root,'ack':ack}
    tails=read(job/'expected-tails.json');points=[];latest=None
    try:
        for which in (0,1):
            value,point=backend.restore(which);payload=dict(value);producer_hash=payload.pop('state_semantic_sha256')
            if semantic_digest(payload)!=producer_hash:raise ValueError('Committed pre-serialization producer state hash differs')
            if value['version']!=11 or value['trained_decisions']!=79641 or value['model_state_revision']!=0 or value['partial_ppo'] is not None:raise ValueError('Failed case is not zero optimizer updates')
            if str(point['generation']) in tails:
                ref=tails[str(point['generation'])]
                if sha(job/ref['path'])!=ref['sha256']:raise ValueError('Existing producer tail changed')
                producer_tail=torch.load(job/ref['path'],map_location='cpu',weights_only=False)
                strict_state(producer_tail,{k:v for k,v in value.items() if k not in ('model','optimizer')})
            else:
                name=f"sealed-tail-{point['generation']:04d}.pt";atomic_torch_save(job/name,{k:v for k,v in value.items() if k not in ('model','optimizer')},replace=False)
                tails[str(point['generation'])]=dict(path=name,sha256=sha(job/name),source='fresh full graph restore matched original pre-serialization semantic hash; failed-attempt retention only')
            required=pending_digests(value['archive_state'])
            commit=dict(generation=point['generation'],cursor=point['cursor'],checkpoint=point['model']['object'],prefix=backend.checked('prefix.json',sha(job/'prefix.json')),pending=[backend.checked(n,d) for n,d in required.items()],framework_v2=point,archive_references=backend.bridge.checkpoints.archive_references())
            outbox.queue_recovery(commit);points.append(dict(generation=point['generation'],producer_semantic_hash_verified=True));latest=value
        class Stream:
            def request(self,operation,payload,*,timeout=None):
                guard()
                if operation=='archive_ack':return outbox.accept_ack(payload['ack'])
                if operation=='archive_chunk':return outbox.chunk(payload['object_sequence'],payload['offset'],payload['count'])
                raise ValueError('Unexpected operation')
        stream=Stream()
        for row in outbox.records():
            if outbox._ack(row) is None:
                ack=receive_one(stream,archive_root,row,catalog.get(row['sha256']),storage_codec='gzip');catalog[row['sha256']]={'archive_root':archive_root,'ack':ack}
        if outbox.pending()['pending_objects']:raise ValueError('Missing full archive ACK')
        write(job/'expected-tails.json',tails);atomic_torch_save(job/'expected.pt',latest,replace=False)
        result=dict(passed=True,points=points,expected_sha256=sha(job/'expected.pt'),scientific_credit=0,optimizer_updates=0,performance_result_valid=False,scope='Retention and cold verification of failed case only; no replay, new BO3 or PPO')
        write(V3/'reports'/f'{label}-sealed.json',result);return result
    finally:backend.close()
