"""Reconstruct completed hybrid model bytes from its durable archive receipt."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ptcg_lab.io import read_json,sha256,atomic_json,utcnow
from ptcg_lab.archive_compressed import verify_archive
from ptcg_lab.remote_lease import validate_no_unreconciled_remote

parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
root=Path(__file__).resolve().parents[1];run=(root/'runs'/args.run).resolve()
if run.parent!=root/'runs':raise ValueError('Training run escaped')
state=read_json(root/'research/autonomy/state.json')
if state.get('active_job') or (root/'research/autonomy/job.lock').exists():raise RuntimeError('Collect active job before materialization')
config=read_json(run/'config.json');metrics=read_json(run/'metrics.json')
if config.get('check_variant') not in ('scientific_continue','fresh_scientific','derived_scientific') or not metrics.get('complete') or not (metrics.get('scientific_continuation') or metrics.get('scientific_training')):
    raise RuntimeError('A completed scientific hybrid continuation is required')
lease=read_json(run/'remote-job.json');validate_no_unreconciled_remote(root,{'remote_job':lease})
archive=run/'online-archive';records=[read_json(p) for p in sorted((archive/run.name/'acks').glob('*.json'))]
if not records:raise RuntimeError('No durable raw archive receipts')
for ack in records:verify_archive(archive,ack)
models=[ack for ack in records if ack['object']=='model.pt']
if len(models)!=1 or models[0]['sha256']!=metrics['model_sha256']:raise ValueError('Candidate model receipt differs')
ack=models[0];data=verify_archive(archive,ack,collect=True)
if hashlib.sha256(data).hexdigest()!=metrics['model_sha256']:raise ValueError('Reconstructed model differs')
target=run/'model.pt'
if target.exists():
    if sha256(target)!=ack['sha256']:raise ValueError('Existing candidate model conflicts')
else:
    tmp=run/'model.from-durable-archive.partial'
    with tmp.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.link(tmp,target);tmp.unlink()
atomic_json(run/'hybrid-materialization.json',{'at':utcnow(),'model_archive_ack':ack,
    'raw_model_sha256':sha256(target),'all_archive_receipts_reconstructed_and_verified':len(records),
    'no_model_training_or_weight_reencoding':True,'promotion_eligible':False})
print({'run_id':run.name,'raw_model_sha256':sha256(target),'verified_archive_receipts':len(records),'evaluation_qualification':False})
