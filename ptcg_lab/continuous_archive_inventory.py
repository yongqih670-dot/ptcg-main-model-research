"""Verify full existing archive bytes before scheduling a delta acceptance."""
import os
from pathlib import Path
import time
from .io import read_json,atomic_json,sha256,utcnow
from .archive_compressed import verify_archive
from .archive_ack import archive_root,safe_target

PARENT='20261027T094058Z_tr10-original-second-actual-counter_2063d8'
FIRST='20261027T073939Z_tr10-lambda1-diagnostic_df4705'
NUMERIC='20261027T132356Z_authorized-parent-numeric-600s_b612a4'


def required_objects(root):
    previous=read_json(root/'research/autonomy/engineering/continuous-authorized-600s-result-20261027.json')
    first=previous['actual_local_ACKs_full_reconstruction_verified'][:2]
    league=read_json(root/'runs'/PARENT/'remote-results/league.json')['snapshots']
    return [{'object':r['object'],'sha256':r['sha256'],'bytes':r['bytes']} for r in first]+[
        {'object':r['path'],'sha256':r['sha256'],'bytes':19529025} for r in league]


def verify_existing(root,required,deadline):
    matches={r['sha256']:[] for r in required}
    for path in (root/'runs').glob('*/online-archive/*/acks/*.json'):
        if time.time()>deadline-15:raise TimeoutError('Archive inventory metadata deadline')
        ack=read_json(path)
        if ack.get('sha256') in matches:matches[ack['sha256']].append(path)
    verified=[];missing=[];rejected=[]
    for row in required:
        started=time.monotonic();selected=None
        candidates=sorted(matches[row['sha256']],key=lambda p:(0 if PARENT in p.parts else 1 if FIRST in p.parts else 2,str(p)))
        for path in candidates:
            if time.time()>deadline-15:raise TimeoutError('Archive full hash deadline')
            try:
                ack=read_json(path);base=path.parents[2]
                if (ack['sha256'],ack['bytes'])!=(row['sha256'],row['bytes']):raise ValueError('Original size differs')
                verify_archive(base,ack)
                encoded=safe_target(archive_root(base),ack['archive_relative']);st=encoded.stat()
                selected={'storage':'gzip_ack','sha256':row['sha256'],'archive_root':base.relative_to(root).as_posix(),'ack':ack}
                verified.append({'required':row,'local_archive_source':selected,
                    'ack_evidence':{'path':path.relative_to(root).as_posix(),'sha256':sha256(path)},
                    'encoded_identity':[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns],
                    'full_encoded_and_original_SHA_verified':True,'verification_seconds':time.monotonic()-started})
                break
            except (OSError,ValueError,EOFError) as exc:
                rejected.append({'path':path.relative_to(root).as_posix(),'error':type(exc).__name__+': '+str(exc)})
        if selected is None:missing.append(row)
    return verified,missing,rejected


def run(config,output):
    root=Path(config['project_path']).resolve();output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':raise RuntimeError('Controlled audit required')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX']);started=time.monotonic()
    if config['maximum_attempts']!=1 or deadline-time.time()>180.5:raise ValueError('One180s local hash audit only')
    for name,digest in config['source_evidence'].items():
        if sha256(root/name)!=digest:raise ValueError('Archive source evidence changed')
    required=required_objects(root)
    if required!=config['required_objects'] or len(required)!=8:raise ValueError('Exact complete parent inventory required')
    verified,missing,rejected=verify_existing(root,required,deadline)
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,
        'required_objects':required,'verified':verified,'missing':missing,'rejected_candidates':rejected,
        'local_archive_sources':[r['local_archive_source'] for r in verified],
        'verified_original_bytes':sum(r['required']['bytes'] for r in verified),
        'missing_original_bytes':sum(r['bytes'] for r in missing),'verification_seconds':time.monotonic()-started,
        'remote_payload_bytes_downloaded':0,'optimizer_steps_executed':0,'new_games':0,
        'prior_GPU_numeric_evidence_reused_not_relabelled_as_new_test':True,'promotion_eligible':False})
