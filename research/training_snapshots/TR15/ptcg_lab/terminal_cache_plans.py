"""Verify complete cold originals before an exact terminal replica retirement."""
import hashlib
from pathlib import Path
from .io import read_json,sha256
from .remote_lease import validate_no_unreconciled_remote


def validate_local_plans(root,plans,allowed_runs):
    root=Path(root).resolve()
    if not plans or len(plans)>2:raise ValueError('Exact two terminal TR08 sources only')
    for plan in plans:
        name=plan['source']['run_id']
        if name not in allowed_runs:raise ValueError('Retirement source is not a closed original TR08 attempt')
        original=root/'runs'/name;validate_no_unreconciled_remote(root,{'remote_job':read_json(original/'remote-job.json')})
        proof=plan['durable_local_archive'];archive=root/'runs'/proof['run_id']
        if archive.resolve().parent!=root/'runs' or sha256(archive/'file-receipts.json')!=proof['receipts_sha256']:
            raise ValueError('Original full cold archive receipts changed')
        metrics=read_json(archive/'metrics.json')
        if not metrics.get('complete') or not metrics.get('all_files_hash_verified_and_fsynced_locally') or metrics.get('source_run_id')!=name:
            raise ValueError('Completed source-specific full cold archive required')
        receipts={r['path']:r for r in read_json(archive/'file-receipts.json')}
        if len(plan['objects'])>128 or sum(r['bytes'] for r in plan['objects'])>2*2**30:raise ValueError('Terminal replica bound exceeded')
        for row in plan['objects']:
            expected=receipts.get(row['path'])
            if expected is None or any(row[k]!=expected[k] for k in ['path','sha256','bytes','file_identity']):raise ValueError('Exact archived remote file identity differs')
            path=archive/'archive'/row['path']
            if path.exists():
                if path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:raise ValueError('Full retained raw original differs')
            else:
                from .evidence_tiering import read_missing
                data=read_missing(path)
                if data is None or len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Full lossless retained original readback differs')
