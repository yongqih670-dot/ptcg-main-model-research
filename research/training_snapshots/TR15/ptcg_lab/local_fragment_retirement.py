"""Exact two-file retirement of reconstructible obsolete public-wheel prefixes."""
import hashlib
import os
from pathlib import Path
import time
from .io import atomic_json,read_json,sha256,utcnow
from .local_cache_retirement import windows_security,owned,identity,LocalReadPacer,digest_stream

PAIRS={
 'runs/20261024T123745Z_hybrid-single-response_8f7571/wheelhouse/nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl.stream.partial':
 'runs/20261024T124736Z_hybrid-digest-assembly_cbadfd/wheelhouse/nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl',
 'runs/20261024T120920Z_hybrid-range-wheelhouse_f7cf24/wheelhouse/nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.partial':
 'runs/20261024T123745Z_hybrid-single-response_8f7571/wheelhouse/nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'}


def verify_pair(root,row,sid,owner,pacer,runs_owner):
    if PAIRS.get(row['partial'])!=row['wheel']:raise ValueError('Retirement target is outside the two exact public prefixes')
    part=owned(root,root/row['partial'],sid,owner,runs_owner)
    wheel=owned(root,root/row['wheel'],sid,owner,runs_owner)
    before=identity(part);retained=identity(wheel)
    if before!=row['file_identity'] or retained!=row['wheel_identity'] or not 0<row['bytes']<row['wheel_bytes']:
        raise ValueError('Original public prefix or full wheel identity differs')
    with part.open('rb') as f:partial_digest,n=digest_stream(f,pacer)
    whole=hashlib.sha256();prefix=hashlib.sha256();left=n;count=0
    with wheel.open('rb') as f:
        while block:=f.read(1024*1024):
            pacer.consume(len(block));whole.update(block);count+=len(block)
            if left:
                piece=block[:left];prefix.update(piece);left-=len(piece)
    if (n!=row['bytes'] or partial_digest!=row['sha256'] or partial_digest!=prefix.hexdigest()
            or count!=row['wheel_bytes'] or whole.hexdigest()!=row['wheel_sha256']
            or identity(part)!=before or identity(wheel)!=retained):
        raise ValueError('Public prefix is not exactly reconstructible from the retained full wheel')
    source=root/'runs'/row['source_run_id'];attempts=source/'attempts.json'
    if (source.name!=Path(row['partial']).parts[1] or sha256(attempts)!=row['source_attempts_sha256']
            or not read_json(attempts) or any(x['status']!='failed' for x in read_json(attempts))):
        raise ValueError('Original failed download terminal evidence differs')
    return dict(row,prefix_equal=True)


def run(config,output):
    root=Path(config['project_path']).absolute();output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':
        raise RuntimeError('Exact local controlled runner required')
    if (config.get('job_kind'),config.get('retirement_variant'),config.get('maximum_attempts'))!=('local_cache_retirement','public_download_prefixes',1):
        raise ValueError('Unexpected prefix retirement role')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if not 0<deadline-time.time()<=180.5:raise ValueError('One180s prefix retirement only')
    if sha256(Path(__file__))!=config['retirement_module_sha256']:raise ValueError('Pinned prefix retirement source differs')
    proof_path=root/config['inspection_evidence']['path']
    if sha256(proof_path)!=config['inspection_evidence']['sha256']:raise ValueError('Read-only prefix proof changed')
    inspection=read_json(proof_path);rows=inspection['rows']
    if {r['partial']:r['wheel'] for r in rows}!=PAIRS or len(rows)!=2:
        raise ValueError('Only the two exact public prefixes are permitted')
    sid,owner=windows_security();allowed=inspection['runs_container_owner_sid_sha256']
    if hashlib.sha256(sid.encode()).hexdigest()!=inspection['owner_sid_sha256']:raise ValueError('Current OS owner differs')
    owned(root,output,sid,owner,allowed)
    pacer=LocalReadPacer(16*2**20,deadline-20)
    # Verify every byte of both retained wheels and both prefixes before any
    # unlink. A rejected second prefix leaves the first one untouched too.
    verified=[verify_pair(root,r,sid,owner,pacer,allowed) for r in rows]
    proof={'at':utcnow(),'all_prefixes_and_full_wheels_verified_before_any_unlink':True,'files':verified,
        'source_failed_attempts_logs_and_all_scientific_outputs_unchanged':True}
    manifest=output/'verified-retirement-manifest.json';atomic_json(manifest,proof)
    atomic_json(output/'retirement-intent.json',{'at':utcnow(),'manifest_sha256':sha256(manifest),
        'maximum_files':2,'maximum_bytes':sum(r['bytes'] for r in verified),'no_directory_deletion':True})
    released=0
    for index,row in enumerate(verified):
        if time.time()>deadline-20:raise TimeoutError('Retirement deadline; preserve previous exact receipts')
        part=owned(root,root/row['partial'],sid,owner,allowed)
        wheel=owned(root,root/row['wheel'],sid,owner,allowed)
        if identity(part)!=row['file_identity'] or identity(wheel)!=row['wheel_identity']:
            raise ValueError('Verified public prefix or retained wheel changed before unlink')
        with part.open('rb') as f:digest,count=digest_stream(f,pacer)
        if count!=row['bytes'] or digest!=row['sha256'] or identity(part)!=row['file_identity']:
            raise ValueError('Prefix changed immediately before own single-file retirement')
        atomic_json(output/f'unlink-{index:02d}.json',{'status':'intent',**row})
        part.unlink();released+=count
        atomic_json(output/f'unlink-{index:02d}.json',{'status':'retired',**row})
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'retired_files':2,
        'retired_logical_bytes':released,'manifest_sha256':sha256(manifest),
        'both_full_retained_wheels_unchanged':True,'all_original_prefix_bytes_exactly_recoverable':True,
        'rehydration':'For each verified row, copy exactly bytes[0:bytes] from retained wheel into its missing partial path; verify stored SHA256.',
        'all_scientific_data_code_weights_original_attempts_and_stops_unchanged':True,'promotion_eligible':False})
