"""Retire proven duplicate copies, preserving all original scientific bytes."""
import hashlib,os,time
from pathlib import Path
from .io import read_json,atomic_json,sha256,utcnow
from .local_cache_retirement import windows_security,identity,LocalReadPacer,digest_stream
from .evidence_tiering import ARCHIVES,TR07_ARCHIVE,TR08_FIRST_ARCHIVE,owned_path,target_relative,source_blocks


def delete_access(path):
    import ctypes
    from ctypes import wintypes as w
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,ctypes.c_void_p,w.DWORD,w.DWORD,w.HANDLE];k.CreateFileW.restype=w.HANDLE
    k.CloseHandle.argtypes=[w.HANDLE]
    # Request DELETE access without marking deletion or changing permissions.
    handle=k.CreateFileW(str(path),0x10000|0x80,7,None,3,0x00200000,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    k.CloseHandle(handle)


def run(config,output):
    root=Path(config['project_path']).absolute();output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':raise RuntimeError('Controlled local runner required')
    if (config.get('job_kind'),config.get('retirement_variant'),config.get('maximum_attempts'))!=('local_cache_retirement','forensic_replicas',1):
        raise ValueError('Unexpected forensic replica maintenance role')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if not 0<deadline-time.time()<=600.5:raise ValueError('One600s exact forensic maintenance scope only')
    if sha256(Path(__file__))!=config['retirement_module_sha256']:raise ValueError('Pinned forensic retirement source changed')
    spec=config['inspection_evidence'];inspection=root/spec['path']
    if sha256(inspection)!=spec['sha256']:raise ValueError('Full original-byte inspection changed')
    proof=read_json(inspection);tokens=proof['trusted_execution_tokens'];rows=proof['rows'];sid,owner=windows_security()
    if hashlib.sha256(sid.encode()).hexdigest()!=tokens['desktop']:raise ValueError('Expected project user execution identity')
    if len(rows)>128 or sum(r['bytes'] for r in rows)>2*2**30:raise ValueError('Exact replica inventory exceeds declared scope')
    if len({r['target'] for r in rows})!=len(rows):raise ValueError('Duplicate replica retirement target')
    scope=set(proof['archive_evidence'])
    if scope not in (ARCHIVES,{TR07_ARCHIVE},{TR08_FIRST_ARCHIVE}):raise ValueError('Unreviewed exact forensic archive scope')
    if any(Path(r['target']).parts[1] not in scope for r in rows):raise ValueError('Tiered target is outside this exact registered scope')
    state=read_json(root/'research/autonomy/state.json')
    if scope=={TR08_FIRST_ARCHIVE}:
        from .remote_lease import validate_no_unreconciled_remote
        source=root/'runs/20261026T085757Z_tr08-actor-lr3e6-CPU-sampling_868ff0'
        validate_no_unreconciled_remote(root,{'remote_job':read_json(source/'remote-job.json')})
        spec=proof['actual_full_recovery_audit'];audit=root/spec['path']
        if sha256(audit)!=spec['sha256'] or not read_json(audit).get('complete') or read_json(audit).get('full_archive_files_verified')!=907:
            raise ValueError('Complete cold TR08 first-source audit required')
        if any(r['retained_source']['kind']!='durable_gzip_ACK' or r['retained_source']['ACK']['run_id']!=source.name for r in rows):
            raise ValueError('TR08 forensic copies require the immutable original first-run primary GZIP')
    for name in (['TR04','TR05','TR06'] if scope==ARCHIVES else ['TR07'] if scope=={TR07_ARCHIVE} else []):
        old=state['research_stage']['hypotheses'][name]
        if old['status']!='invalid' or old.get('remaining_original_training_attempts')!=0:
            raise ValueError('Only closed, nonresumable failed hypotheses may tier forensic duplicates')
    for rid,evidence in proof['archive_evidence'].items():
        if rid not in scope:raise ValueError('Forensic archive outside the exact permitted scope')
        for name,digest in evidence.items():
            if sha256(root/'runs'/rid/name)!=digest:raise ValueError('Original archive receipts or cost/attempt evidence changed')
    pacer=LocalReadPacer(16*2**20,deadline-25)
    for row in rows:
        target_relative(row['target']);path=owned_path(root,root/row['target'],tokens,owner)
        if identity(path)!=row['file_identity'] or hashlib.sha256(owner(path).encode()).hexdigest()!=row['owner_token_sha256']:
            raise ValueError('Original duplicate file identity differs')
        with path.open('rb') as raw:digest,n=digest_stream(raw,pacer)
        if (digest,n)!=(row['sha256'],row['bytes']):raise ValueError('Duplicate file full original bytes differ')
        for _ in source_blocks(root,row,tokens,owner,pacer):pass
        delete_access(path)
    manifest=output/'tiered-evidence-manifest.json'
    atomic_json(manifest,{'at':utcnow(),'all_original_sources_and_targets_verified_before_any_unlink':True,
        'trusted_execution_tokens':tokens,'archive_evidence':proof['archive_evidence'],'rows':rows,
        'original_receipts_code_primary_weights_costs_and_stop_decisions_unchanged':True})
    manifest_ref={'path':manifest.relative_to(root).as_posix(),'sha256':sha256(manifest)}
    catalog_path=root/'research/autonomy/evidence-locations.json'
    catalog=read_json(catalog_path) if catalog_path.exists() else {'project_path':str(root),'trusted_execution_tokens':tokens,'objects':{}}
    if catalog['project_path']!=str(root) or catalog['trusted_execution_tokens']!=tokens:raise ValueError('Existing evidence location catalog differs')
    for row in rows:
        if row['target'] in catalog['objects']:raise ValueError('Original evidence already has another storage transaction')
        catalog['objects'][row['target']]={'sha256':row['sha256'],'bytes':row['bytes'],'manifest':manifest_ref}
    atomic_json(catalog_path,catalog)
    atomic_json(output/'retirement-intent.json',{'at':utcnow(),'manifest':manifest_ref,'files':len(rows),
        'logical_bytes':sum(r['bytes'] for r in rows),'catalog_sha256':sha256(catalog_path),'primary_sources_never_unlinked':True})
    released=0
    for i,row in enumerate(rows):
        if time.time()>deadline-25:raise TimeoutError('Exact tiering deadline; preserve all prior per-file receipts')
        path=owned_path(root,root/row['target'],tokens,owner)
        if identity(path)!=row['file_identity']:raise ValueError('Verified redundant replica changed before unlink')
        with path.open('rb') as raw:digest,n=digest_stream(raw,pacer)
        if (digest,n)!=(row['sha256'],row['bytes']) or identity(path)!=row['file_identity']:
            raise ValueError('Replica changed immediately before its single-file unlink')
        atomic_json(output/f'unlink-{i:03d}.json',{'status':'intent','target':row['target'],'manifest':manifest_ref})
        path.unlink();released+=n
        atomic_json(output/f'unlink-{i:03d}.json',{'status':'retired','target':row['target'],'manifest':manifest_ref})
        atomic_json(output/'progress.json',{'at':utcnow(),'stage':'retiring_verified_duplicate_evidence','files':i+1,'logical_bytes':released})
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'retired_duplicate_files':len(rows),
        'retired_logical_bytes':released,'all_original_bytes_retained_in_primary_raw_or_verified_gzip':True,
        'original_archive_metadata_and_all_costs_attempts_stops_unchanged':True,'read_through_catalog':'research/autonomy/evidence-locations.json',
        'manifest':manifest_ref,'no_source_model_or_primary_archive_object_removed':True,'physical_disk_bytes_saved_not_assumed':True,'promotion_eligible':False})
