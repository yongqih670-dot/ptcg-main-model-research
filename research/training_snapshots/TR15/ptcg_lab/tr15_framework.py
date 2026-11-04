"""New phase admission for unchanged delivered v3 algorithms; no old-ticket opt-in."""
from pathlib import Path
import os,re
from .io import read_json,sha256,atomic_json
from .framework_v3_runtime import flags,ExitEvidence
from .framework_v3_exit import RoleObserver
from .tr15_scope import validate_remote,AUTH

SWITCHES={'control_cache':True,'journal_cache':True,'outbox_cache':True,'capacity_transaction':False,'progress_coalescing':True,'exit_evidence':True}

REQUIRED=('parent_acceptance_passed','checkpoint_model_optimizer_all_RNG_permutation_exact',
          'RNG_permutation_input_counters_exact','PPO_continuation_within_measured_GPU_repeatability',
          'native_selfplay_whole_BO3_snapshot_fixed','full_update_snapshot_refresh_verified',
          'partial_restore_retains_preupdate_snapshot','two_recovery_points_cold_restored',
          'complete_archive_ACK_verified','cache_invalidation_stop_verified')

def validate_manifest_local(root,config):
    root=Path(root);ref=config['framework_v3_manifest']
    if sha256(root/ref['path'])!=ref['sha256']:raise ValueError('Native runtime manifest changed')
    manifest=read_json(root/ref['path'])
    if manifest['authority']!=AUTH or manifest['optimizations']!=SWITCHES:raise ValueError('Native runtime scope changed')
    if sha256(root/manifest['delivery']['path'])!=manifest['delivery']['sha256']:raise ValueError('Original delivered framework changed')
    for name,digest in manifest['runtime_sources'].items():
        if sha256(root/name)!=digest:raise ValueError('Prospective native runtime changed: '+name)
    return manifest

def manifest_fields(root,config):
    import base64
    validate_manifest_local(root,config)
    ref=config['framework_v3_manifest']
    result={'framework_v3_manifest_sha256':ref['sha256'],
            'framework_v3_manifest_base64':base64.b64encode((Path(root)/ref['path']).read_bytes()).decode(),
            'framework_v3_optimizations':dict(SWITCHES),
            'TR15_native_acceptance':config.get('TR15_native_acceptance') is True}
    if result['TR15_native_acceptance']:
        if config.get('engineering_only') is not True:raise ValueError('Disposable native test must be engineering only')
        result['engineering_only']=True
        if config.get('TR15_engineering_recovery'):result['TR15_engineering_recovery']=config['TR15_engineering_recovery']
    else:result['framework_v3_native_acceptance']=config['framework_v3_native_acceptance']
    return result

def validate_report(report):
    if not report.get('complete') or report.get('scientific_decision_credit')!=0 or not all(report.get(k) is True for k in REQUIRED):
        raise ValueError('Actual native/GPU/selfplay differences are incomplete')
    return report

def admit_ticket(root,job,ticket,manifest):
    import base64,hashlib,json
    from .server_workspace import inside
    raw=base64.b64decode(manifest['framework_v3_manifest_base64'],validate=True)
    if hashlib.sha256(raw).hexdigest()!=manifest['framework_v3_manifest_sha256']:raise ValueError('Native manifest transport changed')
    bound=json.loads(raw)
    if bound['authority']!=AUTH or bound['optimizations']!=SWITCHES or manifest['framework_v3_optimizations']!=SWITCHES:
        raise ValueError('Native manifest switches changed')
    package=job/'control'
    for name,digest in bound['runtime_sources'].items():
        path=inside(root,package/name)
        if sha256(path)!=digest:raise ValueError('Native runtime package mismatch: '+name)
    path=job/'framework-v3-native-manifest.json'
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    engineering=manifest.get('TR15_native_acceptance') is True
    if engineering!=bool(manifest['engineering_only']):raise ValueError('Native framework role differs')
    ticket.update(framework_v3_approved=True,framework_v3_manifest_sha256=manifest['framework_v3_manifest_sha256'],
                  TR15_framework_role='engineering' if engineering else 'scientific')
    if not engineering:
        from .retained_source import resolve
        ref=manifest['framework_v3_native_acceptance'];source,prior=resolve(root,ticket['project_id'],ref['source'])
        path=source/'TR15-native-selfplay-acceptance.json'
        if sha256(path)!=ref['sha256']:raise ValueError('Actual native acceptance bytes changed')
        report=validate_report(read_json(path))
        if report['framework_v3_manifest_sha256']!=manifest['framework_v3_manifest_sha256']:
            raise ValueError('Actual native acceptance uses different runtime')
        ticket['TR15_native_acceptance_sha256']=ref['sha256']

def payload_fields(manifest):
    result={k:manifest[k] for k in ('framework_v3_manifest_sha256','framework_v3_optimizations','TR15_native_acceptance')}
    if not result['TR15_native_acceptance']:
        result['framework_v3_native_acceptance_sha256']=manifest['framework_v3_native_acceptance']['sha256']
    if manifest.get('TR15_engineering_recovery'):result['TR15_engineering_recovery']=manifest['TR15_engineering_recovery']
    return result

def admitted_switches(config,output):
    output=Path(output).resolve();ticket=read_json(output/'ticket.json')
    if (ticket['run_id'],str(ticket['lease_epoch']))!=(os.environ.get('PTCG_JOB_RUN_ID'),os.environ.get('PTCG_JOB_LEASE_EPOCH')):
        raise ValueError('TR15 framework process identity differs')
    validate_remote(ticket['TR13_runtime_scope'],ticket['mode'],ticket['new_diagnostic_identity'])
    engineering=config.get('TR15_native_acceptance') is True
    if ticket.get('TR15_framework_role')!=('engineering' if engineering else 'scientific') or ticket.get('framework_v3_approved') is not True:
        raise ValueError('New framework is not admitted for this exact role')
    bound=ticket.get('framework_v3_manifest_sha256')
    if not isinstance(bound,str) or not re.fullmatch('[0-9a-f]{64}',bound) or bound!=config.get('framework_v3_manifest_sha256'):
        raise ValueError('Framework runtime binding is missing')
    path=output/'framework-v3-native-manifest.json'
    if sha256(path)!=bound:raise ValueError('Registered native framework manifest changed')
    manifest=read_json(path);package=Path(__file__).resolve().parent.parent
    if manifest['authority']!=AUTH or manifest['optimizations']!=SWITCHES or flags(config.get('framework_v3_optimizations'))!=SWITCHES:
        raise ValueError('Delivered optimization switches or phase authority changed')
    for relative,digest in manifest['runtime_sources'].items():
        file=(package/relative).resolve()
        if not file.is_relative_to(package) or file.is_symlink() or sha256(file)!=digest:raise ValueError('Registered native runtime source changed')
    if not engineering:
        proof=ticket.get('TR15_native_acceptance_sha256')
        if not re.fullmatch('[0-9a-f]{64}',proof or '') or proof!=config.get('framework_v3_native_acceptance_sha256'):
            raise ValueError('Scientific opt-in requires the completed actual native/GPU/selfplay acceptance')
    os.environ['PTCG_TR15_V3_ACTIVE']='1'
    os.environ['PTCG_TR15_JOB_DIRECTORY']=str(output)
    return dict(SWITCHES)

def observe(role,output,epoch=None):
    output=Path(output).resolve();epoch=int(epoch or os.environ['PTCG_JOB_LEASE_EPOCH'])
    if output.name!=os.environ.get('PTCG_JOB_RUN_ID'):raise ValueError('Exit observer job identity differs')
    process={'pid':os.getpid()}
    if os.name=='posix':
        process.update(start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                       host_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    else:
        # The GPU controller environment deliberately has fewer packages.
        # Bind native Windows creation time without installing dependencies.
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        created,ended,kernel_time,user_time=(wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(kernel.GetCurrentProcess(),*[ctypes.byref(t) for t in (created,ended,kernel_time,user_time)]):
            raise ctypes.WinError(ctypes.get_last_error())
        process['creation_FILETIME_100ns']=(created.dwHighDateTime<<32)|created.dwLowDateTime
    return RoleObserver(ExitEvidence(output,output.name,epoch,output.name+'-'+str(epoch),role,process_identity=process))
