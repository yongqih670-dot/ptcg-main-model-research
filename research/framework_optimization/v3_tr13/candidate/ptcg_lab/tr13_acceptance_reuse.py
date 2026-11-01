"""Reuse one accepted implementation; validate only newly produced recovery state."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from .io import read_json,sha256

AUTH={'path':'research/autonomy/authorizations/TR13-segment-acceptance-reuse-20261030.json',
      'sha256':'9b2a36846c30fc694937ac40cef5f3a1975fa3a5b87e8f222343dd1db4f53fd9'}
CORE=['tr13_train','framework_v2_training','ppo_transaction','rl_core','rl_collector','rl_sampling',
      'rl_league','rl_resume','ppo_accounting','sampling_parity','training_checkpoint','checkpoint_refs']


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def scientific_sources(package_root):
    package_root=Path(package_root)
    paths=[package_root/'ptcg_lab'/(name+'.py') for name in CORE]
    paths+=sorted((package_root/'ptcg_lab/fo_v2').glob('*.py'))
    return {p.relative_to(package_root).as_posix():sha256(p) for p in paths}


def runtime_fingerprint(project_root,gpu_uuid):
    project_root=Path(project_root)
    if Path(sys.prefix).resolve()!=project_root/'.venv':raise ValueError('Reuse requires the same private environment')
    versions={};records={};binaries={}
    for name in ['torch','numpy','kaggle-environments','nvidia-cudnn-cu12','nvidia-cublas-cu12','nvidia-cuda-runtime-cu12']:
        try:
            distribution=importlib.metadata.distribution(name);versions[name]=distribution.version
            for item in distribution.files or []:
                path=Path(distribution.locate_file(item))
                if str(item).endswith('.dist-info/RECORD'):records[str(path)]=sha256(path)
                elif str(item).endswith('.so') or '.so.' in str(item):
                    stat=path.stat();binaries[str(path)]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'ctime_ns':stat.st_ctime_ns,'inode':stat.st_ino}
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    device=subprocess.run(['nvidia-smi','-i',gpu_uuid,'--query-gpu=name,driver_version,compute_cap',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    return {'python':sys.version,'private_prefix':str(Path(sys.prefix).resolve()),'distributions':versions,
            'installation_RECORD_SHA256':records,'installed_binary_identity':binaries,
            'GPU_model_driver_compute_capability':device}


def format_signature(value):
    def shape(item):
        if hasattr(item,'shape') and hasattr(item,'dtype'):return {'shape':list(item.shape),'dtype':str(item.dtype)}
        return {'scalar_type':type(item).__name__}
    return {'model':{k:shape(v) for k,v in value['model'].items()},
        'optimizer_states':{str(k):{n:shape(v) for n,v in row.items()} for k,row in value['optimizer']['state'].items()},
        'optimizer_parameter_groups':[row['params'] for row in value['optimizer']['param_groups']],
        'RNG_keys':sorted(value['rng']),'CUDA_RNG_streams':len(value['rng']['cuda']),
        'CPU_sampler_RNG':shape(value['cpu_sampler_rng'])}


def envelope(document,reference):
    return {'document':document,'canonical_sha256':canonical_hash(document),'reference':reference}


def certificate(scope):
    document=scope['document']
    if scope['canonical_sha256']!=canonical_hash(document) or document.get('authority')!=AUTH or not document.get('complete'):
        raise ValueError('Accepted TR13 reuse certificate is absent or changed')
    if document.get('recovery_index_schema')!=3 or not document.get('latest_only') or not document.get('two_recovery_points_retained'):
        raise ValueError('Recovery reuse scope differs')
    if not {'native_GPU','initial_full_archive','initial_two_cold_points','first_real_CUDA_partial_PPO'}<=set(document.get('accepted_evidence',{})):
        raise ValueError('Necessary first acceptance evidence is incomplete')
    return document


def validate_local(root,scope):
    root=Path(root);doc=certificate(scope);ref=scope['reference']
    if sha256(root/AUTH['path'])!=AUTH['sha256'] or sha256(root/ref['path'])!=ref['sha256'] or read_json(root/ref['path'])!=doc:
        raise ValueError('Direct reuse authority or certificate bytes changed')
    if scientific_sources(root)!=doc['scientific_sources']:raise ValueError('Training implementation changed; full difference acceptance required')
    for name,digest in doc.get('collector_runtime_sources',{}).items():
        if sha256(root/name)!=digest:raise ValueError('Collector timing runtime changed')
    for ref in doc['accepted_evidence'].values():
        if sha256(root/ref['path'])!=ref['sha256']:raise ValueError('Prior accepted evidence changed')
    initial=read_json(root/doc['accepted_evidence']['native_GPU']['path'])
    if not initial.get('parent_acceptance_passed') or initial.get('optimizer_steps_executed')!=0:
        raise ValueError('Original zero-update acceptance did not pass')
    partial=read_json(root/doc['accepted_evidence']['first_real_CUDA_partial_PPO']['path'])
    exact=partial.get('model_optimizer_all_RNG_permutation_logs_exact')
    bounded=partial.get('checkpoint_model_optimizer_all_RNG_permutation_exact') and partial.get('RNG_permutation_input_counters_exact') and partial.get('PPO_continuation_within_measured_GPU_repeatability')
    if not partial.get('complete') or not (exact or bounded):
        raise ValueError('First actual CUDA partial-PPO proof did not pass')
    return doc


def validate_targeted_difference(root,scope,reference):
    """Reuse the frozen basis while one explicitly registered collector fix is tested."""
    root=Path(root);doc=certificate(scope)
    if sha256(root/scope['reference']['path'])!=scope['reference']['sha256']:
        raise ValueError('Original accepted certificate changed')
    if sha256(root/reference['path'])!=reference['sha256']:raise ValueError('Targeted difference registration changed')
    difference=read_json(root/reference['path'])
    allowed=difference['allowed_core_changes']
    if difference['basis_certificate']!=scope['reference'] or allowed not in ([],['ptcg_lab/rl_collector.py'],['ptcg_lab/tr13_train.py']):
        raise ValueError('Recovery fix exceeds the registered collector timing change')
    actual=scientific_sources(root)
    changed=sorted(k for k in actual if actual[k]!=doc['scientific_sources'].get(k))
    if changed!=difference['allowed_core_changes'] or actual!=difference['new_scientific_sources']:
        raise ValueError('Unreviewed training implementation change')
    for name,digest in difference['new_runtime_sources'].items():
        if sha256(root/name)!=digest:raise ValueError('Timing-window runtime changed')
    for ref in doc['accepted_evidence'].values():
        if sha256(root/ref['path'])!=ref['sha256']:raise ValueError('Frozen basis evidence changed')
    return doc


def validate_remote(job,ticket,manifest,engine):
    doc=certificate(manifest['acceptance_reuse'])
    if ticket.get('new_diagnostic_identity')!='TR13' or manifest.get('parent_acceptance_only') or manifest.get('GPU_cursor_difference_required'):
        raise ValueError('Reuse cannot replace the necessary first acceptance')
    if scientific_sources(Path(__file__).resolve().parent.parent)!=doc['scientific_sources']:
        raise ValueError('Server training code changed; reuse is invalid')
    for name,digest in doc.get('collector_runtime_sources',{}).items():
        if sha256(Path(__file__).resolve().parent.parent/name)!=digest:raise ValueError('Collector timing runtime changed')
    if canonical_hash(runtime_fingerprint(ticket['project_root'],ticket['gpu_uuid']))!=doc['runtime_environment_sha256']:
        raise ValueError('Runtime environment changed; reuse is invalid')
    if {k:v['sha256'] for k,v in engine['files'].items()}!=doc['native_engine_files']:
        raise ValueError('Native engine changed; reuse is invalid')
    original=doc['training_config']
    if manifest['training_config']!=original:raise ValueError('PPO or scientific settings changed; reuse is invalid')
    if {row['path']:row['sha256'] for row in manifest['file_map']}!=doc['source_bundle_inputs']:
        raise ValueError('Actor bundle or frozen inputs changed; reuse is invalid')
    return doc


def check_state(value,scope):
    doc=certificate(scope)
    if canonical_hash(format_signature(value))!=doc['model_format_sha256']:raise ValueError('Latest model/optimizer/RNG format changed')
    partial=value.get('partial_ppo')
    if partial is not None:
        from .ppo_transaction import validate
        validate(partial,len(partial['advantages']),doc['training_config']['ppo']['epochs'])
    return True
