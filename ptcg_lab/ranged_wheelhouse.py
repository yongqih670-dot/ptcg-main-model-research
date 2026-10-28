"""Download an already-resolved Linux lock using resumable verified ranges."""
import argparse
import json
import os
from pathlib import Path
import ssl
import time
import urllib.parse
import urllib.request
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import cpython_tags,compatible_tags
from packaging.utils import parse_wheel_filename,canonicalize_name
from packaging.version import Version
from .io import atomic_json,read_json,sha256,utcnow
from .range_fetch import fetch_verified
from .stream_fetch import fetch_single_response
from .wheelhouse_prepare import TORCH_URL,TORCH_SHA256,linux_environment,wheel_record


def supported_tags(glibc_minor):
    platforms=[f'manylinux_2_{n}_x86_64' for n in range(glibc_minor,4,-1)]+['manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64','linux_x86_64']
    return list(dict.fromkeys([*cpython_tags((3,12),abis=['cp312'],platforms=platforms),
                              *compatible_tags((3,12),interpreter='cp312',platforms=platforms)]))


def select_wheel(metadata,requirement,allowed_hashes,tags):
    ranked={tag:i for i,tag in enumerate(tags)};candidates=[]
    for item in metadata['urls']:
        filename=item['filename']
        if not filename.endswith('.whl') or Path(filename).name!=filename or ':' in filename or chr(92) in filename:continue
        name,version,_,wheel_tags=parse_wheel_filename(filename)
        if canonicalize_name(requirement.name)!=canonicalize_name(name):continue
        if version not in requirement.specifier:continue
        if item.get('yanked'):continue
        if item.get('requires_python') and Version('3.12.3') not in SpecifierSet(item['requires_python']):continue
        if allowed_hashes is not None and item['digests']['sha256'] not in allowed_hashes:continue
        common=wheel_tags&set(ranked)
        if common:candidates.append((min(ranked[t] for t in common),filename,item))
    if not candidates:raise ValueError('No hash-approved Linux3.12 wheel: '+str(requirement))
    return min(candidates,key=lambda x:(x[0],x[1]))[2]


def validate_closure(records):
    installed={canonicalize_name(r['name']):r for r in records};extras={name:set() for name in installed}
    # Kaggle's all-games optional ecosystem is deliberately outside the CABT
    # minimal runtime; actual official CABT import/BO3 checks remain mandatory.
    for _ in range(len(records)+1):
        changed=False
        for name,record in installed.items():
            if name=='kaggle-environments':continue
            for text in record['requires_dist']:
                req=Requirement(text)
                active=not req.marker or any(req.marker.evaluate(dict(linux_environment(),extra=x)) for x in ['',*extras[name]])
                if not active:continue
                target=canonicalize_name(req.name)
                if target not in installed or Version(installed[target]['version']) not in req.specifier:
                    raise ValueError('Actual Linux wheel dependency closure missing: '+text)
                before=len(extras[target]);extras[target].update(req.extras);changed=changed or len(extras[target])!=before
        if not changed:return True
    raise ValueError('Dependency extras did not converge')


def check_scope_budget(prior_seconds,remaining_seconds,cap_seconds,attempts):
    if attempts:raise RuntimeError('Post-review range repair permits one controlled attempt')
    if prior_seconds+remaining_seconds>cap_seconds+.5:raise RuntimeError('Inherited preparation cap exceeded')


def publish_manifest(config,output,records,selections,transport):
    validate_closure(records)
    expanded=sum(r['expanded_bytes'] for r in records);compressed=sum(r['bytes'] for r in records)
    largest=max(r['bytes'] for r in records);peak=expanded+largest+config['training_capsule_estimate_bytes']+config['runtime_buffer_bytes']+128*1024**2
    manifest={'at':utcnow(),'target':'CPython3.12 Linux x86_64 glibc2.39','files':records,'source_selections':selections,
              'compressed_bytes':compressed,'expanded_bytes':expanded,'estimated_peak_with_one_staged_wheel':peak,
              'linux_dependency_closure_verified':True,'kaggle_other_game_dependencies_excluded':True,
              'remote_installation_verified':False,'server_ensurepip_absent_offline_pip_bootstrap_required':True,
              'transport':transport,'cache_reuse_requires_full_official_hash':True}
    atomic_json(output/'wheelhouse-manifest.json',manifest)
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'wheels':len(records),
                'compressed_bytes':compressed,'expanded_bytes':expanded,'estimated_server_peak_bytes':peak,
                'estimated_fits_10gib':peak<=10*2**30,'remote_installation_verified':False,
                'linux_dependency_closure_verified':True,'manifest_sha256':sha256(output/'wheelhouse-manifest.json'),
                'promotion_eligible':False})


def run(config,output):
    output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Ranged wheelhouse requires controlled runner')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX']);lock=Path(config['source_lock_path'])
    if sha256(lock)!=config['source_lock_sha256']:raise ValueError('Original Linux dependency lock changed')
    attempts=read_json(output/'attempts.json') if (output/'attempts.json').exists() else []
    check_scope_budget(config['prior_scope_seconds'],deadline-time.time(),config['scope_wall_seconds_max'],attempts)
    output.joinpath('wheelhouse').mkdir(exist_ok=True);output.joinpath('metadata').mkdir(exist_ok=True)
    import certifi
    context=ssl.create_default_context(cafile=certifi.where());tags=supported_tags(config['server_glibc_minor'])
    requirements=[]
    for line in lock.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):continue
        parts=line.split(' --hash=');req=Requirement(parts[0].strip())
        if req.marker:raise ValueError('Download lock must be target-normalized')
        requirements.append((req,{p.strip().split(':',1)[1] for p in parts[1:]}))
    requirements.extend([(Requirement('kaggle-environments==1.33.2'),None),(Requirement('pip==24.2'),None)])
    if config.get('offline_manifest_audit'):
        selection_path=Path(config['audit_selection_path'])
        if sha256(selection_path)!=config['audit_selection_sha256']:raise ValueError('Offline artifact selection changed')
        selections=read_json(selection_path);expected={canonicalize_name(req.name):(req,hashes) for req,hashes in requirements}
        if len(selections)!=len(expected) or {canonicalize_name(x['name']) for x in selections}!=set(expected):
            raise ValueError('Offline wheelhouse does not cover exact lock')
        records=[]
        for item in selections:
            if time.time()>deadline-30:raise TimeoutError('Offline audit deadline')
            if Path(item['file']).name!=item['file'] or ':' in item['file']:raise ValueError('Unsafe offline filename')
            paths=[Path(d)/item['file'] for d in config['audit_wheel_directories'] if (Path(d)/item['file']).is_file()]
            if not paths:raise ValueError('Missing offline wheel: '+item['file'])
            path=paths[0];record=dict(wheel_record(path),source_path=str(path.resolve()))
            if record['bytes']!=item['bytes'] or record['sha256']!=item['sha256']:raise ValueError('Offline artifact differs from official SHA')
            req,hashes=expected[canonicalize_name(item['name'])]
            if canonicalize_name(record['name'])!=canonicalize_name(req.name) or Version(record['version']) not in req.specifier:
                raise ValueError('Offline wheel metadata differs from lock')
            if hashes is not None and record['sha256'] not in hashes:raise ValueError('Offline wheel hash is outside lock')
            if canonicalize_name(req.name)=='torch' and record['sha256']!=TORCH_SHA256:raise ValueError('Offline Torch identity changed')
            records.append(record)
            atomic_json(output/'progress.json',{'at':utcnow(),'stage':'offline_artifact_audit','wheels_checked':len(records),'wheels_total':len(selections)})
        publish_manifest(config,output,records,selections,'offline_full_sha_metadata_audit');return
    transport=config.get('transport','strict_ranges')
    if transport not in ('strict_ranges','pypi_single_response_pytorch_strict_ranges','pypi_digest_quarantine_pytorch_strict_ranges'):
        raise ValueError('Unreviewed transport')
    caches=[]
    if config.get('verified_cache_selection_path'):
        cache_path=Path(config['verified_cache_selection_path'])
        if sha256(cache_path)!=config['verified_cache_selection_sha256']:raise ValueError('Cache selection identity changed')
        caches.append(({x['file']:x for x in read_json(cache_path)},Path(config['verified_cache_directory'])))
    for source in config.get('additional_verified_caches',[]):
        cache_path=Path(source['selection_path'])
        if sha256(cache_path)!=source['selection_sha256']:raise ValueError('Additional cache selection changed')
        caches.append(({x['file']:x for x in read_json(cache_path)},Path(source['directory'])))
    selections=[];last_progress=0;paths=[]
    for req,hashes in requirements:
        if time.time()>deadline-30:raise TimeoutError('Wheelhouse preparation deadline')
        if canonicalize_name(req.name)=='torch':
            if req.url.split('#')[0]!=TORCH_URL or hashes!={TORCH_SHA256}:raise ValueError('Torch source identity changed')
            item={'filename':urllib.parse.unquote(TORCH_URL.rsplit('/',1)[1]),'url':TORCH_URL,'size':820326880,'digests':{'sha256':TORCH_SHA256}}
        else:
            version=next(s.version for s in req.specifier if s.operator=='==')
            meta_path=output/'metadata'/f'{canonicalize_name(req.name)}-{version}.json'
            if meta_path.exists():metadata=read_json(meta_path)
            else:
                url=f'https://pypi.org/pypi/{canonicalize_name(req.name)}/{version}/json'
                with urllib.request.urlopen(url,context=context,timeout=30) as response:metadata=json.loads(response.read(8*1024**2))
                atomic_json(meta_path,metadata)
            item=select_wheel(metadata,req,hashes,tags)
        host=urllib.parse.urlsplit(item['url'])
        if host.scheme!='https' or host.username or host.hostname not in {'files.pythonhosted.org','download-r2.pytorch.org','download.pytorch.org'}:
            raise ValueError('Nonofficial artifact origin')
        selection={'name':req.name,'file':item['filename'],'url':item['url'],'bytes':item['size'],'sha256':item['digests']['sha256']}
        selections.append(selection);atomic_json(output/'selected-wheels.json',selections)
        def progress(done,total):
            nonlocal last_progress
            if time.monotonic()-last_progress>10 or done==total:
                atomic_json(output/'progress.json',{'at':utcnow(),'stage':'artifact_transfer','transport':transport,'package':req.name,
                            'bytes_done':done,'bytes_total':total,'selected_packages':len(selections)});last_progress=time.monotonic()
        cached_path=None;path=output/'wheelhouse'/item['filename']
        for cache,cache_directory in caches:
            cached=cache.get(item['filename'])
            if cached and (cache_directory/item['filename']).is_file():
                if cached!=selection:raise ValueError('Cached artifact selection changed')
                cached_path=cache_directory/item['filename'];break
        if cached_path is not None:
            path=cached_path
            if path.stat().st_size!=selection['bytes'] or sha256(path)!=selection['sha256']:
                raise ValueError('Cached artifact differs from official identity')
            progress(item['size'],item['size'])
        elif transport=='pypi_single_response_pytorch_strict_ranges' and host.hostname=='files.pythonhosted.org':
            fetch_single_response(item['url'],path,item['size'],item['digests']['sha256'],deadline,progress)
        elif transport=='pypi_digest_quarantine_pytorch_strict_ranges' and host.hostname=='files.pythonhosted.org':
            seed=config.get('quarantine_seeds',{}).get(item['filename'])
            fetch_verified(item['url'],path,item['size'],item['digests']['sha256'],deadline,progress,
                           entity_policy='official_digest_quarantine',seed=seed)
        else:
            fetch_verified(item['url'],path,item['size'],item['digests']['sha256'],deadline,progress)
        paths.append(path)
    records=[dict(wheel_record(p),source_path=str(p.resolve())) for p in paths]
    publish_manifest(config,output,records,selections,transport)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(read_json(a.config),a.output)
