"""Controlled cross-platform Linux wheelhouse; never installs into project envs."""
import argparse
from email.parser import Parser
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile
from packaging.markers import default_environment
from packaging.requirements import Requirement
from .io import atomic_json,read_json,sha256,utcnow


TORCH_URL='https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'
TORCH_SHA256='d252cf975fb18c94a85336323ad425f473df56dab35a44b00399bd70c7a3b997'
PINNED_BASE=('filelock','typing_extensions','setuptools','sympy','networkx','Jinja2','MarkupSafe','fsspec',
             'numpy','psutil','jsonschema','attrs','jsonschema-specifications','referencing','rpds-py',
             'requests','certifi','charset-normalizer','idna','urllib3','mpmath','packaging')


def linux_environment():
    return dict(default_environment(),os_name='posix',sys_platform='linux',platform_system='Linux',
                platform_machine='x86_64',python_version='3.12',python_full_version='3.12.3',
                implementation_name='cpython',implementation_version='3.12.3',extra='')


def normalize_lock(text):
    """A Linux lock must not lose CUDA dependencies when pip runs on Windows."""
    logical=[];pending=''
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        pending+=line[:-1].rstrip()+' ' if line.endswith(chr(92)) else line
        if line.endswith(chr(92)):continue
        logical.append(pending);pending=''
    if pending:raise ValueError('Incomplete dependency lock line')
    output=[];names=set()
    for line in logical:
        if line.startswith('-'):raise ValueError('Unexpected lock option')
        parts=line.split(' --hash=');requirement=Requirement(parts[0].strip())
        if requirement.marker and not requirement.marker.evaluate(linux_environment()):continue
        requirement.marker=None
        if not requirement.url and not any(s.operator=='==' for s in requirement.specifier):raise ValueError('Unpinned dependency')
        if len(parts)<2 or any(not re.fullmatch(r'sha256:[0-9a-f]{64}',p.strip()) for p in parts[1:]):
            raise ValueError('Dependency lock lacks SHA256')
        names.add(requirement.name.lower().replace('_','-'))
        output.append(str(requirement)+' '+ ' '.join('--hash='+p.strip() for p in parts[1:]))
    if 'torch' not in names:raise ValueError('Missing locked Torch')
    return '\n'.join(output)+'\n',names


def wheel_record(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if any(n.startswith('/') or '..' in Path(n).parts or chr(92) in n or ':' in n for n in names):raise ValueError('Unsafe wheel member')
        metadata=[n for n in names if n.endswith('.dist-info/METADATA') and len(n.split('/'))==2]
        if len(metadata)!=1:raise ValueError('Invalid wheel metadata')
        parsed=Parser().parsestr(archive.read(metadata[0]).decode('utf-8'))
        expanded=sum(i.file_size for i in archive.infolist())
    return {'file':path.name,'sha256':sha256(path),'bytes':path.stat().st_size,'expanded_bytes':expanded,
            'name':parsed['Name'],'version':parsed['Version'],'requires_dist':parsed.get_all('Requires-Dist',[])}


def run(config,output):
    output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Wheelhouse requires controlled runner')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX']);source=Path(__file__).resolve().parents[1]
    for directory in ['tools','uv-cache','tmp','wheelhouse']:(output/directory).mkdir(exist_ok=True)
    env=dict(os.environ,UV_CACHE_DIR=str(output/'uv-cache'),UV_PYTHON_DOWNLOADS='never',
             UV_NATIVE_TLS='true',PIP_NO_INPUT='1',PIP_DISABLE_PIP_VERSION_CHECK='1',
             PIP_CONFIG_FILE=os.devnull,UV_NO_CONFIG='true',
             TEMP=str(output/'tmp'),TMP=str(output/'tmp'),TMPDIR=str(output/'tmp'))
    def command(argv,label):
        if shutil.disk_usage(output).free<20*2**30:raise RuntimeError('Local free-space reserve reached')
        atomic_json(output/'progress.json',{'at':utcnow(),'stage':label})
        with (output/(label+'.log')).open('a',encoding='utf-8') as log:
            subprocess.run(argv,check=True,timeout=max(1,deadline-time.time()-30),env=env,stdout=log,stderr=log)
    uv=list((output/'tools').rglob('uv.exe'))
    if not uv:
        command([sys.executable,'-m','pip','install','--no-deps','--no-compile','--no-cache-dir','--index-url','https://pypi.org/simple','--target',str(output/'tools'),'uv==0.9.5'],'install_isolated_resolver')
        uv=list((output/'tools').rglob('uv.exe'))
    if len(uv)!=1:raise RuntimeError('Isolated uv executable unavailable')
    base=read_json(config['cpu_manifest_path'])['packages']
    requirements=['torch @ '+TORCH_URL+'#sha256='+TORCH_SHA256]+[name+'=='+base[name] for name in PINNED_BASE]
    (output/'requirements.in').write_text('\n'.join(requirements)+'\n',encoding='utf-8')
    command([str(uv[0]),'pip','compile',str(output/'requirements.in'),'--python-platform','x86_64-unknown-linux-gnu',
             '--python-version','3.12.3','--generate-hashes','--no-header','--no-annotate','--no-build',
             '--index-url','https://pypi.org/simple','-o',str(output/'requirements.linux.lock')],'resolve_linux_requirements')
    normalized,names=normalize_lock((output/'requirements.linux.lock').read_text(encoding='utf-8'))
    if not {'cuda-toolkit','nvidia-cudnn-cu12','nvidia-nccl-cu12','triton'}<=names:
        raise ValueError('Linux CUDA dependency closure missing')
    (output/'requirements.download.lock').write_text(normalized,encoding='utf-8')
    options=[]
    for minor in range(39,4,-1):options+=['--platform',f'manylinux_2_{minor}_x86_64']
    options+=['--platform','manylinux2014_x86_64','--platform','linux_x86_64','--python-version','3.12',
              '--implementation','cp','--abi','cp312','--only-binary=:all:','--no-deps','--no-cache-dir','--dest',str(output/'wheelhouse')]
    command([sys.executable,'-m','pip','download',*options,'--index-url','https://pypi.org/simple','--require-hashes','-r',str(output/'requirements.download.lock')],'download_locked_linux_wheels')
    command([sys.executable,'-m','pip','download',*options,'--index-url','https://pypi.org/simple','kaggle-environments==1.33.2'],'download_official_cabt_distribution')
    records=[wheel_record(p) for p in sorted((output/'wheelhouse').glob('*.whl'))]
    torch=[r for r in records if r['name'].lower()=='torch']
    if len(torch)!=1 or torch[0]['sha256']!=TORCH_SHA256:raise ValueError('Torch official source hash differs')
    expanded=sum(r['expanded_bytes'] for r in records);compressed=sum(r['bytes'] for r in records)
    largest=max(r['bytes'] for r in records);capsule=config['training_capsule_estimate_bytes']
    # One wheel is staged/installed at a time; full compressed wheelhouse stays local.
    peak=expanded+largest+capsule+config['runtime_buffer_bytes']+128*1024**2
    manifest={'at':utcnow(),'target':'CPython3.12 Linux x86_64','files':records,'compressed_bytes':compressed,
              'expanded_bytes':expanded,'estimated_peak_with_one_staged_wheel':peak,
              'peak_model':'sumwheeluncompressed +largestwheel +trainingcapsule +runtimebuffer +128MiBbootstrap',
              'remote_installation_verified':False,'official_cabt_minimal_dependencies':True,
              'cabt_optional_other_game_dependencies_not_installed':True}
    atomic_json(output/'wheelhouse-manifest.json',manifest)
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'wheels':len(records),
                'compressed_bytes':compressed,'expanded_bytes':expanded,'estimated_server_peak_bytes':peak,
                'estimated_fits_10gib':peak<=10*2**30,'remote_installation_verified':False,
                'manifest_sha256':sha256(output/'wheelhouse-manifest.json'),'promotion_eligible':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(read_json(a.config),a.output)
