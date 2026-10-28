"""Bounded, project-local QEMU Linux deployment experiment.

Only the controlled Windows runner launches this module. The guest receives an
allowlisted public payload, never host credentials or the full workspace. TCG
timings are actual emulated-Linux timings, not Kaggle production attestation.
"""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
import zipfile

import psutil
from .io import atomic_json, read_json, sha256, utcnow


QEMU_URL = 'https://qemu.weilnetz.de/w64/qemu-w64-setup-20260811.exe'
QEMU_SHA512 = '5bcf9eed634e8575a37b74f445af41a2fe4106da512d0c30c368301d4c105037fdfab40a5287367a28a957624cddebbc8c07e16c88ab6634f554cdf3d16bf543'
IMAGE_URL = 'https://cloud-images.ubuntu.com/noble/20261017/noble-server-cloudimg-amd64.img'
IMAGE_SHA256 = '04e8f2461a16a44e78a370284471f29616cdd9baf1893df25807b71f985b6c94'
SEVEN_ZIP = Path('C:/Program Files/7-Zip/7z.exe')
MARKER = 'PTCG_LINUX_RESULT:'
RAM_PROFILE = 'ram_guest_v1'
RAM_RUNTIME_MAX = 2 * 2**30
RAM_OVERLAY_MAX = 256 * 2**20


def select_runtime_members(listing):
    """Select the CPU x86 runtime before extracting a pinned NSIS archive."""
    sections=listing.split('----------',1)
    if len(sections)!=2:raise ValueError('Unrecognized installer archive listing')
    result={};seen=set()
    for block in re.split(r'\r?\n\s*\r?\n',sections[1].strip()):
        row=dict(line.split(' = ',1) for line in block.splitlines() if ' = ' in line)
        if 'Path' not in row:continue
        name=row['Path'].strip().replace('\\','/')
        if not safe_payload_name(name) or any(c in name for c in '\n\r*?'):
            raise ValueError('Unsafe installer member')
        key=name.casefold()
        if key in seen:raise ValueError('Duplicate or case-colliding installer member')
        seen.add(key)
        if row.get('Folder')=='+' or row.get('Attributes','').startswith('D'):continue
        p=PurePosixPath(name);suffix=p.suffix.lower()
        selected=p.name.lower() in ('qemu-system-x86_64.exe','qemu-img.exe') or suffix=='.dll'
        selected=selected or suffix in ('.bin','.rom','.fd','.dtb') or 'keymaps' in p.parts
        if not selected:continue
        if not row.get('Size','').isdigit():raise ValueError('Missing runtime member size')
        result[name]=int(row['Size'])
    for required in ('qemu-system-x86_64.exe','qemu-img.exe','bios-256k.bin'):
        if sum(PurePosixPath(n).name.lower()==required for n in result)!=1:
            raise ValueError('Selected runtime must contain exactly one '+required)
    if sum(result.values())>600*2**20:raise ValueError('Selected runtime exceeds600MiB')
    return result


def runtime_capacity(folder, overlay=None):
    """Account actual logical host files; the virtual disk size is not usage."""
    total=0
    for path in Path(folder).rglob('*'):
        st=path.lstat()
        if path.is_symlink() or getattr(st,'st_file_attributes',0)&0x400:
            raise ValueError('Linked runtime file or directory')
        if path.is_file():total+=st.st_size
    overlay_bytes=Path(overlay).stat().st_size if overlay and Path(overlay).exists() else 0
    if total>RAM_RUNTIME_MAX-64*2**20 or overlay_bytes>RAM_OVERLAY_MAX-16*2**20:
        raise OSError('RAM-guest runtime/overlay storage stop boundary reached')
    return {'runtime_logical_bytes':total,'overlay_file_bytes':overlay_bytes,
            'runtime_hard_max_bytes':RAM_RUNTIME_MAX,'overlay_hard_max_bytes':RAM_OVERLAY_MAX}


def checked_download(url, path, algorithm, digest, deadline, byte_limit):
    """No redirects outside HTTPS, bounded bytes/time, immutable verified cache."""
    path = Path(path)
    def check(p):
        h = hashlib.new(algorithm)
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
        return h.hexdigest() == digest
    if path.exists():
        if path.is_symlink() or path.stat().st_size>byte_limit:raise ValueError('Cached download path/size differs')
        if not check(path): raise ValueError('Cached download checksum mismatch')
        return {'url': url, 'bytes': path.stat().st_size, 'algorithm': algorithm, 'digest': digest, 'cached': True}
    partial = path.with_suffix(path.suffix + '.partial')
    total = 0
    with urllib.request.urlopen(url, timeout=30) as response, partial.open('wb') as out:
        if not response.geturl().startswith('https://'): raise ValueError('Non-HTTPS redirect')
        while True:
            if time.time() > deadline - 60: raise TimeoutError('Download deadline')
            block = response.read(1024 * 1024)
            if not block: break
            total += len(block)
            if total > byte_limit: raise ValueError('Download exceeds preregistered byte cap')
            out.write(block)
        out.flush(); os.fsync(out.fileno())
    if not check(partial): raise ValueError('Downloaded checksum mismatch')
    partial.replace(path)
    return {'url': url, 'bytes': total, 'algorithm': algorithm, 'digest': digest, 'cached': False}


def safe_payload_name(name):
    p = PurePosixPath(name)
    return (bool(name) and not p.is_absolute() and '..' not in p.parts and
            str(p) == name and '\\' not in name and ':' not in name)


def verify_runtime_files(folder, receipt):
    paths=[Path(folder),*Path(folder).rglob('*')]
    if any(p.is_symlink() or getattr(p.lstat(),'st_file_attributes',0)&0x400 for p in paths):
        raise ValueError('Unexpected linked QEMU runtime file')
    actual = {p.relative_to(folder).as_posix(): sha256(p) for p in Path(folder).rglob('*')
              if p.is_file() and not p.is_symlink()}
    if actual != receipt['files'] or receipt.get('installer_sha512') != QEMU_SHA512:
        raise ValueError('Extracted QEMU runtime changed')
    return actual


def result_from_serial(text, nonce, run_id, archive_hash):
    records = []
    for line in text.splitlines():
        if MARKER not in line: continue
        encoded = line.split(MARKER, 1)[1].strip()
        try: row = json.loads(base64.b64decode(encoded, validate=True))
        except (ValueError, json.JSONDecodeError): continue
        if row.get('nonce') == nonce: records.append(row)
    if len(records) != 1: raise ValueError('Missing or duplicate guest completion receipt')
    row = records[0]
    if row.get('run_id') != run_id or row.get('archive_sha256') != archive_hash:
        raise ValueError('Guest output identity mismatch')
    if row.get('platform') != 'Linux': raise ValueError('Guest was not Linux')
    return row


def make_payload(source, config, run_id, deadline):
    source = Path(source)
    probe = dict(config['probe'])
    archive = Path(probe['archive_path'])
    if sha256(archive) != probe['archive_sha256']: raise ValueError('Package changed')
    files = {'submission.tar.gz': archive.read_bytes()}
    allowed = [source / 'ptcg_lab']
    for spec in (probe['candidate'], probe['reference']):
        if not safe_payload_name(spec['path']) or not spec['path'].startswith('agents/'):
            raise ValueError('Guest agent path outside allowlisted agents')
        allowed.append(source / spec['path'])
    for root in allowed:
        for file in root.rglob('*'):
            if not file.is_file() or file.is_symlink() or '__pycache__' in file.parts or file.suffix == '.pyc': continue
            if file.suffix not in ('.py', '.json', '.pt', '.npz', '.csv', '.txt'): continue
            relative = file.relative_to(source).as_posix()
            files['source/' + relative] = file.read_bytes()
    probe['archive_path'] = '/opt/ptcg/task/submission.tar.gz'
    files['probe.json'] = json.dumps(probe, sort_keys=True).encode()
    identity = {'run_id': run_id, 'deadline': deadline,
                'archive_sha256': probe['archive_sha256'],
                'files': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    files['payload-manifest.json'] = json.dumps(identity, sort_keys=True).encode()
    if any(not safe_payload_name(k) for k in files): raise ValueError('Unsafe payload path')
    if sum(map(len, files.values())) > 200 * 1024**2: raise ValueError('Payload too large')
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', compression=zipfile.ZIP_STORED) as z:
        for name, value in sorted(files.items()): z.writestr(name, value)
    return data.getvalue(), identity


def guest_script(base_url, payload_hash, nonce, run_id, archive_hash, deadline, ram_guest=False):
    # The script and payload contain only project public code, weights and deck.
    # No login, SSH key, host mount, hostfwd, or credential propagation is used.
    constants = repr({'base_url': base_url, 'payload_hash': payload_hash, 'nonce': nonce,
                      'run_id': run_id, 'archive_sha256': archive_hash, 'deadline': deadline,'ram_guest':ram_guest})
    return '#!/usr/bin/python3\nC = ' + constants + r'''
import base64,hashlib,importlib.metadata,json,os,platform,subprocess,sys,time,traceback,urllib.request,zipfile
from pathlib import Path,PurePosixPath
root=Path('/opt/ptcg/task');root.mkdir(parents=True,exist_ok=True)
report={'nonce':C['nonce'],'run_id':C['run_id'],'archive_sha256':C['archive_sha256'],'platform':platform.system(),'complete':False}
def command(args,cap):
    remaining=min(cap,C['deadline']-time.time()-90)
    if remaining<=0:raise TimeoutError('Guest absolute deadline')
    subprocess.run(args,check=True,timeout=remaining,env=dict(os.environ,DEBIAN_FRONTEND='noninteractive',PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_INPUT='1'))
try:
    apt_options=[]
    if C['ram_guest']:
        command(['mount','-t','tmpfs','-o','size=6G,noswap,mode=700','ptcg-runtime','/opt/ptcg'],30)
        root.mkdir(parents=True,exist_ok=True)
        fs=subprocess.check_output(['findmnt','-n','-o','FSTYPE','--target','/opt/ptcg'],text=True,timeout=10).strip()
        if fs!='tmpfs':raise ValueError('Private guest project is not actual tmpfs')
        options=subprocess.check_output(['findmnt','-n','-o','OPTIONS','--target','/opt/ptcg'],text=True,timeout=10).strip()
        if 'noswap' not in options.split(','):raise ValueError('Guest tmpfs swap isolation was not observed')
        for folder in ['tmp','cache','apt/lists/partial','apt/archives/partial']:(root.parent/folder).mkdir(parents=True,exist_ok=True)
        os.environ.update(TMPDIR=str(root.parent/'tmp'),XDG_CACHE_HOME=str(root.parent/'cache'),PYTHONDONTWRITEBYTECODE='1')
        apt_options=['-o','Dir::State::lists='+str(root.parent/'apt/lists'),'-o','Dir::Cache::archives='+str(root.parent/'apt/archives'),
                     '-o','Dir::Cache::pkgcache=','-o','Dir::Cache::srcpkgcache=']
        report['guest_storage']={'filesystem':fs,'root':'/opt/ptcg','size_limit_bytes':6*2**30,'mount_options':options,'noswap_observed':True,'ephemeral_on_shutdown':True}
    blob=urllib.request.urlopen(C['base_url']+'payload.zip',timeout=60).read(200*1024**2+1)
    if hashlib.sha256(blob).hexdigest()!=C['payload_hash']:raise ValueError('Guest payload checksum')
    (root/'payload.zip').write_bytes(blob)
    with zipfile.ZipFile(root/'payload.zip') as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or sum(i.file_size for i in z.infolist())>200*1024**2:raise ValueError('Unsafe payload size/duplicates')
        for name in names:
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or str(p)!=name or chr(92) in name or ':' in name:raise ValueError('Unsafe payload member')
        z.extractall(root)
    manifest=json.loads((root/'payload-manifest.json').read_text())
    if manifest['run_id']!=C['run_id'] or manifest['archive_sha256']!=C['archive_sha256']:raise ValueError('Guest payload identity')
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Guest file checksum')
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if not Path('/opt/ptcg/venv/bin/python').exists():
        command(['apt-get',*apt_options,'update','-o','Acquire::Retries=1','-o','Acquire::http::Timeout=30'],600)
        command(['apt-get',*apt_options,'install','-y','--no-install-recommends','python3-venv','ca-certificates'],900)
        command(['python3','-m','venv','/opt/ptcg/venv'],120)
    python='/opt/ptcg/venv/bin/python'
    command([python,'-m','pip','install','--no-cache-dir','--timeout','60','--retries','1','--index-url','https://download.pytorch.org/whl/cpu','torch==2.11.0+cpu'],1500)
    command([python,'-m','pip','install','--no-cache-dir','--timeout','60','--retries','1','numpy==2.5.4','psutil==7.2.2','jsonschema==4.26.0','requests==2.34.2'],600)
    command([python,'-m','pip','install','--no-cache-dir','--no-deps','--timeout','60','--retries','1','kaggle-environments==1.33.2'],600)
    output=root/'output'/C['run_id'];output.mkdir(parents=True,exist_ok=True)
    os.environ.update(PTCG_JOB_RUN_ID=C['run_id'],PTCG_JOB_DEADLINE_UNIX=str(C['deadline']-60))
    os.chdir(root/'source')
    command([python,'-B','-m','ptcg_lab.package_probe','--config',str(root/'probe.json'),'--output',str(output)],3600)
    report['metrics']=json.loads((output/'metrics.json').read_text())
    report['series']=[json.loads(p.read_text()) for p in sorted(output.glob('package-series-*.json'))]
    report['environment']=json.loads(subprocess.check_output([python,'-c',"import importlib.metadata as m,json,platform,hashlib; d=m.distribution('kaggle-environments'); files={str(p):hashlib.sha256(d.locate_file(p).read_bytes()).hexdigest() for p in d.files if str(p).startswith('kaggle_environments/envs/cabt/') and not str(p).endswith('.pyc')}; print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in m.distributions()},'official_cabt_files_sha256':files}))"],text=True,timeout=60))
    report['complete']=bool(report['metrics'].get('complete') and report['metrics'].get('linux_execution_observed') and report['metrics'].get('cpu_only_observed'))
except Exception:
    report['error']=traceback.format_exc()
finally:
    report['finished_unix']=time.time()
    (root/'last-receipt.json').write_text(json.dumps(report))
    encoded=base64.b64encode(json.dumps(report).encode()).decode()
    with open('/dev/ttyS0','w') as serial:serial.write('\nPTCG_LINUX_RESULT:'+encoded+'\n');serial.flush()
    os.sync()
    subprocess.run(['shutdown','-h','now'],check=False,timeout=15)
'''


def serve_payload(payload, script, nonce):
    routes = {'/seed/user-data': script.encode(),
              '/seed/meta-data': ('instance-id: ptcg-' + nonce + '\nlocal-hostname: ptcg-linux\n').encode(),
              '/seed/vendor-data': b'', '/seed/network-config': b'', '/payload.zip': payload}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = routes.get(self.path)
            if data is None: self.send_error(404); return
            self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args): pass
    return ThreadingHTTPServer(('127.0.0.1', 0), Handler), routes


def run(config, output):
    output = Path(output).resolve(); source = Path(__file__).resolve().parents[1]
    if os.environ.get('PTCG_JOB_RUN_ID') != output.name: raise RuntimeError('Linux probe requires controlled runner')
    deadline = float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if os.name != 'nt': raise RuntimeError('Host controller requires Windows Job confinement')
    if config.get('job_kind') != 'linux_probe' or config.get('workers') != 1: raise ValueError('Unexpected Linux job config')
    profile=config.get('storage_profile')
    if profile not in (None,RAM_PROFILE):raise ValueError('Unknown Linux storage profile')
    ram_guest=profile==RAM_PROFILE
    if config['probe']['series_count'] != 4: raise ValueError('Engineering probe is fixed at four BO3')
    expected_inputs={'qemu_url':QEMU_URL,'qemu_sha512':QEMU_SHA512,'image_url':IMAGE_URL,'image_sha256':IMAGE_SHA256}
    if config.get('runtime_inputs') != expected_inputs: raise ValueError('Runtime configuration/source pins differ')
    attempts=read_json(output/'attempts.json') if (output/'attempts.json').exists() else []
    if (len(attempts)>=2 or sum(a['seconds'] for a in attempts)+deadline-time.time()>10800.5
            or (attempts and deadline-time.time()>3600.5)):
        raise RuntimeError('Linux setup preregistered attempt/wall cap exceeded')
    if not SEVEN_ZIP.is_file(): raise RuntimeError('Existing 7-Zip unavailable')
    required_free=(22 if ram_guest else 30)*2**30
    if shutil.disk_usage(output).free < required_free: raise RuntimeError('Linux setup free disk reserve unavailable')
    runtime = output / 'runtime'; runtime.mkdir(exist_ok=True)
    nonce = uuid.uuid4().hex
    atomic_json(output/'progress.json', {'at': utcnow(), 'stage': 'verified_downloads', 'nonce': nonce})
    downloads = [checked_download(QEMU_URL, runtime/'qemu-installer.exe', 'sha512', QEMU_SHA512, deadline, 250*1024**2),
                 checked_download(IMAGE_URL, runtime/'ubuntu-base.img', 'sha256', IMAGE_SHA256, deadline, 700*1024**2)]
    atomic_json(output/'download-receipt.json', downloads)
    extracted = runtime/'qemu'
    extraction_receipt=runtime/'qemu-extraction-receipt.json'
    def bounded(args, cap, **kwargs):
        return subprocess.run(args, check=True, timeout=max(1,min(cap,deadline-time.time()-60)), **kwargs)
    if not extraction_receipt.exists():
        # Extract the distribution; never run the installer or change host features.
        listing=bounded([str(SEVEN_ZIP),'l','-slt',str(runtime/'qemu-installer.exe')],60,capture_output=True,text=True).stdout
        selected=select_runtime_members(listing) if ram_guest else None
        extra=[]
        if selected:
            selection_file=runtime/'selected-members.txt';selection_file.write_text('\n'.join(selected)+'\n',encoding='utf-8')
            extra=['-spd','-scsUTF-8','@'+str(selection_file)]
            atomic_json(runtime/'selection-manifest.json',{'storage_profile':profile,'files_and_bytes':selected})
        else:
            section=listing.split('----------',1)
            if len(section)!=2: raise ValueError('Unrecognized installer archive listing')
            for name in re.findall(r'^Path = (.+)$',section[1],re.MULTILINE):
                if not safe_payload_name(name.strip().replace('\\','/')): raise ValueError('Unsafe installer member')
        bounded([str(SEVEN_ZIP),'x','-y','-o'+str(extracted),str(runtime/'qemu-installer.exe'),*extra],180,stdout=subprocess.DEVNULL)
        files={p.relative_to(extracted).as_posix():sha256(p) for p in extracted.rglob('*') if p.is_file() and not p.is_symlink()}
        if selected:
            sizes={n:(extracted/n).stat().st_size for n in files}
            if sizes!=selected:raise ValueError('Extracted runtime differs from selected exact member/size manifest')
        atomic_json(extraction_receipt,{'at':utcnow(),'installer_sha512':QEMU_SHA512,'files':files,'storage_profile':profile})
    if read_json(extraction_receipt).get('storage_profile')!=profile:raise ValueError('Linux runtime storage profile changed on resume')
    verify_runtime_files(extracted,read_json(extraction_receipt))
    matches = list(extracted.rglob('qemu-system-x86_64.exe'))
    if len(matches)!=1: raise RuntimeError('Expected one extracted QEMU system binary')
    qemu=matches[0];qemu_img=qemu.with_name('qemu-img.exe')
    version=bounded([str(qemu),'--version'],30,capture_output=True,text=True).stdout
    overlay=runtime/'guest.qcow2'
    if ram_guest:runtime_capacity(runtime,overlay)
    if overlay.exists():
        checkpoint=read_json(output/'guest-checkpoint.json')
        if (checkpoint.get('run_id')!=output.name or checkpoint.get('overlay')!=str(overlay)
                or checkpoint.get('overlay_sha256')!=sha256(overlay)):
            raise RuntimeError('Guest disk resume identity/hash mismatch')
        previous=read_json(output/'guest-process.json')
        try:
            if abs(psutil.Process(previous['pid']).create_time()-previous['created'])<.01:
                raise RuntimeError('Previous guest remains alive')
        except psutil.NoSuchProcess: pass
    else:
        bounded([str(qemu_img),'create','-f','qcow2','-F','qcow2','-b',str(runtime/'ubuntu-base.img'),str(overlay),'10G'],30,stdout=subprocess.DEVNULL)
    payload,identity=make_payload(source,config,output.name,deadline)
    atomic_json(output/('payload-identity-'+nonce+'.json'),identity)
    server,routes=serve_payload(payload,'',nonce)
    port=server.server_address[1];base_url=f'http://PRIVATE_HOST:{port}/'
    routes['/seed/user-data']=guest_script(base_url,hashlib.sha256(payload).hexdigest(),nonce,output.name,config['probe']['archive_sha256'],deadline,ram_guest=ram_guest).encode()
    threading.Thread(target=server.serve_forever,daemon=True).start()
    serial=output/('guest-serial-'+nonce+'.log')
    guest_memory=12288 if ram_guest else 4096
    command=[str(qemu),'-machine','q35','-accel','tcg,thread=multi','-cpu','max','-smp','2','-m',str(guest_memory),
             '-drive',f'file={overlay},if=virtio,format=qcow2','-nic','user,model=virtio-net-pci',
             '-smbios','type=1,serial=ds=nocloud;s='+base_url+'seed/',
             '-display','none','-serial','file:'+str(serial),'-monitor','stdio','-no-reboot','-rtc','base=utc']
    if ram_guest:
        bios=list(extracted.rglob('bios-256k.bin'))
        if len(bios)!=1:raise ValueError('Selected BIOS path differs')
        command+=['-L',str(bios[0].parent)]
    atomic_json(output/'qemu-command.json',{'argv':command,'qemu_version':version,'binary_sha256':sha256(qemu),
                'nonce':nonce,'guest_memory_mib':guest_memory,'vcpus':2,'acceleration':'tcg','host_forwards':[],
                'storage_profile':profile,'guest_project_tmpfs':ram_guest})
    process=None; clean_shutdown=False
    try:
        with (output/('qemu-'+nonce+'.log')).open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log,text=True)
            created=psutil.Process(process.pid).create_time()
            atomic_json(output/'guest-process.json',{'pid':process.pid,'created':created,'nonce':nonce,'run_id':output.name})
            last=0
            while process.poll() is None and time.time()<deadline-45:
                if ram_guest:
                    usage=runtime_capacity(runtime,overlay)
                    if shutil.disk_usage(output).free<20*2**30+64*2**20:raise OSError('Linux guest physical free reserve stop boundary')
                if time.time()-last>=60:
                    atomic_json(output/'progress.json',{'at':utcnow(),'stage':'linux_guest_running','nonce':nonce,
                                'seconds_remaining':deadline-time.time(),'serial_bytes':serial.stat().st_size if serial.exists() else 0,
                                'storage_profile':profile,'runtime_storage':usage if ram_guest else None})
                    last=time.time()
                time.sleep(1)
            if process.poll() is None:
                process.stdin.write('system_powerdown\n');process.stdin.flush()
                try: process.wait(timeout=max(1,min(30,deadline-time.time()-10)))
                except subprocess.TimeoutExpired: pass
            clean_shutdown=process.poll()==0
    finally:
        if process is not None and process.poll() is None:
            if abs(psutil.Process(process.pid).create_time()-created)>.01: raise RuntimeError('QEMU process identity changed')
            process.kill();process.wait(timeout=5)
        server.shutdown();server.server_close()
        checkpoint={'at':utcnow(),'run_id':output.name,'overlay':str(overlay),'overlay_bytes':overlay.stat().st_size,
                    'overlay_sha256':sha256(overlay),'clean_shutdown':clean_shutdown,'nonce':nonce,
                    'resume_requires_identity_and_disk_verification':True}
        atomic_json(output/('guest-checkpoint-'+nonce+'.json'),checkpoint)
        atomic_json(output/'guest-checkpoint.json',checkpoint)
    guest=result_from_serial(serial.read_text(encoding='utf-8',errors='replace'),nonce,output.name,config['probe']['archive_sha256'])
    atomic_json(output/('guest-receipt-'+nonce+'.json'),guest)
    atomic_json(output/'guest-receipt.json',guest)
    complete=bool(guest.get('complete') and clean_shutdown and guest.get('finished_unix',float('inf'))<deadline)
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':complete,'engineering_only':True,
                'linux_cpu':bool(guest.get('metrics',{}).get('linux_execution_observed') and guest.get('metrics',{}).get('cpu_only_observed')),'archive_sha256':config['probe']['archive_sha256'],
                'clean_shutdown':clean_shutdown,'guest':guest,'kaggle_server_equivalence_verified':False,
                'final_reliability_qualified':False,'promotion_eligible':False})
    if not complete: raise RuntimeError('Linux engineering probe did not complete; inspect guest receipt')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(read_json(args.config),args.output)


if __name__=='__main__': main()
