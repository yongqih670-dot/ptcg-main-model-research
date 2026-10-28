"""Bounded owned-directory inspection and one prospective storage-profile edit."""
import base64
import json
from pathlib import Path
import subprocess
from .io import read_json,atomic_json,sha256,utcnow
from .server_boundary_audit import ssh_command
from .storage_profile import validate,NEW,OLD


def inspect(config,output):
    output=Path(output);root=Path(config['project_path'])
    source=root/'runs'/config['source_run_id'];lease=read_json(source/'remote-job.json')
    profile=config['storage_profile'];validate(profile,NEW)
    params={'root':NEW,'old_root':OLD,'project_id':config['project_id'],'source_run_id':source.name,
        'source_identity':lease['identity'],'source_manifest_sha256':sha256(source/'remote-training-manifest.json'),
        'profile':profile,'previous_profile':config['previous_storage_profile']}
    script='P='+repr(params)+r'''
import pathlib,os,json,hashlib,subprocess,shutil,base64
root=pathlib.Path(P['root']);old=pathlib.Path(P['old_root']);uid=os.getuid()
def owned(path):
 path=pathlib.Path(path)
 if path.is_symlink() or path.resolve()!=path or path.stat().st_uid!=uid:raise ValueError('Owned realpath changed')
 return path
for directory in [root,old]:
 owned(directory);marker=owned(directory/'.ptcg-owner.json')
 if json.loads(marker.read_text())!={'project_id':P['project_id'],'uid':uid,'root':str(directory)}:raise ValueError('Owner marker differs')
 if directory.stat().st_mode&0o077:raise ValueError('Project root is not private')
env=dict(os.environ,XDG_RUNTIME_DIR='/run/user/'+str(uid),DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/'+str(uid)+'/bus')
def props(unit):
 raw=subprocess.run(['systemctl','--user','show',unit,'--property=Id','--property=InvocationID','--property=ControlGroup','--property=ActiveState','--property=SubState','--property=ExecMainStartTimestampMonotonic','--property=Result','--property=ExecMainStatus'],env=env,capture_output=True,text=True,check=True,timeout=15).stdout
 return dict(x.split('=',1) for x in raw.splitlines() if '=' in x)
units=subprocess.run(['systemctl','--user','list-units','ptcg-hybrid-*','--all','--no-legend','--no-pager','--plain'],env=env,capture_output=True,text=True,check=True,timeout=15).stdout
for row in units.splitlines():
 unit=row.split()[0]
 if unit.endswith('.service') and props(unit).get('ControlGroup'):raise ValueError('Another owned project service has live processes')
identity=P['source_identity'];observed=props(identity['unit']);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if boot!=identity['host_boot_id'] or observed.get('InvocationID')!=identity['invocation_id'] or observed.get('ExecMainStartTimestampMonotonic')!=identity['start_monotonic'] or observed.get('ControlGroup') or observed.get('SubState') not in ('exited','dead','failed'):raise ValueError('Source invocation is not positively terminal')
job=owned(owned(root/'jobs')/P['source_run_id'])
if json.loads(owned(job/'identity.json').read_text())!=identity or hashlib.sha256(owned(job/'training-manifest.raw.json').read_bytes()).hexdigest()!=P['source_manifest_sha256']:raise ValueError('Source full-state owner differs')
mount=json.loads(subprocess.run(['findmnt','-T',str(root),'--json','--output','TARGET,SOURCE,FSTYPE,OPTIONS'],capture_output=True,text=True,check=True,timeout=10).stdout)
mount_row=mount['filesystems'][0]
if mount_row['target']!='/mnt/HDD2_3TB' or 'rw' not in mount_row['options'].split(','):raise ValueError('HDD2 write mount differs')
def size(directory):
 total=0
 for folder,dirs,files in os.walk(directory,followlinks=False):
  for name in files+dirs:
   p=pathlib.Path(folder)/name
   if p.is_symlink() or p.is_file():total+=p.lstat().st_size
 return total
used_old=size(old);used_new=size(root);free=shutil.disk_usage(root).free
if used_old+used_new+2**30>80*2**30 or free-2**30<20*2**30:raise OSError('TR12 storage admission failed')
marker=owned(root/'.ptcg-storage.json');old_bytes=marker.read_bytes();existing=json.loads(old_bytes)
changed=False
if existing!=P['profile']:
 if existing!=P['previous_profile']:raise ValueError('Unexpected prior storage profile; never overwrite it')
 receipts=root/'receipts'
 if not receipts.exists():receipts.mkdir(mode=0o700)
 owned(receipts);folder=receipts/'TR12-storage-activation-20261028'
 folder.mkdir(mode=0o700);owned(folder)
 with (folder/'previous-storage-profile.json').open('xb') as f:f.write(old_bytes);f.flush();os.fsync(f.fileno())
 raw=(json.dumps(P['profile'],ensure_ascii=True,indent=2)+'\n').encode()
 temporary=root/'.ptcg-storage-TR12.partial'
 with temporary.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(temporary,marker)
 receipt={'project_id':P['project_id'],'source_terminal_identity':identity,'old_profile_sha256':hashlib.sha256(old_bytes).hexdigest(),'new_profile_sha256':hashlib.sha256(raw).hexdigest(),'used_old_bytes':used_old,'used_new_bytes':used_new,'free_bytes':free,'mount':mount,'all_existing_tickets_and_HDD1_bytes_unchanged':True}
 with (folder/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
 changed=True
files={}
for name in ['checkpoint-commits/index.json','progress.json','league.json','payload-config.json','training-updates.json','inflight.json']:
 p=job/name
 if not p.exists():
  if name=='inflight.json':continue
  raise FileNotFoundError(name)
 owned(p)
 if p.stat().st_size>8*2**20:raise ValueError('Full-state metadata exceeds bounded source envelope')
 raw=p.read_bytes();files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'base64':base64.b64encode(raw).decode()}
print(json.dumps({'complete':True,'source_run_id':job.name,'source_terminal_properties':observed,'source_identity':identity,'host_boot_id':boot,'mount':mount,'owner_uid':uid,'used_old_bytes':used_old,'used_new_bytes':used_new,'free_bytes':free,'profile_changed':changed,'profile':P['profile'],'source_files':files}))
'''
    result=subprocess.run(ssh_command(config),input=script,text=True,encoding='utf-8',capture_output=True,timeout=180)
    if result.returncode:raise RuntimeError('Bounded TR12 ownership/source inspection failed: '+result.stderr[-1500:])
    record=json.loads(result.stdout);files=record.pop('source_files');directory=output/'source-evidence';directory.mkdir()
    references={}
    for name,item in files.items():
        target=directory/name;target.parent.mkdir(parents=True,exist_ok=True)
        raw=base64.b64decode(item['base64'],validate=True)
        import hashlib
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Remote source bytes changed in transfer')
        target.write_bytes(raw);references[name]={'path':target.relative_to(root).as_posix(),'sha256':item['sha256'],'bytes':item['bytes']}
    record.update(at=utcnow(),source_files=references)
    atomic_json(output/'boundary-and-source-receipt.json',record)
    return record
