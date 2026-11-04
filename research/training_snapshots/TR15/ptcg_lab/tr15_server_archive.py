"""Bounded lossless archival of consumed, unreferenced closed server trajectories."""
import ast,base64,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
from .io import read_json,atomic_json,sha256,utcnow
from .server_boundary_audit import ssh_command

REMOTE_HEAD=r'''
import os,sys,json,time,hashlib,gzip,signal,resource,subprocess,re,shutil
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
os.nice(10)
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[-1]})
p=json.loads(sys.stdin.readline(1024*1024));root=Path('/mnt/HDD2_3TB/PROJECT_OWNER/ptcg-hybrid-v3');uid=os.getuid()
assert uid==1005 and root.resolve()==root and root.stat().st_uid==uid and not root.is_symlink()
assert json.loads((root/'.ptcg-owner.json').read_text())=={'project_id':'01a16843-f6fb-7151-accb-d709afdd6aca','uid':uid,'root':str(root)}
assert p['plan']['root']==str(root) and p['plan']['uid']==uid and 1<=len(p['plan']['rows'])<=256
assert sum(x['bytes'] for x in p['plan']['rows'])<=3*2**30
assert 60<p['deadline_unix']-time.time()<=1800
def alarm(*_):raise TimeoutError('Original finite server archival endpoint')
signal.signal(signal.SIGALRM,alarm);signal.alarm(int(p['deadline_unix']-time.time()))
class Pacer:
 def __init__(self):self.until=time.monotonic()
 def consume(self,n):
  self.until=max(self.until,time.monotonic())+n/(4*2**20)
  while time.monotonic()<self.until:time.sleep(min(.05,self.until-time.monotonic()))
pacer=Pacer()
def ident(f):
 s=f.stat();assert not f.is_symlink() and f.resolve().is_relative_to(root) and s.st_uid==uid
 return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid,s.st_mode]
def digest(f,paced=False):
 h=hashlib.sha256();n=0
 with f.open('rb') as stream:
  while data:=stream.read(1024*1024):
   if paced:pacer.consume(len(data))
   n+=len(data);h.update(data)
 return h.hexdigest(),n
def sha256(f):return digest(f)[0]
def atomic(f,value):
 f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.partial')
 with tmp.open('w') as stream:json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
 os.replace(tmp,f);fd=os.open(f.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def emit(value):print(json.dumps(value),flush=True)
job=root/'jobs'/p['plan']['source_run_id'];assert job.resolve()==job and job.stat().st_uid==uid
assert json.loads((job/'identity.json').read_text())==p['plan']['source_identity']
handle=p['plan']['source_identity']
text=subprocess.run(['systemctl','--user','show',handle['unit'],'-p','InvocationID','-p','ExecMainStartTimestampMonotonic','-p','ActiveState','-p','SubState','-p','ControlGroup'],capture_output=True,text=True,check=True,timeout=15).stdout
service=dict(x.split('=',1) for x in text.splitlines() if '=' in x)
assert service['InvocationID']==handle['invocation_id'] and service['ExecMainStartTimestampMonotonic']==handle['start_monotonic']
assert service['ActiveState'] in ('inactive','failed') or service['ActiveState']=='active' and service['SubState']=='exited'
if service['ControlGroup']:
 cgroup=Path('/sys/fs/cgroup')/service['ControlGroup'].lstrip('/');assert not cgroup.exists() or not (cgroup/'cgroup.procs').read_text().strip()
indices={}
for row in p['plan']['all_current_recovery_indices']:
 f=root/row['path'];assert sha256(f)==row['sha256'];indices[row['path']]=ident(f)
jobs_signature=ident(root/'jobs');job_signatures={str(f.relative_to(root)):ident(f) for f in (root/'jobs').iterdir() if f.is_dir() and not f.is_symlink()}
def unchanged_recovery():
 assert ident(root/'jobs')==jobs_signature,'Another project job appeared; refuse retirement'
 for name,value in indices.items():assert ident(root/name)==value,'Recovery index changed; refuse retirement'
 for name,value in job_signatures.items():assert ident(root/name)==value,'Job namespace changed; refuse retirement'
def logical():
 roots=[root,Path('/mnt/HDD1_3TB/PROJECT_OWNER/ptcg-hybrid-v3')]
 return sum((Path(d)/n).lstat().st_size for base in roots for d,_,names in os.walk(base,followlinks=False) for n in names)
before=logical();free_before=shutil.disk_usage(root).free
assert before+256*2**20<80*2**30 and free_before-256*2**20>=20*2**30
folder=root/'lossless-closed-trajectories'/p['run_id'];assert not folder.exists();folder.mkdir(parents=True,mode=0o700)
protected={name:digest(job/name,True)[0] for name in ['model.pt','checkpoint.pt','recovery-v2.json','checkpoint-commits/index.json','training-updates.json','league.json','progress.json','payload-config.json','trial-metrics.json'] if (job/name).exists()}
emit({'event':'started','run_id':p['run_id'],'source_run_id':job.name,'before_logical_bytes':before,'protected_files':protected,
 'remote_process_identity':{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()},'deadline_unix':p['deadline_unix']})
records=[];encoded_total=0;released_total=0
'''

REMOTE_BODY=r'''
for sequence,row in enumerate(p['plan']['rows']):
 unchanged_recovery()
 canonical=root/row['canonical'];assert canonical.is_relative_to(job/'rollouts') and re.fullmatch(r'v[0-9]+-s[0-9]+\.pt',canonical.name)
 aliases=[root/name for name in row['aliases']];assert canonical in aliases and len(aliases)<=3 and len(set(aliases))==len(aliases)
 for f in aliases:
  assert f.is_relative_to(job) and f.stat().st_size==row['bytes']
  assert f==canonical or re.fullmatch('data-'+row['sha256']+r'\.bin',f.name) and f.parent==job/'objects'
 identities={str(f.relative_to(root)):ident(f) for f in aliases}
 packed=folder/(row['sha256']+'.gz');encoded=pack_original(canonical,packed,row['sha256'],row['bytes'],pacer)
 with gzip.open(packed,'rb') as stream:
  h=hashlib.sha256();count=0
  while data:=stream.read(1024*1024):count+=len(data);h.update(data)
 assert count==row['bytes'] and h.hexdigest()==row['sha256'],'Full reconstructed raw bytes differ'
 assert sha256(packed)==encoded['sha256']
 for f in aliases:
  if f!=canonical:assert digest(f,True)==(row['sha256'],row['bytes']),'Independent original alias full hash differs'
  assert ident(f)==identities[str(f.relative_to(root))]
 record={'sequence':sequence,'source_sha256':row['sha256'],'source_bytes':row['bytes'],'original_aliases':row['aliases'],
  'encoded_path':packed.relative_to(root).as_posix(),'encoded_sha256':encoded['sha256'],'encoded_bytes':encoded['bytes'],
  'all_original_alias_full_SHA256_verified':True,'full_decompressed_SHA256_verified':True,'source_identities':identities}
 atomic(folder/('%04d-prepared.json'%sequence),record)
 record_digest=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 emit({'event':'prepared','record':record,'record_digest':record_digest})
 ack=json.loads(sys.stdin.readline(16385));assert ack['sequence']==sequence and ack['record_digest']==record_digest and ack['accepted'] is True and re.fullmatch('[0-9a-f]{64}',ack['local_receipt_sha256'])
 atomic(folder/('%04d-ACK.json'%sequence),ack)
 unchanged_recovery()
 for f in aliases:assert ident(f)==identities[str(f.relative_to(root))]
 for f in aliases:f.unlink()
 amount=row['bytes']*len(aliases);released_total+=amount;encoded_total+=encoded['bytes']
 result={'event':'released','sequence':sequence,'source_sha256':row['sha256'],'raw_logical_bytes_released':amount,'encoded_bytes_retained':encoded['bytes'],
  'server_encoded_copy_retained':True,'local_original_full_hash_archive_retained':True,'ACK_before_original_release':True}
 atomic(folder/('%04d-released.json'%sequence),result);records.append(result);emit(result)
unchanged_recovery()
for name,digest_before in protected.items():assert digest(job/name,True)[0]==digest_before,'Model/optimizer/checkpoint/records changed'
for row in p['plan']['all_current_recovery_indices']:assert sha256(root/row['path'])==row['sha256']
after=logical();result={'event':'complete','complete':True,'source_run_id':job.name,'run_id':p['run_id'],'files':len(records),
 'logical_before_bytes':before,'logical_after_bytes':after,'actual_net_logical_bytes_released':before-after,
 'free_before_bytes':free_before,'free_after_bytes':shutil.disk_usage(root).free,'encoded_bytes_retained':encoded_total,
 'all_original_encoded_and_reconstructed_full_SHA256_verified':True,'all_current_recovery_indices_unchanged':True,
 'models_optimizer_checkpoints_league_and_experiment_records_unchanged':True,'server_and_local_full_data_copies_retained':True,
 'archive_index_directory':folder.relative_to(root).as_posix(),'new_scientific_decisions':0}
atomic(folder/'INDEX.json',dict(result,records=records));emit(result)
'''

def run(config,output):
    root=Path(config['project_path']).resolve();output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':raise ValueError('Controlled archival runner required')
    if sha256(Path(__file__))!=config['retirement_module_sha256']:raise ValueError('Registered archival code changed')
    ref=config['server_trajectory_plan'];assert sha256(root/ref['path'])==ref['sha256'];plan=read_json(root/ref['path'])
    assert plan['authority']==config['TR15_authorization'] and len(plan['rows'])<=256
    # Reuse the previously accepted complete-input compression primitive unchanged.
    tree=ast.parse((root/'ptcg_lab/local_trajectory_archive.py').read_text(encoding='utf-8-sig'))
    pack=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pack_original')
    source=ast.get_source_segment((root/'ptcg_lab/local_trajectory_archive.py').read_text(encoding='utf-8-sig'),pack)
    remote=REMOTE_HEAD+'\n'+source+'\n'+REMOTE_BODY
    bootstrap='import base64;exec(compile(base64.b64decode('+repr(base64.b64encode(remote.encode()).decode())+'),"owned_closed_trajectory_archive","exec"))'
    command=ssh_command(config);command[-3:]=['/usr/bin/python3','-B','-u','-c',shlex.quote(bootstrap)]
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])-15;receipt=output/'server-retirement-receipts';receipt.mkdir()
    with (output/'ssh-errors.log').open('wb') as errors:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,encoding='utf-8',bufsize=1)
        child.stdin.write(json.dumps({'run_id':output.name,'deadline_unix':deadline,'plan':plan})+'\n');child.stdin.flush();final=None
        try:
            for line in child.stdout:
                if len(line)>1024*1024:raise ValueError('Archive receipt envelope exceeded')
                row=json.loads(line);event=row['event']
                if event=='prepared':
                    record=row['record'];path=receipt/('%04d-prepared.json'%record['sequence']);atomic_json(path,row)
                    expected=plan['rows'][record['sequence']]
                    assert record['source_sha256']==expected['sha256'] and record['source_bytes']==expected['bytes'] and record['original_aliases']==expected['aliases']
                    assert hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()==row['record_digest']
                    assert record['all_original_alias_full_SHA256_verified'] and record['full_decompressed_SHA256_verified']
                    assert record['encoded_path'].startswith('lossless-closed-trajectories/'+output.name+'/') and 0<record['encoded_bytes']<=record['source_bytes']+2**20
                    ack={'sequence':record['sequence'],'record_digest':row['record_digest'],'accepted':True,'local_receipt_sha256':sha256(path)}
                    child.stdin.write(json.dumps(ack)+'\n');child.stdin.flush()
                elif event=='started':atomic_json(output/'server-retirement-started.json',row)
                elif event=='released':atomic_json(receipt/('%04d-released.json'%row['sequence']),row)
                elif event=='complete':final=row
                atomic_json(output/'progress.json',{'at':utcnow(),'stage':'server_lossless_archive_'+event,'sequence':row.get('sequence',row.get('record',{}).get('sequence'))})
            code=child.wait(timeout=20)
            if code or not final:raise RuntimeError('Bounded archival stopped; per-file originals or verified gzip+ACK receipts retained; inspect ssh-errors.log')
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=15)
    atomic_json(output/'metrics.json',dict(final,at=utcnow(),engineering_only=True,scientific_credit=0))
