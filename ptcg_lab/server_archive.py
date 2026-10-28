"""Read-only bounded binary return of one terminal owned training job."""
import argparse
import gzip
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import struct
import zlib

from .io import atomic_json,read_json,sha256,utcnow
from .remote_lease import validate_no_unreconciled_remote
from .server_boundary_audit import ssh_command
from .terminal_journal import journal_terminal,journal_failed_exit



def terminal_proven(properties,handle,ticket_deadline,prior,now,journal=None,expected_uid=1005,failed_exit=False):
    def terminal(props):
        return props.get('ActiveState') in ('failed','inactive') or props.get('ActiveState')=='active' and props.get('SubState')=='exited'
    def matches(props):
        return props.get('InvocationID')==handle['invocation_id'] and props.get('ExecMainStartTimestampMonotonic')==handle['start_monotonic']
    if properties.get('LoadState')=='not-found':
        if journal is not None:
            if failed_exit:
                try:
                    proof=journal_failed_exit(journal,handle,expected_uid,now)
                    return (matches(prior) and prior.get('ActiveState')=='failed' and prior.get('Result')=='exit-code'
                        and prior.get('ExecMainStatus')==proof['ExecMainStatus'] and prior.get('ControlGroup')=='')
                except (ValueError,KeyError,TypeError):return False
            try:journal_terminal(journal,handle,expected_uid,now);return True
            except (ValueError,KeyError,TypeError):return False
        return now>=ticket_deadline+60 and matches(prior) and terminal(prior)
    return matches(properties) and terminal(properties)


REMOTE=inspect.getsource(journal_terminal)+'\n'+inspect.getsource(journal_failed_exit)+'\n'+inspect.getsource(terminal_proven)+r'''
import os,pathlib,json,sys,subprocess,signal,time,hashlib,re,struct,zlib,resource
if sys.flags.optimize:raise RuntimeError('Archive validation requires enabled checks')
resource.setrlimit(resource.RLIMIT_CPU,(120,120))
resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
p=PARAMETERS;root=pathlib.Path(p.get('source_root','OLD_READONLY_ROOT/ptcg-hybrid-v3'));uid=os.getuid()
started_unix=time.time()
if abs(started_unix-p['client_unix'])>=5:raise ValueError('Read-only archive initial clock/transport handshake exceeded5seconds: '+str(started_unix-p['client_unix']))
assert str(root) in ('OLD_READONLY_ROOT/ptcg-hybrid-v3','SERVER_PROJECT_ROOT')
assert not root.is_symlink() and root.resolve()==root and root.stat().st_uid==uid and not root.stat().st_mode&0o077
marker=root/'.ptcg-owner.json';assert not marker.is_symlink() and marker.stat().st_uid==uid
assert json.loads(marker.read_text())=={'project_id':p['project_id'],'uid':uid,'root':str(root)}
assert re.fullmatch(r'[A-Za-z0-9_-]{12,100}',p['source_run_id'])
job=root/'jobs'/p['source_run_id'];assert not (root/'jobs').is_symlink() and not job.is_symlink() and job.resolve()==job and job.stat().st_uid==uid
ticket=json.loads((job/'ticket.json').read_text());assert ticket['run_id']==p['source_run_id'] and ticket['project_id']==p['project_id']
handle=p['identity'];assert ticket['unit']==handle['unit'] and json.loads((job/'identity.json').read_text())==handle
assert re.fullmatch(r'ptcg-hybrid-[a-z0-9-]{8,80}\.service',handle['unit'])
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==handle['host_boot_id']
env=dict(os.environ,XDG_RUNTIME_DIR='/run/user/'+str(uid),DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/'+str(uid)+'/bus')
r=subprocess.run(['systemctl','--user','show',handle['unit'],'--property=LoadState','--property=InvocationID','--property=ExecMainStartTimestampMonotonic','--property=ActiveState','--property=SubState'],env=env,text=True,capture_output=True,timeout=10,check=True)
props=dict(x.split('=',1) for x in r.stdout.splitlines())
journal=None
if p.get('journal_terminal_proof'):
 result=subprocess.run(['journalctl','--user','-u',handle['unit'],'-n','20','--no-pager','-o','json'],env=env,text=True,capture_output=True,check=True,timeout=12)
 assert len(result.stdout)<=256*1024
 journal=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
 parser=journal_failed_exit if p.get('failed_exit_journal') else journal_terminal
 assert parser(journal,handle,uid,time.time())==p['journal_terminal_proof']
assert terminal_proven(props,handle,ticket['deadline_unix'],p['prior_terminal_properties'],time.time(),journal,uid,p.get('failed_exit_journal',False))
if not 30<p['deadline_unix']-time.time()<=1800:raise ValueError('Original finite archive deadline exhausted')
def timed_out(*_):raise TimeoutError('Finite read-only archive deadline')
signal.signal(signal.SIGALRM,timed_out);signal.alarm(max(1,int(p['deadline_unix']-time.time())))
paths=[]
for path in job.rglob('*'):
 rel=path.relative_to(job)
 if rel.parts[0] in ('control','stage','requests','acks','evictions','receipts') or not path.is_file():continue
 assert not path.is_symlink() and path.stat().st_uid==uid and root in path.resolve().parents
 assert path.stat().st_size<=128*2**20
 paths.append(path)
paths.sort(key=lambda x:(x.name!='checkpoint.pt',str(x)))
maximum=p.get('maximum_source_files',512)
assert maximum in (512,1024)
assert len(paths)<=maximum and sum(x.stat().st_size for x in paths)<=2*2**30
if maximum==1024:
 actual=[]
 for f in paths:
  s=f.stat();actual.append({'path':f.relative_to(job).as_posix(),'bytes':s.st_size,'identity':[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]})
 actual.sort(key=lambda r:r['path'])
 assert len(actual)==p['expected_file_count'] and sum(r['bytes'] for r in actual)==p['expected_bytes']
 assert hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest()==p['expected_inventory_sha256']
print(json.dumps({'run_id':p['source_run_id'],'file_count':len(paths),'bytes':sum(x.stat().st_size for x in paths),'read_only':True,'identity':handle,'current_service_properties':props,'prior_terminal_proof_used':props.get('LoadState')=='not-found'}),flush=True)
next_io=time.monotonic()
for path in paths:
 st=path.stat();before=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns);rel=path.relative_to(job).as_posix()
 reused=p.get('reuse',{}).get(rel)
 if reused:
  assert reused['bytes']==st.st_size
  if not reused.get('accept_new_source_identity'):assert reused['file_identity']==list(before)
 codec='reuse' if reused else 'zlib' if p.get('lossless_wire_compression') else 'raw'
 print(json.dumps({'path':rel,'bytes':st.st_size,'file_identity':before,'codec':codec}),flush=True)
 compressor=zlib.compressobj(1) if codec=='zlib' else None
 digest=hashlib.sha256();count=0
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb',buffering=0) as source:
  actual=os.fstat(source.fileno());assert (actual.st_dev,actual.st_ino,actual.st_size,actual.st_mtime_ns)==before
  while count<st.st_size:
   n=min(256*1024,st.st_size-count);next_io=max(next_io,time.monotonic())+n/(4*2**20)
   while True:
    delay=next_io-time.monotonic()
    if delay<=0:break
    if time.time()>p['deadline_unix']:raise TimeoutError('Archive pacing deadline')
    time.sleep(min(.05,delay))
   data=source.read(n)
   if not data:raise EOFError('Immutable archive file truncated')
   if codec=='raw':sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
   elif codec=='zlib':
    packed=compressor.compress(data)+compressor.flush(zlib.Z_SYNC_FLUSH)
    sys.stdout.buffer.write(struct.pack('>I',len(packed))+packed);sys.stdout.buffer.flush()
   digest.update(data);count+=len(data)
 if codec=='zlib':
  packed=compressor.flush(zlib.Z_FINISH)
  sys.stdout.buffer.write(struct.pack('>I',len(packed))+packed+struct.pack('>I',0));sys.stdout.buffer.flush()
 st=path.stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==before
 if reused:assert reused['sha256']==digest.hexdigest()
 print(json.dumps({'path':rel,'bytes':count,'sha256':digest.hexdigest()}),flush=True)
print(json.dumps({'complete':True,'source_files_untouched':True}),flush=True)
'''


def frame(stream):
    line=stream.readline(16385)
    if not line or len(line)>16384 or not line.endswith(b'\n'):raise ValueError('Invalid archive frame')
    return json.loads(line)


def destination(directory,relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Archive path escaped destination')
    target=directory/rel
    if directory.resolve() not in target.resolve().parents:raise ValueError('Archive realpath escaped destination')
    return target


def compressed_chunks(stream,size):
    """Decode a framed bounded zlib stream, rejecting truncation and expansion."""
    decoder=zlib.decompressobj();received=0
    while True:
        header=stream.read(4)
        if len(header)!=4:raise EOFError('Compressed archive frame truncated')
        length=struct.unpack('>I',header)[0]
        if length==0:break
        if length>512*1024:raise ValueError('Compressed archive block too large')
        packed=stream.read(length)
        if len(packed)!=length:raise EOFError('Compressed archive block truncated')
        data=decoder.decompress(packed,min(256*1024,size-received)+1)
        received+=len(data)
        if received>size or len(data)>256*1024 or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError('Compressed archive expansion or trailing data')
        if data:yield data
    if received!=size or not decoder.eof:raise ValueError('Compressed archive size or termination differs')


def connect(config,parameters,output,errors,deadline):
    """Retry only a pre-inventory SSH255 close, within the original deadline."""
    records=[];error_path=output/'remote-read-errors.log'
    for attempt in range(1,4):
        parameters['client_unix']=time.time()
        payload=('PARAMETERS='+repr(parameters)+'\n'+REMOTE).encode()
        errors.flush();offset=error_path.stat().st_size
        child=subprocess.Popen(ssh_command(config),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,bufsize=0)
        try:
            view=memoryview(payload)
            while view:
                written=child.stdin.write(view)
                if not written:raise EOFError('Archive control stdin closed')
                view=view[written:]
            child.stdin.close();stream=io.BufferedReader(child.stdout,buffer_size=256*1024)
            inventory=frame(stream)
            records.append({'at':utcnow(),'connection_attempt':attempt,'inventory_received':True})
            atomic_json(output/'archive-connection-attempts.json',records)
            return child,stream,inventory
        except (BrokenPipeError,EOFError,ValueError) as exc:
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=5)
            errors.flush()
            with error_path.open('rb') as log:log.seek(offset);detail=log.read(8192).decode(errors='replace')
            records.append({'at':utcnow(),'connection_attempt':attempt,'inventory_received':False,
                            'exit_code':child.returncode,'error':type(exc).__name__,'stderr':detail})
            atomic_json(output/'archive-connection-attempts.json',records)
            child.stdout.close()
            try:child.stdin.close()
            except BrokenPipeError:pass
            if (child.returncode!=255 or 'Connection closed by ' not in detail
                    or attempt==3 or time.time()>deadline-60):raise
            time.sleep(3*attempt)
    raise RuntimeError('Unreachable bounded connection loop')


def inventory_bound(project,config):
    maximum=config.get('maximum_source_files',512)
    if maximum==512:return maximum,{}
    if maximum!=1024:raise ValueError('Unreviewed archive file bound')
    spec=config['source_inventory_evidence'];path=(Path(project)/spec['path']).resolve()
    if Path(project).resolve() not in path.parents or sha256(path)!=spec['sha256']:raise ValueError('Measured terminal inventory changed')
    proof=read_json(path)
    if proof['source_run_id']!=config['source_run_id']:raise ValueError('Terminal inventory names another source')
    measured=proof['actual'];rows=sorted(measured['rows'],key=lambda r:r['path'])
    if (not measured.get('read_only_metadata_only') or measured.get('owner_uid')!=1005
            or not 512<len(rows)<=1024 or measured['file_count']!=len(rows)
            or sum(r['bytes'] for r in rows)!=measured['bytes'] or measured['bytes']>2*2**30
            or any(not 0<=r['bytes']<=128*2**20 for r in rows)):
        raise ValueError('Measured terminal inventory exceeds the bounded extension')
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return maximum,{'maximum_source_files':maximum,'expected_file_count':len(rows),'expected_bytes':measured['bytes'],
                    'expected_inventory_sha256':digest}


def materialize_compressed_reuse(project,output,rows,deadline):
    """Decode exact durable originals once; the server still cold-hashes each.

    This is part of the same controlled archive job, with its original deadline
    and local storage supervision. No training source or existing archive changes.
    """
    from .archive_compressed import verify_archive
    from .archive_ack import archive_root,safe_target
    result={};decoded={};cache=output/'reused-originals'
    if not rows:return result,decoded
    cache.mkdir()
    for row in rows:
        spec=row['ACK'];ack_path=(project/spec['path']).resolve()
        if project.resolve() not in ack_path.parents or sha256(ack_path)!=spec['sha256']:
            raise ValueError('Durable compressed ACK changed')
        ack=read_json(ack_path);base=(project/row['archive_root']).resolve()
        if project.resolve() not in base.parents or (ack['sha256'],ack['bytes'])!=(row['sha256'],row['bytes']):
            raise ValueError('Compressed original identity differs')
        digest=row['sha256'];target=cache/(digest+'.blob')
        if digest not in decoded:
            verify_archive(base,ack)
            original=safe_target(archive_root(base),ack['archive_relative']);h=hashlib.sha256();count=0
            with gzip.open(original,'rb') if ack.get('storage') else original.open('rb') as src,target.open('xb') as dst:
                while block:=src.read(1024*1024):
                    if time.time()>deadline-30:raise TimeoutError('Compressed reuse decode deadline')
                    count+=len(block)
                    if count>row['bytes']:raise ValueError('Compressed original expansion differs')
                    dst.write(block);h.update(block)
                dst.flush();os.fsync(dst.fileno())
            if count!=row['bytes'] or h.hexdigest()!=digest:raise ValueError('Decoded exact original SHA differs')
            decoded[digest]={'path':target,'bytes':count,'first_object':row['path']}
        result[row['path']]={k:row[k] for k in ['path','bytes','sha256']}
        result[row['path']].update(local_source=str(target),accept_new_source_identity=True)
    atomic_json(output/'compressed-reuse-preparation.json',{'complete':True,'unique_originals_decoded':len(decoded),
        'source_paths':len(result),'raw_network_bytes_avoided':sum(r['bytes'] for r in rows),
        'all_original_ACKs_and_raw_SHA_verified':True})
    return result,decoded


def run(config,output):
    output=Path(output).resolve();project=Path(config['project_path'])
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Archive requires controlled runner')
    deadline=min(float(os.environ['PTCG_JOB_DEADLINE_UNIX']),config.get('archive_original_absolute_deadline',float('inf')))
    if sha256(Path(__file__))!=config['archive_source_sha256']:raise ValueError('Archive program changed')
    if config.get('archive_repair_parent_run_ids'):
        spent=sum(a['seconds'] for run_id in config['archive_repair_parent_run_ids'] for a in read_json(project/'runs'/run_id/'attempts.json'))
        if spent+deadline-time.time()>config.get('archive_scope_seconds_max',1800)+.5:raise RuntimeError('Archive repair cannot regain spent scope')
    state=read_json(project/'research/autonomy/state.json');validate_no_unreconciled_remote(project,state)
    source=project/'runs'/config['source_run_id'];terminal_path=source/'remote-terminal.json'
    if config.get('source_terminal_evidence'):
        ref=config['source_terminal_evidence'];terminal_path=(project/ref['path']).resolve()
        if source.resolve() not in terminal_path.parents or sha256(terminal_path)!=ref['sha256']:
            raise ValueError('Exact preregistered terminal receipt path/hash differs')
    terminal=read_json(terminal_path)
    if source.resolve().parent!=(project/'runs').resolve():raise ValueError('Archive source run escaped project')
    parameters={'project_id':config['project_id'],'source_run_id':source.name,'identity':terminal['identity'],
                'client_unix':time.time(),'deadline_unix':deadline-20,'prior_terminal_properties':terminal.get('service_properties',{})}
    parameters['source_root']=config.get('source_root','OLD_READONLY_ROOT/ptcg-hybrid-v3')
    maximum_files,expected_inventory=inventory_bound(project,config);parameters.update(expected_inventory)
    if terminal.get('terminal_proof_kind')=='exact_invocation_systemd_journal':parameters['journal_terminal_proof']=terminal['journal_terminal_proof']
    elif terminal.get('terminal_proof_kind')=='exact_invocation_systemd_failed_exit_journal':
        parameters.update(journal_terminal_proof=terminal['journal_terminal_proof'],failed_exit_journal=True)
    reusable={}
    for prior_run in config.get('reuse_archive_run_ids',[]):
        prior=(project/'runs'/prior_run).resolve()
        if prior.parent!=(project/'runs').resolve():raise ValueError('Prior archive escaped project')
        inventory=read_json(prior/'inventory.json')
        if inventory['run_id']!=source.name or inventory['identity']!=terminal['identity']:
            raise ValueError('Prior archive source identity differs')
        for row in read_json(prior/'file-receipts.json'):
            old=destination(prior/'archive',row['path'])
            if old.stat().st_size!=row['bytes'] or sha256(old)!=row['sha256']:
                raise ValueError('Reusable durable archive file changed')
            if row['path'] in reusable and reusable[row['path']]['sha256']!=row['sha256']:
                raise ValueError('Prior archive receipts conflict')
            reusable[row['path']]={**row,'local_source':str(old)}
    # An immutable original may predate this remote run. Verify its full local
    # bytes now; the terminal server still cold-hashes its exact current file
    # identity before a new receipt/link can be published. No prior source inode
    # is invented and no unknown remote content is skipped.
    for row in config.get('reuse_local_sources',[]):
        old=(project/row['local_path']).resolve()
        if project.resolve() not in old.parents or old.is_symlink() or old.stat().st_size!=row['bytes'] or sha256(old)!=row['sha256']:
            raise ValueError('Reusable local original archive changed or escaped')
        destination(output/'archive',row['path'])
        previous=reusable.get(row['path'])
        if previous and previous['sha256']!=row['sha256']:raise ValueError('Reusable local archive paths conflict')
        reusable[row['path']]={**row,'local_source':str(old),'accept_new_source_identity':True}
    compressed,decoded=materialize_compressed_reuse(project,output,config.get('reuse_compressed_local_sources',[]),deadline)
    for name,row in compressed.items():
        if name in reusable and row['sha256']!=reusable[name]['sha256']:raise ValueError('Compressed reused source conflict')
        reusable[name]=row
    parameters['reuse']={key:{k:v for k,v in row.items() if k!='local_source'} for key,row in reusable.items()}
    parameters['lossless_wire_compression']=bool(config.get('lossless_wire_compression'))
    target_root=output/'archive';target_root.mkdir()
    receipts=[];start=time.monotonic();last=0;child=None
    with (output/'remote-read-errors.log').open('wb') as errors:
        child,stream,inventory=connect(config,parameters,output,errors,deadline)
        try:
            atomic_json(output/'inventory.json',inventory)
            if inventory['run_id']!=source.name or inventory['identity']!=terminal['identity'] or not 0<=inventory['file_count']<=maximum_files or inventory['bytes']>2*2**30:
                raise ValueError('Archive source inventory identity differs')
            if expected_inventory and (inventory['file_count']!=expected_inventory['expected_file_count'] or inventory['bytes']!=expected_inventory['expected_bytes']):
                raise ValueError('Measured terminal archive envelope changed')
            for _ in range(inventory['file_count']):
                item=frame(stream);path=destination(target_root,item['path']);path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_name(path.name+'.incoming');digest=hashlib.sha256();left=item['bytes']
                if path.exists() or not 0<=left<=128*2**20:raise ValueError('Archive file envelope or duplicate')
                if item.get('codec')=='reuse':
                    prior=reusable.get(item['path'])
                    if not prior or (not prior.get('accept_new_source_identity') and prior['file_identity']!=item['file_identity']) or prior['bytes']!=left:
                        raise ValueError('Immutable reused file identity differs')
                    footer=frame(stream)
                    if footer['path']!=item['path'] or footer['bytes']!=left or footer['sha256']!=prior['sha256']:
                        raise ValueError('Reused server file hash differs')
                    expected=config.get('required_hashes',{}).get(item['path'])
                    if expected and expected!=footer['sha256']:raise ValueError('Reused prior verified hash differs')
                    os.link(prior['local_source'],path)
                    receipts.append({**item,'sha256':footer['sha256'],'local_source':prior['local_source']})
                    atomic_json(output/'file-receipts.json',receipts)
                    continue
                with temporary.open('xb') as local:
                    chunks=iter(compressed_chunks(stream,left)) if item.get('codec')=='zlib' else None
                    while left:
                        if time.time()>deadline-25:raise TimeoutError('Archive receive deadline')
                        data=next(chunks,b'') if chunks is not None else stream.read(min(left,256*1024))
                        if not data:raise EOFError('Archive binary stream ended early')
                        local.write(data);digest.update(data);left-=len(data)
                        if time.monotonic()-last>=5:
                            atomic_json(output/'progress.json',{'at':utcnow(),'stage':'receiving','path':item['path'],
                                'file_bytes_done':item['bytes']-left,'file_bytes':item['bytes'],'archived_files':len(receipts)})
                            last=time.monotonic()
                    if chunks is not None:
                        if next(chunks,None) is not None:raise ValueError('Compressed archive excess bytes')
                    local.flush();os.fsync(local.fileno())
                footer=frame(stream)
                if footer['path']!=item['path'] or footer['bytes']!=item['bytes'] or footer['sha256']!=digest.hexdigest():
                    raise ValueError('Archive full file hash differs')
                expected=config.get('required_hashes',{}).get(item['path'])
                if expected and expected!=digest.hexdigest():raise ValueError('Archive differs from prior verified checkpoint')
                temporary.replace(path);receipts.append({**item,'sha256':digest.hexdigest()})
                atomic_json(output/'file-receipts.json',receipts)
            tail=frame(stream)
            if not tail.get('complete') or child.wait(timeout=20)!=0:raise RuntimeError('Read-only archive did not finish')
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=10)
    if sum(x['bytes'] for x in receipts)!=inventory['bytes']:raise ValueError('Archive byte total differs')
    required=config.get('required_hashes',{})
    if any(not any(row['path']==name and row['sha256']==digest for row in receipts) for name,digest in required.items()):
        raise ValueError('Required verified artifact missing')
    decoded_cache_retired=0
    for digest,row in decoded.items():
        path=row['path'];other=destination(target_root,row['first_object'])
        if output not in path.resolve().parents or path.is_symlink() or sha256(other)!=digest or sha256(path)!=digest:
            raise ValueError('Exact redundant decoded cache cannot be safely retired')
        if (path.stat().st_dev,path.stat().st_ino)!=(other.stat().st_dev,other.stat().st_ino):raise ValueError('Archive/cache hardlink identity differs')
        path.unlink();decoded_cache_retired+=row['bytes']
    compression=None
    sample=next((x for x in receipts if x['path'].startswith('rollouts/') and x['path'].endswith('.pt')),None)
    if sample:
        raw=target_root/sample['path'];compressed=output/'lossless-rollout-sample.pt.gz'
        with raw.open('rb') as src,compressed.open('xb') as dst:
            with gzip.GzipFile(fileobj=dst,mode='wb',filename='',mtime=0,compresslevel=1) as packed:
                while data:=src.read(256*1024):packed.write(data)
            dst.flush();os.fsync(dst.fileno())
        h=hashlib.sha256()
        with gzip.open(compressed,'rb') as unpacked:
            while data:=unpacked.read(256*1024):h.update(data)
        if h.hexdigest()!=sample['sha256']:raise ValueError('Lossless rollout reconstruction differs')
        compression={'source_path':sample['path'],'source_bytes':sample['bytes'],'compressed_bytes':compressed.stat().st_size,
                     'ratio':compressed.stat().st_size/sample['bytes'],'exact_reconstruction_sha256':h.hexdigest()}
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'source_run_id':source.name,
        'archived_files':len(receipts),'archived_bytes':inventory['bytes'],'source_files_untouched':True,
        'all_files_hash_verified_and_fsynced_locally':True,'required_prior_hashes_verified':required,
        'archive_receipts_sha256':sha256(output/'file-receipts.json'),'seconds':time.monotonic()-start,
        'lossless_rollout_sample':compression,'lossless_wire_compression':bool(config.get('lossless_wire_compression')),
        'reused_files':len(reusable),'decoded_redundant_cache_bytes_retired_after_full_archive':decoded_cache_retired,
        'promotion_eligible':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(read_json(args.config),args.output)
