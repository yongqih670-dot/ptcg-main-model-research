"""Finite project-owned migration endpoint; no shell actions or foreign writes."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import io
import zipfile

OLD='OLD_READONLY_ROOT/ptcg-hybrid-v3';NEW='SERVER_PROJECT_ROOT'
PROJECT='PROJECT_ID'
session=None


def acceptance_initialize(p):
    global session
    root=Path(NEW);uid=os.getuid();rid=p['run_id']
    if session is not None or not re.fullmatch(r'[A-Za-z0-9_-]{12,100}',rid):raise ValueError('Unexpected acceptance session')
    if root.is_symlink() or root.resolve()!=root or root.stat().st_uid!=uid or root.stat().st_mode&0o077:
        raise ValueError('Target root isolation differs')
    if json.loads((root/'.ptcg-owner.json').read_text())!={'project_id':PROJECT,'uid':uid,'root':NEW}:raise ValueError('Target owner differs')
    if abs(time.time()-p['client_unix'])>5 or not 120<p['deadline_unix']-time.time()<=1800:raise ValueError('Acceptance deadline differs')
    copyjob=root/'jobs'/p['copy_run_id']
    if copyjob.is_symlink() or copyjob.resolve()!=copyjob or copyjob.stat().st_uid!=uid:raise ValueError('Copy job owner differs')
    manifest=copyjob/'full-copy-manifest.json.gz'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=p['copy_manifest_sha256']:raise ValueError('Actual full copy receipt differs')
    if json.loads((copyjob/'copy-progress.json').read_text()).get('complete') is not True:raise ValueError('Copy not fully complete')
    profile=json.loads((root/'.ptcg-storage.json').read_text())
    if profile!=p['storage_profile']:raise ValueError('Forward storage authority differs')
    job=root/'jobs'/rid;job.mkdir(mode=0o700);control=job/'control';control.mkdir(mode=0o700)
    packed=base64.b64decode(p['code_zip_base64'],validate=True)
    if hashlib.sha256(packed).hexdigest()!=p['code_zip_sha256']:raise ValueError('Acceptance source archive changed')
    with zipfile.ZipFile(io.BytesIO(packed)) as z:
        if sum(i.file_size for i in z.infolist())>4*2**20:raise ValueError('Acceptance code closure too large')
        for info in z.infolist():
            rel=Path(info.filename)
            if rel.is_absolute() or '..' in rel.parts or not (rel.as_posix().startswith('ptcg_lab/') and rel.suffix=='.py' or rel.as_posix()=='acceptance_worker.py'):
                raise ValueError('Unexpected acceptance source')
            target=control/rel;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(z.read(info));f.flush();os.fsync(f.fileno())
    sys.path.insert(0,str(control))
    from ptcg_lab.storage_profile import validate
    from ptcg_lab.server_workspace import owned_root,admit,inside
    from ptcg_lab.systemd_job import inspect_service,identity,terminal
    from ptcg_lab.resource_profile import verify_actual_machine
    from ptcg_lab.io import atomic_json
    validate(profile,root);owned_root(OLD,PROJECT);verify_actual_machine(p['server_resource_profile'])
    metadata=inside(root,root/'jobs/20261026T131453Z_tr09-epochs4-CPU32_382e9e/control/ptcg_lab/public_cg/metadata.json')
    if hashlib.sha256(metadata.read_bytes()).hexdigest()!=p['public_metadata_sha256']:
        raise ValueError('Original public card metadata SHA differs')
    target_metadata=control/'ptcg_lab/public_cg/metadata.json'
    target_metadata.parent.mkdir(mode=0o700,parents=True,exist_ok=True);os.link(metadata,target_metadata)
    copyhandle=json.loads((copyjob/'identity.json').read_text());boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    props=inspect_service(copyhandle['unit'])
    prior=p['copy_terminal_receipt'];prior_props=prior['service_properties']
    if (prior.get('identity')!=copyhandle or prior.get('host_boot_id')!=boot
            or identity(prior_props,boot)!=copyhandle or not terminal(prior_props)
            or prior_props.get('Result')!='success' or prior_props.get('ExecMainStatus')!='0' or prior_props.get('ControlGroup')!=''):
        raise ValueError('Archived positive exact copy exit0/emptycgroup receipt required')
    if props.get('LoadState')=='not-found':
        from ptcg_lab.terminal_journal import read_journal,journal_terminal
        events=read_journal(copyhandle);proof=journal_terminal(events,copyhandle,uid,time.time())
        atomic_json(job/'copy-terminal-handoff-proof.json',{'original_positive_receipt':prior,'actual_exact_invocation_journal':events,'proof':proof,
            'not_found_defaults_not_used_as_success_evidence':True})
    elif identity(props,boot)!=copyhandle or not terminal(props) or props.get('Result')!='success' or props.get('ExecMainStatus')!='0':
        raise ValueError('Exact copy invocation must be positively terminal')
    units=subprocess.run(['systemctl','--user','list-units','ptcg-hybrid-*','--all','--no-legend','--no-pager'],
        capture_output=True,text=True,check=True,timeout=10,
        env=dict(os.environ,XDG_RUNTIME_DIR='/run/user/'+str(uid),DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/'+str(uid)+'/bus')).stdout
    for line in units.splitlines():
        tokens=line.replace('●','').split()
        if tokens and tokens[0]!=copyhandle['unit']:raise RuntimeError('Other project service must be reconciled')
    originals=[];adaptations=[]
    candidates=[root/'.venv/pyvenv.cfg',*[x for x in (root/'.venv/bin').iterdir() if x.is_file() and not x.is_symlink()]]
    for path in candidates:
        inside(root,path)
        if path.stat().st_size>2*2**20:raise ValueError('Unexpected large environment entrypoint')
        old=path.read_bytes()
        if OLD.encode() not in old:continue
        new=old.replace(OLD.encode(),NEW.encode());relative=path.relative_to(root).as_posix()
        originals.append({'path':relative,'original_base64':base64.b64encode(old).decode(),'sha256':hashlib.sha256(old).hexdigest()})
        temporary=path.with_name(path.name+'.relocation-incoming')
        with temporary.open('xb') as f:f.write(new);f.flush();os.fsync(f.fileno())
        os.chmod(temporary,path.stat().st_mode&0o777);temporary.replace(path)
        adaptations.append({'path':relative,'before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(new).hexdigest()})
    atomic_json(job/'environment-entrypoint-originals.json',originals)
    atomic_json(job/'environment-adaptation.json',{'changes':adaptations,'only_copied_environment_entrypoints_changed':True,
        'historical_scientific_job_tickets_manifests_checkpoints_unchanged':True})
    chain=[]
    for previous in p.get('previous_engineering_run_ids',[]):
        if not re.fullmatch(r'[A-Za-z0-9_-]{12,100}',previous):raise ValueError('Unsafe previous engineering record')
        directory=inside(root,root/'jobs'/previous)
        records={}
        for name in ['environment-adaptation.json','environment-entrypoint-originals.json']:
            path=directory/name
            if path.exists():
                inside(root,path)
                if path.stat().st_size>256*1024:raise ValueError('Environment adaptation history too large')
                records[name]=json.loads(path.read_text())
        chain.append({'run_id':previous,'records':records})
    atomic_json(job/'environment-adaptation-chain.json',chain)
    rows=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.splitlines()
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout
    from ptcg_lab.server_trial_agent import choose_idle_gpu
    gpu_required=p.get('gpu_restore_required',True)
    selected=choose_idle_gpu(rows,apps) if gpu_required else None
    if gpu_required and selected is None:raise RuntimeError('No authorized idle GPU for full CUDA RNG restore; no training started')
    ticket={'kind':'storage_acceptance','mode':'migration_full_restore','run_id':rid,'lease_epoch':p['lease_epoch'],
        'project_id':PROJECT,'project_root':NEW,'source_root':OLD,'unit':p['unit'],'python_kind':'venv',
        'job_directory':str(job),'ticket_path':str(job/'ticket.json'),'control_root':str(control),
        'deadline_unix':p['deadline_unix'],'memory_max_bytes':p['server_resource_profile']['memory_max_bytes'],
        'cpu_quota_cores':p['server_resource_profile']['cpu_quota_cores'],'server_resource_profile':p['server_resource_profile'],
        'workspace_hard_max_bytes':50*2**30,'storage_profile':profile,'stop_grace_seconds':45,
        'gpu_index':int(selected[0]) if selected else None,'gpu_uuid':selected[1] if selected else None,'gpu_restore_required':gpu_required,
        'cpu_throughput_probe':bool(p.get('cpu_throughput_probe',False))}
    atomic_json(job/'ticket.json',ticket);session={'ticket':ticket,'job':job}
    atomic_json(job/'gpu-start-probe.json',{'selected_row':selected,'all_authorized_observed_rows':[r for r in rows if int(r.split(',')[0]) in (4,5,6,7)],'no_foreign_compute_on_selected_GPU':True})
    return {'ticket':ticket,'capacity':admit(root,512*2**20),'environment_adaptation':adaptations,'selected_idle_gpu':selected}


def operate(p):
    global session
    root=Path(NEW);source=Path(OLD);uid=os.getuid()
    if uid!=1005:raise ValueError('Unexpected server UID')
    if p['operation']=='acceptance_initialize':return acceptance_initialize(p)
    if p['operation']=='initialize':
        if session is not None:raise RuntimeError('Migration session already initialized')
        rid=p['run_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{12,100}',rid):raise ValueError('Invalid run identity')
        parent=root.parent
        if (parent.is_symlink() or parent.resolve()!=parent or parent.stat().st_uid!=uid
                or not os.access(parent,os.W_OK|os.X_OK) or root.exists()):
            raise ValueError('Migration requires the absent target beneath owned real parent')
        if abs(time.time()-p['client_unix'])>5:raise ValueError('Server clock differs')
        if not 120<p['deadline_unix']-time.time()<=7200:raise ValueError('Migration deadline differs')
        if shutil.disk_usage(parent).free<40*2**30:raise OSError('Insufficient target reserve')
        manager_env=dict(os.environ,XDG_RUNTIME_DIR='/run/user/'+str(uid),DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/'+str(uid)+'/bus')
        units=subprocess.run(['systemctl','--user','list-units','ptcg-hybrid-*','--all','--no-legend','--no-pager'],
            capture_output=True,text=True,check=True,timeout=10,env=manager_env).stdout
        if units.strip():raise RuntimeError('Existing project unit must be reconciled before migration')
        root.mkdir(mode=0o700)
        owner={'project_id':PROJECT,'uid':uid,'root':NEW}
        def write(path,obj):
            with path.open('x',encoding='utf-8') as f:json.dump(obj,f);f.flush();os.fsync(f.fileno())
        write(root/'.ptcg-owner.json',owner)
        for name in ['jobs','tmp','migration-control']: (root/name).mkdir(mode=0o700)
        control=root/'migration-control'/rid;control.mkdir(mode=0o700);(control/'ptcg_lab').mkdir(mode=0o700)
        allowed={'ptcg_lab/'+x for x in ['__init__.py','io.py','systemd_job.py','resource_profile.py','storage_profile.py','server_workspace.py','storage_copy_worker.py']}|{'migration_worker.py'}
        if set(p['files'])!=allowed:raise ValueError('Unexpected migration code closure')
        for relative,item in p['files'].items():
            data=base64.b64decode(item['base64'],validate=True)
            if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Staged source digest differs')
            with (control/relative).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        sys.path.insert(0,str(control))
        from ptcg_lab.storage_profile import validate
        from ptcg_lab.server_workspace import owned_root,admit
        from ptcg_lab.resource_profile import verify_actual_machine
        validate(p['storage_profile'],root);owned_root(source,PROJECT);write(root/'.ptcg-storage.json',p['storage_profile'])
        machine=verify_actual_machine(p['server_resource_profile']);capacity=admit(root,20*2**30)
        job=root/'jobs'/rid;job.mkdir(mode=0o700)
        ticket={'kind':'storage_migration','mode':'cold_copy','run_id':rid,'lease_epoch':p['lease_epoch'],
            'project_id':PROJECT,'project_root':NEW,'source_root':OLD,'unit':p['unit'],
            'job_directory':str(job),'ticket_path':str(job/'ticket.json'),'control_root':str(control),
            'deadline_unix':p['deadline_unix'],'memory_max_bytes':p['server_resource_profile']['memory_max_bytes'],
            'cpu_quota_cores':p['server_resource_profile']['cpu_quota_cores'],'server_resource_profile':p['server_resource_profile'],
            'workspace_hard_max_bytes':50*2**30,'storage_profile':p['storage_profile'],'stop_grace_seconds':45,
            'io_bytes_per_second':p['io_bytes_per_second']}
        write(job/'ticket.json',ticket);session={'ticket':ticket,'job':job}
        return {'ticket':ticket,'machine':machine,'capacity':capacity,'parent_dev':parent.stat().st_dev,
            'target_dev':root.stat().st_dev,'source_dev':source.stat().st_dev,'target_uid':root.stat().st_uid,
            'target_realpath':str(root.resolve()),'target_mode':oct(root.stat().st_mode&0o777)}
    if session is None:raise RuntimeError('Migration session not initialized')
    from ptcg_lab.systemd_job import start_verified,inspect_service,identity,stop_verified,terminal
    from ptcg_lab.io import atomic_json,sha256
    job=session['job'];ticket=session['ticket'];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if p['operation']=='start':
        if 'identity' in session:raise RuntimeError('Same job cannot be launched twice')
        if ticket['kind']=='storage_acceptance' and ticket.get('gpu_restore_required',True):
            rows=subprocess.run(['nvidia-smi','-i',ticket['gpu_uuid'],'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip().split(',')
            apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout
            if int(rows[0])>64 or int(rows[1])>1 or ticket['gpu_uuid'] in apps:raise RuntimeError('Authorized GPU no longer idle; no launch')
        handle,properties=start_verified(ticket);session['identity']=handle;atomic_json(job/'identity.json',handle)
        return {'identity':handle,'service_properties':properties,'host_boot_id':boot,'server_unix':time.time()}
    handle=session['identity'];properties=inspect_service(ticket['unit'])
    def collected_proof():
        if boot!=handle['host_boot_id']:raise RuntimeError('Host boot changed')
        if 'collected_proof' not in session:
            from ptcg_lab.terminal_journal import read_journal,journal_terminal
            events=read_journal(handle);proof=journal_terminal(events,handle,uid,time.time())
            session['collected_proof']={'terminal_proof_kind':'exact_invocation_systemd_journal',
                'terminal_journal':events,'journal_terminal_proof':proof}
        return session['collected_proof']
    if properties.get('LoadState')=='not-found':
        if p['operation'] not in ('status','evidence'):raise RuntimeError('Collected invocation cannot be signaled')
        collected_proof()
    elif properties.get('LoadState')!='loaded' or identity(properties,boot)!=handle:raise RuntimeError('Migration invocation differs')
    if p['operation']=='stop':
        properties=stop_verified(handle);session['same_invocation_stop_issued']=True
        if properties.get('LoadState')=='not-found':collected_proof()
    if p['operation'] in ('status','stop'):
        result={'identity':handle,'service_properties':properties,'host_boot_id':boot,'server_unix':time.time(),
            'terminal':terminal(properties),'same_invocation_stop_issued':session.get('same_invocation_stop_issued',False),
            **session.get('collected_proof',{})}
        if ticket['kind']=='storage_acceptance' and not result['terminal']:
            cg=properties.get('ControlGroup','')
            if not cg or '..' in cg.split('/'):raise ValueError('Missing owned acceptance cgroup')
            group=Path('/sys/fs/cgroup')/cg.lstrip('/');members={int(x) for x in (group/'cgroup.procs').read_text().split()}
            result['own_resources']={name:(group/name).read_text().strip() for name in ['cpu.stat','memory.current','memory.peak','io.stat'] if (group/name).exists()}
            apps=subprocess.run(['nvidia-smi','-i',ticket['gpu_uuid'],'--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.splitlines() if ticket.get('gpu_restore_required',True) else []
            parsed=[[v.strip() for v in row.split(',')] for row in apps]
            result['own_GPU_processes']=[r for r in parsed if len(r)==3 and r[0]==ticket['gpu_uuid'] and int(r[1]) in members]
            result['foreign_GPU_processes']=[r for r in parsed if len(r)==3 and r[0]==ticket['gpu_uuid'] and int(r[1]) not in members]
            result['GPU_background']=subprocess.run(['nvidia-smi','-i',ticket['gpu_uuid'],'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip() if ticket.get('gpu_restore_required',True) else 'CUDA hidden for CPU-only acceptance'
        for name in ['copy-progress.json','copy-failure.json','actual-resource-limits.json','acceptance-metrics.json','full-restore.json','fresh-process-restore.json','simulator-check.json']:
            if (job/name).exists():result[name]=json.loads((job/name).read_text())
        if result['terminal']:
            cg=properties.get('ControlGroup','')
            if cg:
                path=Path('/sys/fs/cgroup')/cg.lstrip('/')/'cgroup.procs'
                if path.exists() and path.read_text().strip():raise RuntimeError('Terminal copy cgroup still populated')
        return result
    if p['operation']=='evidence':
        if not terminal(properties):raise RuntimeError('Evidence read requires terminal copy')
        if p['name'] not in ['full-copy-manifest.json.gz','actual-resource-limits.json','copy-progress.json','stdout.log','stderr.log','copy-failure.json',
            'acceptance-metrics.json','full-restore.json','fresh-process-restore.json','simulator-check.json','hard-protection.json',
            'environment-entrypoint-originals.json','environment-adaptation.json','engine-manifest.json','engine.log','gpu-start-probe.json','copy-terminal-handoff-proof.json',
            'CPU-throughput-benchmark.json','cpu-throughput-2.log','cpu-throughput-4.log','engineering-selfplay-threads2.pt','engineering-selfplay-threads4.pt','environment-adaptation-chain.json','restored-training-state.pt']:
            raise ValueError('Unexpected evidence file')
        path=job/p['name'];offset=p.get('offset',0);size=p.get('size',512*1024)
        if type(offset) is not int or offset<0 or type(size) is not int or not 0<size<=512*1024:raise ValueError('Invalid evidence chunk')
        if not path.exists():return {'absent':True}
        st=path.stat();file_identity=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_uid,st.st_mode)
        cached=session.setdefault('immutable_evidence_hashes',{}).get(path.name)
        if cached is not None and cached[0]!=file_identity:raise RuntimeError('Terminal evidence changed during transfer')
        if cached is None:
            cached=(file_identity,sha256(path));session['immutable_evidence_hashes'][path.name]=cached
        with path.open('rb') as f:f.seek(offset);data=f.read(size)
        return {'name':path.name,'bytes':path.stat().st_size,'offset':offset,'base64':base64.b64encode(data).decode(),
            'file_sha256':cached[1],'chunk_sha256':hashlib.sha256(data).hexdigest()}
    raise ValueError('Unknown migration operation')


for line in sys.stdin.buffer:
    p=json.loads(line)
    try:result={'sequence':p['sequence'],'ok':True,'result':operate(p)}
    except BaseException as exc:result={'sequence':p['sequence'],'ok':False,'error':type(exc).__name__+': '+str(exc)}
    print(json.dumps(result,separators=(',',':')),flush=True)
