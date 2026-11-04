"""Hash-only migration inventory; no archive duplication, network, or training."""
import argparse
import os
from pathlib import Path, PurePosixPath
import time

from .io import atomic_json, read_json, sha256, utcnow


TREES=('agents','configs','ptcg_lab','scripts','tests','runs','data/competition',
       'research/autonomy','research/environment','research/opponents','research/proposals','research/public_sources')
SECRET_NAMES={'.env','kaggle.json','credentials','credentials.json','client_secret.json',
              'service_account.json','id_rsa','id_ed25519','id_ecdsa','id_dsa'}
FORBIDDEN_DIRS={'.ssh','.kaggle','.aws','.codex','.git'}


def relative_name(name):
    name=str(name).replace(chr(92),'/');path=PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or str(path)!=name or ':' in name:
        raise ValueError('Unsafe migration-relative path')
    return name


def check_asset(root,path):
    root=Path(root).resolve();path=Path(path)
    name=relative_name(path.relative_to(root).as_posix())
    parts=PurePosixPath(name).parts
    if any(p.lower() in FORBIDDEN_DIRS for p in parts):raise ValueError('Private configuration in migration input')
    if (path.name.lower() in SECRET_NAMES or path.name.lower().startswith('.env.')
            or path.suffix.lower() in {'.pem','.key','.ppk','.p12','.pfx'}):
        raise ValueError('Credential-like file in migration input')
    current=root
    for part in parts:
        current=current/part
        if current.is_symlink():raise ValueError('Linked migration input')
    if not path.is_file() or not path.resolve().is_relative_to(root):raise ValueError('Migration input outside workspace')
    return name


def inventory_paths(root,output,ledger_snapshot):
    root=Path(root).resolve();output=Path(output).resolve();selected={}
    for tree in TREES:
        folder=root/tree
        if not folder.exists():continue
        for path in folder.rglob('*'):
            if path.is_symlink():raise ValueError('Linked migration input')
            if path.is_dir():continue
            if output==path or output in path.parents:continue
            if '__pycache__' in path.parts or path.suffix in {'.pyc','.tmp'}:continue
            if path in [root/'research/autonomy/state.json',root/'research/autonomy/job.lock']:continue
            name=check_asset(root,path);selected[name]=path
    for name in ['AGENTS.md','research/AUTONOMOUS_RESEARCH.md','research/START_AUTONOMOUS.md']:
        path=root/name
        if path.exists():selected[check_asset(root,path)]=path
    for path in (root/'research').iterdir():
        if path.is_file() and path.suffix in {'.md','.py','.json'}:selected[check_asset(root,path)]=path
    check_asset(root,ledger_snapshot)
    selected['research/autonomy/state.json']=Path(ledger_snapshot)
    return dict(sorted(selected.items()))


def verify_training_journal(root,run,entries):
    run=Path(run);prefix=run.relative_to(root).as_posix();source=read_json(run/'source-manifest.json')
    for name,digest in source.items():
        key=prefix+'/source/'+relative_name(name)
        if entries.get(key,{}).get('sha256')!=digest:raise ValueError('Frozen training source changed')
    league=read_json(run/'league.json')
    for snapshot in league['snapshots']:
        if entries.get(prefix+'/'+relative_name(snapshot['path']),{}).get('sha256')!=snapshot['sha256']:
            raise ValueError('Historical opponent changed')
    ids=set();count=0
    for path in sorted((run/'series').glob('[0-9][0-9][0-9][0-9][0-9][0-9].json')):
        row=read_json(path);sid=row['series_id']
        if sid in ids or sid!=int(path.stem) or row['statuses']!=['DONE','DONE']:
            raise ValueError('Invalid committed BO3 identity/status')
        ids.add(sid);relative=relative_name(row['trajectory_path'])
        if not relative.startswith('rollouts/'):raise ValueError('Trajectory outside journal')
        if entries.get(prefix+'/'+relative,{}).get('sha256')!=row['trajectory_sha256']:
            raise ValueError('Committed trajectory changed')
        count+=1
    import torch
    checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    required={'model','optimizer','rng','version','trained_decisions','next_series_id','updates'}
    if not required<=checkpoint.keys():raise ValueError('Incomplete checkpoint payload')
    if not {'python','numpy','torch','cuda'}<=checkpoint['rng'].keys():raise ValueError('Missing RNG domains')
    if not checkpoint['optimizer']['state']:raise ValueError('Missing trained optimizer state')
    progress=read_json(run/'progress.json')
    for key in ['version','trained_decisions','next_series_id']:
        if checkpoint[key]!=progress[key]:raise ValueError('Checkpoint/progress mismatch')
    return {'committed_bo3':count,'frozen_source_files':len(source),'league_snapshots':len(league['snapshots']),
            'checkpoint_keys_verified':sorted(required),'model_tensors':len(checkpoint['model']),
            'optimizer_parameter_states':len(checkpoint['optimizer']['state']),
            'checkpoint_progress':{k:checkpoint[k] for k in ['version','trained_decisions','next_series_id']},
            'all_payload_tensors_loaded_on_cpu':all(t.device.type=='cpu' for t in checkpoint['model'].values())}


def verify_inventory(root,manifest):
    root=Path(root).resolve()
    for name,entry in manifest['files'].items():
        name=relative_name(name);path=root/name
        if check_asset(root,path)!=name or path.stat().st_size!=entry['bytes'] or sha256(path)!=entry['sha256']:
            raise ValueError('Transferred inventory mismatch: '+name)
    return {'verified_files':len(manifest['files']),'verified_bytes':manifest['total_bytes']}


def run(config,output):
    output=Path(output).resolve();root=Path(config['project_path']).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Handoff requires controlled runner')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX']);started=time.monotonic()
    ledger=root/relative_name(config['ledger_snapshot_path'])
    if sha256(ledger)!=config['ledger_snapshot_sha256']:raise ValueError('Sealed ledger changed')
    state=read_json(ledger)
    if state.get('active_job'):raise ValueError('Cannot freeze an active heavy job')
    files=inventory_paths(root,output,ledger);entries={};last=0;total=0
    for name,path in files.items():
        if time.time()>deadline-30:raise TimeoutError('Handoff inventory hard deadline approaching')
        before=path.stat();digest=sha256(path);after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('Migration input changed during hashing')
        entries[name]={'sha256':digest,'bytes':after.st_size,'local_source':path.relative_to(root).as_posix()}
        total+=after.st_size
        if time.monotonic()-last>=15:
            atomic_json(output/'progress.json',{'at':utcnow(),'stage':'hash_inventory','files_done':len(entries),
                        'files_total':len(files),'bytes_done':total});last=time.monotonic()
    run_path=root/'runs'/config['training_run_id'];journal=verify_training_journal(root,run_path,entries)
    manifest={'schema_version':1,'at':utcnow(),'status':'local_inventory_verified_remote_transfer_not_started',
              'training_run_id':config['training_run_id'],'files':entries,'total_bytes':total,
              'ledger_snapshot_sha256':config['ledger_snapshot_sha256'],'journal':journal,
              'remote_destination':'user-specified server; dedicated project directory still to be verified',
              'environment_directories_excluded':['.venv','.venv-gpu'],
              'training_portability_not_yet_verified':True,'remote_runtime_activated':False,
              'ledger_requires_final_post_job_reconciliation':True}
    atomic_json(output/'transfer-manifest.json',manifest)
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'files':len(entries),
                'total_bytes':total,'journal':journal,'seconds':time.monotonic()-started,
                'manifest_sha256':sha256(output/'transfer-manifest.json'),'remote_transfer_completed':False,
                'linux_restore_verified':False,'promotion_eligible':False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(read_json(args.config),args.output)


if __name__=='__main__':main()
