"""Finish missing immutable bytes/ACKs from one exact stopped TR13 service."""
from pathlib import Path
import time
from .io import read_json,atomic_json,sha256,utcnow
from .remote_lease import validate_no_unreconciled_remote
from .ssh_stream import open_readonly_stream
from .archive_outbox import drain_available


def complete(root,config,output):
    root=Path(root);output=Path(output);source=root/'runs'/config['source_run_id']
    lease=read_json(source/'remote-job.json');validate_no_unreconciled_remote(root,{'remote_job':lease})
    connection=read_json(source/'config.json');deadline=float(__import__('os').environ['PTCG_JOB_DEADLINE_UNIX'])-20
    package=root/'ptcg_lab';stream=open_readonly_stream(connection,package,output,deadline)
    existing={}
    for catalog in config.get('prior_archive_catalogs',[]):
        if sha256(root/catalog['path'])!=catalog['sha256']:raise ValueError('Accepted archive catalog changed')
        for row in read_json(root/catalog['path'])['objects']:
            existing[row['sha256']]={'archive_root':root/row['archive_root'],'ack':row['ack']}
    archive=source/'online-archive'
    for path in (archive/source.name/'acks').glob('*.json'):
        ack=read_json(path);existing[ack['sha256']]={'archive_root':archive,'ack':ack}
    hashes=read_json(source/'remote-source.json')
    try:
        result=stream.request('attach_terminal',{'lease':lease,'manifest_sha256':sha256(source/'remote-training-manifest.json'),
            'source_checker_hashes':{n:hashes[n] for n in ['ptcg_lab/systemd_job.py','ptcg_lab/archive_outbox.py']},'deadline_unix':deadline})
        if not result.get('attached_terminal_only'):raise ValueError('Archive endpoint did not attach read/ACK-only')
        began=time.monotonic();count=0;last=0
        def progress(row,offset):
            nonlocal last
            if time.monotonic()-last>=5:
                atomic_json(output/'progress.json',{'at':utcnow(),'stage':'completing_only_missing_terminal_archive',
                    'source_run_id':source.name,'object':row['object'],'received_bytes':offset,'expected_bytes':row['bytes']});last=time.monotonic()
        while True:
            if time.time()>deadline-10:raise TimeoutError('Original finite missing-archive allowance exhausted')
            outcome=drain_available(stream,archive,source.name,lease['lease_epoch'],existing,progress,storage_codec='gzip')
            count+=outcome['transferred_objects']
            if not outcome['pending_objects']:break
        atomic_json(output/'terminal-archive-completion.json',{'at':utcnow(),'complete':True,'source_run_id':source.name,
            'completed_missing_objects':count,'actual_seconds':time.monotonic()-began,'original_server_files_retained':True,'service_never_started_or_signaled':True})
        stream.request('close')
    finally:stream.close()
