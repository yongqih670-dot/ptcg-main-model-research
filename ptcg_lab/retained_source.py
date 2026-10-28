"""Read-only access to one hash-bound, terminal owned server capsule."""
from pathlib import Path
import os
import time
from .io import read_json,sha256
from .archive_prefix import relative_path,canonical_hash
from .server_workspace import inside
from . import systemd_job


def resolve(root,project_id,spec):
    run=relative_path(spec['run_id'])
    if '/' in run:raise ValueError('Retained source run escaped project')
    source=inside(root,root/'jobs'/run)
    if sha256(source/'training-manifest.raw.json')!=spec['manifest_sha256'] or read_json(source/'identity.json')!=spec['identity']:
        raise ValueError('Retained source manifest or invocation changed')
    ticket=read_json(source/'ticket.json')
    if (ticket['run_id'],ticket['unit'],ticket['lease_epoch'],ticket['project_id'])!=(run,spec['identity']['unit'],spec['lease_epoch'],project_id):
        raise ValueError('Retained ticket and authoritative lease differ')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot!=spec['identity']['host_boot_id']:raise ValueError('Retained source boot changed')
    properties=systemd_job.inspect_service(ticket['unit'])
    if properties.get('LoadState')=='not-found':
        if spec.get('journal_failed_exit_proof'):
            from .terminal_journal import journal_failed_exit,read_journal
            proof=journal_failed_exit(read_journal(spec['identity']),spec['identity'],os.getuid(),time.time())
            prior=spec['prior_terminal_properties']
            if (proof!=spec['journal_failed_exit_proof'] or prior.get('InvocationID')!=spec['identity']['invocation_id']
                    or prior.get('ExecMainStartTimestampMonotonic')!=spec['identity']['start_monotonic']
                    or prior.get('ActiveState')!='failed' or prior.get('Result')!='exit-code'
                    or prior.get('ControlGroup')!='' or prior.get('ExecMainStatus')!=proof['ExecMainStatus']):
                raise ValueError('Original failed-exit manager proof and failed empty-group receipt differ')
        elif spec.get('journal_terminal_proof'):
            from .terminal_journal import journal_terminal,read_journal
            if journal_terminal(read_journal(spec['identity']),spec['identity'],os.getuid(),time.time())!=spec['journal_terminal_proof']:
                raise ValueError('Retained original positive terminal journal differs')
        elif time.time()<ticket['deadline_unix']+60:raise ValueError('Retained source absence before original deadline is insufficient')
    elif not systemd_job.terminal(properties) or systemd_job.identity(properties,boot)!=spec['identity']:
        raise ValueError('Retained source invocation is not terminal')
    return source,read_json(source/'training-manifest.raw.json')


def checkpoint_reference(source,spec):
    index=read_json(source/'checkpoint-commits/index.json')
    if index['run_id']!=spec['run_id'] or index['lease_epoch']!=spec['lease_epoch']:raise ValueError('Retained recovery owner differs')
    point=index['commits'][-1]
    if canonical_hash({k:v for k,v in point.items() if k!='commit_id'})!=point['commit_id']:raise ValueError('Retained recovery commit changed')
    if point['checkpoint']!=spec['checkpoint_reference'] or point['cursor']!=spec['cursor']:raise ValueError('Original scientific checkpoint/cursor differs')
    return point
