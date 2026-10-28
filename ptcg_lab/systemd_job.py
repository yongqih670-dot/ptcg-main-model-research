"""Finite, identity-checked user services for the hybrid server worker only.

No timers, recurring jobs, account changes, or arbitrary shell commands.
Actual service/cgroup behavior must still be validated on the target Linux host.
"""
import os
from pathlib import Path,PurePosixPath
import re
import subprocess
import time


KINDS={'environment_setup':'environment_worker.py','training_check':'training_check.py','train_ppo':'training_worker.py',
       'supervisor_audit':'boundary_worker.py','resource_audit':'resource_worker.py',
       'storage_migration':'migration_worker.py','storage_acceptance':'acceptance_worker.py'}
PROPERTIES=('LoadState','ActiveState','SubState','InvocationID','ExecMainPID','ExecMainCode','ExecMainStatus',
            'ExecMainStartTimestampMonotonic','ExecMainExitTimestampMonotonic','MemoryPeak','MemoryMax',
            'RuntimeMaxUSec','TimeoutStopUSec','ControlGroup','Result')


def _unit(value):
    if not re.fullmatch(r'ptcg-hybrid-[a-z0-9-]{8,80}\.service',value):raise ValueError('Unowned systemd unit name')
    return value


def _inside(root,value):
    root=PurePosixPath(root);path=PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or root not in path.parents:raise ValueError('Server worker path outside project')
    return str(path)


def service_command(ticket,now=None):
    now=time.time() if now is None else now;unit=_unit(ticket['unit']);kind=ticket['kind']
    if kind not in KINDS:raise ValueError('Unsupported server execution role')
    root=PurePosixPath(ticket['project_root'])
    if not root.is_absolute() or '..' in root.parts:raise ValueError('Invalid project root')
    remaining=ticket['deadline_unix']-now
    clock=ticket.get('active_clock_scope');clock_check=False
    if clock and not ticket.get('TR12_runtime_scope'):
        from .active_clock import AUTH,validate_scope
        validate_scope(clock)
        clock_check=(kind=='training_check' and ticket.get('mode')=='active_clock_acceptance'
            and ticket.get('active_clock_acceptance_authorization')==AUTH and clock['execution_limit_seconds']<=30
            and clock['hard_window_seconds']<=120 and ticket['cpu_quota_cores']<=1
            and ticket['memory_max_bytes']<=256*2**20)
        if not clock_check:raise ValueError('Unscoped active clock service')
    maximum=7200
    if ticket.get('TR12_runtime_scope'):
        if kind!='train_ppo':raise ValueError('TR12 extended window is only scientific training')
        from .volume_scope import validate_remote
        maximum=validate_remote(ticket['TR12_runtime_scope'],ticket.get('mode'),ticket.get('new_diagnostic_identity'))
        if ticket.get('cpu_quota_cores')!=3:raise ValueError('TR12 requires the authorized3CPU quota')
        if ticket.get('active_clock_scope'):
            from .active_clock import validate_scope
            validate_scope(ticket['active_clock_scope']);maximum=ticket['active_clock_scope']['hard_window_seconds']
    if ticket.get('TR11_runtime_scope'):
        if ticket.get('TR12_runtime_scope'):raise ValueError('Runtime scopes cannot be combined')
        if kind!='train_ppo':raise ValueError('TR11 extended window is only a scientific training service')
        from .continuous_runtime import validate_remote_scope
        maximum=validate_remote_scope(ticket['TR11_runtime_scope'],ticket.get('mode'),ticket.get('new_diagnostic_identity'))
        if ticket.get('cpu_quota_cores')!=3:raise ValueError('TR11 extended ticket needs exact3CPU quota')
    if not 60<=remaining<=maximum+.5:raise ValueError('Server wall limit outside project policy')
    memory=ticket['memory_max_bytes'];workspace=ticket['workspace_hard_max_bytes']
    grace=ticket.get('stop_grace_seconds',45)
    cpu=ticket.get('cpu_quota_cores',2)
    if type(cpu) not in (int,float) or not 0<cpu<=4:raise ValueError('Shared-host CPU quota exceeds conservative profile')
    if type(grace) is not int or not 1<=grace<=45:raise ValueError('Invalid bounded stop grace')
    from .resource_profile import ticket_limits
    explicit=ticket_limits(ticket)
    memory_ceiling=explicit[1] if explicit is not None else 22*2**30
    if type(memory) is not int or not 0<memory<=memory_ceiling:raise ValueError('Server memory cap exceeds project policy')
    from .storage_profile import ticket_ceiling
    if type(workspace) is not int or not 0<workspace<=ticket_ceiling(ticket):raise ValueError('Workspace cap exceeds hybrid policy')
    control=PurePosixPath(ticket.get('control_root',str(root/'control')))
    _inside(root,control)
    worker=_inside(root,control/KINDS[kind]);jobdir=_inside(root,ticket['job_directory'])
    ticket_path=_inside(root,ticket['ticket_path'])
    if ticket.get('python_kind','system') not in ('system','venv'):raise ValueError('Invalid worker interpreter')
    if kind=='train_ppo' and ticket.get('python_kind')!='venv':raise ValueError('PPO requires project Python environment')
    python=str(root/'.venv/bin/python') if ticket.get('python_kind')=='venv' else '/usr/bin/python3'
    environment={'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                 'PTCG_JOB_RUN_ID':ticket['run_id'],'PTCG_JOB_DEADLINE_UNIX':str(ticket['deadline_unix']),
                 'TMPDIR':str(root/'tmp'),'PIP_NO_INDEX':'1','PIP_NO_CACHE_DIR':'1',
                 'PIP_CONFIG_FILE':'/dev/null','PIP_DISABLE_PIP_VERSION_CHECK':'1','PIP_NO_INPUT':'1',
                 'PYTHONNOUSERSITE':'1','XDG_CACHE_HOME':str(root/'tmp/cache'),'TORCH_HOME':str(root/'tmp/torch'),
                 'MPLCONFIGDIR':str(root/'tmp/matplotlib'),'XDG_CONFIG_HOME':str(root/'tmp/config'),
                 'PIP_CACHE_DIR':str(root/'tmp/pip'),'TRITON_CACHE_DIR':str(root/'tmp/triton'),
                 'CUDA_CACHE_PATH':str(root/'tmp/cuda')}
    if ticket.get('project_id'):
        environment.update(PTCG_PROJECT_ROOT=str(root),PTCG_PROJECT_ID=ticket['project_id'])
    if ticket.get('storage_profile'):
        environment.update(HOME=str(root/'tmp/home'),TORCH_EXTENSIONS_DIR=str(root/'tmp/torch-extensions'))
    if 'lease_epoch' in ticket:
        if type(ticket['lease_epoch']) is not int or ticket['lease_epoch']<1:raise ValueError('Invalid fenced job epoch')
        environment['PTCG_JOB_LEASE_EPOCH']=str(ticket['lease_epoch'])
    if kind=='train_ppo' or kind=='storage_acceptance' and ticket.get('gpu_restore_required',True):
        gpu=ticket.get('gpu_uuid','')
        if not re.fullmatch(r'GPU-[0-9a-f-]{36}',gpu):raise ValueError('Unverified GPU UUID')
        environment['CUDA_VISIBLE_DEVICES']=gpu
        if ticket.get('mode') in ('gpu_existing_restore','migration_full_restore'):environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    else:environment['CUDA_VISIBLE_DEVICES']=''
    if clock:
        runtime=min(remaining,clock['execution_limit_seconds']-(now-clock['job_started_unix'])+clock.get('startup_wait_seconds',0))
        if runtime<=(2 if clock_check else 100):raise ValueError('Insufficient active allowance to start service safely')
    else:runtime=remaining
    command=['systemd-run','--user','--quiet','--unit='+unit,'--service-type='+('notify' if clock else 'exec'),'--expand-environment=no',
             '--remain-after-exit','--working-directory='+str(root),
             '--property=RuntimeMaxSec='+str(int(runtime-(1 if clock_check else 50))),
             '--property=TimeoutStopSec='+str(grace)+'s','--property=KillMode=control-group','--property=SendSIGKILL=yes',
             '--property=MemoryMax='+str(memory),'--property=MemorySwapMax=0','--property=TasksMax=512',
             '--property=CPUQuota='+str(cpu*100)+'%','--property=CPUWeight=20','--property=Nice=10','--property=UMask=0077',
             '--property=StandardOutput=append:'+jobdir+'/stdout.log','--property=StandardError=append:'+jobdir+'/stderr.log']
    for key,value in environment.items():
        if '\n' in value or '\r' in value:raise ValueError('Invalid service environment value')
        command+=['--setenv='+key+'='+value]
    if clock:command+=['--property=NotifyAccess=main','--property=TimeoutStartSec=30s']
    return command+[python,'-B',worker,'--ticket',ticket_path]


def user_manager_environment():
    if os.name!='posix':raise RuntimeError('User service operations require target Linux')
    runtime=Path('/run/user')/str(os.getuid());bus=runtime/'bus'
    if not bus.exists():raise RuntimeError('Existing user service manager unavailable')
    return dict(os.environ,XDG_RUNTIME_DIR=str(runtime),DBUS_SESSION_BUS_ADDRESS='unix:path='+str(bus))


def inspect_service(unit):
    _unit(unit)
    command=['systemctl','--user','show',unit,'--no-pager',*[f'--property={p}' for p in PROPERTIES]]
    result=subprocess.run(command,text=True,capture_output=True,timeout=15,env=user_manager_environment())
    properties=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode and properties.get('LoadState')!='not-found':raise RuntimeError('Cannot inspect owned service')
    properties['unit']=unit;return properties


def identity(properties,boot_id):
    if properties.get('LoadState')!='loaded' or not re.fullmatch('[0-9a-f]{32}',properties.get('InvocationID','')):
        raise ValueError('Service invocation identity not established')
    return {'unit':properties['unit'],'invocation_id':properties['InvocationID'],
            'start_monotonic':properties['ExecMainStartTimestampMonotonic'],'host_boot_id':boot_id}


def terminal(properties):
    return (properties.get('LoadState')=='not-found' or properties.get('ActiveState') in ('failed','inactive')
            or (properties.get('ActiveState')=='active' and properties.get('SubState')=='exited'))


def stop_verified(handle):
    properties=inspect_service(handle['unit']);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if identity(properties,boot)!=handle:raise RuntimeError('Service identity changed; refusing stop')
    subprocess.run(['systemctl','--user','stop',handle['unit']],check=True,timeout=55,env=user_manager_environment())
    after=inspect_service(handle['unit'])
    if not terminal(after):raise RuntimeError('Owned service did not terminate')
    return after


def start_verified(ticket):
    command=service_command(ticket);previous=inspect_service(ticket['unit'])
    if previous.get('LoadState')!='not-found':raise RuntimeError('Owned unit name already exists; query it instead of launching twice')
    subprocess.run(command,check=True,capture_output=True,text=True,timeout=20,env=user_manager_environment())
    properties=inspect_service(ticket['unit']);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return identity(properties,boot),properties


def freeze_verified(handle,frozen):
    properties=inspect_service(handle['unit']);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if identity(properties,boot)!=handle:raise RuntimeError('Service identity changed; refusing freeze/thaw')
    if terminal(properties):raise RuntimeError('Cannot pace a terminal service')
    operation='freeze' if frozen else 'thaw'
    subprocess.run(['systemctl','--user',operation,handle['unit']],check=True,timeout=10,env=user_manager_environment())
    after=inspect_service(handle['unit'])
    if identity(after,boot)!=handle:raise RuntimeError('Service changed during freeze/thaw')
    group=after.get('ControlGroup','')
    if not group or '..' in group.split('/'):raise RuntimeError('Missing owned control group')
    events=(Path('/sys/fs/cgroup')/group.lstrip('/')/'cgroup.events').read_text()
    actual=dict(line.split() for line in events.splitlines())
    if actual.get('frozen')!=('1' if frozen else '0'):raise RuntimeError('Kernel freezer did not reach requested state')
    return {'operation':operation,'identity':handle,'kernel_frozen':actual['frozen'],'control_group':group}
