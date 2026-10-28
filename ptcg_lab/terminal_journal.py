"""Positive systemd-manager evidence for one already-collected invocation."""


def journal_terminal(events,handle,expected_uid,now):
    import re
    if not isinstance(events,list) or not 1<=len(events)<=64:raise ValueError('Bounded original-unit journal required')
    unit=handle['unit'];invocation=handle['invocation_id'];boot=handle['host_boot_id'].replace('-','')
    if not re.fullmatch(r'ptcg-hybrid-[a-z0-9-]{8,80}\.service',unit) or not re.fullmatch(r'[0-9a-f]{32}',invocation):
        raise ValueError('Unowned journal unit or invocation')
    started=[];stopped=[];exited=[];failed=[];last_mono=-1
    for e in events:
        if e.get('USER_UNIT')!=unit:raise ValueError('Journal contains another unit')
        if (e.get('_COMM')!='systemd' or e.get('_EXE') not in ('/usr/lib/systemd/systemd','/lib/systemd/systemd')
                or e.get('_UID')!=str(expected_uid) or e.get('_BOOT_ID','').replace('-','')!=boot
                or e.get('USER_INVOCATION_ID')!=invocation):
            raise ValueError('Journal manager credentials/boot/invocation differ')
        mono=int(e['__MONOTONIC_TIMESTAMP']);real=int(e['__REALTIME_TIMESTAMP'])
        if mono<last_mono or not e.get('__CURSOR') or real/1e6>now+5:raise ValueError('Journal order/time/cursor differs')
        last_mono=mono;message=e.get('MESSAGE_ID')
        if message=='39f53479d3a045ac8e11786248231fbf':started.append(e)
        elif message=='9d1aaa27d60140bd96365438aad20286':stopped.append(e)
        elif message=='98e322203f7a4ed290d09fe03c09fe15':exited.append(e)
        elif message=='d9b373ed55a64feb8242e02dbe79a49c':failed.append(e)
    if len(started)!=1 or len(stopped)!=1:raise ValueError('One positive original started/stopped event pair required')
    start=started[0];stop=stopped[0];sm=int(start['__MONOTONIC_TIMESTAMP']);em=int(stop['__MONOTONIC_TIMESTAMP'])
    if abs(sm-int(handle['start_monotonic']))>1_000_000 or em<sm:raise ValueError('Original kernel start or stop identity differs')
    sr=int(start['__REALTIME_TIMESTAMP']);er=int(stop['__REALTIME_TIMESTAMP'])
    if er<sr or abs((er-sr)-(em-sm))>5_000_000:raise ValueError('Original journal clocks differ')
    if any(int(e['__MONOTONIC_TIMESTAMP'])>em for e in started+exited+failed):raise ValueError('A later runtime event follows the declared stop')
    status=exited[-1].get('EXIT_STATUS','unknown') if exited else 'unknown'
    result='exit-code' if status not in ('0','unknown') else 'failed' if failed else 'success' if status=='0' else 'unknown'
    return {'unit':unit,'invocation_id':invocation,'host_boot_id':handle['host_boot_id'],'uid':expected_uid,
        'start_monotonic':handle['start_monotonic'],'started_cursor':start['__CURSOR'],'stopped_cursor':stop['__CURSOR'],
        'started_unix':sr/1e6,'stopped_unix':er/1e6,'Result':result,'ExecMainStatus':status,
        'positive_stopped_event_from_original_systemd_user_manager':True}


def read_journal(handle):
    import json,subprocess
    from .systemd_job import user_manager_environment
    result=subprocess.run(['journalctl','--user','-u',handle['unit'],'-n','20','--no-pager','-o','json'],
        capture_output=True,text=True,check=True,timeout=12,env=user_manager_environment())
    if len(result.stdout)>256*1024:raise ValueError('Own-unit journal exceeds bounded evidence scope')
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]


def journal_failed_exit(events,handle,expected_uid,now):
    """Positive failed main-process exit; never manufacture a Stopped event.

    A failed unit can be garbage-collected without a Stopped notification.
    Archive admission additionally requires the original matching failed
    service receipt with empty ControlGroup. This proof alone cannot unlock
    training or replace the legacy started/stopped receipt contract.
    """
    import re
    if not isinstance(events,list) or not 1<=len(events)<=64:raise ValueError('Bounded failed invocation journal required')
    unit=handle['unit'];invocation=handle['invocation_id'];boot=handle['host_boot_id'].replace('-','')
    if not re.fullmatch(r'ptcg-hybrid-[a-z0-9-]{8,80}\.service',unit) or not re.fullmatch('[0-9a-f]{32}',invocation):
        raise ValueError('Unowned failed invocation')
    groups={k:[] for k in ['39f53479d3a045ac8e11786248231fbf','98e322203f7a4ed290d09fe03c09fe15','d9b373ed55a64feb8242e02dbe79a49c']};last=-1
    for e in events:
        if (e.get('USER_UNIT')!=unit or e.get('USER_INVOCATION_ID')!=invocation or e.get('_BOOT_ID','').replace('-','')!=boot
                or e.get('_UID')!=str(expected_uid) or e.get('_COMM')!='systemd' or e.get('_EXE') not in ('/usr/lib/systemd/systemd','/lib/systemd/systemd')):
            raise ValueError('Failed journal manager credentials/boot/invocation differ')
        mono=int(e['__MONOTONIC_TIMESTAMP']);real=int(e['__REALTIME_TIMESTAMP'])
        if mono<last or not e.get('__CURSOR') or real/1e6>now+5:raise ValueError('Failed invocation journal order/time differs')
        last=mono
        if e.get('MESSAGE_ID') in groups:groups[e['MESSAGE_ID']].append(e)
    if any(len(v)!=1 for v in groups.values()):raise ValueError('Exactly one Started, Exited and Failed manager event required')
    start,exit,failed=[groups[k][0] for k in groups]
    sm,em,fm=[int(e['__MONOTONIC_TIMESTAMP']) for e in [start,exit,failed]]
    sr,er,fr=[int(e['__REALTIME_TIMESTAMP']) for e in [start,exit,failed]]
    if abs(sm-int(handle['start_monotonic']))>1_000_000 or not sm<=em<=fm or not sr<=er<=fr or abs((er-sr)-(em-sm))>5_000_000:
        raise ValueError('Original failed invocation clocks/start differ')
    status=exit.get('EXIT_STATUS','')
    if not re.fullmatch('[1-9][0-9]{0,2}',status) or exit.get('EXIT_CODE','exited')!='exited':raise ValueError('Positive nonzero normal main-process exit required')
    return {'unit':unit,'invocation_id':invocation,'host_boot_id':handle['host_boot_id'],'uid':expected_uid,
        'start_monotonic':handle['start_monotonic'],'started_cursor':start['__CURSOR'],'exited_cursor':exit['__CURSOR'],
        'failed_cursor':failed['__CURSOR'],'started_unix':sr/1e6,'exited_unix':er/1e6,'failed_unix':fr/1e6,
        'Result':'exit-code','ExecMainStatus':status,'positive_failed_exit_from_original_systemd_user_manager':True}
