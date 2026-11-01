"""Finite test-child supervision; the Popen handle is the exit authority."""
import json,os,subprocess,time
from pathlib import Path
import psutil
from ptcg_lab.processes import KillOnCloseJob

def sample_family(process,created,bound,job):
    """A root/descendant may exit at every psutil call; PID reuse is an error."""
    if process.create_time()!=created:raise RuntimeError('Owned pilot PID identity changed')
    family=[process,*process.children(recursive=True)];io_total=[0,0];rss=0
    for p in family:
        try:
            observed=p.create_time()
            if p.pid not in bound:
                if not p.is_running():continue
                job.assign(p.pid);bound[p.pid]=observed
            if observed!=bound[p.pid]:raise RuntimeError('Owned descendant PID reused')
            rss+=p.memory_info().rss;io=p.io_counters();io_total[0]+=io.read_bytes;io_total[1]+=io.write_bytes
        except psutil.NoSuchProcess:continue
    return rss,io_total

def _write(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.partial');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,p)

def bind_job(child,job):
    if job.handle:
        if not job.kernel.AssignProcessToJobObject(job.handle,int(child._handle)):
            import ctypes
            error=ctypes.get_last_error()
            if child.poll() is None:raise ctypes.WinError(error)
            return False  # actual handle confirms exit before job assignment
    else:job.assign(child.pid)
    return True

def run(command,*,cwd,console,receipt,progress,stop_file,context=None,maximum_seconds=3600):
    started=time.monotonic();job=KillOnCloseJob();samples=[];peak=0;last_signature=None;last_progress=time.monotonic();stalled=False
    child=None;created=None;monitor_error=None;code=None;exits_during_sample=0;stop_reason=None
    report=dict(schema=2,status='launching',command=list(command),started_at=time.time(),administrative_total_time_cap=None,context=context)
    _write(receipt,report)
    try:
        with Path(console).open('wb') as output:
            child=subprocess.Popen(command,cwd=cwd,stdout=output,stderr=subprocess.STDOUT,start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
            bind_job(child,job)
            try:process=psutil.Process(child.pid);created=process.create_time()
            except psutil.NoSuchProcess:process=None
            report.update(status='running',PID=child.pid,created=created);_write(receipt,report)
            bound={child.pid:created}
            while child.poll() is None:
                if time.monotonic()-started>=maximum_seconds or Path(stop_file).exists():
                    stop_reason='deadline' if time.monotonic()-started>=maximum_seconds else 'explicit_stop'
                    stalled=True;break
                try:
                    if process is None:raise psutil.NoSuchProcess(child.pid)
                    rss,io_total=sample_family(process,created,bound,job)
                except psutil.NoSuchProcess:
                    # Exit can race poll/create_time/children. Only wait knows code.
                    exits_during_sample+=1
                    if child.poll() is not None:break
                    time.sleep(.01);continue
                peak=max(peak,rss)
                if rss>8*2**30 or psutil.virtual_memory().available<8*2**30:
                    stop_reason='owned_supervisor_resource_stop';Path(stop_file).write_text(json.dumps({'reason':stop_reason,'owned_tree_RSS':rss}),encoding='utf-8');stalled=True;break
                stamp=Path(progress).stat().st_mtime_ns if Path(progress).exists() else 0
                signature=(*io_total,stamp)
                if signature!=last_signature:last_signature=signature;last_progress=time.monotonic()
                if time.monotonic()-last_progress>600:
                    stop_reason='no_progress_600_seconds'
                    Path(stop_file).write_text(json.dumps({'reason':'own engineering worker made no I/O or phase/minibatch progress for600s','source':'finite pilot supervisor'}),encoding='utf-8');stalled=True;break
                if len(samples)<4096:samples.append(dict(elapsed=time.monotonic()-started,RSS=rss,read=io_total[0],write=io_total[1],system_available_RAM_bytes=psutil.virtual_memory().available))
                time.sleep(1)
            if stalled:
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name=='nt':job.close()
                    else:
                        import signal
                        if process.create_time()!=created or os.getpgid(child.pid)!=child.pid:raise RuntimeError('Owned child identity/group changed')
                        os.killpg(child.pid,signal.SIGTERM)
                    child.wait(timeout=30)
            code=child.wait()
    except BaseException as error:
        monitor_error=repr(error);job.close()
        if child is not None:
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:code=None
        raise
    finally:
        job.close()
        report.update(status='monitor_error' if monitor_error else 'terminal',returncode=code,exitcode_source='Popen.wait' if code is not None else None,
            seconds=time.monotonic()-started,PID=child.pid if child else None,created=created,monitor_error=monitor_error,
            owned_tree_peak_RSS_observed_bytes=peak,samples=samples,no_progress_stop=stop_reason=='no_progress_600_seconds',stop_reason=stop_reason,exits_during_sample=exits_during_sample,finished_at=time.time())
        _write(receipt,report)
    return report
