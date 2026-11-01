"""One finite45-second observation, no recurring monitor or automatic launch."""
import json,time
from common import *
def main():
    import psutil
    began=time.time();stable=None;rows=[]
    for _ in range(46):
        available=psutil.virtual_memory().available;now=time.time();rows.append(dict(at=now,available_RAM_bytes=available))
        if available>=int(8.5*2**30):stable=stable or now
        else:stable=None
        if stable and now-stable>=30:break
        if now-began>=45:break
        time.sleep(1)
    report=dict(at=time.time(),ready=stable is not None and time.time()-stable>=30,minimum_available_RAM_bytes=min(r['available_RAM_bytes'] for r in rows),samples=rows,no_job_started=True,no_other_process_touched=True,
                waiting_cost_exemption_claimed=False,reason='Unavailable process command/cwd prevents attribution; preserve conservative accounting')
    write(V3/'reports/memory-observations'/f'{int(began*1000)}.json',report)
    write(V3/'reports/MEMORY_GATE.json',report);print({k:v for k,v in report.items() if k not in ('samples',)})
if __name__=='__main__':main()
