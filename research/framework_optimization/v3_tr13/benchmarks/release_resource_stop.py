"""Release only our own closed resource stop after a fresh stable RAM gate."""
from common import *
def main():
    import psutil
    gate=read(V3/'reports/MEMORY_GATE.json');owner=read(V3/'reports/owner.json');path=V3/'STOP';raw=path.read_bytes();value=json.loads(raw)
    if not gate['ready'] or time.time()-gate['at']>90 or psutil.virtual_memory().available<int(8.5*2**30):raise MemoryError('RAM gate not freshly satisfied')
    if value.get('reason')!='owned_supervisor_resource_stop' or owner['phase']!='terminal':raise ValueError('Never release a user stop or a live job stop')
    if read(ROOT/'research/autonomy/state.json').get('active_job'):raise ValueError('Sole scientific controller has priority')
    evidence=V3/'repairs'/('resource-stop-'+str(int(time.time()*1000)));evidence.mkdir()
    (evidence/'STOP.json').write_bytes(raw);write(evidence/'release.json',dict(at=time.time(),reason='Actual own stopped job terminal and RAM>=8.5GiB for30s; no original fees reset',memory_gate=gate,scientific_credit=0))
    if path.read_bytes()!=raw:raise ValueError('Stop changed before release')
    path.unlink();print('Only our closed RAM-resource stop released; all evidence retained')
if __name__=='__main__':main()
