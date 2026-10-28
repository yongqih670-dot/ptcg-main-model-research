"""Read-only whole-host block I/O upper bound, never a kernel io.max claim.

Linux 6.8 sources root-cgroup io.stat from system-wide disk statistics.
Counting every device includes other users and layered devices conservatively;
the caller may act only on its own separately verified worker session.
"""
from pathlib import Path


def parse_block_io(text):
    result={}
    for line in text.splitlines():
        parts=line.split()
        if not parts:continue
        device=parts[0]
        if len(device.split(':'))!=2 or not all(v.isdigit() for v in device.split(':')):
            raise RuntimeError('Invalid kernel block device identity')
        if device in result:raise RuntimeError('Duplicate kernel device counters')
        fields=dict(value.split('=',1) for value in parts[1:])
        if fields and not {'rbytes','wbytes'}.issubset(fields):raise RuntimeError('Incomplete kernel block counters')
        values=[int(fields.get(name,0)) for name in ['rbytes','wbytes']]
        if any(v<0 for v in values):raise RuntimeError('Negative block accounting')
        result[device]=values
    if not result:raise RuntimeError('Missing kernel block accounting')
    return result


class HostBlockCounters:
    def __init__(self,project_device,path='/sys/fs/cgroup/io.stat',boot_path='/proc/sys/kernel/random/boot_id'):
        self.path=Path(path);self.boot_path=Path(boot_path);self.boot=self.boot_path.read_text().strip()
        self.device=project_device;self.baseline=parse_block_io(self.path.read_text());self.previous=self.baseline
        if self.device not in self.baseline:raise RuntimeError('Project filesystem has no kernel block accounting')
    def sample(self):
        if self.boot_path.read_text().strip()!=self.boot:raise RuntimeError('Kernel block accounting boot identity changed')
        current=parse_block_io(self.path.read_text())
        if current.keys()!=self.baseline.keys():raise RuntimeError('Kernel block device set changed')
        if any(current[d][i]<self.previous[d][i] for d in current for i in (0,1)):
            raise RuntimeError('Kernel block counters regressed')
        self.previous=current
        totals=[sum(current[d][i]-self.baseline[d][i] for d in current) for i in (0,1)]
        project=[current[self.device][i]-self.baseline[self.device][i] for i in (0,1)]
        return {'read_bytes':totals[0],'write_bytes':totals[1],'project_device':self.device,
                'project_device_delta':project,'devices':current,'baseline':self.baseline,
                'boot_id':self.boot,'source':str(self.path),'scope':'whole_host_upper_bound'}
