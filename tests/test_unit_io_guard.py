import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
from ptcg_lab.unit_io_guard import IOBudget,IOGuard,UnitCounters,parse_io


class UnitIOGuardTests(unittest.TestCase):
    def test_rate_debt_freezes_then_thaws_after_refill(self):
        budget=IOBudget(100,100,burst_bytes=100,stop_debt_bytes=200,max_gap_seconds=10)
        self.assertEqual(budget.observe(0,0,0)['action'],'run')
        self.assertEqual(budget.observe(.1,0,150)['action'],'freeze')
        self.assertEqual(budget.observe(.2,0,150)['action'],'hold')
        self.assertEqual(budget.observe(1.2,0,150)['action'],'thaw')
    def test_hard_burst_or_missing_monitor_interval_requires_stop(self):
        budget=IOBudget(100,100,burst_bytes=100,stop_debt_bytes=200)
        budget.observe(0,0,0)
        self.assertEqual(budget.observe(.1,0,400)['action'],'stop')
        other=IOBudget(100,100);other.observe(0,0,0)
        self.assertEqual(other.observe(1,0,0)['action'],'stop')
    def test_both_cached_character_and_physical_bytes_are_conservative(self):
        self.assertEqual(parse_io('rchar: 100\nwchar: 20\nread_bytes: 80\nwrite_bytes: 60\n'),(100,60))
        with self.assertRaises(ValueError):parse_io('rchar: 1\n')
    def test_unreadable_process_accounting_stops_only_bound_handle(self):
        guard=object.__new__(IOGuard);guard.handle={'unit':'owned','invocation_id':'exact'};guard.reason=None
        guard.counters=type('BadCounters',(),{'sample':lambda self:(_ for _ in ()).throw(PermissionError('missing'))})()
        reports=[];guard.report=reports.append
        with patch('ptcg_lab.systemd_job.stop_verified',return_value={'ActiveState':'failed'}) as stop:
            self.assertFalse(guard.step());stop.assert_called_once_with(guard.handle)
        self.assertTrue(reports[0]['decision']['reason'].startswith('guard_unverified'))
    def test_departed_child_accounting_is_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cg=root/'cg/owned/job';cg.mkdir(parents=True);proc=root/'proc';proc.mkdir()
            (cg/'cgroup.events').write_text('frozen 0\n');(cg/'cgroup.procs').write_text('100\n101\n')
            for pid,count in [(100,1000),(101,4000)]:
                p=proc/str(pid);p.mkdir();(p/'stat').write_text(str(pid)+' (owned) '+' '.join(['S']+['0']*18+[str(pid)]))
                (p/'cgroup').write_text('0::/owned/job\n');(p/'io').write_text(f'rchar: 0\nwchar: {count}\nread_bytes: 0\nwrite_bytes: {count}\n')
            counters=UnitCounters('/owned/job',(proc/'100').stat().st_uid,proc_root=proc,cgroup_root=root/'cg')
            self.assertEqual(counters.sample()['write_bytes'],5000)
            (cg/'cgroup.procs').write_text('100\n');(proc/'100/io').write_text('rchar: 0\nwchar: 6000\nread_bytes: 0\nwrite_bytes: 6000\n')
            self.assertGreaterEqual(counters.sample()['write_bytes'],6000)
            (cg/'cgroup.procs').write_text('')
            with self.assertRaises(RuntimeError):counters.sample()
