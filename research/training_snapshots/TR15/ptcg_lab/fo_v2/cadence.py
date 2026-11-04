"""Measure completed reliable commits; never infer cadence from config alone."""
import time


class Cadence:
    def __init__(self,*,maximum_gap=600,commit_reserve=60,clock=time.monotonic):
        if not 0<commit_reserve<maximum_gap:raise ValueError('Invalid cadence reserve')
        self.maximum=maximum_gap;self.reserve=commit_reserve;self.clock=clock;self.last=clock();self.gaps=[]
    def committed(self):
        now=self.clock();gap=now-self.last;self.gaps.append(gap);self.last=now
        return dict(reliable_commit_gap_seconds=gap,within_600_seconds=gap<=self.maximum)
    def must_checkpoint(self):return self.clock()-self.last>=self.maximum-self.reserve
    def interrupt_inflight(self,series_id,completed_pending_ids):
        if not self.must_checkpoint():return None
        if series_id in set(completed_pending_ids):raise ValueError('Complete BO3 cannot be excluded')
        return dict(excluded_unfinished_BO3=series_id,next_series_id=series_id+1,
                    reason='series_boundary_required_before_reliable_checkpoint',optimized_decisions_credit=0)
