"""Bounded phase spans: union of intervals, never sum concurrent wall times."""
from contextlib import contextmanager
import json
import os
import threading
import time


def union_seconds(intervals):
    total = 0.0
    end = None
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError('Negative span')
        total += b - max(a, end if end is not None else a) if end is None or b > end else 0
        end = b if end is None else max(end, b)
    return total


class Trace:
    def __init__(self, enabled=False, maximum_spans=100000):
        self.enabled = enabled
        self.maximum_spans = maximum_spans
        self.start = time.perf_counter()
        self.spans = []
        self.dropped = 0
        self.lock = threading.Lock()

    @contextmanager
    def span(self, phase, cuda_device=None, **metadata):
        if not self.enabled:
            yield
            return
        # Explicit synchronization is opt-in and perturbs execution. No claim
        # that a Python submission duration is CUDA kernel duration.
        if cuda_device is not None:
            import torch
            torch.cuda.synchronize(cuda_device)
        start = time.perf_counter()
        status = 'complete'
        try:
            yield
        except BaseException:
            status = 'error'
            raise
        finally:
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
            end = time.perf_counter()
            row = dict(phase=phase, start=start-self.start, end=end-self.start,
                       pid=os.getpid(), thread=threading.get_ident(), status=status,
                       cuda_synchronized=cuda_device is not None, **metadata)
            with self.lock:
                if len(self.spans) < self.maximum_spans:
                    self.spans.append(row)
                else:
                    self.dropped += 1

    def report(self):
        with self.lock:
            spans = list(self.spans)
        wall = time.perf_counter()-self.start
        phases = {p: union_seconds([(s['start'], s['end']) for s in spans if s['phase']==p])
                  for p in sorted({s['phase'] for s in spans})}
        covered = union_seconds([(s['start'], s['end']) for s in spans])
        return dict(wall_seconds=wall, union_covered_seconds=covered,
                    residual_seconds=max(0., wall-covered), phases_union_seconds=phases,
                    phase_sums_are_not_additive=True, dropped_spans=self.dropped, spans=spans)

    def write(self, path):
        path.write_text(json.dumps(self.report(), indent=2), encoding='utf-8')
