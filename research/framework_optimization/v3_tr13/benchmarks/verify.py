"""Targeted isolated tests; never runs production dispatcher or science."""
import contextlib,io,json,os,sys,time,unittest
from pathlib import Path
V3=Path(__file__).resolve().parents[1];ROOT=V3.parents[2];sys.path[:0]=[str(V3/'candidate'),str(V3/'tests')]
def main():
    start=time.monotonic()
    import psutil
    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[-3:])
    import torch
    torch.set_num_threads(2)
    loader=unittest.TestLoader();suite=loader.discover(str(V3/'tests'),pattern='test_*.py');output=io.StringIO()
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),wall_seconds=time.monotonic()-start,scientific_decisions=0,GPU_seconds=0)
    (V3/'reports/tests.txt').write_text(output.getvalue(),encoding='utf-8');(V3/'reports/tests.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
    if not result.wasSuccessful():print(output.getvalue());raise SystemExit(1)
if __name__=='__main__':main()
