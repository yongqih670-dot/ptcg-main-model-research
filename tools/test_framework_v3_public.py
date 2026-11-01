"""Run the original synthetic fixtures; explicitly skip the private actor fixture."""
import json,os,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];V3=ROOT/'research/framework_optimization/v3_tr13'
def leaves(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from leaves(item)
        else:yield item
def main():
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.path[:0]=[str(V3/'candidate'),str(V3/'benchmarks')]
    import psutil,torch
    torch.set_num_threads(2)
    me=psutil.Process()
    if hasattr(me,'cpu_affinity'):me.cpu_affinity(me.cpu_affinity()[-3:])
    if psutil.virtual_memory().available<8*2**30:raise MemoryError('Keep the original8GiB host RAM reserve')
    (V3/'work').mkdir(exist_ok=True)
    suite=unittest.defaultTestLoader.discover(str(V3/'tests'),pattern='test_*.py')
    for case in leaves(suite):
        if case.id().endswith('.test_real_actor_features_and_full_optimizer_load'):
            cls=type(case);name=case._testMethodName;setattr(cls,name,unittest.skip('Original full actor/optimizer fixture needs private weights; no private data is loaded')(getattr(cls,name)))
    began=time.monotonic();result=unittest.TextTestRunner(verbosity=1).run(suite)
    report=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,passed_tests=result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),skipped=len(result.skipped),skip_reasons=[reason for _,reason in result.skipped],errors=len(result.errors),failures=len(result.failures),wall_seconds=time.monotonic()-began,scope='Public synthetic unit fixtures; separate from original38-test private acceptance',scientific_decisions=0,GPU_seconds=0)
    print(json.dumps(report,ensure_ascii=False))
    if '--record' in sys.argv:(V3/'reports/public-tests.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
if __name__=='__main__':main()
