"""One bounded local verification command; never starts training or remote jobs."""
import argparse
import ast
import contextlib
import io
import json
import time
import unittest
from common import WORK,ROOT,check_baseline,digest


def main():
    check_baseline();start=time.time();capture=io.StringIO()
    for folder in ('source','tests','benchmarks'):
        for path in (WORK/folder).rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    loader=unittest.TestLoader();suite=loader.discover(str(WORK/'tests'),pattern='test_*.py')
    # Reuse actual safety contracts, not simplified reimplementations.
    for name in ['tests.test_archive_barrier','tests.test_checkpoint_refs','tests.test_training_checkpoint',
                 'tests.test_rl_resume','tests.test_archive_outbox','tests.test_systemd_job',
                 'tests.test_ppo_accounting','tests.test_batched_actor_prototype']:
        suite.addTests(loader.loadTestsFromName(name))
    result=unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
    elapsed=time.time()-start
    (WORK/'reports/tests.txt').write_text(capture.getvalue(),encoding='utf-8')
    report={'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),
        'errors':len(result.errors),'skipped':len(result.skipped),'seconds':elapsed,
        'scope':'CPU engineering fixtures and production safety unit tests; no server/GPU acceptance',
        'new_scientific_decisions':0,'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p)
            for folder in [WORK/'source',WORK/'tests',WORK/'benchmarks'] for p in folder.rglob('*.py')}}
    (WORK/'reports/tests.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
    if not result.wasSuccessful():print(capture.getvalue());raise SystemExit(1)


if __name__=='__main__':main()
