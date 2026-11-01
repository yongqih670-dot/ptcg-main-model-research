"""Actual separate server input mapping was not in remote Python source packet."""
import time
from common import *
def main():
    report=V3/'repairs/missing-public-metadata-01';report.mkdir(exist_ok=False)
    snapshot=V3/'benchmarks/replay.py';(report/'replay.py').write_bytes(snapshot.read_bytes())
    run=ROOT/read(V3/'reports/BASELINE.json')['actual_source_run'];name='ptcg_lab/public_cg/metadata.json'
    manifest=read(run/'source-manifest.json');digest=manifest.get(name.replace('/','\\')) or manifest[name]
    source=run/'source'/name
    if sha(source)!=digest:raise ValueError('Actual separately uploaded input changed')
    for arm in ('baseline','candidate'):
        target=V3/arm/name
        if target.exists():raise ValueError('Do not replace asset')
        target.write_bytes(source.read_bytes())
    failure=read(V3/'reports/B-baseline-qual-process.json')
    write(report/'diagnosis.json',dict(at=time.time(),cause='Public metadata is separately sent by server_trial.input_objects, absent in remote-source.py packet; isolated harness did not copy it',
         failed_receipt='reports/B-baseline-qual-process.json',actual_returncode=failure['returncode'],charged_seconds=failure['seconds'],optimizer_steps_executed=0,
         remedy='Add byte-exact original separately uploaded metadata to both isolated arms. No model/PPO/runtime arithmetic changed. Explicit second qualification attempt, failed evidence retained.',source_sha256=digest,source=str(source.relative_to(ROOT))))
    write(V3/'reports/RUNTIME_INPUTS.json',dict(public_metadata=dict(path=name,sha256=digest,bytes=source.stat().st_size),actual_separate_upload_function='server_trial.input_objects',scientific_credit=0))
    print('Separately uploaded actual metadata restored to BOTH isolated copies')
if __name__=='__main__':main()
