"""Local 8 MiB file publication microbenchmark, complete hashes kept on both paths."""
import json
import platform
from pathlib import Path
import statistics
import tempfile
import time
from common import WORK,check_baseline,digest
from fo.checkpoint_refs import commit_checkpoint,read_index
from ptcg_lab.bounded_io import environment_context


def main(output=None,workspace=None):
    output=output or WORK/'reports/publication_benchmark.json'
    workspace=workspace or WORK/'benchmarks'
    check_baseline();result={False:[],True:[]};size=8*2**20
    context=environment_context()
    def write(path,data):
        if context is None:path.write_bytes(data)
        else:context.atomic_write(path,lambda stream:stream.write(data))
    with tempfile.TemporaryDirectory(dir=workspace) as temp:
        root=Path(temp);source=root/'source';payload=bytes(range(256))*(size//256);write(source,payload)
        digest_value=digest(source)
        for repeat in range(7):
            for flag in ([False,True] if repeat%2==0 else [True,False]):
                run=root/f'{flag}-{repeat}';run.mkdir();(run/'prefix.json').write_text('{}')
                # File preparation is explicitly outside publication timing.
                write(run/'checkpoint.pt',payload)
                start=time.perf_counter()
                receipt=commit_checkpoint(run,'run',1,0,'checkpoint.pt','prefix.json',[],
                    dict(version=0,trained_decisions=0,next_series_id=0),immutable_hardlink=flag)
                result[flag].append(time.perf_counter()-start)
                assert read_index(run,'run',1)['commits'][0]==receipt
                assert receipt['checkpoint']['sha256']==digest_value
    report={'type':'file_publication_microbenchmark','platform':platform.system(),'bytes':size,'input_sha256':digest_value,
        'timings':{str(k):{'n':len(v),'median_seconds':statistics.median(v),'min_seconds':min(v),
                         'max_seconds':max(v),'samples_seconds':v} for k,v in result.items()},
        'end_to_end_speedup':None,'application_IO_context_active':context is not None,
        'limitations':['8 MiB synthetic file on the reported filesystem; production full hashes retained.',
            'No serialization, network transfer, or complete training measured; pacing status is reported separately.',
            'Candidate requires atomic-replace-only writer and same filesystem; otherwise copy fallback.']}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
