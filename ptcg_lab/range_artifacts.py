"""Two bounded HTTPS range readers, one artifact, original full SHA authority."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import re
import threading
import time
from urllib.request import Request, urlopen

from .io_pacing import BytePacer
from .server_workspace import inside
from .staged_environment import official_url


def file_identity(path):
    stat=path.stat()
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns


def fetch_range(url,start,end,total,pacer,lock,check):
    for attempt in range(2):
        check()
        try:
            request=Request(official_url(url),headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
            with urlopen(request,timeout=15) as response:
                official_url(response.url)
                match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                if response.status!=206 or not match or tuple(map(int,match.groups()))!=(start,end,total):
                    raise ValueError('Exact range identity differs')
                result=bytearray()
                while len(result)<end-start+1:
                    check();count=min(256*1024,end-start+1-len(result))
                    with lock:pacer.consume(count)
                    data=response.read(count)
                    if not data:raise EOFError('Incomplete bounded range')
                    result.extend(data)
                return bytes(result)
        except (TimeoutError,OSError,EOFError):
            if attempt:raise
    raise AssertionError('Unreachable artifact retry')


def download(root,job,item,deadline,check,capacity,progress):
    spec=item['range_repair']
    if spec.get('workers') not in (2,4) or spec.get('range_bytes')!=1024**2:
        raise ValueError('Unreviewed artifact concurrency/range size')
    url=official_url(item['url']);target=inside(root,job/'stage'/item['file'])
    temporary=inside(root,target.with_name(target.name+'.partial'))
    pacer=BytePacer(4*2**20,deadline-60);lock=threading.Lock()
    computed=hashlib.sha256();offset=0;started=time.monotonic();last_report=0
    with temporary.open('xb') as output:
        resume=spec.get('resume')
        if resume:
            source=inside(root,root/resume['relative_path']);identity=file_identity(source)
            if not 0<resume['bytes']<item['bytes'] or identity[2]!=resume['bytes']:
                raise ValueError('Resume fragment length differs')
            check();prefix=hashlib.sha256()
            with source.open('rb') as old:
                while data:=old.read(256*1024):
                    check();pacer.consume(len(data));output.write(data);prefix.update(data);computed.update(data);offset+=len(data)
            if prefix.hexdigest()!=resume['prefix_sha256'] or file_identity(source)!=identity:
                raise ValueError('Old fragment differs from independently pinned local wheel prefix')
            output.flush();os.fsync(output.fileno())
            progress('verified_partial_reused',bytes_done=offset,bytes_total=item['bytes'])
        with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
            network_start=time.monotonic();network_offset=offset
            futures={};next_offset=offset
            def schedule():
                nonlocal next_offset
                while len(futures)<spec['workers'] and next_offset<item['bytes']:
                    end=min(item['bytes']-1,next_offset+1024**2-1)
                    futures[next_offset]=pool.submit(fetch_range,url,next_offset,end,item['bytes'],pacer,lock,check)
                    next_offset=end+1
            schedule()
            while futures:
                check();data=futures.pop(offset).result(timeout=40)
                output.write(data);computed.update(data);offset+=len(data);schedule()
                if time.monotonic()-last_report>=5:
                    output.flush();capacity()
                    elapsed=time.monotonic()-network_start
                    if elapsed>120 and (offset-network_offset)/elapsed<64*1024:
                        raise TimeoutError('Own artifact throughput too low for bounded recovery; retain evidence')
                    progress('downloading_bounded_ranges',bytes_done=offset,bytes_total=item['bytes'],
                             elapsed_seconds=time.monotonic()-started,concurrent_ranges_max=spec['workers'])
                    last_report=time.monotonic()
        output.flush();os.fsync(output.fileno())
    if offset!=item['bytes'] or computed.hexdigest()!=item['sha256']:
        raise ValueError('Recovered full artifact differs from original pinned SHA')
    temporary.replace(target)
    return target
