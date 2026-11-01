"""One-response, digest-pinned artifacts; never joins different representations."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import ssl
import time
import urllib.request

from .io import sha256


def fetch_single_response(url,destination,size,digest,deadline,progress=None):
    destination=Path(destination)
    if not url.startswith('https://') or not re.fullmatch('[0-9a-f]{64}',digest):
        raise ValueError('Invalid artifact identity')
    if type(size) is not int or size<=0:raise ValueError('Invalid artifact size')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if destination.stat().st_size!=size or sha256(destination)!=digest:raise ValueError('Existing artifact differs')
        return destination
    partial=destination.with_suffix(destination.suffix+'.stream.partial')
    if partial.exists():raise RuntimeError('Single-response transfer cannot resume a partial response')
    if time.time()>deadline-30:raise TimeoutError('Artifact transfer deadline')
    if shutil.disk_usage(destination.parent).free-size<20*2**30:raise OSError('Artifact disk reserve reached')
    import certifi
    context=ssl.create_default_context(cafile=certifi.where())
    request=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
    computed=hashlib.sha256();received=0
    with urllib.request.urlopen(request,context=context,timeout=min(30,max(1,deadline-time.time()-20))) as response:
        if response.status!=200:raise ValueError('Single-response transfer requires HTTP200')
        if response.headers.get('Content-Length')!=str(size):raise ValueError('Artifact Content-Length differs')
        if response.headers.get('Content-Encoding') not in (None,'identity'):raise ValueError('Unexpected content encoding')
        with partial.open('xb') as output:
            while received<size:
                if time.time()>deadline-30:raise TimeoutError('Artifact transfer deadline')
                if shutil.disk_usage(destination.parent).free-1024**2<20*2**30:raise OSError('Artifact disk reserve reached')
                block=response.read(min(1024**2,size-received))
                if not block:raise OSError('Truncated artifact response')
                output.write(block);computed.update(block);received+=len(block)
                if progress:progress(received,size)
            if response.read(1):raise ValueError('Artifact response exceeds official size')
            output.flush();os.fsync(output.fileno())
    if computed.hexdigest()!=digest or sha256(partial)!=digest:
        raise ValueError('Complete artifact differs from official SHA256')
    partial.replace(destination)
    return destination
