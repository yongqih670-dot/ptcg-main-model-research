"""Resumable, bounded HTTP ranges with exact Content-Range and final SHA256."""
import hashlib
from pathlib import Path
import re
import shutil
import ssl
import time
import urllib.request
from .io import atomic_json,read_json,sha256


def fetch_verified(url,destination,size,digest,deadline,progress=None,chunk_bytes=4*1024**2,
                   entity_policy='strict_etag',seed=None):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if not re.fullmatch('[0-9a-f]{64}',digest) or type(size) is not int or size<=0:raise ValueError('Invalid download identity')
    if not url.startswith('https://'):raise ValueError('Only HTTPS artifacts allowed')
    if destination.exists():
        if destination.stat().st_size!=size or sha256(destination)!=digest:raise ValueError('Existing artifact checksum differs')
        return destination
    partial=destination.with_suffix(destination.suffix+'.partial');journal=destination.with_suffix(destination.suffix+'.ranges.json')
    if entity_policy not in ('strict_etag','official_digest_quarantine'):raise ValueError('Unknown entity policy')
    state={'url':url,'bytes':size,'sha256':digest,'etag':None,'chunks':[],'entity_policy':entity_policy}
    offset=0
    if seed and not journal.exists() and not partial.exists():
        source=Path(seed['path']);seed_bytes=seed['bytes']
        if entity_policy!='official_digest_quarantine' or source.resolve()==partial.resolve():raise ValueError('Invalid quarantine seed')
        if type(seed_bytes) is not int or not 0<seed_bytes<size:raise ValueError('Invalid seed length')
        if source.stat().st_size!=seed_bytes or sha256(source)!=seed['sha256']:raise ValueError('Quarantine seed changed')
        if shutil.disk_usage(destination.parent).free-size<20*2**30:raise OSError('Seed disk reserve reached')
        with source.open('rb') as src,partial.open('xb') as dst:
            while block:=src.read(chunk_bytes):
                if time.time()>deadline-30:raise TimeoutError('Seed copy deadline')
                dst.write(block);state['chunks'].append({'offset':offset,'bytes':len(block),
                                                       'sha256':hashlib.sha256(block).hexdigest(),'etag':None})
                offset+=len(block)
            dst.flush();__import__('os').fsync(dst.fileno())
        state['seed']={k:seed[k] for k in ['path','bytes','sha256']};atomic_json(journal,state)
    if journal.exists():
        saved=read_json(journal)
        if any(saved[k]!=state[k] for k in ['url','bytes','sha256']):raise ValueError('Download resume identity changed')
        if saved.get('entity_policy','strict_etag')!=entity_policy:raise ValueError('Cannot reinterpret a previous entity policy')
        offset=0
        with partial.open('rb') as src:
            for chunk in saved['chunks']:
                if chunk['offset']!=offset:raise ValueError('Noncontiguous download journal')
                block=src.read(chunk['bytes'])
                if len(block)!=chunk['bytes'] or hashlib.sha256(block).hexdigest()!=chunk['sha256']:
                    raise ValueError('Committed download chunk changed')
                offset+=len(block)
        state=saved
        if partial.stat().st_size!=offset:
            # Only an uncommitted append in this private download is discarded.
            with partial.open('r+b') as dst:dst.truncate(offset)
    elif partial.exists():raise ValueError('Unidentified partial artifact')
    import certifi
    context=ssl.create_default_context(cafile=certifi.where())
    while offset<size:
        if time.time()>deadline-30:raise TimeoutError('Download cap reached with resumable chunks preserved')
        count=min(chunk_bytes,size-offset)
        if shutil.disk_usage(destination.parent).free-count<20*2**30:raise OSError('Download disk reserve reached')
        headers={'Range':f'bytes={offset}-{offset+count-1}','Accept-Encoding':'identity'}
        if state['etag'] and entity_policy=='strict_etag':headers['If-Range']=state['etag']
        error=None
        for attempt in range(3):
            if time.time()>deadline-30:raise TimeoutError('Download deadline reached')
            try:
                request=urllib.request.Request(url,headers=headers)
                with urllib.request.urlopen(request,timeout=min(45,max(1,deadline-time.time()-20)),context=context) as response:
                    expected=f'bytes {offset}-{offset+count-1}/{size}'
                    if response.status!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError('Server did not honor exact artifact byte range')
                    etag=response.headers.get('ETag')
                    if entity_policy=='strict_etag' and state['etag'] and etag!=state['etag']:
                        raise ValueError('Artifact entity changed during download')
                    if response.headers.get('Content-Encoding') not in (None,'identity'):raise ValueError('Unexpected range encoding')
                    block=response.read(count+1)
                    if len(block)!=count:raise OSError('Truncated artifact byte range')
                break
            except ValueError:raise
            except (OSError,TimeoutError) as exc:
                error=exc
                if attempt==2:raise
                time.sleep(1)
        with partial.open('ab') as dst:dst.write(block);dst.flush();__import__('os').fsync(dst.fileno())
        state['etag']=state['etag'] or etag
        state['chunks'].append({'offset':offset,'bytes':count,'sha256':hashlib.sha256(block).hexdigest(),'etag':etag})
        offset+=count;atomic_json(journal,state)
        if progress:progress(offset,size)
    if sha256(partial)!=digest:raise ValueError('Complete artifact differs from official SHA256')
    partial.replace(destination)
    return destination
