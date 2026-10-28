"""Bounded lossless resource history sized for the admitted two-hour jobs."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import zlib
from .server_workspace import inside

MAX_LOG_BYTES=8*2**20
MAX_RECORD_BYTES=8*2**10
MAX_RAW_RECORD_BYTES=128*2**10
SAMPLE_SECONDS=10


def capacity_profile(maximum_job_seconds=7200,maximum_children=4,extended_TR11=False,extended_TR12=False,wait_exempt=False):
    if extended_TR11 and extended_TR12:raise ValueError('Runtime journal scopes cannot be combined')
    if wait_exempt and not extended_TR12:raise ValueError('Wait window belongs only to TR12')
    ceiling=86400 if wait_exempt else 28800 if extended_TR12 else 30000 if extended_TR11 else 7200
    journal_bytes=128*2**20 if wait_exempt else 32*2**20 if extended_TR11 or extended_TR12 else MAX_LOG_BYTES
    if not 0<maximum_job_seconds<=ceiling or not 1<=maximum_children<=4:
        raise ValueError('Resource journal time/child scope differs')
    records=math.ceil(maximum_job_seconds/SAMPLE_SECONDS)+maximum_children
    if records*MAX_RECORD_BYTES>journal_bytes:raise ValueError('Journal cannot cover admitted job duration')
    return {'maximum_job_seconds':maximum_job_seconds,'sample_interval_seconds':SAMPLE_SECONDS,
            'maximum_record_bytes':MAX_RECORD_BYTES,'maximum_raw_record_bytes':MAX_RAW_RECORD_BYTES,
            'maximum_journal_bytes':journal_bytes,'planned_record_count_upper':records,
            'planned_bytes_upper':records*MAX_RECORD_BYTES,'all_records_losslessly_retained':True}


def encode(sample):
    raw=(json.dumps(sample,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(raw)>MAX_RAW_RECORD_BYTES:raise ValueError('Resource metadata exceeds bounded raw envelope')
    if len(raw)<=MAX_RECORD_BYTES:return raw
    envelope={'resource_record_codec':'zlib+base64','raw_bytes':len(raw),
              'raw_sha256':hashlib.sha256(raw).hexdigest(),
              'data':base64.b64encode(zlib.compress(raw,1)).decode()}
    result=(json.dumps(envelope,separators=(',',':'))+'\n').encode()
    if len(result)>MAX_RECORD_BYTES:raise ValueError('Resource metadata exceeds bounded encoded envelope')
    return result


def decode(line):
    value=json.loads(line)
    if 'resource_record_codec' not in value:return value
    if value['resource_record_codec']!='zlib+base64' or not 0<value['raw_bytes']<=MAX_RAW_RECORD_BYTES:
        raise ValueError('Unknown/oversized resource record codec')
    inflater=zlib.decompressobj();packed=base64.b64decode(value['data'],validate=True)
    raw=inflater.decompress(packed,MAX_RAW_RECORD_BYTES+1)
    if len(raw)!=value['raw_bytes'] or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail or hashlib.sha256(raw).hexdigest()!=value['raw_sha256']:
        raise ValueError('Lossless resource record integrity mismatch')
    return json.loads(raw)


def append(root,job,sample,maximum_bytes=None):
    if maximum_bytes is None:maximum_bytes=MAX_LOG_BYTES
    if maximum_bytes not in (MAX_LOG_BYTES,32*2**20,128*2**20):raise ValueError('Unreviewed resource journal envelope')
    path=inside(root,Path(job)/'resource-samples.jsonl');data=encode(sample)
    if path.exists() and path.stat().st_size+len(data)>maximum_bytes:
        raise OSError('Bounded resource journal full outside admitted capacity envelope')
    with path.open('ab') as records:
        records.write(data);records.flush();os.fsync(records.fileno())
    return len(data)
