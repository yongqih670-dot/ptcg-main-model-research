"""Small staged-install helpers; no training or global host modifications."""
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile
import zlib

from .io import atomic_json, read_json, sha256, utcnow
from .io_pacing import BytePacer
from .server_workspace import inside

OFFICIAL_HOSTS = {'files.pythonhosted.org', 'download.pytorch.org', 'download-r2.pytorch.org'}
# Per-job exact-byte mirror only; every artifact retains the original official
# filename, size and full SHA. This never changes pip's global configuration.
EXACT_BYTE_MIRRORS = {'mirrors.aliyun.com'}


def official_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or parsed.hostname not in OFFICIAL_HOSTS|EXACT_BYTE_MIRRORS
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError('Artifact URL is not an approved HTTPS source')
    return value


def recovery_descriptor(project, config, items):
    """Bind existing install receipts and expected files to the original wheels."""
    old = Path(project) / config['recovery_parent_run_id']
    if not old.is_dir():
        old = Path(project) / 'runs' / config['recovery_parent_run_id']
    manifest = old / 'remote-wheel-manifest.json'
    if sha256(manifest) != config['recovery_parent_manifest_sha256']:
        raise ValueError('Original installation manifest changed')
    result = {'parent_run_id': old.name, 'parent_manifest_sha256': sha256(manifest), 'prefix': []}
    for index, item in enumerate(items[:18]):
        receipt = old / 'remote-archive' / f'{index:03d}' / 'receipt.json'
        data = read_json(receipt)
        if data['source_sha256'] != item['sha256'] or data['manifest_sha256'] != sha256(manifest):
            raise ValueError('Archived installation prefix differs')
        expected = []
        with zipfile.ZipFile(item['source_path']) as wheel:
            record = [n for n in wheel.namelist() if n.count('/') == 1 and n.endswith('.dist-info/RECORD')]
            if len(record) != 1:
                raise ValueError('Ambiguous wheel RECORD')
            for name, digest, size in csv.reader(io.StringIO(wheel.read(record[0]).decode())):
                if not digest or '.data/' in name:
                    continue  # Installed RECORD below covers generated scripts too.
                if name.startswith('/') or '..' in name.split('/') or not digest.startswith('sha256='):
                    raise ValueError('Unsafe wheel RECORD')
                expected.append([name, digest, int(size)])
        result['prefix'].append({'index': index, 'name': item['name'], 'version': item['version'],
                                 'receipt_sha256': sha256(receipt), 'expected_files': expected})
    return result


def validate_recovery_owner(root, manifest):
    if 'recovery' not in manifest:
        decoder=zlib.decompressobj()
        content=decoder.decompress(base64.b64decode(manifest['recovery_zlib_base64'],validate=True),600001)
        if len(content)>600000 or not decoder.eof or decoder.unused_data:
            raise ValueError('Oversized or invalid installed-prefix proof')
        manifest['recovery']=json.loads(content)
    proof = manifest['recovery']; owner = read_json(inside(root, root / 'environment-owner.json'))
    if (owner.get('creator_run_id') != proof['parent_run_id']
            or owner.get('manifest_sha256') != proof['parent_manifest_sha256']
            or owner.get('status') != 'creating'):
        raise ValueError('Existing environment is not the explicitly registered partial install')
    inside(root, root / '.venv')
    parent = inside(root, root / 'jobs' / proof['parent_run_id'])
    if sha256(parent / 'wheel-manifest.json') != proof['parent_manifest_sha256']:
        raise ValueError('Remote original manifest differs')
    if len(proof['prefix']) != 18:
        raise ValueError('Recovery prefix must contain exactly the archived 18 wheels')
    for index, row in enumerate(proof['prefix']):
        path = inside(root, parent / 'receipts' / f'{index:03d}.json')
        if row['index'] != index or sha256(path) != row['receipt_sha256']:
            raise ValueError('Remote installed-prefix receipt differs')
    return owner


def kernel_snapshot(ticket):
    group = [s.split(':', 2)[2] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')]
    if len(group) != 1 or group[0].split('/')[-1] != ticket['unit']:
        raise RuntimeError('Installer is outside the identity-bound user service')
    cg = Path('/sys/fs/cgroup') / group[0].lstrip('/')
    values = {name: (cg / name).read_text().strip() for name in
              ['cpu.max', 'cpu.stat', 'memory.max', 'memory.current', 'memory.peak', 'memory.swap.max', 'pids.max']}
    quota, period = values['cpu.max'].split()
    from .resource_profile import ticket_limits
    explicit=ticket_limits(ticket)
    cpu_cap,memory_cap=explicit if explicit is not None else (2,4*2**30)
    if quota == 'max' or not 0 < int(quota) / int(period) <= cpu_cap:
        raise RuntimeError('CPU kernel cap is absent or excessive')
    if values['memory.max'] != str(memory_cap) or values['memory.swap.max'] != '0':
        raise RuntimeError('Memory/swap kernel caps differ')
    if values['pids.max'] == 'max' or int(values['pids.max']) > 512:
        raise RuntimeError('Process cap is absent or excessive')
    values['own_io_stat'] = (cg / 'io.stat').read_text().strip() if (cg / 'io.stat').exists() else None
    # The host counter is context only, never an attribution or stop condition.
    host = Path('/sys/fs/cgroup/io.stat')
    values['host_io_background'] = host.read_text().strip() if host.exists() else None
    return values


def verify_prefix_item(root, row, check):
    distribution = importlib.metadata.distribution(row['name'])
    if distribution.version != row['version']:
        raise ValueError('Installed prefix version differs')
    verified = 0
    def verify(path, encoded, size):
        nonlocal verified
        check(); path = inside(root, path)
        if root / '.venv' not in path.parents or path.stat().st_size != int(size):
            raise ValueError('Installed prefix file path/size differs')
        algorithm, value = encoded.split('=', 1)
        if algorithm != 'sha256':
            raise ValueError('Unsupported installed hash')
        expected = base64.urlsafe_b64decode(value + '=' * (-len(value) % 4)).hex()
        if sha256(path) != expected:
            raise ValueError('Installed prefix content differs from its pinned hash')
        verified += 1
    for name, encoded, size in row['expected_files']:
        verify(distribution.locate_file(name), encoded, size)
    for entry in distribution.files or []:
        if entry.hash:
            # Direct entries already verified against the immutable wheel above.
            if str(entry) not in {x[0] for x in row['expected_files']}:
                path = Path(os.path.abspath(distribution.locate_file(entry)))
                verify(path, entry.hash.mode + '=' + entry.hash.value, entry.size)
    if not verified:
        raise ValueError('Empty installed file verification')
    return verified


def download_wheel(root, job, item, deadline, check, capacity, progress):
    """One artifact at a time, 4 MiB/s, finite ranges, exact pinned full hash."""
    url = official_url(item['url']); destination = inside(root, job / 'stage' / item['file'])
    temporary = inside(root, destination.with_name(destination.name + '.partial'))
    pacer = BytePacer(4 * 2**20, deadline - 60)
    digest = hashlib.sha256(); offset = 0; requests = 0; last_report = 0
    with temporary.open('xb') as output:
        while offset < item['bytes']:
            check(); capacity(item['bytes'] - offset + item['expanded_bytes'])
            end = min(item['bytes'] - 1, offset + 16 * 2**20 - 1)
            requests += 1
            if requests > (item['bytes'] // (64 * 1024) + 20):
                raise RuntimeError('Excessive artifact response fragmentation')
            request = Request(url, headers={'Range': f'bytes={offset}-{end}', 'Accept-Encoding': 'identity'})
            with urlopen(request, timeout=20) as response:
                official_url(response.url)
                if response.status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if (not match or int(match[1]) != offset or int(match[3]) != item['bytes']
                            or not offset <= int(match[2]) <= end):
                        raise ValueError('Artifact Content-Range differs')
                    remaining = int(match[2]) - offset + 1
                elif response.status == 200 and offset == 0:
                    remaining = item['bytes']
                else:
                    raise ValueError('Artifact server ignored resumed range')
                while remaining:
                    check(); count = min(256 * 1024, remaining); pacer.consume(count)
                    block = response.read(count)
                    if not block:
                        raise EOFError('Incomplete artifact response; keep partial as evidence')
                    output.write(block); digest.update(block); offset += len(block); remaining -= len(block)
                    if time.monotonic() - last_report >= 5:
                        output.flush(); capacity(); progress('downloading_official_wheel', bytes_done=offset, bytes_total=item['bytes'])
                        last_report = time.monotonic()
        output.flush(); os.fsync(output.fileno())
    if offset != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise ValueError('Full artifact hash differs from original pinned wheel')
    temporary.replace(destination)
    return destination
