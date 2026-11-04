import hashlib
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ptcg_lab.staged_environment import download_wheel, official_url, kernel_snapshot
from ptcg_lab.io_pacing import BytePacer


class Response(io.BytesIO):
    def __init__(self, data, start, end, total):
        super().__init__(data)
        self.status = 206
        self.url = 'https://files.pythonhosted.org/package.whl'
        self.headers = {'Content-Range': f'bytes {start}-{end}/{total}'}


class StagedEnvironmentTests(unittest.TestCase):
    def test_pacer_does_not_sleep_negative_when_clock_crosses_target(self):
        ticks=iter([0.,0.,.2,.4,.5])
        delays=[]
        pacer=BytePacer(100,time.time()+30,clock=lambda:next(ticks),sleep=delays.append)
        pacer.consume(30)
        self.assertTrue(all(d>=0 for d in delays))
        self.assertEqual(len(delays),1)
    def test_clipped_ranges_preserve_exact_file_hash(self):
        data = b'official pinned wheel content' * 250
        offsets = []
        def request(req, timeout):
            start = int(req.headers['Range'].split('=')[1].split('-')[0])
            offsets.append(start)
            end = min(len(data), start + 701)
            return Response(data[start:end], start, end-1, len(data))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); (root/'stage').mkdir()
            item = {'url': 'https://files.pythonhosted.org/package.whl', 'file': 'package.whl',
                    'bytes': len(data), 'expanded_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            with patch('ptcg_lab.staged_environment.urlopen', request):
                result = download_wheel(root, root, item, time.time()+120, lambda: None, lambda *a: None, lambda *a, **kw: None)
            self.assertEqual(result.read_bytes(), data)
            self.assertGreater(len(offsets), 1)
            self.assertFalse((root/'stage/package.whl.partial').exists())

    def test_wrong_hash_keeps_partial_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); (root/'stage').mkdir()
            item = {'url': 'https://files.pythonhosted.org/package.whl', 'file': 'package.whl',
                    'bytes': 3, 'expanded_bytes': 3, 'sha256': '0'*64}
            with patch('ptcg_lab.staged_environment.urlopen', return_value=Response(b'bad', 0, 2, 3)):
                with self.assertRaisesRegex(ValueError, 'Full artifact hash'):
                    download_wheel(root, root, item, time.time()+120, lambda: None, lambda *a: None, lambda *a, **kw: None)
            self.assertFalse((root/'stage/package.whl').exists())
            self.assertEqual((root/'stage/package.whl.partial').read_bytes(), b'bad')

    def test_unapproved_download_or_redirect_host_rejected(self):
        for url in ['http://files.pythonhosted.org/x', 'https://example.com/x',
                    'https://token@files.pythonhosted.org/x', 'https://files.pythonhosted.org:444/x']:
            with self.assertRaises(ValueError): official_url(url)

    def test_kernel_caps_checked_and_host_io_only_observed(self):
        unit = 'ptcg-hybrid-test-13.service'
        data = {'cgroup': '0::/user.slice/'+unit, 'cpu.max': '200000 100000',
                'cpu.stat': 'usage_usec 100', 'memory.max': str(4*2**30), 'memory.current': '100',
                'memory.peak': '100', 'memory.swap.max': '0', 'pids.max': '512',
                'io.stat': '8:0 rbytes=999999999999 wbytes=999999999999'}
        with patch.object(Path, 'read_text', lambda p: data[p.name]), patch.object(Path, 'exists', lambda p: True):
            self.assertIn('host_io_background', kernel_snapshot({'unit': unit}))
            data['cpu.max'] = 'max 100000'
            with self.assertRaisesRegex(RuntimeError, 'CPU kernel cap'):
                kernel_snapshot({'unit': unit})
            data['cpu.max'] = '200000 100000'; data['memory.swap.max'] = 'max'
            with self.assertRaisesRegex(RuntimeError, 'Memory/swap'):
                kernel_snapshot({'unit': unit})
            with self.assertRaisesRegex(RuntimeError, 'identity-bound'):
                kernel_snapshot({'unit': 'somebody-else.service'})
