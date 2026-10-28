import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ptcg_lab.resource_journal import encode,decode,append,capacity_profile,MAX_LOG_BYTES


class ResourceJournalTests(unittest.TestCase):
    def test_capacity_covers_entire_two_hour_multi_child_job_without_old_cap_failure(self):
        profile=capacity_profile()
        self.assertEqual(profile['planned_record_count_upper'],724)
        self.assertLess(profile['planned_bytes_upper'],MAX_LOG_BYTES)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job=root/'job';job.mkdir()
            sample={'kernel':{'host_io_background':'x'*6900},'actual_own_memory':123}
            for i in range(profile['planned_record_count_upper']):append(root,job,dict(sample,sequence=i))
            path=job/'resource-samples.jsonl'
            self.assertGreater(path.stat().st_size,512*1024);self.assertLess(path.stat().st_size,MAX_LOG_BYTES)
            records=[decode(line) for line in path.read_text().splitlines()]
            self.assertEqual([x['sequence'] for x in records],list(range(724)))
            self.assertTrue(all(x['kernel']==sample['kernel'] for x in records))

    def test_oversized_background_is_losslessly_encoded_without_losing_own_metrics(self):
        sample={'kernel':{'host_io_background':'rbytes=123 wbytes=456\n'*2000},'own_memory':34,'GPU':{'own':12}}
        encoded=encode(sample);self.assertLess(len(encoded),8192);self.assertEqual(decode(encoded),sample)
        row=json.loads(encoded);row['raw_sha256']='0'*64
        with self.assertRaises(ValueError):decode(json.dumps(row))
        row['resource_record_codec']='unknown'
        with self.assertRaises(ValueError):decode(json.dumps(row))

    def test_log_guard_and_invalid_scope_still_stop(self):
        for seconds in [0,7201]:
            with self.assertRaises(ValueError):capacity_profile(seconds)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job=root/'job';job.mkdir()
            with patch('ptcg_lab.resource_journal.MAX_LOG_BYTES',60):
                append(root,job,{'sample':'x'*30})
                before=(job/'resource-samples.jsonl').read_bytes()
                with self.assertRaises(OSError):append(root,job,{'sample':'x'*30})
                self.assertEqual((job/'resource-samples.jsonl').read_bytes(),before)


if __name__=='__main__':unittest.main()
