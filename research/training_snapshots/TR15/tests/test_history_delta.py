import hashlib
import unittest
from ptcg_lab.history_delta import encode,decode


class HistoryDeltaTests(unittest.TestCase):
    def test_changing_and_shrinking_serialized_bytes_preserve_external_hashes(self):
        previous=b''
        for raw in [bytes(range(256))*1000,b'zip-header'+bytes(range(256))*1000,b'tiny',b'',b'grow-again'*200]:
            packed,record=encode(raw,previous)
            actual=decode(packed,previous,record)
            self.assertEqual(hashlib.sha256(actual).hexdigest(),hashlib.sha256(raw).hexdigest())
            previous=raw
    def test_wrong_parent_corruption_and_expansion_fail(self):
        packed,record=encode(b'stable model weights'*1000,b'previous')
        with self.assertRaises(ValueError):decode(packed,b'wrong',record)
        with self.assertRaises(ValueError):decode(packed+b'trailing',b'previous',record)
        with self.assertRaises(ValueError):decode(packed,b'previous',{**record,'delta_bytes':record['delta_bytes']-1})
