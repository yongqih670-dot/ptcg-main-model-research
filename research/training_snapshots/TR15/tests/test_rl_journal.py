import json
from pathlib import Path
import tempfile
import unittest

from ptcg_lab.io import sha256
from ptcg_lab.rl_journal import committed_fixed_results


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'series').mkdir()
        (self.root / 'rollouts').mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def row(self, sid, reward, fixed=True):
        shard = self.root / 'rollouts' / f's{sid:06d}.pt'
        shard.write_bytes(f'committed-{sid}'.encode())
        data = {'series_id': sid, 'statuses': ['DONE', 'DONE'], 'learner_seat': 1,
                'rewards': [-reward, reward], 'assignment': {'fixed': fixed, 'opponent_id': 'fixed-a'},
                'trajectory_path': shard.relative_to(self.root).as_posix(), 'trajectory_sha256': sha256(shard)}
        path = self.root / 'series' / f'{sid:06d}.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        return shard, path, data

    def test_rebuild_is_idempotent_without_existing_statistics(self):
        self.row(0, -1); self.row(1, 1); self.row(2, 0); self.row(3, -1, fixed=False)
        (self.root / 'series' / '000004-interrupted.json').write_text('{}')
        expected = {'fixed-a': {'n': 3, 'losses': 1.5}}
        self.assertEqual(committed_fixed_results(self.root), expected)
        self.assertEqual(committed_fixed_results(self.root), expected)

    def test_changed_shard_fails_closed(self):
        shard, _, _ = self.row(0, -1)
        shard.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'trajectory changed'):
            committed_fixed_results(self.root)

    def test_incomplete_or_misidentified_result_fails_closed(self):
        _, path, data = self.row(0, -1)
        for change in ({'statuses': ['ERROR', 'DONE']}, {'series_id': 1}):
            path.write_text(json.dumps(dict(data, **change)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'complete BO3'):
                committed_fixed_results(self.root)

    def test_external_shard_fails_closed(self):
        _, path, data = self.row(0, -1)
        data['trajectory_path'] = '../outside.pt'
        path.write_text(json.dumps(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'outside rollout'):
            committed_fixed_results(self.root)


if __name__ == '__main__':
    unittest.main()
