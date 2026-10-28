import copy
import unittest
from pathlib import Path

from ptcg_lab.ppo_accounting import checkpoint_decisions, update_decisions


def trace(count, batches, stop=None, epochs=2):
    steps = []
    for epoch in range(epochs):
        for offset in range(0, count, batches):
            skip = (epoch, offset) == stop
            steps.append(dict(epoch=epoch, offset=offset, batch=min(batches, count-offset), skipped_for_kl=skip))
            if skip:
                return dict(version=1, decisions=count, steps=steps)
    return dict(version=1, decisions=count, steps=steps)


class ActualTrainingAccounting(unittest.TestCase):
    def test_writer_delta_rejects_any_change_to_archive_or_checkpoint_logic(self):
        from ptcg_lab.ppo_counter_compatibility import compatible_source
        root=Path(__file__).resolve().parents[1]
        before=(root/'runs/20261027T073939Z_tr10-lambda1-diagnostic_df4705/source/ptcg_lab/train_ppo.py').read_text(encoding='utf-8')
        after=(root/'runs/20261027T094058Z_tr10-original-second-actual-counter_2063d8/source/ptcg_lab/train_ppo.py').read_text(encoding='utf-8')
        self.assertTrue(compatible_source(before,after))
        # The new diagnostic has additional reviewed instrumentation. It must
        # never borrow TR10's historical counter-only compatibility receipt.
        current=(root/'ptcg_lab/train_ppo.py').read_text(encoding='utf-8')
        with self.assertRaises(ValueError):compatible_source(before,current)
        with self.assertRaises(ValueError):
            compatible_source(before,after.replace('outbox.evict_consumed(', 'outbox.unreviewed_eviction('))

    def test_real_TR10_early_stop_does_not_count_unused_or_repeated_decisions(self):
        first = trace(8371, 256)
        second = trace(8280, 256)
        third = trace(8272, 256, stop=(0, 5632))
        rows = [first, second, third]
        for version, row in enumerate(rows, 1):
            row['version'] = version
        self.assertEqual([update_decisions(r) for r in rows], [8371, 8280, 5632])
        saved = dict(version=3, trained_decisions=24923, updates=rows)
        self.assertEqual(checkpoint_decisions(saved), 22283)
        saved['unique_PPO_trained_decisions'] = 22283
        self.assertEqual(checkpoint_decisions(saved), 22283)
        saved['unique_PPO_trained_decisions'] = 24923
        with self.assertRaises(ValueError):
            checkpoint_decisions(saved)

    def test_full_first_epoch_counts_once_even_if_second_skips(self):
        self.assertEqual(update_decisions(trace(10, 4, stop=(1, 4))), 10)
        self.assertEqual(update_decisions(trace(10, 4, epochs=4)), 10)
        self.assertEqual(update_decisions(trace(10, 4, stop=(0, 0))), 0)

    def test_reject_incomplete_or_ambiguous_commits(self):
        good = trace(10, 4)
        cases = []
        for key, value in [('offset', 1), ('epoch', 1), ('batch', 0), ('skipped_for_kl', 1)]:
            bad = copy.deepcopy(good)
            bad['steps'][0][key] = value
            cases.append(bad)
        cases.append({**good, 'steps': good['steps'][:2]})
        cases.append({**good, 'unique_PPO_trained_decisions': 20})
        bad = copy.deepcopy(good)
        bad['steps'][0]['skipped_for_kl'] = True
        cases.append(bad)
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                update_decisions(bad)

    def test_restored_history_does_not_reset_consumed_cursor_or_add_epochs(self):
        row = trace(10, 4, stop=(0, 4))
        saved = dict(version=1, trained_decisions=10, updates=[row])
        for _ in range(3):
            self.assertEqual(checkpoint_decisions(saved), 4)
            self.assertEqual(saved['trained_decisions'], 10)
        saved['version'] = 2
        with self.assertRaises(ValueError):
            checkpoint_decisions(saved)


if __name__ == '__main__':
    unittest.main()
