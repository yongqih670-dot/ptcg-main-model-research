import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from ptcg_lab.io import sha256
from ptcg_lab.train_catboost import load_dataset


class BCDatasetTests(unittest.TestCase):
    def test_whole_series_splits_and_duplicate_states_are_kept_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'dataset').mkdir();manifest={'shards':[]}
            for split in ('train','validation','test'):
                path=root/'dataset'/f'{split}.npz'
                # Each split has one unique decision and one state shared by all.
                np.savez(path,X=np.zeros((4,224),np.float32),y=[1,0,0,1],offsets=[0,2,4],contexts=[0,7],fingerprints=[split,'shared'])
                manifest['shards'].append({'shard':path.name,'shard_sha256':sha256(path),'split':split,'group_id':split+'-bo3'})
            data,audit=load_dataset(root,manifest)
            self.assertEqual([len(data[s]['groups']) for s in ('train','validation','test')],[1,1,2])
            self.assertEqual(audit['cross_split_unique_visible_observations'],1)
            for d in data.values():
                for a,b in zip(d['offsets'][:-1],d['offsets'][1:]):self.assertAlmostEqual(float(d['weight'][a:b].sum()),1)
            manifest['shards'][1]['group_id']='train-bo3'
            with self.assertRaisesRegex(ValueError,'BO3 crossed'):load_dataset(root,manifest)


if __name__=='__main__':unittest.main()
