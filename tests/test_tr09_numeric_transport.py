import copy,hashlib,tempfile
from pathlib import Path
import unittest
from ptcg_lab.server_trial import bind_clean_recovery_inputs


class NumericTransportTests(unittest.TestCase):
    def test_reference_upload_does_not_modify_master_input_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'audit/numeric-reference.pt';path.parent.mkdir();path.write_bytes(b'controlled-test-reference')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            c={'hypothesis_id':'TR09','check_variant':'derived_scientific','consumed_reference_barrier_required':True,
                'source_clean_checkpoint_recovery':{'local_audit':{'sha256':'a'*64}},'ACK_lag_barrier_acceptance':{'sha256':'b'*64},
                'numeric_reference':{'path':'audit/numeric-reference.pt','bytes':path.stat().st_size,'sha256':digest}}
            original_map=[{'path':'agents/actor/model.pt','sha256':'c'*64,'bytes':42}]
            manifest={'files':[],'file_map':copy.deepcopy(original_map)};objects={}
            bind_clean_recovery_inputs(root,c,manifest,objects)
            self.assertEqual(manifest['file_map'],original_map);self.assertEqual(objects,{digest:path.resolve()})
            self.assertEqual(manifest['files'],[{'file':digest+'.blob','sha256':digest,'bytes':path.stat().st_size,'expanded_bytes':0}])
            for bad in [{**c,'hypothesis_id':'TR08'},{**c,'check_variant':'fresh_scientific'},{**c,'consumed_reference_barrier_required':False}]:
                with self.assertRaises(ValueError):bind_clean_recovery_inputs(root,bad,{},{} )
            for field,value in [('sha256','d'*64),('bytes',0),('bytes',64*2**20+1),('path','../numeric-reference.pt')]:
                bad=copy.deepcopy(c);bad['numeric_reference'][field]=value
                with self.assertRaises(ValueError):bind_clean_recovery_inputs(root,bad,{},{} )


if __name__=='__main__':unittest.main()
