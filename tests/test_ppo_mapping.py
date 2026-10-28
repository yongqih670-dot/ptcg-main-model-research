import unittest
import torch
from ptcg_lab.public_agent import map_actor_seat_tensor


class PPOMappingTests(unittest.TestCase):
    def test_only_selected_feature_changes_and_input_is_not_mutated(self):
        # [physical seat, first decided, I am first, arbitrary visible feature]
        x=torch.tensor([[0.,1.,1.,7.],[1.,1.,0.,8.],[1.,1.,1.,9.],[0.,0.,0.,10.]])
        original=x.clone()
        zero=map_actor_seat_tensor(x,0,'zero',1,2)
        order=map_actor_seat_tensor(x,0,'turn_order',1,2)
        self.assertTrue(torch.equal(x,original))
        self.assertEqual(zero[:,0].tolist(),[0,0,0,0])
        self.assertEqual(order[:,0].tolist(),[0,1,0,0])
        self.assertTrue(torch.equal(zero[:,1:],x[:,1:]))
        self.assertTrue(torch.equal(order[:,1:],x[:,1:]))


if __name__=='__main__':unittest.main()
