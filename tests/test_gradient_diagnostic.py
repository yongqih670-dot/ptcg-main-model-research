import copy
import types
import unittest
import numpy as np
import torch
from ptcg_lab.gradient_diagnostic import measure


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.trunk=torch.nn.Linear(1,1,bias=False);self.value_mlp=torch.nn.Linear(1,1)
        with torch.no_grad():self.trunk.weight.fill_(1);self.value_mlp.weight.fill_(.5);self.value_mlp.bias.fill_(0)
    def forward_eval(self,x):
        h=self.trunk(x[:,:1]);return torch.cat([h,-h,h*0],1),self.value_mlp(h)


class GradientDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.model=TinyActor();self.features=types.SimpleNamespace(b=types.SimpleNamespace(STOP=2,OFFSETS={'oracle':(1,2)}))
        decisions=[]
        for i,reward in enumerate([1.,-1.]):
            rows=np.asarray([[i+1.,0]],np.float32)
            with torch.no_grad():scores,_=self.model.forward_eval(torch.from_numpy(rows));old=float(scores.log_softmax(-1)[0,i])
            decisions.append({'steps':[{'rows':rows,'mapping':np.asarray([[0,1,2]]),'n':2,'target':i}],
                'old_logp':old,'old_value':0.,'terminal':True,'reward':reward})
        self.data={'series_id':1,'policy_version':0,'summary':{'statuses':['DONE','DONE'],'rounds':2},'decisions':decisions}
        self.settings={'gamma':1.,'gae_lambda':.99,'clip':.2,'entropy_coefficient':.01,'value_loss_coefficient':.5}
    def test_reports_component_gradients_without_mutating_weights_or_parameter_grads(self):
        before=copy.deepcopy(self.model.state_dict());report=measure(self.model,self.features,[self.data],self.settings)
        self.assertEqual(report['optimizer_steps'],0);self.assertTrue(report['complete'])
        self.assertGreater(report['rows'][0]['weighted_value_shared_gradient_l2'],0)
        for name,value in before.items():self.assertTrue(torch.equal(value,self.model.state_dict()[name]))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
    def test_hidden_features_and_incomplete_BO3_are_rejected(self):
        self.data['decisions'][0]['steps'][0]['rows'][0,1]=1
        with self.assertRaises(ValueError):measure(self.model,self.features,[self.data],self.settings)
        self.data['decisions'][0]['steps'][0]['rows'][0,1]=0;self.data['summary']['statuses'][1]='ACTIVE'
        with self.assertRaises(ValueError):measure(self.model,self.features,[self.data],self.settings)
