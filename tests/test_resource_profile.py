import copy
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch
from ptcg_lab.io import atomic_json,read_json,sha256
from ptcg_lab.resource_profile import limits,ticket_limits,verify_actual_machine,validate_local_authority,bind_manifest
from ptcg_lab.staged_environment import kernel_snapshot
from ptcg_lab.systemd_job import service_command


class ResourceProfileTests(unittest.TestCase):
    def profile(self):
        return {'schema':'per_GPU_CPU_RAM_share_v1','authorization_sha256':'a'*64,'machine_evidence_sha256':'b'*64,
                'observed_machine':{'logical_CPU':32,'GPU_count':8,'RAM_total_bytes':540673888256},
                'cpu_quota_cores':2,'memory_max_bytes':32*2**30}

    def ticket(self):
        return {'unit':'ptcg-hybrid-profile-12345678.service','kind':'train_ppo','project_root':'/home/user/ptcg',
                'job_directory':'/home/user/ptcg/jobs/run','ticket_path':'/home/user/ptcg/jobs/run/ticket.json',
                'run_id':'run','deadline_unix':1000,'memory_max_bytes':32*2**30,'workspace_hard_max_bytes':10*2**30,
                'cpu_quota_cores':2,'python_kind':'venv','gpu_uuid':'AUTHORIZED_GPU_UUID',
                'server_resource_profile':self.profile()}

    def test_half_share_defaults_and_full_share_ceiling_without_unbounded_memory(self):
        self.assertEqual(limits(self.profile()),(2,32*2**30))
        full={**self.profile(),'cpu_quota_cores':4,'memory_max_bytes':62*2**30}
        self.assertEqual(limits(full),(4,62*2**30))
        for delta in [{'cpu_quota_cores':4.1},{'memory_max_bytes':63*2**30},{'cpu_quota_cores':float('nan')},
                      {'memory_max_bytes':True},{'authorization_sha256':''},
                      {'observed_machine':{'logical_CPU':8,'GPU_count':8,'RAM_total_bytes':540673888256}}]:
            with self.assertRaises(ValueError):limits({**self.profile(),**delta})

    def test_legacy_service_does_not_gain_large_memory_cap_from_new_code(self):
        t=self.ticket();command=service_command(t,now=400)
        self.assertIn('--property=MemoryMax='+str(32*2**30),command)
        self.assertIn('--property=CPUQuota=200%',command)
        del t['server_resource_profile']
        with self.assertRaises(ValueError):service_command(t,now=400)
        t['memory_max_bytes']=4*2**30
        self.assertIn('--property=MemoryMax='+str(4*2**30),service_command(t,now=400))
        t=self.ticket();t['memory_max_bytes']=33*2**30
        with self.assertRaises(ValueError):ticket_limits(t)

    def test_kernel_monitor_demands_ticket_bound_limit_and_keeps_legacy_four_GiB(self):
        t=self.ticket();data={'cgroup':'0::/user.slice/'+t['unit'],'cpu.max':'200000 100000','cpu.stat':'usage_usec 100',
            'memory.max':str(32*2**30),'memory.current':'100','memory.peak':'100','memory.swap.max':'0','pids.max':'512','io.stat':''}
        with patch.object(Path,'read_text',lambda p:data[p.name]),patch.object(Path,'exists',lambda p:True):
            self.assertEqual(kernel_snapshot(t)['memory.max'],str(32*2**30))
            data['memory.max']=str(4*2**30)
            with self.assertRaisesRegex(RuntimeError,'Memory/swap'):kernel_snapshot(t)
            self.assertEqual(kernel_snapshot({'unit':t['unit']})['memory.max'],str(4*2**30))
            data['cpu.max']='500000 100000'
            with self.assertRaisesRegex(RuntimeError,'CPU kernel cap'):kernel_snapshot(t)

    def test_target_hardware_is_reobserved_before_launch(self):
        with patch('ptcg_lab.resource_profile.os.cpu_count',return_value=32),patch('ptcg_lab.resource_profile.os.sched_getaffinity',return_value=set(range(32)),create=True), \
             patch.object(Path,'read_text',return_value='MemTotal: 528001844 kB\n'), \
             patch('ptcg_lab.resource_profile.subprocess.run',return_value=types.SimpleNamespace(stdout='\n'.join(map(str,range(8))))):
            self.assertTrue(verify_actual_machine(self.profile())['actual_per_GPU_share_verified'])
            with patch('ptcg_lab.resource_profile.os.cpu_count',return_value=8):
                with self.assertRaises(ValueError):verify_actual_machine(self.profile())

    def fixture(self,root):
        actual={'logical_cpu_total':32,'logical_cpu_affinity':32,'gpu_count':8,'memory':{'MemTotal':540673888256}}
        probe=root/'probe.json';atomic_json(probe,{'actual':actual})
        authority=root/'authorization.json';atomic_json(authority,{'actual_machine_evidence':{'path':'probe.json','sha256':sha256(probe)},
            'authorized_future_ceiling':{'logical_CPU_quota':4,'memory_max_GiB':62}})
        spec={'path':'authorization.json','sha256':sha256(authority)}
        policy={'status':'active','activation':{'executable':True},'execution':{'hybrid':{'shared_server_safety':{'per_GPU_resource_authorization':spec}}}}
        atomic_json(root/'research/autonomy/policy.json',policy);state={'server_resource_profile_authorization':spec}
        atomic_json(root/'research/autonomy/state.json',state)
        profile=self.profile();profile.update(authorization_sha256=spec['sha256'],machine_evidence_sha256=sha256(probe))
        return profile,state,policy

    def test_production_manifest_binding_requires_real_evidence_and_activated_authority(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);p,s,policy=self.fixture(root)
            self.assertEqual(bind_manifest(root,{'server_resource_profile':p},{'mode':'fresh_scientific'})['server_resource_profile'],p)
            with self.assertRaisesRegex(ValueError,'forward-activated'):validate_local_authority(root,p,{})
            changed=copy.deepcopy(p);changed['observed_machine']['GPU_count']=4
            with self.assertRaisesRegex(ValueError,'measured capacity'):validate_local_authority(root,changed,s)
            policy['status']='proposed';atomic_json(root/'research/autonomy/policy.json',policy)
            with self.assertRaisesRegex(ValueError,'not active'):bind_manifest(root,{'server_resource_profile':p},{})
            self.assertEqual(bind_manifest(root,{},{}),{})


if __name__=='__main__':unittest.main()
