"""Zero-history compact PPO bootstrap, without changing actor or sampling RNG."""
from pathlib import Path
import random
import numpy as np
import torch
from .archive_prefix import canonical_hash,load_prefix,validate_prefix
from .io import atomic_json,read_json,sha256,utcnow
from .tensor_io import atomic_torch_save,load_torch
from .rl_core import rng_state,restore_rng


def settings_identity(config):
    return canonical_hash({k:config[k] for k in ['hypothesis_id','parent_model_sha256','seed','target_decisions','ppo','league']})


def initialize(output,config,model,optimizer):
    output=Path(output);receipt=output/'fresh-compact-origin.json';identity=settings_identity(config)
    if receipt.exists():
        row=read_json(receipt)
        if row['settings_sha256']!=identity or row['origin_run_id']!=output.name:raise ValueError('Fresh compact settings/origin changed')
        spec=row['archive_restore'];prefix=load_prefix(output/spec['prefix_path'],spec['prefix_sha256'])
        if prefix['consumed'] or prefix['tail'] or prefix['checkpoint']['version']!=0:raise ValueError('Fresh origin contains inherited training data')
        return spec
    if (output/'checkpoint.pt').exists() or (output/'checkpoint-commits/index.json').exists():
        raise ValueError('Fresh initialization cannot adopt an unregistered checkpoint')
    value={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),
        'version':0,'trained_decisions':0,'next_series_id':0,'updates':[]}
    atomic_torch_save(output/'checkpoint.pt',value,replace=False);digest=sha256(output/'checkpoint.pt')
    prefix={'schema_version':1,'origin_run_id':output.name,'archive_inventory_type':'fresh_empty_history',
        'archive_inventory_sha256':canonical_hash([]),'origin_config_sha256':identity,
        'origin_source_manifest_sha256':config['source_manifest_sha256'],
        'checkpoint':{'sha256':digest,'version':0,'trained_decisions':0,'next_series_id':0},
        'consumed':[],'tail':[],'consumed_records_root':canonical_hash([]),
        'consumed_fixed_statistics':{},'full_fixed_statistics':{}}
    validate_prefix(prefix);atomic_json(output/'archive-prefix.json',prefix)
    spec={'prefix_path':'archive-prefix.json','prefix_sha256':sha256(output/'archive-prefix.json'),
        'initial_checkpoint_sha256':digest}
    atomic_json(receipt,{'at':utcnow(),'origin_run_id':output.name,'settings_sha256':identity,'archive_restore':spec,
        'initialization':'Full original H002, fresh optimizer, zero terminal-history; no inherited TR04 data',
        'no_rng_draw_or_optimizer_step':True})
    return spec


def _draws(state):
    restore_rng(state)
    result={'python':random.random(),'numpy':np.random.randint(0,1000000,8),'torch':torch.rand(8),
        'cuda':torch.rand(16,device='cuda').cpu()}
    restore_rng(state);return result


def audit_gpu_restart(output,model,optimizer,saved,gpu_uuid,bootstrap_only):
    output=Path(output)
    if saved['version']!=0 or saved['trained_decisions']!=0:
        if bootstrap_only:raise ValueError('Bootstrap must precede every optimizer update')
        return
    if len(saved['model'])!=127 or saved['optimizer']['state']:raise ValueError('Fresh full model/empty optimizer expected')
    if any(not torch.equal(value.cpu(),saved['model'][key]) for key,value in model.state_dict().items()):raise ValueError('Fresh full model restore differs')
    if optimizer.state_dict()['param_groups']!=saved['optimizer']['param_groups']:raise ValueError('Fresh optimizer groups changed')
    draws=_draws(saved['rng']);path=output/'fresh-gpu-reference.pt'
    if bootstrap_only:
        atomic_torch_save(path,{'draws':draws,'gpu_uuid':gpu_uuid,'model_tensors':127},replace=False)
        atomic_json(output/'fresh-gpu-bootstrap.json',{'at':utcnow(),'complete':True,'model_tensors':127,
            'optimizer_states':0,'gpu_uuid':gpu_uuid,'sampling_rng_restored_after_probe':True,'optimizer_steps':0})
    else:
        previous=load_torch(path,map_location='cpu',weights_only=False);a=previous['draws'];b=draws
        if previous['gpu_uuid']!=gpu_uuid or a['python']!=b['python'] or not np.array_equal(a['numpy'],b['numpy']) or not torch.equal(a['torch'],b['torch']) or not torch.equal(a['cuda'],b['cuda']):
            raise ValueError('Fresh GPU process RNG or device differs')
        atomic_json(output/'fresh-gpu-restore.json',{'at':utcnow(),'complete':True,'model_tensors':127,'optimizer_states':0,
            'same_GPU_fresh_process_rng_exact':True,'gpu_uuid':gpu_uuid,'optimizer_steps_before_restore':0,
            'full_H002_and_empty_optimizer_verified':True,'sampling_rng_restored_after_probe':True})
