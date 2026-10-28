from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'research/framework_optimization'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK/'source'))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_baseline():
    manifest=json.loads((WORK/'reports/input_manifest.json').read_text())
    for row in manifest['source_files']:
        if not row.get('reference_only') and digest(ROOT/row['path'])!=row['sha256']:
            raise ValueError('Source drift; review before benchmark: '+row['path'])


def real_fixture():
    import torch
    from ptcg_lab.rl_core import RLFeatures
    report=ROOT/'runs/20261028T093904Z_tr12-full-state-tail_e16975/verified-recovery-report.json'
    ref=json.loads(report.read_text())['numerical_reference'];path=ROOT/ref['path']
    if digest(path)!=ref['sha256']:raise ValueError('Numeric fixture checksum mismatch')
    value=torch.load(path,map_location='cpu',weights_only=False)
    config=json.loads((ROOT/'runs/20261027T160313Z_tr11-authorized-second-posix-fixed_1eae55/config.json').read_text())
    # The actual bundle path is bound in completed metrics, not guessed.
    metrics=json.loads((ROOT/'runs/20261027T160313Z_tr11-authorized-second-posix-fixed_1eae55/metrics.json').read_text())
    bundle=ROOT/metrics['actual_config']['bundle_path']
    features=RLFeatures(bundle)
    from ptcg.rl.model import PTCGTransformer,arch_from_state_dict
    model=PTCGTransformer(**arch_from_state_dict(value['model'],heads=8),critic='blind')
    model.load_state_dict(value['model'],strict=True);model.eval()
    return model,features,value['fixture'],{'path':ref['path'],'sha256':ref['sha256'],
        'model_source_checkpoint_sha256':value['source_checkpoint_sha256'],
        'model_source_path':str(bundle/'ptcg/rl/model.py'),
        'model_source_sha256':digest(bundle/'ptcg/rl/model.py'),'decisions':len(value['fixture'])}
