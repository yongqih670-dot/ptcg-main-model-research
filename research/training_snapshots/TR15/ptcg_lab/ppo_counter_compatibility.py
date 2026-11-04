"""Exact source delta review; never certifies new archive behavior by old tests."""
from pathlib import Path
from .io import read_json, sha256


def compatible_source(before, after):
    edits = [
        ('from .training_control import TrainingControl,ControlledTrainingStop\n',
         'from .training_control import TrainingControl,ControlledTrainingStop\nfrom .ppo_accounting import ACCOUNTING,checkpoint_decisions,update_decisions\n'),
        ('version=0;trained=0;series_id=0;', 'version=0;trained=0;unique_trained=0;series_id=0;'),
        ("series_id=saved['next_series_id'];updates=saved['updates']\n",
         "series_id=saved['next_series_id'];updates=saved['updates']\n        unique_trained=checkpoint_decisions(saved)\n"),
        ("'pending_decisions':len(decisions),'next_series_id':series_id,",
         "'unique_PPO_trained_decisions':unique_trained,'decision_accounting':ACCOUNTING,\n            'pending_decisions':len(decisions),'next_series_id':series_id,"),
        ("value={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),\n",
         "value={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),\n               'unique_PPO_trained_decisions':unique_trained,'decision_accounting':ACCOUNTING,\n"),
        ("while trained<config['target_decisions']:", "while unique_trained<config['target_decisions']:"),
        ("config['target_decisions']-trained)", "config['target_decisions']-unique_trained)"),
        ('            updates.append(row);trained+=len(decisions);',
         "            row['unique_PPO_trained_decisions']=update_decisions(row)\n            unique_trained+=row['unique_PPO_trained_decisions']\n            updates.append(row);trained+=len(decisions);"),
        ("'updates':version,'completed_series':", "'unique_PPO_trained_decisions':unique_trained,'decision_accounting':ACCOUNTING,\n            'updates':version,'completed_series':"),
    ]
    expected = before
    for old, new in edits:
        if expected.count(old) != 1:
            raise ValueError('Original approved writer source differs from exact reviewed delta')
        expected = expected.replace(old, new)
    if expected != after:
        raise ValueError('Writer change extends beyond unique-decision accounting')
    return True


def validate(root, evidence, approved_before_sha256):
    root = Path(root).resolve()
    path = (root/evidence['path']).resolve()
    if root not in path.parents or sha256(path) != evidence['sha256']:
        raise ValueError('Counter compatibility evidence changed')
    proof = read_json(path)
    before = (root/proof['before_source']['path']).resolve()
    after = root/'ptcg_lab/train_ppo.py'
    if (proof.get('scope') != 'TR10_original_second_exact_counter_only_delta'
            or root not in before.parents or sha256(before) != approved_before_sha256
            or proof['before_source']['sha256'] != approved_before_sha256
            or sha256(after) != proof['after_source_sha256']):
        raise ValueError('Exact reviewed archive writer lineage differs')
    compatible_source(before.read_text(encoding='utf-8'), after.read_text(encoding='utf-8'))
    audit=root/proof['actual_cold_recovery_audit']['path']
    if root not in audit.resolve().parents or sha256(audit)!=proof['actual_cold_recovery_audit']['sha256'] or not read_json(audit).get('complete'):
        raise ValueError('Actual full recovery proof required')
    return proof
