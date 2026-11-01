"""One count per scientific family, not per resumed/copied checkpoint or epoch."""
from pathlib import Path
from .io import read_json,sha256,atomic_json,utcnow


def build(root,output,tr09):
    root=Path(root);rows=[];state=read_json(root/'research/autonomy/state.json')
    completed={'TR01':'20261023T141824Z_tr01-blind-ppo-32k_25713f',
        'TR02':'20261023T144339Z_tr02-ppo-lr1e5_9a0e14','TR03':'20261023T151043Z_tr03-ppo-262k_a347a7'}
    def evidence(path):return {'path':path.relative_to(root).as_posix(),'sha256':sha256(path)}
    for hypothesis,rid in completed.items():
        source=root/'runs'/rid;config=read_json(source/'config.json');metrics=read_json(source/'metrics.json')
        if not metrics.get('complete') or config['parent_model_sha256']!='959b220daf6777f45ad5d6aa14c35d3274bf3958ed502c7551679453e5f4aaea':
            raise ValueError('Completed independent original H002 training family differs')
        updates=metrics['updates']
        if type(updates) is int:
            count=updates;details=source/'training-updates.json'
            if details.exists():
                journal=read_json(details)
                if len(journal)!=count or sum(u['decisions'] for u in journal)!=metrics['trained_decisions']:
                    raise ValueError('Completed scalar update count differs from its full journal')
        elif isinstance(updates,list):
            count=len(updates)
            if sum(u['decisions'] for u in updates)!=metrics['trained_decisions']:raise ValueError('Committed PPO family sum differs')
        else:raise ValueError('Unrecognized historical PPO update representation')
        rows.append({'hypothesis_id':hypothesis,'canonical_training_family':rid,'trained_unique_decisions':metrics['trained_decisions'],
            'PPO_updates':count,'training_target_completed':True,
            'scientific_status':state['research_stage']['hypotheses'][hypothesis]['status'],
            'evidence':[evidence(source/'metrics.json'),evidence(source/'config.json')]})
    for hypothesis in ['TR04','TR05','TR06','TR07','TR08']:
        path=root/'research/autonomy'/f'{hypothesis}-result.json';m=read_json(path);h=state['research_stage']['hypotheses'][hypothesis]
        if m.get('classification')!='invalid' or h['status']!='invalid' or h.get('remaining_original_training_attempts')!=0:
            raise ValueError('Old closed family status or attempt scope changed')
        # These are highest published family totals from their full forensics,
        # already including any inherited portion of the original second.
        # Do not add the two attempts' cumulative counters together.
        rows.append({'hypothesis_id':hypothesis,'canonical_training_family':h['run_ids'][0],
            'trained_unique_decisions':m['trained_decisions'],'PPO_updates':m.get('committed_PPO_updates',m.get('PPO_updates')),
            'training_target_completed':False,'scientific_status':'invalid','evidence':[evidence(path)],
            'invalid_does_not_erase_actual_updates_or_qualify_model':True,'attempt_totals_not_double_counted':True})
    rows.append({'hypothesis_id':'TR09','canonical_training_family':tr09['counting_family_root'],
        'trained_unique_decisions':tr09['unique_actual_PPO_training_decisions'],'PPO_updates':tr09['PPO_updates'],
        'complete_BO3':tr09['complete_BO3'],'individual_games':tr09['individual_completed_games'],
        'sampled_decisions':tr09['sampled_unique_decisions'],'pending_untrained_decisions':tr09['pending_decisions'],
        'scientific_status':'started_incomplete_target','training_target_completed':False,
        'evidence':[evidence(Path(output)/'analysis.json')]})
    total=sum(r['trained_unique_decisions'] for r in rows)
    result={'at':utcnow(),'scope':'TR01-TR09 actual committed unique selfplay decisions, one canonical family total each',
        'goal_unique_actual_PPO_training_decisions':10000000,'actual_unique_training_decisions':total,
        'remaining_unique_training_decisions':max(0,10000000-total),'families':rows,
        'excluded':'Engineering trials/342newCPUprobe samples, all pending/untrained data, model copies, restored prefixes counted again, repeated PPO epochs, external H002 pretraining with no local count.',
        'historical_total_games_and_sampled_decisions_not_filled_from_estimates':True,
        'known_PPO_updates_lower_bound':sum(r['PPO_updates'] or 0 for r in rows),
        'training_volume_is_not_model_strength_or_research_success':True,'promotion_eligible':False}
    atomic_json(Path(output)/'scale-ledger.json',result);return result
