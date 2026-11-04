"""Frozen complete-BO3 TR15 inference; no promotion or extra sample loop."""
from collections import Counter
from pathlib import Path
from .io import read_json,read_series,atomic_json,sha256,utcnow
from .volume_statistics import bootstrap_cells


def finish(root,proof,evaluations):
    import numpy as np
    root=Path(root);frozen=read_json(root/'research/autonomy/TR15_EVALUATION_FROZEN.json')
    for name in ('primary_reference','development_reference'):
        for key in ('rows','config'):
            ref=frozen[name][key]
            if sha256(root/ref['path'])!=ref['sha256']:raise ValueError('Frozen parent BO3 evidence changed')
    def checked(path,count):
        rows=read_series(path)
        if len(rows)!=count or len({r['series_id'] for r in rows})!=count:raise ValueError('Incomplete or duplicated frozen evaluation')
        if any(r['statuses']!=['DONE','DONE'] or r['result'] not in ('win','loss','draw') or r['error']
                or r['policy_invalid_count'] or r['internal_random_fallback_count'] for r in rows):
            raise ValueError('Endpoint evaluation contains invalid BO3 or fallbacks')
        return rows
    reliability=checked(root/'runs'/evaluations['reliability']/'series.jsonl',50)
    new_primary=checked(root/'runs'/evaluations['primary']/'series.jsonl',1600)
    new_panel=checked(root/'runs'/evaluations['development']/'series.jsonl',800)
    old_primary=[r for r in read_series(root/frozen['primary_reference']['rows']['path']) if r['cell_id']=='endpoint-vs-ppo']
    old_panel=[r for r in read_series(root/frozen['development_reference']['rows']['path']) if r['cell_id'].startswith('endpoint-vs-')]
    def scores(rows):return [1. if r['result']=='win' else .5 if r['result']=='draw' else 0. for r in rows]
    for rows in (old_primary,new_primary):
        if Counter(r['candidate_seat'] for r in rows)!=Counter({0:800,1:800}):raise ValueError('Primary seat balance changed')
    groups={'parent':scores(old_primary),'endpoint':scores(new_primary)}
    means,samples=bootstrap_cells(groups,{'parent':1600,'endpoint':1600},frozen['bootstrap_seed'])
    gain=(sum(groups['endpoint'])-sum(groups['parent']))/1600;ci=np.quantile(samples['endpoint']-samples['parent'],[.025,.975]).tolist()
    passed=gain>=.05 and ci[0]>0
    panel_groups={};opponents=[]
    for cell in frozen['endpoint_development_cells']:
        name=cell['cell_id'];opponents.append(name)
        for label,rows in [('parent',old_panel),('endpoint',new_panel)]:
            selected=[r for r in rows if r['cell_id']==name]
            if len(selected)!=100 or Counter(r['candidate_seat'] for r in selected)!=Counter({0:50,1:50}):
                raise ValueError('Panel sample or seat balance changed')
            panel_groups[label+'|'+name]=scores(selected)
    panel_means,panel_samples=bootstrap_cells(panel_groups,{k:100 for k in panel_groups},frozen['bootstrap_seed'])
    parent_mean=sum(panel_means['parent|'+n] for n in opponents)/8
    endpoint_mean=sum(panel_means['endpoint|'+n] for n in opponents)/8
    draws=sum((panel_samples['endpoint|'+n]-panel_samples['parent|'+n])/8 for n in opponents)
    result={'at':utcnow(),'phase':'TR15','complete':True,'classification':'improved_learning_signal' if passed else 'regressed' if ci[1]<0 else 'inconclusive',
        'actual_additional_unique_decisions':proof['new_unique'],'actual_lineage_unique_decisions':proof['actual_unique'],
        'target_overshoot':proof['actual_unique']-407037,'new_PPO_updates':proof['new_PPO_updates'],
        'inherited_pending_first_epoch_coverage':proof['inherited_pending_first_epoch_coverage'],
        'new_sampling_first_epoch_coverage':proof['new_sampling_first_epoch_coverage'],
        'primary':{'parent_score':means['parent'],'endpoint_score':means['endpoint'],'gain':gain,'ci95':ci,
            'BO3_per_strategy':1600,'seats_per_strategy':[800,800],'learning_signal_passed':passed,
            'required_gain':.05,'required_CI_lower_above':0,'bootstrap_samples':20000,'bootstrap_seed':frozen['bootstrap_seed']},
        'development':{'parent_score':parent_mean,'endpoint_score':endpoint_mean,'gain':endpoint_mean-parent_mean,
            'ci95':np.quantile(draws,[.025,.975]).tolist(),'BO3_per_strategy':800,'weights':{n:1/8 for n in opponents},
            'cell_scores':panel_means,'descriptive_unadjusted_multiple_comparisons':True},
        'reliability':{'complete_BO3':len(reliability),'errors':0,'invalid_actions':0,'internal_fallbacks':0},
        'evaluation_runs':evaluations,'new_endpoint_BO3':2450,'parent_BO3_reused':2400,
        'limitations':['Training amount and latest-frozen-self opponent schedule changed together; no fixed-opponent training control, so their causal contributions cannot be separated.',
            'The primary evaluation opponent overlaps earlier inherited training history; TR15 itself trains against frozen self only. No general-strength claim.',
            'Native environment seed is not exposed; independent unpaired complete-BO3 bootstrap, no paired reproducibility claim.',
            'Eight-opponent development results are descriptive; final pool was not accessed.'],
        'promotion_eligible':False,'champion_changed':False,'submission_made':False,'no_further_training':True,
        'full_training_and_recovery_data_retained_local_server_only':True}
    path=root/'research/autonomy/TR15-result.json';atomic_json(path,result)
    report=root/'research/autonomy/TR15_ANALYSIS.md'
    report.write_text(
        '# TR15 固定训练量终点评测\n\n'
        f"结论：{result['classification']}。新增有效唯一决策 {proof['new_unique']:,}，累计谱系 {proof['actual_unique']:,}；完整更新边界超量 {result['target_overshoot']:,}。\n\n"
        f"继承 pending 中实际首 epoch 成功覆盖 {proof['inherited_pending_first_epoch_coverage']:,}，新增采样中实际成功覆盖 {proof['new_sampling_first_epoch_coverage']:,}。重复 epoch、未优化数据和未提交部分更新均未计入。\n\n"
        f"固定对手：父版 {means['parent']:.4%}，终点 {means['endpoint']:.4%}，差值 {gain:+.4%}，95% CI [{ci[0]:+.4%}, {ci[1]:+.4%}]；预定 +5 个百分点且 CI 下界大于0的门槛{'通过' if passed else '未通过'}。各1600完整BO3，席位800/800；20000次非配对完整BO3 bootstrap。\n\n"
        f"八对手面板：父版 {parent_mean:.4%}，终点 {endpoint_mean:.4%}，差值 {endpoint_mean-parent_mean:+.4%}；各800 BO3、每对手100、席位50/50。多对比为描述性结果。50 BO3可靠性检查无错误、非法动作或内部回退。\n\n"
        '本阶段同时增加训练量并改用每次完整更新后刷新的冻结自身策略，缺少固定对手训练对照，无法分离两者贡献。主评测对手与继承训练历史重叠；不能据此宣称泛化提升。native 环境未暴露 seed，不宣称配对复现。未窥视最终池，未择优checkpoint，未追加样本，未自动晋级或提交。\n\n'
        '终点已停止训练；两可靠恢复点和完整原字节哈希归档均已核验，完整训练与恢复数据继续仅保留本地和服务器。\n\n'
        '[详细结果](TR15-result.json) · [冻结协议](TR15_PROTOCOL.json) · [冻结评测输入](TR15_EVALUATION_FROZEN.json)\n',encoding='utf-8')
    state=read_json(root/'research/autonomy/state.json')
    if state.get('active_job'):raise ValueError('Cannot finalize while a job is active')
    phase=state['forward_phases']['TR15'];phase.update(status='complete',completed_at=utcnow(),no_further_training=True,
        classification=result['classification'],result={'path':path.relative_to(root).as_posix(),'sha256':sha256(path)},evaluation_runs=evaluations)
    previous_streak=state.get('consecutive_valid_runs_without_promotion',0)
    state['consecutive_valid_runs_without_promotion']=previous_streak+1
    phase.update(completed_valid_independent_hypotheses=1,
                 global_nonpromotion_counter_before=previous_streak,global_nonpromotion_counter_after=previous_streak+1)
    state.update(status='completed_requested_stage',current_phase='TR15_complete_training_stopped',updated_at=utcnow(),
        next_action='Requested independent TR15 stage complete; retain all data and do not continue training.')
    atomic_json(root/'research/autonomy/state.json',state)
    ledger_path=root/'research/autonomy/CURRENT_VOLUME.json';ledger=read_json(ledger_path)
    for family in ledger['families']:
        if family['hypothesis_id']=='TR15':
            family.update(scientific_status=result['classification'],independent_evaluation_result=phase['result'])
    atomic_json(ledger_path,ledger)
    policy_path=root/'research/autonomy/policy.json';policy=read_json(policy_path)
    cadence=policy['execution']['framework_v2_checkpoint_cadence']
    cadence.update(formal_training_driver_integrated=True,
        actual_native_GPU_acceptance=phase['native_GPU_acceptance'],actual_GPU_cursor_difference=phase['GPU_cursor_difference'],
        applies_to='prospective framework v2 component packet, assessment, and the hash-bound TR15 native route')
    policy['policy_version']=utcnow()[:10]+'-v9-TR15-complete'
    atomic_json(policy_path,policy)
    continuation=read_json(root/'research/autonomy/latest-continuation.json')
    continuation.update(at=utcnow(),current_phase=state['current_phase'],active_run_id=None,
        TR15=phase,next_action=state['next_action'],budget_snapshot=state['budget_used'])
    atomic_json(root/'research/autonomy/latest-continuation.json',continuation)
