# PTCG 精选研究材料

> **2026-11-09 新框架状态：尚未正式启用。** 最新隔离候选 v24 为 `NOT_READY`：开发检查已有通过项，唯一正式 CPU 验收失败；B 四案的 **15% 性能门槛尚未测得**。[优化结果、验收边界与时间费用](research/framework_optimization/tr17_candidate_20261106/README.md)。历史 v3 的 29.09% 不代表本候选已达标。本次仅公开结果摘要。

公开仓库保留精选源码、冻结配置、实验结果、来源许可和离线统计复现材料。**最新已公开科学结果为TR16 v107终点评测（2026-11-04）。TR16 训练已停止，不再续训。** 终点累计839,120、本阶段新增431,968个有效唯一决策、53次完整PPO更新；4,174个未优化决策保留但不计入训练量。原新增600,000目标未完成，缺168,032；该终点在评测前登记，没有按胜率选择。

**TR16主评测增益+1.6875个百分点，95% CI [-1.7500, +5.1250]，未达预设门槛，结论为inconclusive；八对手面板增益+6.1250个百分点，95% CI [+2.1250, +10.1250]，仅支持该开发面板上的改善。TR14、TR15同为inconclusive，没有更新champion。** TR13此前通过学习信号门槛；独立最终池与其他晋级条件仍未完成，champion保持H002。

| 阶段 | 新增有效唯一决策 | 主模型累计 | 固定对手终点得分 | 相对父版增益（百分点）及95% CI | 八对手面板终点得分 |
|---|---:|---:|---:|---|---:|
| TR13 | 100,022 | 206,957 | 60.500% | +5.8125 [ +2.3750, +9.3125 ] | 72.250% |
| TR14 | 100,080 | 307,037 | 60.000% | −0.5000 [ −3.9375, +2.8750 ] | 72.125% |
| TR15 | 100,115 | 407,152 | 60.625% | +0.6250 [ −2.7500, +4.0000 ] | 68.875% |
| TR16 v107 | 431,968 | 839,120 | 62.3125% | +1.6875 [ −1.7500, +5.1250 ] | 75.000% |

主比较每策略1600完整BO3、席位800/800；八对手面板每策略800BO3、每对手100、席位50/50。预注册主门槛为增益至少5个百分点且95% CI下界大于0，20,000次完整BO3非配对bootstrap。TR14、TR15和TR16各新增2450个终点评测BO3，均正常结束；父版2400个既有BO3复用。固定主对手与当前或继承训练历史重叠，开发面板不是独立最终池，不直接代表Kaggle天梯胜率或泛化达标。

TR15改为最新冻结自身策略作对手，完整PPO更新后刷新；新增训练量和对手分布同时变化，不能分离其因果贡献。TR16继续沿用TR15冻结自身对手刷新机制，进一步增加训练量。TR13–TR16是同一连续谱系，不是多个独立训练种子。历史截至TR15的全分支汇总见旧报告；当前TR16主线为839,120个有效唯一决策/107次更新，不能把分支总量当作主模型训练量。

## 报告与下载

- **[TR16 v107最终评测与分析报告](docs/TR16_FINAL_REPORT.md)** · [机器可读结果](research/autonomy/TR16-result.json) · [终点登记](research/autonomy/TR16_REGISTERED_ENDPOINT.json) · [精简结局](evidence/TR16/)。

- **[TR15最终报告](docs/TR15_FINAL_REPORT.md)** · [机器可读结果](research/autonomy/TR15-result.json) · [冻结自博弈配置](research/autonomy/TR15-training-frozen.json) · [精简结局](evidence/TR15/)。
- **[TR14最终报告](docs/TR14_FINAL_REPORT.md)** · [机器可读结果](research/autonomy/TR14-result.json) · [固定对手配置](research/autonomy/TR14-training-frozen.json) · [精简结局](evidence/TR14/)。
- [TR13最终报告](docs/TR13_FINAL_REPORT.md) · [TR13已完成的平台试提交记录](docs/TR13_KAGGLE_TRIAL.md) · [TR12历史最终报告](docs/TR12_FINAL_REPORT.md)。
- [TR15实际冻结源码、测试与执行入口](research/training_snapshots/TR15/README.md) · [TR15实际框架与恢复验收](docs/TR15_FRAMEWORK_ACCEPTANCE.md)。
- [v3原始工程候选及本地CPU验收](research/framework_optimization/v3_tr13/README.md) · [原工程报告](research/framework_optimization/v3_tr13/REPORT.md)。原29.09%耗时改善仅对应该CPU固定回放。
- [实验结论](docs/RESULTS.md) · [复现说明](docs/REPRODUCE.md) · [公开范围](docs/DATA.md) · [来源与许可](docs/THIRD_PARTY.md)。
- [下载当前精选源码与报告ZIP](https://github.com/yongqih670-dot/ptcg-main-model-research/archive/refs/heads/main.zip)，或 `git clone https://github.com/yongqih670-dot/ptcg-main-model-research.git`。
- [文件SHA-256清单](manifests/curated-files.json) · [TR16来源清单](manifests/tr16-publication.json) · [TR14/TR15历史来源清单](manifests/tr14-tr15-publication.json) · [发布范围](manifests/publication-scope.json)。

```powershell
python -B tools/verify_curated.py
python -B tools/verify_tr16_results.py
python -B tools/verify_tr14_tr15_results.py
python -B tools/verify_tr15_snapshot.py
```

统计工具只需NumPy，仅重算公开结局，不下载数据、不加载模型、不启动对战或训练。根目录生产代码保留历史基础快照；TR15实际实现位于独立冻结目录，二者的版本边界见复现说明。

**完整训练数据、原始回放、权重、checkpoint、optimizer/RNG和恢复归档不上传GitHub。** 原件继续保留本地和服务器；旧数据Release及附件已撤回，完整数据上传和自动重试继续禁用。历史实现、配置、验收及预算记录不构成启动训练的授权。
