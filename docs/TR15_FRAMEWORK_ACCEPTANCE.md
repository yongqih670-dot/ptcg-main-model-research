# TR15 框架与实际恢复验收

2026-11-01公开的v3工程候选，最初只完成本地CPU同输入固定回放验收，中位端到端耗时减少29.09%。该数值继续只代表该固定CPU工程比较，不代表GPU训练提速或棋力增益。

TR15随后在实际服务器完成native、自博弈快照刷新、精确CUDA/optimizer/全部RNG与中断恢复差异验收后才进入科学训练。使用已冻结v3开关；`capacity_transaction=False`，未启用AMP。对应实现已补入[TR15冻结源码](../research/training_snapshots/TR15/README.md)，公开的[原始工程候选](../research/framework_optimization/v3_tr13/README.md)保持独立历史版本。

精选证据：[native验收摘要](../evidence/TR15/native-acceptance-summary.json)、[已验收源码哈希](../evidence/TR15/accepted-source-hashes.json)、[终点完整恢复摘要](../evidence/TR15/recovery-acceptance-summary.json)。TR15科学终点为v54、407152有效唯一决策、pending0，13次新增完整更新；最长可靠完成提交间隔约781.16秒，满足该阶段800秒上限。完整恢复图、文件ACK与载荷继续保留本地/服务器，公开摘要不能代替这些输入。

工程启动曾出现计时记录重复初始化、存储预留接口与容量预测不匹配等失败，修复后复用未变验收，仅补必要差异。原失败及成本保留。科学终点评测仍为证据不足，不将工程通过等同模型晋级。后续TR16的计时、连接与依赖修订不在本次TR15冻结实现内。
