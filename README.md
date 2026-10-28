# PTCG 精选研究材料

公开仓库保留训练与评测源码、冻结配置、实验结果、依赖版本、来源许可和必要复现说明。2026-10-29 按所有者要求停止完整训练数据与恢复归档的上传及重试，并删除原数据 Release 的全部附件。

**不提供完整训练数据、原始对局回放、模型权重、optimizer/RNG/checkpoint 或恢复归档下载。** 原始数据继续保留在所有者本地及服务器；本次整理不修改训练作业。源码和科学结果采用 2026-10-28 的固定快照，历史运行描述不是实时状态。

当前科学结论：TR11 证据不足，TR12 尚无最终结论；champion / release anchor 仍为 H002。训练量、工程恢复或局部微基准均不能代替独立强度证据。

## 下载与阅读

- [下载精选源码 ZIP](https://github.com/yongqih670-dot/ptcg-main-model-research/archive/refs/heads/main.zip)；也可使用 `git clone https://github.com/yongqih670-dot/ptcg-main-model-research.git`。
- [复现说明](docs/REPRODUCE.md)：环境、配置、研究流程及复现边界。
- [实验结果](docs/RESULTS.md)：真实结论、统计门槛和局限。
- [材料范围](docs/DATA.md)：保留与未公开材料、旧下载入口撤回情况。
- [来源与许可](docs/THIRD_PARTY.md)：第三方代码、牌组、卡表的来源。
- [清理收据](manifests/publication-scope.json)及[文件哈希清单](manifests/curated-files.json)。

`ptcg_lab/` 保留生产模块；`scripts/` 保留通用研究、导出、统计与验收入口；`configs/` 保留训练/评测参数；`agents/` 保留适配代码、牌组、版本规格和来源；`tests/` 保留相应测试源码。`research/framework_optimization/` 为未部署的工程候选，不能视为 TR12 已使用的优化。

原完整数据下载命令和分包链接已撤回；不要使用旧的 `snapshot-20261028` 数据下载说明。本仓库没有活动训练票据，历史配置和预算记录不构成在其他机器启动训练的授权。
