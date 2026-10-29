# PTCG 精选研究材料

公开仓库保留精选源码、冻结配置、实验结果、依赖版本、来源许可和统计复现材料。基础代码快照为 2026-10-28；本次于 2026-10-29 同步 TR12 最终评测与分析。历史运行记录不代表实时状态。

**TR12 已按用户决定提前停止训练，完整状态恢复及 5,650 BO3 评测已完成，结论为证据不足（inconclusive）。** 终点固定为最后提交的 version15，未按胜率选择 checkpoint。champion / release anchor 仍为 H002，未晋级或提交。

| 项目 | TR12 提前停止终点的实际结果 |
|---|---|
| 训练量 | 新增 **57,762** 条唯一首 epoch 成功优化决策 / 7 更新，为原计划 320,768 条的 **18.01%**；继承后累计 106,935 条。原训练量目标未完成。 |
| 固定对手主比较 | TR11 **50.625%** → 终点 **54.6875%**；增益 **+4.0625 个百分点**，95% CI **[+0.625, +7.5625]**。CI 为正，但未达原定增益至少 5 点的门槛。每臂 1,600 完整 BO3，各 800/800 席位。 |
| 八对手开发面板 | 两者均 **67.125%**；增益 **0.00 点**，95% CI **[−4.25, +4.25]**。没有整体提升证据；描述性比较未作多重比较校正。 |
| 恢复与可靠性 | 两个源断点及 CPU/GPU 完整状态恢复通过，6,807 条未优化尾部保留；50 BO3 可靠性检查及全部评测运行检查通过。历史严格 600 秒持久化间隔未获认证，详见报告。 |

固定主对手与训练对手重叠，开发池也不是未触碰最终池。native 引擎未暴露 seed，不宣称逐局配对复现；训练量和工程恢复不能代替独立强度证据。

## 报告与下载

- **[TR12 最终评测与分析报告](docs/TR12_FINAL_REPORT.md)**：逐对手结果、训练健康、实际耗时、停止根因、恢复限制及费用。
- [TR12 机器可读结果](research/autonomy/TR12-result.json)、[提前停止协议修订](research/autonomy/TR12_EARLY_STOP_PROTOCOL.json)、[精选结果来源清单](manifests/tr12-final-publication.json)。
- [精简 BO3 结局表](evidence/TR12/)及[离线统计重算工具](tools/verify_tr12_results.py)：仅包含系列编号、席位、胜平负及运行检查，不含对局动作或回放。
- [全部实验结论](docs/RESULTS.md)、[复现说明](docs/REPRODUCE.md)、[公开材料范围](docs/DATA.md)、[来源与许可](docs/THIRD_PARTY.md)。
- [下载当前精选源码与报告 ZIP](https://github.com/yongqih670-dot/ptcg-main-model-research/archive/refs/heads/main.zip)，或 `git clone https://github.com/yongqih670-dot/ptcg-main-model-research.git`。
- [文件 SHA-256 清单](manifests/curated-files.json)及[发布范围收据](manifests/publication-scope.json)。

```powershell
python -B tools/verify_curated.py
python -B tools/verify_tr12_results.py
```

第二条命令需要 NumPy，只重算已公开结局的统计，不下载数据、不加载模型、不启动对战或训练。

**完整训练数据、原始回放、模型权重、checkpoint、optimizer/RNG 及恢复归档不上传。** 原件继续保留在所有者本地和服务器；完整数据上传及重试保持禁用，旧数据 Release 与附件已撤回。TR12 不再续训；本仓库中的历史实现、配置与预算记录不构成启动训练的授权。`research/framework_optimization/` 中的候选未部署于 TR12。
