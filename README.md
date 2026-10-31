# PTCG 精选研究材料

公开仓库保留精选源码、冻结配置、实验结果、依赖版本、来源许可和统计复现材料。基础代码快照为 2026-10-28；最新于 2026-10-31 同步 TR13 最终结果，保留 TR12 历史报告。历史运行记录不代表实时状态。

**TR13 已完成：version15 → version28，新增100,022条有效唯一决策，学习信号门槛通过。训练已停止；独立最终池尚未验证，champion仍为H002。**

| 项目 | TR13 最终结果 |
|---|---|
| 训练量 | 新增100,022条 / 13次更新；累计206,957条；完整更新越过目标22条 |
| 固定对手，各1600 BO3 | 54.6875% → **60.50%**；+5.8125个百分点，95% CI [+2.375,+9.3125] |
| 8对手面板，各800 BO3 | 67.125% → **72.25%**；+5.125个百分点，95% CI [+1.125,+9.125] |
| 可靠性 | 全部2450场终点评测无错误、超时、非法动作或内部随机回退 |

主对手与训练对手重叠，开发面板不是最终池；不能据此宣称泛化达标。用户另行授权的一次平台试提交见[提交状态](docs/TR13_KAGGLE_TRIAL.md)，不等同研究晋级。TR12历史结论仍为证据不足。

## 报告与下载

- **[TR13 最终报告](docs/TR13_FINAL_REPORT.md)**、[机器可读结果](research/autonomy/TR13-result.json)、[精简结局](evidence/TR13/)与[统计重算工具](tools/verify_tr13_results.py)。

- **[TR12 最终评测与分析报告](docs/TR12_FINAL_REPORT.md)**：逐对手结果、训练健康、实际耗时、停止根因、恢复限制及费用。
- [TR12 机器可读结果](research/autonomy/TR12-result.json)、[提前停止协议修订](research/autonomy/TR12_EARLY_STOP_PROTOCOL.json)、[精选结果来源清单](manifests/tr12-final-publication.json)。
- [精简 BO3 结局表](evidence/TR12/)及[离线统计重算工具](tools/verify_tr12_results.py)：仅包含系列编号、席位、胜平负及运行检查，不含对局动作或回放。
- [全部实验结论](docs/RESULTS.md)、[复现说明](docs/REPRODUCE.md)、[公开材料范围](docs/DATA.md)、[来源与许可](docs/THIRD_PARTY.md)。
- [下载当前精选源码与报告 ZIP](https://github.com/yongqih670-dot/ptcg-main-model-research/archive/refs/heads/main.zip)，或 `git clone https://github.com/yongqih670-dot/ptcg-main-model-research.git`。
- [文件 SHA-256 清单](manifests/curated-files.json)及[发布范围收据](manifests/publication-scope.json)。

```powershell
python -B tools/verify_curated.py
python -B tools/verify_tr12_results.py
python -B tools/verify_tr13_results.py
```

统计重算命令需要 NumPy，只重算已公开结局的统计，不下载数据、不加载模型、不启动对战或训练。

**完整训练数据、原始回放、模型权重、checkpoint、optimizer/RNG 及恢复归档不上传。** 原件继续保留在所有者本地和服务器；完整数据上传及重试保持禁用，旧数据 Release 与附件已撤回。TR12 不再续训；本仓库中的历史实现、配置与预算记录不构成启动训练的授权。`research/framework_optimization/` 中的候选未部署于 TR12。
