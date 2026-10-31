# 复现说明

本仓库可用于审阅实现、参数、研究方法和已记录结果。完整训练数据、推理权重及恢复状态未公开，因此不能仅凭本仓库恢复原训练进程或逐字节重放全部实验。已公布的模型/数据哈希用于校验合法取得的对应输入。

## 环境与检查

原环境使用 Python 3.12。CPU 依赖版本记录在 `research/environment/requirements-cabt.lock`；CUDA 运行时须按对应服务器版本另行准备。原平台引擎、公开上游权重和其他第三方资料需按其来源与许可自行取得，不能以 Windows 原生依赖替代 Linux 依赖。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-deps -r research/environment/requirements-cabt.lock
.\.venv\Scripts\python.exe tools/verify_curated.py
```

依赖文件是历史环境锁定记录，不保证未来各平台仍有完全相同的 wheel。测试源码位于 `tests/`；涉及模型、原 run 或服务器的集成验收需要未公开输入，不能将缺失数据造成的跳过当成验收通过。

## 研究流程与输入

1. 以 `scripts/run_experiment.py` 为统一运行入口，在自己的授权资源中建立独立 policy/state/预算；参考 `configs/reproduction/policy.reference.json`，不要直接复用历史运行票据。
2. `configs/training/`、`configs/evaluation/`、`research/opponents/` 和 `research/autonomy/preregistrations/` 保留科学配置与预登记。服务器、账户、私钥路径和绝对路径均使用模板占位符，需要自行填写并重新执行身份、路径、容量和硬时限门禁。
3. 原训练由服务器完成 BO3 采样及 PPO，本机负责控制、完整归档、独立评测和部署验收。actor 只使用合法可见 observation；按完整 BO3 划分、评测和 bootstrap。
4. 精确续训必须由所有者持有的完整 actor/critic/optimizer、全部 RNG、league、cursor、pending 及两个可靠恢复点组成。本仓库不含这些恢复载荷，不授权从更早父状态替代正更新后的续训。
5. 完整归档须逐文件核对原始字节和压缩字节 SHA-256、解压验证，完整 ACK 且无恢复引用后才可释放原件。相关实现保留，但此次发布未执行任何原件释放。

## 结果及边界

TR11 未通过预注册提升门槛。TR12 已主动提前停止训练并完成冻结评测：固定对手有小幅正向信号但未达 +5 点门槛，八对手平均持平，整体证据不足，不晋级。详见 [最终报告](TR12_FINAL_REPORT.md)。原训练量目标未完成。固定 TR12 终点评测协议见 `research/autonomy/activations/TR12-volume-320768-20261028/TR12_PROTOCOL.json`；native 引擎未暴露 seed，不宣称配对复现。不能追加样本、修改对手权重或选择中间 checkpoint 来改变结论。

历史统计摘要位于 `evidence/runs/`，结论在 `docs/RESULTS.md`。这只包含选定的统计报告，没有原始对局、训练轨迹或隐藏真值。完整输入由所有者本地及服务器保留，未提供公开下载。

## TR12 统计复现（不启动训练或对战）

提前停止的最后提交终点在独立评测前冻结，见 `research/autonomy/TR12_EARLY_STOP_PROTOCOL.json`。`configs/TR12/early-stop-*.json` 仅为冻结科学输入的参考投影，不是活动运行票据。

公开的 `evidence/TR12/*-outcomes.jsonl` 保留全体 5,650 个评测 BO3 的精简结局，原始回放、训练轨迹和模型权重不在其中。安装原依赖锁中对应的 NumPy 后，可执行：

```powershell
python -B tools/verify_curated.py
python -B tools/verify_tr12_results.py
```

第二条命令检查数量、系列身份、席位和运行标志，并使用公开的 `ptcg_lab/volume_statistics.py` 重算固定的 20,000 次独立完整 BO3 bootstrap，核对主比较和全部开发面板结果。它不访问网络、不加载模型，也不调用训练或对战入口。统计可由公开结局表重算，不等于仅凭公开子集重放 native 对局或恢复训练。

完整训练数据及恢复归档继续保留在所有者本地和服务器，上传与重试已取消；TR12 不再续训。


TR13离线统计重算：`python -B tools/verify_tr13_results.py`。依赖NumPy，输入仅公开精简BO3结局。
