# Framework v3：独立候选与本地 CPU 验收

2026-11-01 同步。固定输入工程对照已完成，端到端中位耗时由 **5350.954 秒降至 3794.181 秒，减少 29.09%**，通过原定 15% 门槛及资源子门槛。完整分项、费用和限制见 [REPORT.md](REPORT.md) 与 [PERFORMANCE.json](reports/PERFORMANCE.json)。

这是独立保存的工程候选及真实 TR13 源码依赖快照。仓库根目录的历史训练实现与科学结果保留；本次同步未启用服务器 v3、未启动训练，也不改变 champion。验收范围为 Windows CPU 固定轨迹回放，不能据此推定 native／GPU／AMP 或实际服务器训练的收益。

## 优化与默认行为

- 控制文件、日志和归档队列使用身份校验后的缓存，减少重复读取及目录枚举；停止、错误、ACK 和检查点证据仍须及时检查。
- 合并短时间内的非关键进度写盘；关键事件立即发布。
- 补充进程退出证据，保留真实 Popen.wait 返回码与 PID／创建时间身份检查。
- 候选 API 的开关默认全部关闭。通过本地性能验收的配置见 [profile.local-accepted.json](profile.local-accepted.json)：control_cache、journal_cache、outbox_cache、progress_coalescing、exit_evidence 开启，**capacity_transaction 关闭**，原强制容量扫描保留。

容量事务组件及服务器零更新模板属于独立原型，未纳入本地通过配置。原 [server-zero-update.TEMPLATE.json](server-zero-update.TEMPLATE.json) 仅为未签发模板，含 capacity_transaction=true；它不能替代本地已通过配置或用于借用旧科学票据。

## 正确性和对照口径

四例按 baseline／candidate／candidate／baseline 顺序执行；每例复用相同 48 完整 BO3、8203 决策、原 PPO、RNG、两次 epoch、minibatch256、2 Torch 线程和 3 核 affinity。各例首 epoch 成功覆盖 8203／8203，均完成 66 次 PPO 更新；这些更新仅发生在一次性工程副本，科学新增量为零。

8 个终点冷恢复点精确匹配保存前状态；按 generation 跨四案完整逻辑状态一致，模型、优化器及全部 RNG 摘要一致。原资格验收还检查了实际 partial PPO 完整恢复与下一优化步严格一致。800 秒可靠提交完成间隔通过；内存、冷恢复耗时及保留包体积三项资源子门槛均通过。

主耗时包含完整工作器、独立两点冷恢复及无损归档保留。每臂仅两次重复，结果为描述性工程测量。首案 warm/cold 早于 TR14，其余步骤晚于 TR14；背景工作集和系统状态跨期变化。首个候选的未完成尝试因关机中断，经第6／7代零更新复核后重测；旧现场、期限与费用保留，未完成尝试不作为完整性能样本。

原首案 warm/cold 的120GiB容量事实保留；本次首案待完成归档和剩余三案经用户明确授权前向使用200GiB。冻结协议原字节未改。累计 CPU 保守记账11.638055小时、GPU为0，包含准备、失败、恢复、归档与交付核对上界，费用明细见 [SUPPORT_LEDGER.json](reports/SUPPORT_LEDGER.json) 与 [JOB_LEDGER.json](reports/JOB_LEDGER.json)。

## 文件与复核

- `candidate/`：验收候选完整文本依赖快照，包含新增组件；`baseline/`：对应冻结基线。
- `patches/candidate.diff`：相对该基线的候选差异；`SOURCE_PROVENANCE.json`：原文件 SHA-256。
- `benchmarks/`、`tests/`：原验收驱动与测试源码，保留原控制边界。
- `reports/`：冻结协议、选择的原验收结果、四例完整状态摘要、费用及测试记录。
- `PUBLICATION_FILES.json`：本次公开目录及离线复核入口的逐文件 SHA-256。

在仓库根目录运行：

```powershell
python -B tools/verify_framework_v3.py
python -B tools/test_framework_v3_public.py
```

第一条命令只使用 Python 标准库，核对公开文件字节、候选清单及结果算术，不读取模型、不启动引擎或训练。它验证公开收据的一致性，不能替代从原始私有状态重做完整验收。

第二条命令需要已有 CPU PyTorch、NumPy 和 psutil，运行原合成夹具，并明确跳过需要私有 actor／optimizer 权重的一项。此次同步的实际结果见 `reports/public-tests.json`；原本地完整38项验收结果另保留为 `reports/tests.json`。默认执行不重写这两份历史结果。

完整训练数据、轨迹、权重、checkpoint、optimizer／RNG 状态和恢复归档均未公开。公开子集不具备独立重做完整数值回放所需的输入；原重任务驱动及历史配置不是新的运行授权。来源与许可沿用仓库 [THIRD_PARTY.md](../../../docs/THIRD_PARTY.md)。
