# 框架优化候选交付（2026-10-28）

本目录是独立工程候选。现用训练源码、配置、研究 policy/state/账本、TR12 路由均未修改。没有启动服务器任务、科学 PPO、子代理、Goal 或定时任务。源文件、输入证据、环境与交付哈希在 reports/ 中。

## 已完成及证据边界

| 模块 | 可运行交付 | 验证状态 | 当前决定 |
|---|---|---|---|
| M0 | 历史日志分析、重叠区间并集计时、30 秒可关闭 Python profiler | 已分析真实 TR11 日志；有本地采样开销测量 | 可用于后续受控验收，历史缺失字段仍为 unknown |
| M1 | 有界单消费者传输队列、完整状态冻结、原子写入方的可选硬链接发布 | CPU 故障/恢复测试及本地文件微基准通过 | 默认关闭，服务器存储/限速路径待验收 |
| M2 | 原始候选索引预整理、FP64 条件概率与 entropy 向量化、独立 PPO 适配器 | 真实 48 决策前向/梯度和合成单步 optimizer/KL 对照通过 | 候选保留；GPU 与端到端待验收 |
| M3 | 1/2/3 环境协调器、独立 sampler RNG、同版本推理合批、完整 BO3 pending/cursor | 模拟引擎合同测试通过；复用已有条件采样原型 | native 多环境运行与生产 journal 桥接未验收，不接 TR12 |
| M4 | FP32 / BF16 / FP16+GradScaler，默认 FP32，scaler 状态恢复 | CPU 默认路径通过；已提供受控 GPU 回放 payload | AMP 无 CUDA 验收数据，保持关闭 |

可运行不等于已获得端到端收益，也不等于已获生产接入许可。

## 真实结果

完整数值见 reports/baseline.json、packed_benchmark.json、publication_benchmark.json、real_gradient_parity.json、tests.json。

- TR11 作业墙钟 10172.641 秒，新增实际唯一成功优化决策 16391，速率 **1.6113 决策/作业秒**。采集 969.876 秒、更新 46.016 秒；两计时器之外 9156.749 秒（90.013%），不能归因于单一环节。
- 稀疏观察包含资源暂停。一个完整屏障观察耗时 327.559 秒，其中 ACK 等待 0.0287 秒、最后一次保存 327.513 秒；保存内部的序列化、写入、hash、状态验证份额仍未知。不会把这个局部屏障推广为整个 run 的归档占比。
- 当前版本在真实 48 决策、2 CPU torch 线程、7 次交错顺序测量中：基线前向与概率汇总中位 **130.35 ms**，向量化含整理 **123.79 ms**，冻结布局复用 **123.89 ms**。约 **5% 局部耗时下降**。首轮探索测量也保留为 packed_benchmark_initial.json；两轮波动说明缓存额外收益未获证明。
- 真实 fixture 的 log-prob、value、entropy 最大误差均为 0；梯度最大绝对差为 1.3411e-7（预设梯度 atol=2e-6 / rtol=1e-5）。log-prob 1e-4、value 2e-4 门槛未调整。实际 actor 未进行本地 PPO 更新；optimizer 更新对照只在小型合成模型上进行。
- 8 MiB 本地文件发布、保留完整哈希的 7 次测量：复制 40.71 ms，硬链接候选 28.04 ms。包含独立输出验证；不包含序列化、服务器限速、传输、采样或 PPO。不能用约 31.1% 的局部耗时下降预测训练改善。
- profiler 的短窗口会扰动性能；本次被分析前向的中位约增加 9%，样本仅 3 次。默认关闭，不用于给候选计算未经校正的加速比。

## 本地复现

从项目根执行，使用既有项目环境，不安装依赖：

```powershell
& .venv/Scripts/python.exe -B research/framework_optimization/benchmarks/analyze_logs.py --run runs/20261027T160313Z_tr11-authorized-second-posix-fixed_1eae55 --output research/framework_optimization/reports/baseline.json
& .venv/Scripts/python.exe -B research/framework_optimization/benchmarks/verify_delivery.py
& .venv/Scripts/python.exe -B research/framework_optimization/benchmarks/bench_packed.py --repeats 7
& .venv/Scripts/python.exe -B research/framework_optimization/benchmarks/bench_publication.py
```

验证脚本先检查主线源码 SHA；漂移时失败，不能覆盖主线来让测试通过。测试包含：130 原始候选、唯一 STOP、混合子步、梯度/optimizer/KL；半包/断连/丢 ACK/重复 ACK/背压；原子替换后两恢复点；storage alias/stride/offset/模型 metadata；完整 BO3/未完成排除/worker RNG/cursor/pending/策略屏障。另复用生产的 checkpoint、archive、systemd 身份与唯一覆盖计数测试。详细测试名在 reports/tests.txt。

## 独立开关和回退

`source/fo/backend.py` 中 Options 默认 vectorized=False、precision='fp32'、archive_queue=False、immutable_hardlink=False、workers=1。没有 import hook、sys.modules 替换或默认训练路由变更。

- 数值候选：显式 `step(Options(vectorized=True), device='cpu')` 返回隔离适配函数；关闭 vectorized 返回原始 ppo_step。minibatch、epoch、随机排列和 KL 逻辑由原调用者保留。`evaluate_packed` 只接受该冻结快照，不同时接收可能错配的 decisions 列表。
- 数据缓存只在调用方明确冻结的输入快照内复用；没有跨更新的 GPU rollout cache。Pinned memory、non_blocking、双缓冲、TF32、compile、fused optimizer 均未启用。
- 归档队列仅搬运已在生产 outbox 登记的不可变对象，采用既有传输/完整 ACK 回调，最大 8 对象/512 MiB，单消费者。ACK 在持有者线程验证并接受；不提供文件删除功能。传输回调必须遵守原限速、取消与 deadline，外层进程硬限时仍必需。进程重启从原 outbox 的未 ACK 项重建；不依赖内存队列恢复。
- 状态冻结必须位于训练写入安全点，复制完整 CPU 存储且保留别名、偏移、stride 和 metadata；不能让后台序列化引用仍被 optimizer 改写的 tensor。此功能有额外内存/复制成本，未宣称提速。
- 硬链接只用于原子替换写入方及同文件系统。候选保留所有完整 hash、原始 ACK/引用计数/两恢复点/新鲜删除前校验。原地写入会破坏同 inode 的归档；因此必须验证所有写入方使用 atomic_torch_save/atomic_json 后才可启用。不支持链接时回退原 paced copy。
- 多环境按同一冻结版本收集波次，只有引擎 IPC read 等待在线程中并行，网络前向合批，opponent 与 RNG 管理由主线程执行。没有 actor/learner 异步训练。各 BO3 完整输出后按 series_id 保存，未完成 native BO3 标记排除；不能序列化 native 的中途环境。上限 540 秒的波次须配合调用者检查点与清理预算，以保留 600 秒要求。
- Coordinator 的 state_dict 必须作为完整 checkpoint 的新字段，与 actor/critic/optimizer/全部 RNG/league 一起原子提交；pending 的完整轨迹与 RNG 写入 journal 并 ACK 后才能 acknowledge_pending；advance_policy 还要求匹配的完整 BO3 和生产 first-epoch 覆盖日志。当前只交付接口，未改生产 journal schema。多环境 RNG 分配不能作为与旧单环境完全等价的续训，TR12 不启用。
- AMP 仅作为独立数值候选。先建立 packed FP32 对照，再单独比较 BF16/FP16。概率、ratio 等保持高精度，FP32 master weights 不变。FP16 的 scaler 必须纳入 checkpoint；缺失或模式变更应拒绝恢复。未通过 log-prob/value、loss、梯度、clip、KL/early-stop 和恢复验收即拒绝，不能放宽门槛。

回退在作业终态和完整身份核验后执行：关闭相应开关；有正更新时仅从最新完整状态恢复，不从父状态重开。AMP/M3 的 schema 若已产生新状态，不能丢弃 scaler/worker RNG 强行改旧 schema，须先在安全节点做明确可逆转换验收。当前候选未接入，主线无需回退操作。

## 唯一研究控制器的待验收工作

当前没有本线程的工程票据/预算预留，也未新增调度链。现行 dispatcher 尚无本候选 payload 的准入分支。研究控制器须先接入其既有受控工程入口，绑定下列输入、明确预算、单 GPU 及所有权/存储/实时共享门禁后才运行。

1. 优先 600 秒 GPU 工程票据：3 CPU/32 GiB，单张启动空闲的 GPU4–7（优先6/7），零科学量；验证 packed FP32 与 AMP 的固定轨迹输出、单步、梯度、scaler/RNG 恢复和同步 GPU 计时。所有准备/归档/恢复计入票据，预留不足则缩短测量窗口，绝不延长原 deadline。
2. 900 秒 CPU 工程票据：相同 IO 限速和 HDD2 文件系统，测量序列化/完整 hash/发布/旋转/ACK 分项；注入中断、进程退出、源原子替换、跨文件系统回退。先独立验证硬链接，不把队列、限速等一起改变。队列性能暂未证明。
3. 1800 秒 CPU 工程票据：native 1/2/3 环境，每项 300 秒窗口和剩余准备/恢复预算；全体进程共用 3 CPU/32 GiB（实时每 GPU 份额更低时缩减）。比较相同输入面板与合批；native 无 seed，不声称逐局配对。完成 journal 桥接、两恢复点和中途退出验收前，不接科学续训。
4. 以上通过后另由控制器在已授权且完整身份核验后的安全节点安排端到端比较；维持同一 PPO/对手/样本计数。中间 checkpoint 不择优评测，工程吞吐不能作为棋力改进。当前无法确认 >=20% 端到端目标。

这些是所需上界，合计至多 GPU 0.1667 作业小时、CPU 0.75 作业小时；尚未预留、未使用，不修改权威账本，也不占受保护最终预算。每项都受剩余预算与原绝对 deadline 的更紧限制。

GPU 回放入口（仅在上述现行控制器准入完成、真实 ticket 与 manifest 已绑定后，作为其子 payload 执行）：

```bash
.venv/bin/python -B research/framework_optimization/benchmarks/hardware_acceptance.py --ticket "$PTCG_ENGINEERING_TICKET"
.venv/bin/python -B research/framework_optimization/benchmarks/cpu_acceptance.py --ticket "$PTCG_ENGINEERING_TICKET" --kind publication
.venv/bin/python -B research/framework_optimization/benchmarks/cpu_acceptance.py --ticket "$PTCG_ENGINEERING_TICKET" --kind native
```

这些命令分别属于三张票据，不可并发或在同张票据跨 CPU/GPU 执行。它们不是直接 SSH 启动命令；不能手工伪造 PTCG_JOB 环境绕过控制器。payload 校验 Linux/HDD2 所有权、job/lease、input manifest SHA、显式 engineering 标志、设备、deadline，并调用现行 TrainingControl。manifest 中 framework_optimization_acceptance 需绑定输入；ticket 必须有 framework_optimization_approved=true。字段见 reports/acceptance_requirements.json，它不是有效票据。CPU 驱动支持真实文件系统发布及 native 1/2/3 环境波次，保留采样输出且零优化量；native 回放快照最多每配置 8 波，完整原件在外层归档 ACK 前不删除。Native 记录的是包含创建进程/保存/同进程恢复的采样速率，不是完整 PPO 吞吐。生产 journal 桥接、跨进程恢复和外层 ACK 的集成仍须控制器完成准入验收。

## 科学与工程未决项

原控制器退出根因仍未知。本次只读记录证明服务终态/归档恢复结果，缺少控制器退出证据，未做猜测修复。历史 90% 时间缺口仍需完整阶段日志；硬链接无法单独解释或消除 327 秒保存。没有端到端收益、GPU AMP 合格或 native 多环境恢复成功的结论。

本机实际 API 为 PyTorch 2.11.0+cpu；未升级。AMP API 已与 [PyTorch 2.11 官方文档](https://docs.pytorch.org/docs/2.11/amp.html)核对。后续 CUDA 活动与 CPU 提交时间须分开，参见 [PyTorch Profiler 官方说明](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。
