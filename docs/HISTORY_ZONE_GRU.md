# 历史联赛、分区读出与 GRU 开发分支

分支：`codex/history-zone-gru`。这是基于当前主线的可选研究实现，未改变冻结的 TR12/TR13、framework v3 候选或 champion，未启动服务器任务。

## 三项实现

1. **历史 checkpoint 联赛**：沿用 `League` 的不可变快照和逐场哈希核验；新增严格混合权重检查、历史采样窗口及无历史时回退到最近冻结策略。示例为最近冻结策略 80%、历史 20%，历史均匀采样最近 8 个非当前快照。一场完整 BO3 内对手身份保持不变。窗口只限制采样，不删除任何恢复或归档依赖。
2. **分区多头读出**：在继承 Transformer 输出上，分别读取己方场上、对方场上、己方手牌、公开/可推断资源区。使用现有 zone 和 owner 编码；不引入对手真实手牌、牌库顺序或隐藏奖赏牌。空区域用零哨兵避免全 mask 的 attention。两个动作指针继续对应原实体。
3. **GRU**：每个玩家每小局独立维护状态，在每个非强制决策的首次选择更新一次；后续多选及 overflow chunk 共享本次记忆，强制选择不推进。Tracker 继续记录强制事件，小局结束重置，不跨 BO3 共享隐状态。它是决策时钟的可学习摘要，不替代精确牌数和持续效果追踪。

Actor/value 新残差的最终输出层均为零初始化，初始 logits/value 与父模型一致；已有主干、卡牌/招式表示不修改。新分支学习后可利用区域和记忆。所有开关由 `model_extensions` 显式指定，原配置不指定时保持旧路径。

## 训练及恢复语义

GRU 模式不会随机打散小局内决策：只打乱完整小局的顺序，按决策预算打包；一场长小局可以超过 nominal minibatch，不截断、不沿用过期 hidden state。每次 PPO forward 从零记忆按当前权重完整重放，跨决策完整反传，多选 log probability 仍按原条件选择求和。小局奖励和 GAE 保持原设置。

每次更新前必须复算完整合法动作分布、选中 logp 和 value；额外保存 `old_full_logp`。拒绝残缺、重复、打乱、混合版本的小局。完整序列进入首 epoch 的成功覆盖量沿用原计数，重复 epoch 不重复增加训练量。完整小局过大时可能 OOM；缩小 nominal minibatch 不保证容纳单场，不能静默改成截断训练。

新结构从匹配父 `model.pt` 继承权重（保留 value 输出），**创建新的 Adam 和联赛**，不是原 TR13 完整状态的无缝续训；没有实现把旧 Adam 迁移到新增参数的功能。需要记录新研究血缘。扩展模型自身 checkpoint 保存模型选项、Adam、RNG、league/cursor/pending，后续按相同配置恢复；禁止借 resume 改结构。CPU 采样副本继承全部新增 tensor，构造不推进采样 RNG。进行中的 BO3 不作为恢复状态；仅完整轨迹提交。

历史模型每场从新 Tracker 开始，记忆不写在共享模型实例上。league manifest 绑定模型语义，防止相同 tensor 形状但不同 attention heads 的模型混用。

## 配置与入口

`configs/development/history-zone-gru.json` 是 overlay，不是可直接开训的完整票据；训练器会拒绝直接执行它。将选定 ablation、league 设置并入新的完整配置，再冻结父权重哈希、源码、预算和评测。三组建议：

| 组 | model_extensions | 对手 |
|---|---|---|
| history_only | 不设置 | 80% latest / 20% history |
| history_zone | zone_readout=true，memory_size=0，heads=4 | 同上 |
| history_zone_gru | zone_readout=true，memory_size=128，heads=4 | 同上 |

先选一项变量做对照，不将三项同时开启后的成绩称为单项收益。原生引擎、私有权重、文本向量、服务器票据不在本次开发中生成或获取。

实现已接入 `train_ppo`、CPU replica、冻结 league、更新前 parity，以及候选导出/部署验证打包。扩展候选使用 `kind=zone_memory_rl`，通过 `ZoneMemoryAgent` 显式重置并携带记忆，不能交给旧的无状态 adapter。打包不代表允许上传比赛或证明 CPU 时限通过。

## 本地验证与局限

运行 `python -B -m unittest tests.test_policy_extensions tests.test_critic_gradient_route tests.test_ppo_mapping tests.test_gradient_diagnostic tests.test_journal_policy_version tests.test_phase_timing -v`。

新测试使用实际继承模型代码、公开元数据和合成的零文本通道/局面，不加载私有权重，不执行 native 对局。覆盖初始行为精确一致、空区/隐私/非法序列、条件多选与 65 候选、跨决策梯度、全分布复算、整局 minibatch、历史加载/篡改拒绝、CPU RNG、PPO 更新与下一步精确恢复、实际训练循环的合成 BO3 采集/提交/恢复，以及隔离进程的部署包模型加载。

旧 `test_ppo_accounting.test_writer_delta_rejects_any_change_to_archive_or_checkpoint_logic` 依赖未公开的 `runs/...` 私有历史源码，在本公开克隆中因缺文件无法执行；未改断言或伪造历史材料。旧公开清单记录冻结发布的原字节，开发改动不能通过修改旧清单冒充原验收。

尚未完成真实 CABT、CUDA、原始文本向量/父权重兼容、真实部署预算及强度验证；本地通过不代表提高胜率。新增 GRU 模型不能直接使用旧非序列 pending 尾部。正式实验前必须在新研究中进行相应验收。
