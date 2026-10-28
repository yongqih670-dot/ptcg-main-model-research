# 公开来源审查（2026-10-23）

目的：引入真实对手与可运行基线，不以标题中的 900+/1300+ 当成本项目实测强度。所有 Notebook 仅解析源码，未执行其安装、删除、打包或提交单元。

| 来源 | 血缘与当前处理 |
|---|---|
| matterhorn3838/pokemon-challenge-ultra-900 | 与下载的 yinghualengyue/ptcg-strong-agent-build 单元源码完全相同；归于同一 Dipam PPO 血缘 |
| yinghualengyue/ptcg-strong-agent-build | 不作为额外独立策略计数；其中嵌入的其他作者策略改从原作者 Notebook 获取 |
| cristobaltudela/ptcg-ai-battle-playground-new-cards-900 | Dipam / Team Unown Gradiant 的 PPO 模型及 Playground 兼容修改；同血缘，不计第三个来源 |
| cggonzalezsc/blast-radius-mega-lucario-agent | 官方路卡利欧示例的扩展规则和内嵌升前场树模型；原始 main.py 原样提取；训练时是否启用可选 oracle 模式未得到独立证明 |
| woominyo/ptcg-alakazam-dusknoir-ready-to-run-agent | 胡地／黑夜魔灵规则策略；原文声明独立实现并列出机制参考；源码和牌组与作者声明 SHA256 一致 |
| masamikobayashi/a-sample-archaludon-75-wr-vs-my-1300-starmie | 铝钢桥龙／闪焰王牌规则策略；原始源码与牌组原样提取 |
| masamikobayashi/prize-card-tracking-1300-starmie | 本次下载仅含机制文章，没有可运行策略，不能算已接入对手 |
| jazivxt/rising-tide-fleet-agents | 仅下载打包入口并查询数据文件列表；尚未审查底层代码，不能算已接入对手 |

三套规则代码来源和哈希见 extraction-manifest.json。各自拥有不同主战牌组和规则实现；路卡利欧继承官方示例，另外两者虽参考公开机制文章，未发现是该代码的复制。独立来源是实现血缘判断，并非作者之间从未交流的证明。

本地运行适配：使用官方 SDK 的 dataclass 和转换函数，只提供静态公开卡表；不加载第二份 native engine，不提供搜索、battle pointer 或 hidden state。各座位分别加载模块，在 BO3 新一局重建策略状态；固定牌组文件，保留原始源码。路卡利欧仅允许 policy 模式，禁止 QCOL_RIVDECK、QCOL_OUT、Q_STATS_FILE 环境开关。铝钢桥龙原有异常随机回退会单独计数，不能把合法回退当成策略正确性证明。

牌组通过 60 张、卡 ID、同名非基础能量最多 4 张、ACE SPEC 最多 1 张、至少一只基础宝可梦检查；官方引擎另行验证。静态结果见 deck-audit.json。

三份规则 Notebook 明示可复制运行，但 CLI metadata 未提供明确许可证名称；目前用于本地对手研究，若作为最终可再分发包的一部分，仍需核实适用条款并保留署名。PPO 数据集另有明确 MIT LICENSE（原作者 Dipam Chakraborty），代码与权重合计约 20 MB，原始包保存于 ppo/。

PPO 源码审查：cfg=None 走纯 policy；actor `_core` 不读取 oracle block，oracle critic 是另一个分支。后续必须做数值上的 oracle block 扰动不变性测试，并显式 `weights_only=True` 加载权重。尚未运行本地 PPO 前向或完整对局，不能声称已通过。原入口只在整个 episode 重置 tracker，BO3 局间重置由本项目适配器补齐；不得重写 current.yourIndex。

本项目的 actor 训练与评测只接受决策者可见 observation。完整官方回放的 visualize 部分包含双方隐藏牌：只能作受限的赛后诊断，禁止直接作为 actor 特征或监督目标。
