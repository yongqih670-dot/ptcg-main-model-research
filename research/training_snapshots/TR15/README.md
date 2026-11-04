# TR15 冻结实现与复现边界

本目录来自2026-11-02成功TR15科学训练作业的不可变源码快照：完整 `ptcg_lab` 模块、对应测试以及10个通用/连续训练执行入口。包含TR13–TR15新增的继承、PPO事务、冻结自博弈、框架v2/v3、共享计时和完整状态归档实现；未复制一次性授权/激活脚本、任何原生引擎二进制或数据载荷。

目录保持原包结构，避免用旧仓库根代码冒充TR15实际执行代码。仓库根 `ptcg_lab/` 仍是历史基础快照，既有框架工程候选也保留其原版本。本目录中的服务器地址、账户和个人路径已替换成占位符；原始SHA与公开字节SHA均见根目录 `manifests/tr14-tr15-publication.json`，有替换的文件不宣称逐字节等同原运行代码。

## 科学设置

TR14沿用固定合法PPO对手；TR15使用最新冻结自身策略，每次完整PPO提交后刷新、整场BO3固定，既有FrozenOpponent采用greedy动作模式。PPO、牌组和精度保持，未启用AMP。配置见根目录 `research/autonomy/TR14-training-frozen.json` 和 `TR15-training-frozen.json`。原生环境seed未暴露，不能复现相同逐局抽牌流。

## 检查与使用

从仓库根执行 `python -B tools/verify_tr15_snapshot.py` 检查精选源码哈希与Python语法，执行 `python -B tools/verify_tr14_tr15_results.py` 重算公开统计。统计工具只依赖NumPy，不调用本目录的训练入口。

若研究训练实现，需要将本目录作为独立项目根，按自身资源与路径建立新policy/state/授权，合法取得官方CABT环境、公开上游材料和对应模型输入，再核验实际环境。原始模型、optimizer、全部RNG、league/cursor/pending和训练轨迹不公开；因此此目录不支持凭公开子集恢复原训练或精确重放。历史authority哈希和资源票据不授予运行权限。

`tests/` 含原始测试源码，一部分依赖未公开模型、原run、服务器或Windows环境，不能把输入缺失时的跳过视为通过。此次发布验证范围为文件完整性、语法和公开结局统计，不声称重跑了完整科学训练或私有恢复验收。
