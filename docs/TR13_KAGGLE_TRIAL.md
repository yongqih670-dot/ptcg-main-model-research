# TR13 Kaggle 平台试提交

**已上传并获 Kaggle 接收，提交编号57825976。** 提交时间2026-10-31 12:09（北京时间）。这是用户另行授权的一次平台试提交，不代表研究晋级；champion仍为H002，训练保持停止。

- [Kaggle比赛提交页](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/submissions)
- 标记：`TR13-v28-206957-20752e097d54`
- 模型：version28，累计206,957个有效唯一决策。
- 模型SHA-256：`c30773b4c390ffa17719e99002aac81449b21368a3b0d84839d2107b782a238b`
- 提交包SHA-256：`20752e097d54be4dabef77576f21395065b7c598d0f12eade04c0ce668a2b2c7`，19,034,804字节。

提交前私有Kaggle Linux CPU验收通过：2核亲和性、4场完整BO3、875次动作与冻结源策略一致；峰值RSS702,664,704字节（约670MiB），动作耗时p99约0.200秒，最大单次约4.726秒（含首次模型加载）。测试固定官方`kaggle-environments==1.33.2`。Notebook预装引擎观测缺少`current.round`，初次对局验收因此失败；固定引擎版本后通过，同一模型与提交包哈希未变。Notebook与实际比赛容器并非同一环境，仍需平台自测。

最新查询尚无平台验证或正式对局记录，评分待产生。上传成功不等于平台自测或排行榜强度验证通过；没有追加第二次提交。后续平台状态应以比赛页面为准。

[提交收据](../evidence/TR13/kaggle-trial-receipt.json) · [Linux部署验收](../evidence/TR13/linux-deployment-probe.json) · [TR13科学结果](TR13_FINAL_REPORT.md)

公开仓库仅同步精选结果与必要复现材料；推理权重未上传GitHub。完整训练数据、轨迹、optimizer/RNG/checkpoint和恢复归档继续保留本地与服务器。
