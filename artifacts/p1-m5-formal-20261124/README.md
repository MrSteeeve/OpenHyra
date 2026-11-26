# Stochopia × OpenHyra P1-M5 实验证据

留出复验已落盘。 本轮单独登记 36 日训练集、24 日开发集、20 日新留出集及顺延日期。实际留出日由冻结后的对账回执确定。候选源码以 `sources/<sha256>.py` 保存。

- `development_plan.json` 记录锁码、数据与试验参数。
- `handshakes/` 记录两家服务的连接验证和用量。
- `development-trials.csv` 保存编译、预算、配对得分、单侧 p05/p10 下界与方向及幅度判定。
- `study_manifest.json` 保存全部开发记录、冻结选择、候选源码与复验清单。
- `verification/report.json` 与 `verification/results/` 保存独立复验及跨股重训。 `verification/test-data-reconciliation.json` 保存实测日期与顺延理由。
- `holdout-endpoints.csv`、`rejected-audit.csv`、`cross-regime.csv` 汇总复验端点、未证实涨分候选和迁移表现。

锁码 Stochopia `dd33659cd4d223b37cf564e17252954a7ddda54f`、OpenHyra `ce8e1b502007f4f07b21e6b2129cdfe98fbdad05`。

原始市场切片、训练权重、调用的原始响应和提示均留在库外。

## M5 收尾精简复验

`verification/closing-registration.json` 在留出集获取前冻结 A/B/C 入选名单及开发清单哈希；保留原始 `study_manifest.json`。`secondary-best-guarded.csv` 明确为冻结后次要审计，未测的 `X_1` 标记 `not_scheduled`。原始档案中登记的全量首轮和弱候选迁移不再执行。

复验按目标股票与源码 SHA256 去重；同一源码的不同直接父模型与事前预测仍独立判定。CSV 的 `delta_*` 为直接父模型配对差值，`seed_delta_*` 为同股 seed 配对差值。跨股矩阵的对角线使用 B 集同一原始轮廓与同股 seed 构造配对，不再重训；两个方向分别在目标股训练，权重不共享。

## 公开证据版本范围

公开版完整保留冻结清单与精简复验登记（原始字节及 SHA256 不变），以及所有选中角色的配对统计和按股票与源码计数的终态。逐笔原始交易所序号、单条行情最大偏差时刻（`max_deviation_time_ms`）及其他原始订单/成交标识从对账回执删除；完整原始回执留在库外。保留全部登记日期、有效会话、正价差、逐日事件及快照总量、盘口匹配率、深度误差与远端文件元数据。诊断中的本机路径匿名化，候选源码不改动。

[正式实验报告](https://github.com/MrSteeeve/Stochopia/blob/main/docs/research/P1-M5-High-Power-Study-Report.md)
