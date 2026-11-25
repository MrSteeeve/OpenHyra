# Stochopia × OpenHyra P1-M4 实验证据

90 次开发搜索与 30 项留出复验已封存。每条轨迹的首次、末次和审计任务见 `study_manifest.json`。候选源码的 SHA-256 对应 `sources/<sha256>.py`。只有沙箱通过的候选才可进入复验。同一候选出现在多个登记角色时共用一次配对结果。

`stochopia-p1-m4-v1` 在完成 40/90 次开发试验后因协议字段与指标极性歧义中止。当时留出集保持封存，账本记有 42 次 API 请求、74,560 个输入 token 和 26,650 个输出 token。原登记见 `development_plan_v1_aborted.json`，原账本外置于 `p1-m4-v1-aborted-20261124/`。修复与冒烟核验后独立登记 `stochopia-p1-m4-v2`。

API 冒烟原批次通过两组，9 次尝试共 14,145 个 token。补测只覆盖 `evidence_gpt` 和 `metrics_ds`，4 次尝试共 6,626 个 token，两组均通过。随后锁码。

- `smoke_gate.json` 保留两批 API 冒烟和 `no_llm` 配对回放的通过记录。
- `baselines/` 保存两只股票的开发种子汇总与种子定义。
- `development-trials.csv` 汇总 90 次候选的编译、预算、护栏和双预测结果。
- `api-attempts.csv` 按请求记录 token、时长及 HTTP 重试。服务端未报告用量的单元格为空。
- `compiler-attempts.csv` 记录匹配源码与修复错误摘要。
- `verification/report.json` 与 `verification/results/` 保存冻结任务复验和断点记录。报告的 `audit_only_rejected_unverified` 是单独的证伪审计集。
- `verification/test-data-reconciliation.json` 记录测试日重构与窗口盘口核对。
- `holdout-endpoints.csv`、`rejected-audit.csv` 和 `cross-regime.csv` 汇总正式复验结果及跨股矩阵。备用架构在表中标为 `auxiliary`。

## 复现入口

Stochopia 锁码提交为 `400443b5039ea1b073bbc9a1156cac079ad15e3f`。OpenHyra 锁码提交为 `66533dae928e0a64eda8232147c92fda4326e32e`。完整树标识、数据版本、日期和配对种子随清单登记。公开回执保留逐日对账、窗口和价差证据，交易所原始序号留在库外原始回执。公开诊断路径已匿名化。原始行情、含实际请求消息和原始响应的工作账本、训练权重与逐笔轨迹保存在仓库外。

从锁码版本、两仓的 Python 环境和获准的库外行情目录运行以下命令。重现开发期需准备原冒烟回执、模型配置与计数器，使用独立运行目录。`$MARKET` 下需有 M2 的 `fetch_fast_m2.py`。基于现有冻结证据继续复验时，从获取与复验两步开始。开始下载留出交易日前，先完成开发搜索并确认封存清单。复验中断时沿原目录恢复。

```bash
export PYTHONPATH="$OPENHYRA:$STOCHOPIA"
"$STOCHOPIA/.venv/bin/python" -m tasks.stochopia_p1.formal dev \
  --artifact-root "$ARTIFACTS" --market-root "$MARKET" --cache-root "$RUN" --tokenizers-dir "$TOKENIZERS"
"$STOCHOPIA/.venv/bin/python" -m tasks.stochopia_p1.formal freeze \
  --artifact-root "$ARTIFACTS" --market-root "$MARKET" --cache-root "$RUN"
"$STOCHOPIA/.venv/bin/python" -m tasks.stochopia_p1.holdout_data \
  --artifact-root "$ARTIFACTS" --market-root "$MARKET"
"$STOCHOPIA/.venv/bin/python" -m tasks.stochopia_p1.formal verify \
  --artifact-root "$ARTIFACTS" --market-root "$MARKET" --cache-root "$RUN"
```

[正式实验报告](https://github.com/MrSteeeve/Stochopia/blob/main/docs/research/P1-M4-Formal-Study-Report.md)

## V2 资源失败补测审计

原始入选清单与已完成测量保留。补测登记在 `verification/retry-v2-resource/registration.json`，重跑前原始报告及失败结果保存在该目录的 `before/`，独立补测结果保存在 `results/`，完成回执为 `completed.json`。

原始测量任务 20 项，仅重跑 10 项已封存失败任务。恢复 9 项，仍失败 1 项。补测后测量任务 29 项。训练单种子墙钟上限 180 秒，评估墙钟上限 1,800 秒，累计 CPU 秒上限为墙钟上限的四倍。本次仅修复资源预算，沿用原 V2 的统计判据。

补测代码提交为 Stochopia `58336c39a8cb6c8aca8e442f6bfee00a9f78cbdc` 和 OpenHyra `c71cacfb7d4ff74995cea0dcaa89638a05959e36`，冻结码和初选决策见原清单。

### 同资源串行续测

首批并发补测中墙钟超时的 1 项随后在同一代码、同一 180/1800 秒资源配置下以单 worker 补测。两次尝试均保存在 `verification/retry-v2-resource/serial/` 与原补测目录。最终完成测量 30 项。
