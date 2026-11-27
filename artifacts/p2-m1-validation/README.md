# P2-M1 只读回顾性验证

本包记录特锐德（300001.SZ）四个已冻结模型在 P1-M5 原 20 个已开启留出交易日上的新度量回放。不是新留出实验，不重新选择 P1 模型，也不修改既有 P1 封存记录。

- `registration.json`：固定模型/源码/权重 SHA256、数据缓存 SHA256、全部日期、设备、随机数、统计与运行配置。
- `report.json`：四模型 K=1 及 M0/M3 K=4/8 的聚合测量、2,000 次日级配对重抽统计和运行计时。
- `legacy-compatibility.json`：真实市场默认 P1/K=1 与 M0a 数值等价的聚合核验。

完整回放曲线缓存、原始行情和训练权重均保留在库外；本包仅含聚合与来源证据。

## 复现

需使用登记的两仓代码版本及已经拥有的 P1-M5 本地重构行情、CPU 权重缓存，不下载新数据，不调用 LLM API。以下四个环境路径由研究者设定：`STOCHOPIA_REPO`、`MARKET_ROOT`、`P1_MARKETS_CACHE`（M5 运行目录的 markets/）、`M1_PRIVATE_WORK`（独立的库外目录）。

从 OpenHyra 代码目录运行：

```sh
PYTHONPATH="$STOCHOPIA_REPO:$PWD" python scripts/verify_p2_m1_szse_300001.py \
  --market-root "$MARKET_ROOT" --p1-cache-root "$P1_MARKETS_CACHE" \
  --stochopia-root "$STOCHOPIA_REPO" --work-root "$M1_PRIVATE_WORK" \
  --artifact-root artifacts/p2-m1-validation --jobs 2 --batch-windows 2 --timeout-s 7200

PYTHONPATH="$STOCHOPIA_REPO:$PWD" python -m experiments.p2_m1_report \
  artifacts/p2-m1-validation/report.json \
  "$STOCHOPIA_REPO/docs/research/P2-M1-Measurement-and-Variance-Report.md"
```

P2 使用显式 `p2-m1` 和 `rollout_protocol_v2`；旧 P1/K=1 保持原协议。不同 K 使用同一地址化随机流的前缀，每日始终是唯一独立重抽单位。连续可用报价面板与严格 C1–C5 评分面板分开保存，缺失报价从不当作零。

[正式报告](https://github.com/MrSteeeve/Stochopia/blob/main/docs/research/P2-M1-Measurement-and-Variance-Report.md)
