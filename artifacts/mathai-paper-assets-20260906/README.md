# Math-AI 论文图表与数据包

状态：已完成。图中英文标签用于论文；本页说明数据来源与结论边界。

## 数据分组

- 原始搜索：2026-09-05/06 的真实模型记录，3 seeds、两轮、24 次生成尝试，5 次超时。
- 原始隐藏审计补全：七类经典基线使用原来的隐藏请求重评；属于回顾性补充。
- 新校准面板：冻结程序统一计算 gap，供反馈对照使用；不属于最终测试。
- 新独立测试：四个实例、512 训练路径、2048 定价路径、256 outer × 16 inner；全部选择冻结后才执行。
- 新反馈对照：相同历史、父程序、模型、操作符与预算上限；结构化反馈 vs 只提供分数。3 seed × 2 操作符，共 6 对。是固定历史的反馈消融，不是新增多轮 RSI。

## 阅读规则

核心数值为上下界间隙。gap score 是归一化置信间隙的均值 + 0.25 × 90 分位数；不是价格相对误差。每个图保留底层 CSV。跨 seed 点/SD 只有 3 个独立区组；路径级置信区间不能解释为跨任务统计显著性。失败保留在尝试与成本表中。

原始搜索过程图上方是当时真实使用的公开收益评分；下方是对同一公开选中 incumbent 的事后 gap 重评。不能将其称作原实验的 gap 反馈曲线。初始种子策略始终作为 incumbent 候选保留。

[Excel 数据总表](data/paper_data.xlsx) · [逐种子结果](data/main_results_by_seed.csv) · [汇总表](tables/main_results.md) · [实验预登记](protocol.json)

## 新独立测试主表

| label                            |   lower_normalized_mean |   upper_normalized_mean |   score_mean |   score_sd |
|:---------------------------------|------------------------:|------------------------:|-------------:|-----------:|
| Ridge                            |                 0.15105 |                 0.25537 |      0.18527 |    0.09656 |
| PCA + Ridge                      |                 0.15124 |                 0.25537 |      0.18504 |    0.09621 |
| Gated Ridge                      |                 0.15034 |                 0.25537 |      0.18601 |    0.09597 |
| MLP                              |                 0.14170 |                 0.25537 |      0.19560 |    0.09292 |
| Ridge + MLP                      |                 0.15075 |                 0.25537 |      0.18570 |    0.09567 |
| Threshold policy                 |                 0.05858 |                 0.25537 |      0.28960 |    0.07331 |
| Policy iteration                 |                 0.15124 |                 0.25537 |      0.18504 |    0.09621 |
| Original Context: best generated |                 0.14988 |                 0.25537 |      0.18656 |    0.09578 |
| Original direct: best generated  |                 0.15014 |                 0.25537 |      0.18637 |    0.09549 |
| Original Context incumbent       |                 0.15054 |                 0.25537 |      0.18590 |    0.09517 |
| Original direct incumbent        |                 0.15041 |                 0.25537 |      0.18604 |    0.09606 |
| Structured-memory incumbent      |                 0.15105 |                 0.25537 |      0.18527 |    0.09656 |
| Score-only incumbent             |                 0.15068 |                 0.25537 |      0.18565 |    0.09590 |

## 实验记录与使用方式

首次运行有 10 个已完成的 Context 计划在候选工作目录准备阶段遇到只读文件错误，另有 2 次 Context 超时。原始记录全部保留；恢复说明在 recovery_amendment.json。恢复只继续尚未发生的 Proposal 调用，没有重新生成 Context、修改候选或更换种子。

主表同时给出原始搜索中最好的已生成算法，以及允许保留初始参考算法的最终 incumbent；后者更接近实际可部署的搜索结果。新反馈组按校准 gap 选取 incumbent，独立测试只做报告。

[指标定义与文献](METHODS.md) · [字段字典](data/DICTIONARY.md) · [成本与失败记录](data/cost_and_reliability.csv) · [恢复说明](recovery_amendment.json) · [核验结果](verification.json)

每幅图均提供 PDF、可编辑 SVG 和 300 dpi PNG。HTML 图册可直接查看；Excel 保留逐方法、逐 seed、逐实例数据。重建图表使用已冻结的 inputs 与 raw 数据，不需要再次调用模型。图中的架构描述以实际运行流程为准；没有将示意图当作实验结果。

[本轮实验结论](FINDINGS.md)

## 图表

### fig01_pipeline

![fig01_pipeline](/Users/williamqian/GitHub/OpenHyra/artifacts/mathai-paper-assets-20260906/figures/fig01_pipeline.png)

Executed original pipeline. Public payoff LCB drives search. Candidate selection freezes before independent primal–dual evaluation; hidden gap is not fed back. The upper-bound construction is shared across Python policies.

[PDF](figures/fig01_pipeline.pdf) · [SVG](figures/fig01_pipeline.svg)

### fig02_feedback_design

![fig02_feedback_design](/Users/williamqian/GitHub/OpenHyra/artifacts/mathai-paper-assets-20260906/figures/fig02_feedback_design.png)

New fixed-history, fixed-parent feedback ablation. Both arms allow one Context and one Proposal call per attempt, the same model, operator, parent and evaluator budget; a failed Context prevents a Proposal call. Only memory contents differ. Ten frozen Context plans continue after a documented pre-generation workspace recovery; two Context timeouts are retained. This isolates feedback representation, not complete architecture or recursive self-improvement.

[PDF](figures/fig02_feedback_design.pdf) · [SVG](figures/fig02_feedback_design.svg)

### fig03_bounds_and_results

![fig03_bounds_and_results](/Users/williamqian/GitHub/OpenHyra/artifacts/mathai-paper-assets-20260906/figures/fig03_bounds_and_results.png)

Fresh holdout comparison. Left: raw normalized lower and upper estimates with component ±1.96 SE bars on the preselected seed-17 / hidden-000 case. They are conditional Monte Carlo intervals, not population uncertainty. Right: gap-score differences from Ridge under matched seed/instance budgets; each point is one seed, no unsupported significance claim. New ablation incumbents are selected solely by calibration scores with the parent retained as an option.

[PDF](figures/fig03_bounds_and_results.pdf) · [SVG](figures/fig03_bounds_and_results.svg)

### fig04_search_trajectory

![fig04_search_trajectory](/Users/williamqian/GitHub/OpenHyra/artifacts/mathai-paper-assets-20260906/figures/fig04_search_trajectory.png)

Original two-round search, four generation attempts per seed and pipeline. Lines include the initial best reference and follow public-score selection; dots are individual generated candidates, crosses are generation failures. Bottom panels independently re-evaluate those same incumbents on the supplemental calibration panel. These are post-hoc gap diagnostics, not evidence that the original search consumed gap feedback. Model time includes Context calls and timeouts, not parallel elapsed time.

[PDF](figures/fig04_search_trajectory.pdf) · [SVG](figures/fig04_search_trajectory.svg)

### fig05_feedback_ablation

![fig05_feedback_ablation](/Users/williamqian/GitHub/OpenHyra/artifacts/mathai-paper-assets-20260906/figures/fig05_feedback_ablation.png)

Matched feedback ablation with three seed blocks and two prescribed operators. Left: mean proposal gap reduction within seed across both operators; generation failures retain the parent (zero reduction) and remain in the denominator. Right: fresh-holdout reduction of the calibration-selected incumbent, allowing retention of the initial parent; small horizontal offsets distinguish overlapping seeds. Lines pair the two feedback arms within a seed. No claim of statistical superiority from three blocks.

[PDF](figures/fig05_feedback_ablation.pdf) · [SVG](figures/fig05_feedback_ablation.svg)

### fig06_stopping_behavior

![fig06_stopping_behavior](/Users/williamqian/GitHub/OpenHyra/artifacts/mathai-paper-assets-20260906/figures/fig06_stopping_behavior.png)

Observed stopping-time distributions of original frozen winners at seed 17, averaged across two repeats on the original public panel. The last date includes all remaining paths at maturity. This is a behavior illustration, not proof of why a mechanism worked or failed.

[PDF](figures/fig06_stopping_behavior.pdf) · [SVG](figures/fig06_stopping_behavior.svg)

### fig07_cost_and_reliability

![fig07_cost_and_reliability](/Users/williamqian/GitHub/OpenHyra/artifacts/mathai-paper-assets-20260906/figures/fig07_cost_and_reliability.png)

Actual generation reliability and model-call duration, including failed and timed-out calls. Hatched segments are failures. Original pipelines and the new feedback experiment are separate experiments, not matched against each other. New arms share call/time caps; realized calls, tokens and duration can differ. Context calls from the infrastructure-interrupted run are counted once; recovery issues only previously uncalled Proposals. Missing token reports are explicitly retained in the cost table.

[PDF](figures/fig07_cost_and_reliability.pdf) · [SVG](figures/fig07_cost_and_reliability.svg)

