# Bermudan 单任务真实模型实验报告

实验于 2026-09-05 至 2026-09-06 完成。真实 Context→Proposal 与直接 program generation 均完成三个 seed、两轮搜索；反馈、执行、训练、隐藏审计和导出包数值复现已经接通。实验没有支持 Context 具有统计优势，也没有证明发现了新算法。

## 运行与验证结果

- 48 个公开候选记录，43 次成功评估、5 次生成超时；包含 24 个 guided/control 对。失败记录全部保留。
- 四种操作符各实际调度 6 次。成功完成评估的 restart 为 4 次、mutation 为 3 次、crossover 为 6 次、subsystem rewrite 为 6 次。
- 7 个完整程序对照 × 3 个 seed，共 21 次基线评估全部成功；每次包含 4 个重新训练的 instance/repeat 单元。
- 6 次隐藏审计全部完成，6/6 lookahead probe 与 6/6 fresh-fit deterministic replay 均通过。
- 从迁移后的冻结包、全新输出目录重跑了 70 次评估：43 个公开成功候选、6 个隐藏审计、21 个基线。源码摘要、得分、模型文件摘要及可用的逐路径收益全部一致；隐藏审计中的独立 probes 也再次通过。
- 导出包重建核验共 438 项全部通过，包含独立重算配对均值/标准误与三个 Context 反馈表的逐字节哈希复原。
- 完整本地测试：530 passed，36 subtests passed。实验代码 PR 的 macOS / Python 3.11 与 3.12 检查均通过。

## 固定实验设计

seed 为 17、29、41。每个 pipeline 每个 seed 运行两轮，每轮两对候选。第一轮分配 whole_program_restart 与 ast_mutation，第二轮分配 ast_crossover 与 subsystem_rewrite。操作符类别预先排定；Context 在这些类别内生成机制、目标分片、预测、反驳条件和后续行动，并依据已完成结果选择 parent。它并未学习一个不受约束的操作符选择策略。

每对候选共享实际 parent、候选 seed、评估请求、训练/定价路径、30 秒 fit 上限与 180 秒 Proposal 上限。control 原样保留 parent，不调用生成模型。每个公开评估使用 2 个实例 × 2 次重复，每单元 256 条训练路径和 512 条独立定价路径。每个 mode/seed 按固定公开得分选出最佳成功 guided 候选；全部公开选择冻结后，才执行 2 个隐藏实例 × 1 次重复的审计，使用 128 条 outer paths、每步 4 条 inner paths。审计结果不回流 Context。

两条完整 pipeline 的 Proposal 上限和 evaluator 预算一致，Context 额外产生 6 次模型调用，而且下一轮 parent 可随证据改变；直接生成沿用初始最佳基线。因此这不是等总成本、且仅改变 Context 单一因素的因果消融。隐藏请求种子按公开协议确定，实例与评分结果在选择后评估；本实验不声称对能读取协议源码的模型提供密码学意义的秘密测试集。

## 得分与效果

公开得分越高越好，是固定测试单元上、候选相对于 evaluator 基线的 strike-normalized payoff improvement 的 95% LCB（均值 − 1.96 × SE）。隐藏审计得分越低越好，为 normalized primal–dual confidence gap 的均值 + 0.25 × 第 90 百分位。两个阶段指标不同，不能直接相减。

| Pipeline | Seed | 公开赢家 | 公开 LCB ↑ | 隐藏 gap score ↓ |
|---|---:|---|---:|---:|
| Context→Proposal | 17 | sol_0007 | -0.000748633 | 0.281527586 |
| Context→Proposal | 29 | sol_0005 | -0.000680710 | 0.476395844 |
| Context→Proposal | 41 | sol_0009 | -0.001606962 | 0.619354633 |
| 直接生成 | 17 | sol_0003 | -0.001144314 | 0.283475918 |
| 直接生成 | 29 | sol_0007 | -0.000281204 | 0.473320181 |
| 直接生成 | 41 | sol_0009 | -0.000380547 | 0.619919264 |

公开得分中直接生成在 2/3 个 seed 上更高；隐藏 gap 指标中 Context→Proposal 在 2/3 个 seed 上更低。仅三个 seed、两个实例的结果不足以判断哪条 pipeline 普遍更优。所有公开赢家的 LCB 都低于零，尚未建立相对于 evaluator 基线的显著正改进。隐藏上下界的观测顺序全部正常，但较宽的 gap 不能视为高精度定价认证。

预测兑现率以每条 pipeline 的全部 12 对为分母，包含执行失败。正方向命中均为 3/12（25%），达到 95% 区间严格大于零的支持率均为 0/12。Context 为 3 次反驳、7 次无定论、2 次执行失败；直接生成为 1 次反驳、8 次无定论、3 次执行失败。生成超时不构成对收益机制的统计反驳。

每对目标分片的 effect 为 guided − unchanged control 的 strike-normalized payoff；使用同一 pricing path 的逐路径差值计算 SE，保留协方差。CI 是固定分片、固定训练结果条件下的 Monte Carlo 区间，不是跨 seed 总体区间；未对 24 次适应性搜索比较做多重检验校正。完整表另见 [paired_results.csv](paired_results.csv)。

| Pipeline | Seed / round | Operator | Target | Effect | SE | Verdict |
|---|---|---|---|---:|---:|---|
| Context→Proposal | 17 / 0 | whole_program_restart | instance:public-put-high-vol | 0.000341883 | 0.00123212 | inconclusive |
| Context→Proposal | 17 / 0 | ast_mutation | instance:public-put-high-vol | — | — | execution_failed |
| Context→Proposal | 17 / 1 | ast_crossover | instance:public-put-high-vol | 0.000610498 | 0.000423877 | inconclusive |
| Context→Proposal | 17 / 1 | subsystem_rewrite | instance:public-put-high-vol | -4.86801e-05 | 0.00053812 | inconclusive |
| Context→Proposal | 29 / 0 | whole_program_restart | instance:public-put-high-vol | -0.00785053 | 0.00224138 | refuted |
| Context→Proposal | 29 / 0 | ast_mutation | instance:public-put-atm | 0.000606846 | 0.00109391 | inconclusive |
| Context→Proposal | 29 / 1 | ast_crossover | instance:public-put-atm | -0.00109455 | 0.000398634 | refuted |
| Context→Proposal | 29 / 1 | subsystem_rewrite | instance:public-put-atm | -0.00171054 | 0.000667662 | refuted |
| Context→Proposal | 41 / 0 | whole_program_restart | instance:public-put-high-vol | — | — | execution_failed |
| Context→Proposal | 41 / 0 | ast_mutation | instance:public-put-high-vol | -0.000992424 | 0.00175861 | inconclusive |
| Context→Proposal | 41 / 1 | ast_crossover | instance:public-put-high-vol | -0.000510005 | 0.000976992 | inconclusive |
| Context→Proposal | 41 / 1 | subsystem_rewrite | instance:public-put-high-vol | -0.000274197 | 0.00137264 | inconclusive |
| 直接生成 | 17 / 0 | whole_program_restart | instance:public-put-atm | 0.000687878 | 0.000852509 | inconclusive |
| 直接生成 | 17 / 0 | ast_mutation | instance:public-put-atm | -0.00220558 | 0.00127005 | inconclusive |
| 直接生成 | 17 / 1 | ast_crossover | instance:public-put-atm | 0.00216417 | 0.00121876 | inconclusive |
| 直接生成 | 17 / 1 | subsystem_rewrite | instance:public-put-atm | -0.000958185 | 0.00125265 | inconclusive |
| 直接生成 | 29 / 0 | whole_program_restart | instance:public-put-atm | — | — | execution_failed |
| 直接生成 | 29 / 0 | ast_mutation | instance:public-put-atm | — | — | execution_failed |
| 直接生成 | 29 / 1 | ast_crossover | instance:public-put-atm | 0.000523068 | 0.00111038 | inconclusive |
| 直接生成 | 29 / 1 | subsystem_rewrite | instance:public-put-atm | -0.00160021 | 0.00063934 | refuted |
| 直接生成 | 41 / 0 | whole_program_restart | instance:public-put-atm | -0.00170913 | 0.000905088 | inconclusive |
| 直接生成 | 41 / 0 | ast_mutation | instance:public-put-atm | — | — | execution_failed |
| 直接生成 | 41 / 1 | ast_crossover | instance:public-put-atm | -0.00218755 | 0.00127216 | inconclusive |
| 直接生成 | 41 / 1 | subsystem_rewrite | instance:public-put-atm | -0.000380388 | 0.000611385 | inconclusive |

## Context 证据链

每个 seed 的第二轮真实 Context 调用均读取上一轮 4 条预测记录。原始表可由 append-only prediction ledger 与 matched-control ledger 重建，重建摘要与调用前记录完全一致；对应文件保存在 verification/。完整模型提示、返回 JSON、操作符 materialization receipt、源码和 parent lineage 都保留在 runs/。

- Seed 17：`7ed975c8c3407dedaf5bef24f8c950c97df806b5e60570fdc9b14bb91e00c6d1`。第二轮机制为 crossover_uncertainty_gate，引用 sol_0002, sol_0003, sol_0004；rewrite_fit_oof_residual，引用 sol_0002, sol_0005, sol_0006。
  记录的决策依据：Completed evidence supports testing complementary composition and a bounded fit-only revision while retaining sol_0002 as the unchanged control.
- Seed 29：`5e55020441f4c908812d2e93f7cda2f1753126550ce4ccc1bab071368c27f0a9`。第二轮机制为 boundary_gated_ridge_irls_crossover，引用 sol_0000, sol_0005, sol_0006；crossfit_shrinkage_fit_rewrite，引用 sol_0000, sol_0005, sol_0006。
  记录的决策依据：Matched evidence supports cautious composition and subsystem isolation, not another kNN restart.
- Seed 41：`198392d6bf5130332a2238dee1616847c27470ea1baf51f0fc926dc138a890d7`。第二轮机制为 compose_ridge_residual_validation_gate，引用 sol_0000, sol_0002, sol_0005, sol_0006；rewrite_fit_boundary_calibration，引用 sol_0002, sol_0005, sol_0006。
  记录的决策依据：Completed evidence rejects the prior mutation’s direction but supports testing complementary composition and a localized fit rewrite.

这里发生的是根据外部记录调整下一次提示、机制和 parent，不是更新 Context 模型权重；未出现过的 hypothesis ID 也不被解释为算法语义新颖。机制中提出的额外 gate 激活率或 fold-regret 等探针，不自动算作已完成观测：本实验实际闭合的是 evaluator 提供的目标分片收益、SE、exercise/stop-time 行为和执行失败。

## 训练与最小独立验证

所有成功的候选、instance 和 repeat 都重新 fit：公开候选 172 个训练单元，基线 84 个，隐藏审计 12 个，共 268 个主评估训练单元；另有 6 次独立 fresh-fit replay 与 2 次可检查的 MLP/hybrid 训练。每个单元记录路径、payoff、输入文件、evaluator target stream、原始 path seed、实际折叠后的 fit seed、模型文件摘要及训练时间。配对内共享路径不代表这些单元在统计上彼此独立。

训练监督来自 Monte Carlo 路径上的 discounted payoff 与候选自身构造的 backward continuation targets，不需要人工标签。MLP 在候选代码中更新两层网络权重；hybrid 先拟合 ridge，再用实际梯度训练 MLP residual。training_validation/ 保留输入、模型、真实 backward target 数组、loss/gradient trace 和预测数组；独立导出检查观察到非零权重更新、训练损失下降、终端 backward target 等于 MC terminal payoff、预测有限。任意生成程序的内部标签仍不属于 evaluator 的可信数学观测。

候选 fit/predict 均通过 native Seatbelt 执行，输入只读，评分代码位于独立可信进程。所有本次训练记录的 research_fallback 都为 false；research_mode 仅为 provenance 标记，不允许在 Seatbelt 启动失败后无隔离重试。验证范围是实际运行的 macOS 环境，不扩展为跨平台安全结论。

lookahead probe 在一个审计实例、一个中间时点、8 条路径上固定历史前缀并更换未来后缀，实际调用两次 predict，验证预测一致。fresh-fit replay 在新沙箱中使用相同源码、输入和 seed，比对模型摘要与预测。六次审计两类 probes 均 passed，额外 probe 与 replay 时间均大于零。这是有限的独立观测，不是普遍 no-lookahead 定理，也不能替代更大规模 evaluator 数学验证。

## 成本与失败

模型为 backend 实际记录的 gpt-5.6-sol，共 30 次调用：6 次 Context 成功、19 次 Proposal 成功、5 次 Proposal 在 180 秒上限失败。5 次超时都缺少 token 用量回报，因此下列 token 总数仅为已报告部分的下界；不推算缺失 token 或货币成本。部分超时包含后端重连，部分在生成后的自检阶段耗尽预算，不能全部归因于算法质量。

| Pipeline | Generated failures | Generation s | Evaluator s | Fit s（包含在 evaluator 中） | Audit s | Total s | Reported tokens ≥ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Context→Proposal | 2/12 | 1690.567 | 71.553 | 10.592 | 7.090 | 1769.210 | 362,639 |
| 直接生成 | 3/12 | 1588.661 | 67.395 | 9.772 | 7.161 | 1663.217 | 273,605 |

上述 total 为生成、公开评估与隐藏审计的实测时长之和，不重复加入已包含的 fit 时间。共同的 21 次基线评估耗时 70.430 秒，另计；迁移包数值重放耗时 220.210 秒，也不计入两条搜索 pipeline 的比较。前一配置失败的 real-model 尝试完整保留在 ../mathai-bermudan-live-20260905/，其 RUN_STATUS 明确排除该次尝试；历史 deterministic fixture 同样不能混入本次统计。

## 复现与文件

运行环境：macOS，Python 3.12.6，NumPy 2.4.0。先进入本 bundle 目录。native Seatbelt 必须能启动；外层受限环境若禁止启动，应在允许原生沙箱的环境运行，不允许直接绕过候选隔离。

```bash
python3 reproduction/experiments/replay_bermudan_bundle.py --bundle "$PWD" --output /tmp/openhyra-fresh-replay
python3 reproduction/experiments/audit_bermudan_bundle.py --bundle "$PWD" --replay-dir /tmp/openhyra-fresh-replay --output /tmp/openhyra-fresh-audit
python3 reproduction/experiments/verify_bermudan_training.py --output /tmp/openhyra-fresh-training
```

重放目录必须全新且为空，否则命令拒绝执行。重放使用 bundle 内冻结的 evaluator 与候选源码，不再调用生成模型；LLM 文本的随机生成和网络超时不保证再次相同，原始响应与失败原因作为固定实验记录保留。核验工具重建配对统计和 Context 表；已有 numerical_replay/ 是本次实际重跑的记录，不能把重新读取它误称为新一次运行。

- manifest.json：原实验请求、版本、预算及冻结源码哈希。git_base 是冻结时 checkout 的 HEAD，文件级 code_sha256 才是实际执行源版本的依据。
- supplemental_verification.json：协议冻结后增加的复现/核验工具及其摘要；不改变实验生成程序或 evaluator。
- candidate_ledger.jsonl、summary.json、paired_results.csv：全部候选、配对效果、失败与成本。
- runs/：实际提示、模型输出、Context 决策、经验库、源码和预测记录；public_selection_frozen.json 记录审计前的赢家。
- references/、audits/、training_validation/：完整基线、隐藏评估和可检查训练证据。
- numerical_replay/、verification/：迁移后实际重跑结果、逐项核验报告和复原的 Context 输入表。
- artifact_hashes.json：导出文件的 SHA-256；排除可重新生成的 __pycache__、.pyc、.DS_Store 和该哈希文件本身。

本次可支持的论文表述是：在固定小规模 Bermudan 环境中，真实 Context→Proposal 能提出机制、生成开放 Python 程序、接受独立 evaluator 评分，并把可追踪的分片结果带入下一轮；该实现及本次数值证据可复现。不能据此声称达到 AlphaEvolve、发现人类未知算法、证明 Context 更优、完成跨任务验证或具备生产部署能力。
