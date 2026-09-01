# OpenHyra System v5 — 程序搜索、行为表征与可干预类比

> Status: Draft v5 for review
>
> Date: 2026-09-01
>
> Scope: Candidate Program、Proposal Agent、Context Agent、Verifier/Evaluator、Experience Bank、搜索调度与科学验证
>
> Previous document: `PRD_CONTEXT_AGENT_V2.md`（保留，不修改）

---

## 0. 文档定位

本文件是在 `PRD_CONTEXT_AGENT_V2.md` 基础上形成的系统级 PRD。旧文件继续作为 Context Agent v2、per-instance 训练、可信 Runner 和 Experience Bank 初版设计的历史记录，不删除、不覆盖。

本文件对新系统设计具有更高优先级。两份文件冲突时：

1. 已实现且经过可信验证的安全约束优先于任何 PRD 描述；
2. 新系统设计以本文件为准；
3. 旧文件中未被本文件否定的 evaluator、sandbox、private audit 约束继续有效；
4. 任何候选协议扩展必须显式升级 schema，不得通过放宽旧 schema 隐式兼容。

### 0.1 从旧 PRD 继承的核心决策

- 训练自由、推理可信：候选代码只能在不可信沙箱中运行，候选代码不得进入 evaluator 进程。
- 候选训练完成后必须导出受限、可验证的冻结工件，由 evaluator-owned trusted runner 推理。
- 公开搜索集、扩展开发集和私有审计集严格分离；私有审计结果不得回流 Experience Bank、Context Agent 或 Proposal Agent。
- 百慕大最优停时的停止决策、payoff、折现、路径模拟、primal-dual 构造和最终评分继续由 evaluator 独占。
- 开放搜索轨可以评估“固定资源约束下哪个候选程序更好”，但不能声称测量受控样本效率。
- 只有 evaluator 控制训练实现和路径预算的封闭科学轨可以研究样本效率。
- Experience Bank 的事实记录保持 append-only，不得删除或修改已经 commit 的记录。

### 0.2 本版本替换的关键决策

- 岛不再等同于 Ridge、MLP、Transformer 等算法语义家族；岛是独立演化种群。
- 语义标签、行为单元、岛归属和类比关系成为四套相互正交的数据结构。
- 得分相关性、行权边界重合或 LLM 判断不再被直接解释为“两个算法具有相同机制”。
- Context Agent 不再吸收代码生成职责；Proposal Agent 继续拥有候选代码生成和修复职责。
- 新增显式 `MechanismCard`、`BehaviorProfile`、`AnalogyHypothesis` 和 `AnalogyResult`。
- “算法理解”以可检验的行为预测和干预迁移为操作性定义，不以自然语言解释是否流畅为判据。
- 所有新候选以代码为生成表面，但代码不被视为算法的完整语义表征。

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [背景、当前边界与问题定义](#2-背景当前边界与问题定义)
3. [产品目标、科学问题与非目标](#3-产品目标科学问题与非目标)
4. [系统原则与不可破坏的不变量](#4-系统原则与不可破坏的不变量)
5. [系统架构](#5-系统架构)
6. [核心对象模型](#6-核心对象模型)
7. [候选程序与可信协议](#7-候选程序与可信协议)
8. [Verifier、Evaluator 与行为采集](#8-verifierevaluator-与行为采集)
9. [Experience Bank：事实账本与派生视图](#9-experience-bank事实账本与派生视图)
10. [独立种群岛与行为单元](#10-独立种群岛与行为单元)
11. [语义记忆与 Analogy Graph](#11-语义记忆与-analogy-graph)
12. [Context Agent 与 Proposal Agent](#12-context-agent-与-proposal-agent)
13. [Verifier 回流与 Context 检索预算](#13-verifier-回流与-context-检索预算)
14. [混合算法与组合算子](#14-混合算法与组合算子)
15. [科学验证方案](#15-科学验证方案)
16. [安全、完整性与隐私边界](#16-安全完整性与隐私边界)
17. [可观测性、验收指标与停止线](#17-可观测性验收指标与停止线)
18. [迁移与实施阶段](#18-迁移与实施阶段)
19. [维护、模块边界与版本策略](#19-维护模块边界与版本策略)
20. [风险登记与应对](#20-风险登记与应对)
21. [决策记录](#21-决策记录)
22. [参考文献](#22-参考文献)

---

## 1. 执行摘要

OpenHyra 当前是一个有严格候选工件边界、可信 evaluator 和 Experience Bank 回流机制的任务级研究循环。它已经能让 Context Agent 根据历史实验提出方向，让 Proposal Agent生成候选，让 sandbox 和 evaluator执行并评分，再把结果写回 Experience Bank。

当前系统的主要天花板不是“能不能再生成更多候选”，而是缺少三种中间能力：

1. **异构算法的共同观察语言。** Ridge、显式规则、数值方法、MLP 和未来的序列模型拥有不同代码结构和训练方式，系统目前只能比较最终分数，不能稳定描述它们在不同问题切片上的行为。
2. **跨算法的显式类比。** 系统没有对象记录“源算法中的哪个机制可能迁移到目标算法、预期改善什么、什么结果会证伪”。
3. **由验证结果驱动的研究策略更新。** Experience Bank 记录发生了什么，但没有可靠地把事实、行为指纹、LLM 推断、迁移假设和反例区分开，也没有测量这些信息是否真正提升下一步搜索效率。

v5 将 OpenHyra 重构为五个闭环层：

```text
候选程序搜索
    ↓
沙箱构建冻结工件
    ↓
可信 Verifier/Evaluator + 标准化行为采集
    ↓
Experience Bank 事实账本 + 可重建派生视图
    ↓
Context Agent 形成可证伪假设 → Proposal Agent 生成下一份代码
```

v5 的核心研究命题是：

> 在固定的百慕大最优停时问题族和匹配的实验预算下，显式的跨算法结构类比能否比纯代码进化、语义分类或行为近邻检索更快找到有效候选？

这里的“类比”不是相似度标签，而是一次预先记录的干预预测：从源算法抽取一个可迁移机制，在目标算法中实施，并与匹配对照比较。只有当预测在预定问题切片上成立，系统才把该关系记为 `transfer_supported`。

---

## 2. 背景、当前边界与问题定义

### 2.1 当前系统能力

当前主任务使用 `bermudan-lsmc-feature-ir.v1`：候选只能修改 `feature_program.json`，evaluator 固定 Ridge、路径模拟、payoff、折现、停止规则、primal-dual 构造和评分。

当前 Context Agent 具备：

- Experience Bank 一致性快照；
- 最优、近期失败、证据等级和方向多样性的代表性采样；
- 全局状态与方向统计；
- 前一轮分析记忆；
- 活跃方向去重；
- LLM 失败后的确定性 fallback；
- 停止请求的外部 Controller 复核。

当前 Experience Bank 具备：

- append-only `records.jsonl`；
- 每个候选的源码快照和评估结果；
- `parent`、`repair_of`、`duplicate_of`、`numeric_duplicate_of` 等关系；
- evaluator metrics、source hash、artifact hash 和 run provenance。

实验性但尚未成为默认主路径的能力包括：

- `openhyra-policy-spec.v1` MLP 冻结权重工件；
- per-instance sandbox training pipeline；
- 候选算法 bundle 的 Top-K freeze 和 per-cell provenance；
- 训练沙箱的附加隔离与资源观测。

### 2.2 当前关键缺口

| 缺口 | 当前表现 | 后果 |
|---|---|---|
| 搜索空间 | 主要是 Feature IR | 无法系统探索训练算法和混合算法 |
| 算法表征 | 代码文本、描述和最终分数 | 不同算法不可比较，代码相似容易被误当成机制相似 |
| 行为采集 | 聚合分数和少量 per-instance metrics | 无法定位互补性、脆弱性和条件性优势 |
| 类比 | 隐式依赖 LLM 上下文 | 无预测、无证伪、无迁移增益归因 |
| 岛模型 | 旧 PRD 按算法类别分岛 | 可能提前隔离异构算法，阻碍混合与迁移 |
| 研究记忆 | LLM 摘要与规则 | 事实和推断边界不够明确，重建不确定 |
| Context 检索 | 固定数量代表性记录 | 不能围绕当前假设按证据价值检索 |
| 科学归因 | 下一轮是否提升 | 无法排除更多 token、更多调用和额外计算造成的收益 |

### 2.3 三个必须避免的概念替换

1. **代码统一不等于表征统一。** 对显式公式，代码接近完整算法；对神经网络，算法还包括训练数据、随机性、训练过程和冻结权重。
2. **行为相似不等于机制相同。** 两个策略可能因为共同任务约束产生相似输出，但内部机制和可迁移干预完全不同。
3. **语义分类不等于岛模型。** 语义标签描述“它是什么”；岛模型控制“搜索种群如何保持独立、多样并迁移”。

---

## 3. 产品目标、科学问题与非目标

### 3.1 产品目标

v5 必须实现：

- 让所有新搜索候选以受版本控制的代码 bundle 生成；
- 支持传统显式方法、训练型模型和至少一种受限混合模型共享同一 evaluator-owned `continuation` 语义；
- 在 Verifier 后生成可比较的 `BehaviorProfile`；
- 用独立种群岛维持搜索多样性，用行为单元保存不同策略行为；
- 用 `MechanismCard` 区分可验证事实与 LLM 推断；
- 用 `AnalogyHypothesis` 和 matched control 验证类比迁移；
- 让 Context Agent 围绕证据缺口制定实验，让 Proposal Agent 生成代码；
- 让所有事实、假设、干预和结果均可追溯到候选、数据、模型、prompt 和 evaluator 版本；
- 在不泄漏 private audit 的前提下，使公开验证结果累积成下一轮可用经验。

### 3.2 核心科学问题

主问题：

> 在固定百慕大最优停时问题族内，显式的跨算法结构类比，能否提升智能体在异构求解器之间的假设迁移效率，并在固定实验预算下更快找到有效候选？

次问题：

1. 行为表征是否比代码语义标签更能预测一个干预会不会迁移？
2. 语义机制卡和行为指纹结合，是否优于任何单一信息源？
3. 类比引导的混合算法是否优于等参数、等计算和等编辑幅度的控制候选？
4. 独立种群岛是否减少搜索坍缩，并提高有效行为区域覆盖？
5. 不同 Context 证据包的大小和构成如何影响搜索收益与 token 成本？
6. EB 累积的异构算法实验数据中，是否存在基于合约特征（moneyness、volatility、维度等）的条件性算法优势模式？该模式能否提升搜索资源分配效率？

### 3.3 操作性定义

在本 PRD 中：

- **算法理解**：模型能够基于已有证据预测算法在未直接观察的受控切片或干预下的行为，并给出可证伪条件。
- **结构类比**：源算法和目标算法之间存在一个显式关系映射，该映射支持一项预先记录的可迁移干预预测。
- **类比成功**：类比引导干预相对 matched control 取得正的 `TransferGain`，且改善方向与预注册预测一致。
- **互补**：两个算法在预定义切片上存在可重复的条件性优势，并且一个受限组合算子能在匹配预算下保留至少一部分优势。
- **行为等价**：两个冻结策略在版本化 probe suite 上落入同一量化行为单元；这不是数学上的全局等价。

### 3.4 非目标

v5 不声称：

- 证明基础模型已经具备通用科学类比能力；
- 一次性支持任意 Python 推理代码或任意神经网络架构；
- 在开放搜索轨测量受控样本效率；
- 让 LLM 解释替代统计验证、因果干预或 evaluator 事实；
- 把 private audit 变成搜索反馈；
- 在第一阶段支持 Decoder-only Transformer；
- 自动发现任意领域的通用算法本体；
- 将百慕大期权结果外推到所有金融任务或科学发现任务。

---

## 4. 系统原则与不可破坏的不变量

### 4.1 信任原则

1. 候选源码、候选日志、候选解释和候选导出的任何文件均为不可信输入。
2. 候选代码只在 sandbox 运行，不得被 evaluator import。
3. trusted runner 只读取严格 schema 的 data-only artifact。
4. evaluator 独占最终输入、随机流、停止决策、对偶构造和评分。
5. 所有 `.npy` 使用 `allow_pickle=False`，拒绝 NaN、Inf、未知 dtype、未知形状、额外 payload 和未知文件。
6. private audit 候选必须在 private seed 产生前冻结并绑定完整 provenance。

### 4.2 证据原则

1. `ExperimentEvent` 中 evaluator 产生的字段是可信观察。
2. 代码解析和规范化得到的字段是确定性派生事实。
3. LLM 生成的标签、解释、机制和类比是版本化推断，不得写成 evaluator 事实。
4. 所有规则必须引用支持和反驳记录；仅有计数不足以升级为“confirmed”。
5. 独立性按不同候选谱系、不同随机种子和不同实例切片显式记录，不以“出现三次”代替。
6. 相关性、聚类和解释只能提出假设，不能单独建立因果或机制结论。

### 4.3 数据原则

1. `records.jsonl` 继续 append-only。
2. 原始源码、冻结工件、probe 结果和训练轨迹使用内容寻址存储。
3. 确定性派生索引必须可位精确重建。
4. LLM annotation 作为独立追加事件保存，不要求重新调用 LLM 重建。
5. 删除派生索引不得删除事实记录或工件。
6. schema 升级使用新版本号，不原地改变旧字段含义。

### 4.4 搜索原则

1. 岛负责种群独立性，不负责定义算法类别。
2. 每个候选有一个产生它的 island epoch，但可以拥有多个 parent 和 inspiration。
3. 语义标签不决定岛归属、合并或删除。
4. 行为单元由确定性 descriptor 和量化规则产生。
5. 跨岛迁移、类比和组合必须保留完整 lineage。
6. 失败、早停、重复和反例也进入 Experience Bank。

---

## 5. 系统架构

### 5.1 总体数据流

```text
Experience Bank facts + derived views
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Context Agent                                               │
│  1. Analyst：事实、冲突、覆盖与失败模式                     │
│  2. Analogy Hypothesizer：迁移映射、预测、反例和控制         │
│  3. Experiment Planner：目标岛、parent、预算与验收条件       │
└──────────────────────────────┬──────────────────────────────┘
                               │ ExperimentPlan
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Proposal Agent                                              │
│  读取 parent + 最多两个 inspiration diff + 协议文档         │
│  生成或修改候选代码 bundle                                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ AlgorithmBundle
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Sandbox Builder                                             │
│  执行不可信候选代码，输出 data-only FrozenPolicyArtifact     │
└──────────────────────────────┬──────────────────────────────┘
                               │ trust boundary
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Trusted Verifier/Evaluator                                  │
│  schema验证 → trusted runner → public/dev评分 → behavior探针 │
└──────────────────────────────┬──────────────────────────────┘
                               │ ExperimentEvent + BehaviorProfile
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Experience Bank                                             │
│  immutable ledger + artifact store                          │
│  population islands + behavior cells + semantic annotations │
│  analogy graph + evidence projections                       │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 控制面与数据面

| 平面 | 组件 | 作用 |
|---|---|---|
| 控制面 | Context Agent | 选择研究问题和实验设计 |
| 控制面 | Island Scheduler | 分配种群和迁移/重置时机 |
| 控制面 | Stop Controller | 审核停止请求和预算门槛 |
| 生成面 | Proposal Agent | 把 ExperimentPlan 转换成候选代码 |
| 执行面 | Sandbox Builder | 运行不可信候选代码并产出冻结工件 |
| 真值面 | Verifier/Evaluator | 产生可信评分、合法性和行为测量 |
| 记忆面 | Experience Bank | 保存事实、工件、关系和派生视图 |

### 5.3 组件所有权

| 能力 | 唯一 owner | 明确不负责 |
|---|---|---|
| 事实写入 | Experience Bank | 不判断算法好坏 |
| 合法性与得分 | Verifier/Evaluator | 不提出下一步实验 |
| 行为探针 | Behavior Profiler（可信 evaluator 子组件） | 不解释机制 |
| 岛调度 | Island Scheduler | 不做语义分类 |
| 机制与类比假设 | Context Agent | 不执行代码、不改评分 |
| 候选代码 | Proposal Agent | 不决定事实是否成立 |
| 最终停止 | Stop Controller | 不替代 Context 的研究分析 |

---

## 6. 核心对象模型

### 6.1 `AlgorithmBundle`

`AlgorithmBundle` 是 Proposal Agent 的提交对象，也是候选程序搜索的基本遗传单位。

```json
{
  "schema": "openhyra-algorithm-bundle.v1",
  "entrypoint": "train.py",
  "artifact_protocol": "openhyra-policy-spec.v1",
  "source_files": ["train.py", "manifest.json"],
  "parent_ids": ["sol_0031"],
  "inspiration_ids": ["sol_0019", "sol_0041"],
  "generation_operator": "analogy_transfer",
  "experiment_plan_id": "plan_0048",
  "candidate_seed": 48000007
}
```

约束：

- `source_files` 必须与实际 bundle 文件精确一致；
- 允许哪些文件由 task protocol 决定，候选不能通过 bundle 字段扩张文件集合；
- v1 继续使用 `train.py` 以复用已经存在的 sandbox training pipeline；
- `train.py` 在概念上是“artifact builder”，可以执行训练，也可以直接构造解析/数值工件；
- bundle 仅声明候选源码，不包含运行后产生的权重；
- `parent_ids` 表示代码谱系，`inspiration_ids` 表示用于类比或组合但未直接继承代码的候选；
- 每个 bundle 在进入评估前计算 canonical tree hash；
- bundle 内容在 commit 后只读。

### 6.2 `FrozenPolicyArtifact`

冻结策略工件是 sandbox 和 evaluator 之间唯一允许的数据边界。

```json
{
  "schema": "openhyra-frozen-policy-artifact.v1",
  "protocol": "openhyra-policy-spec.v1",
  "instance_id": "put_1d_atm_vol20",
  "repeat": 0,
  "artifact_sha256": "...",
  "files": [
    {"path": "normalization.json", "sha256": "..."},
    {"path": "step_000.npy", "sha256": "..."}
  ]
}
```

v1 不在 per-instance artifact 目录中接受 `training_meta.json`。训练时间、内存、日志、收敛曲线和其他 telemetry 由 sandbox supervisor 在工件目录外记录，避免候选用可选 metadata 扩张可信输入面。

### 6.3 `ExperimentEvent`

`ExperimentEvent` 是每次候选评估写入事实账本的可信事件。

```json
{
  "schema": "openhyra-experiment-event.v1",
  "record_id": "sol_0048",
  "algorithm_bundle_sha256": "...",
  "experiment_plan_id": "plan_0048",
  "island_epoch_id": "island_02_epoch_03",
  "status": "ok",
  "score": 0.0182,
  "score_metric": "paired_lower_bound_lcb",
  "per_instance_metrics_ref": "sha256:...",
  "behavior_profile_ref": "sha256:...",
  "runtime_metrics_ref": "sha256:...",
  "parent_ids": ["sol_0031"],
  "inspiration_ids": ["sol_0019"],
  "created_at": "2026-09-01T00:00:00Z"
}
```

可信字段必须由 harness、sandbox supervisor 或 evaluator 写入。候选无权提供 `score`、`status`、`behavior_profile_ref` 或 provenance hash。

### 6.4 `BehaviorProfile`

`BehaviorProfile` 是异构算法的共同观察语言，不是代码或机制的统一表征。

```json
{
  "schema": "openhyra-behavior-profile.v1",
  "probe_suite": "bermudan-behavior-probe.v1",
  "probe_suite_sha256": "...",
  "policy_artifact_sha256": "...",
  "performance": {
    "per_instance_improvement": [0.01, -0.01, 0.02, 0.00, 0.03, -0.02, 0.01, 0.01],
    "paired_mean": 0.006,
    "paired_standard_error": 0.002
  },
  "outcome_distribution": {
    "loss_definition": "negative_paired_discounted_payoff_improvement",
    "mean_loss": -0.006,
    "var_95": 0.012,
    "cvar_95": 0.019
  },
  "policy_geometry": {
    "exercise_rate_by_instance": [0.21, 0.34, 0.18, 0.29, 0.42, 0.37, 0.25, 0.31],
    "boundary_monotonicity_violations": 0,
    "reference_boundary_agreement": 0.82
  },
  "sensitivity": {
    "moneyness": 0.41,
    "volatility": 0.17,
    "correlation": 0.08,
    "time_to_maturity": 0.23
  },
  "robustness": {
    "input_scale_invariance_error": 0.003,
    "state_perturbation_lipschitz_proxy": 1.7,
    "seed_instability": 0.011
  },
  "compute": {
    "training_seconds": 5.4,
    "peak_memory_bytes": 170000000,
    "inference_microseconds_per_state": 8.2,
    "parameter_count": 2433
  }
}
```

规则：

- VaR/CVaR 必须绑定明确的 `loss_definition`，不得对含义不明的“算法收益”直接计算；
- `BehaviorProfile` 只使用公开搜索集和扩展开发集数据；
- 同一 profile 的所有策略比较使用 common random numbers 和相同 probe suite；
- profile 保存完整工件，Context 默认只读取压缩摘要；
- attribution、SHAP、局部 surrogate 等解释信号不得写入可信行为事实区，只能进入 annotation 区。

### 6.5 `MechanismCard`

`MechanismCard` 分离确定性事实、可信观察和 LLM 推断。

```json
{
  "schema": "openhyra-mechanism-card.v1",
  "record_id": "sol_0048",
  "deterministic_facts": {
    "protocol": "openhyra-policy-spec.v1",
    "input_representation": ["normalized_state", "time_to_maturity"],
    "objective": "backward_continuation_regression",
    "optimizer": "adam",
    "regularization": ["weight_decay"],
    "composition": ["symbolic_features", "mlp_residual"]
  },
  "trusted_observations": {
    "strong_slices": ["high_volatility"],
    "weak_slices": ["deep_otm"],
    "failure_modes": ["seed_instability"]
  },
  "llm_inferences": [
    {
      "claim": "symbolic normalization may reduce the residual learner's burden",
      "confidence": 0.63,
      "evidence_record_ids": ["sol_0019", "sol_0031", "sol_0048"],
      "annotation_event_id": "annotation_0082"
    }
  ]
}
```

`deterministic_facts` 只能来自 AST、manifest、受限配置或可信训练包装器。无法可靠静态解析的目标函数或优化器必须标记为 `unknown`，不得由 LLM 推测后放入事实区。

### 6.6 `AnalogyHypothesis`

```json
{
  "schema": "openhyra-analogy-hypothesis.v1",
  "id": "analogy_0017",
  "source_record_ids": ["sol_0019"],
  "target_parent_id": "sol_0031",
  "relation_mapping": [
    {
      "source_role": "log_moneyness_feature",
      "target_role": "mlp_input_normalization",
      "shared_relation": "stabilize_state_scale_under_high_volatility"
    }
  ],
  "non_correspondence": [
    "ridge_coefficients_do_not_map_to_mlp_hidden_weights"
  ],
  "transferable_intervention": "add_log_moneyness_to_mlp_input_and_keep_other_training_choices_fixed",
  "predicted_effect": {
    "metric": "high_volatility_slice_improvement",
    "direction": "positive",
    "minimum_effect": 0.003
  },
  "falsifier": "paired_effect_lcb_le_0_on_high_volatility_slice",
  "matched_control": {
    "operator": "capacity_matched_irrelevant_feature",
    "same_parent": true,
    "same_runner_type": true,
    "same_training_budget": true,
    "same_edit_size_tolerance": 0.2
  },
  "status": "preregistered"
}
```

在目标候选执行前，`relation_mapping`、`predicted_effect`、`falsifier` 和 `matched_control` 必须冻结。执行后不得回写修改原假设，只能追加 `AnalogyResult`。

### 6.6.1 Analogy 的轻量与完整两层

完整 `AnalogyHypothesis`（含 matched control）是高质量因果证据，但每个假设需要 guided + control 两个候选 slot，且 invalid control 比例可能达 20-30%。在候选数量有限的阶段，两层并行：

- **轻量 inspiration tracking（P1 即可用）：** Proposal Agent 在 `AlgorithmBundle` 中记录 `inspiration_ids` 和 `generation_operator`，Context Agent 在 `ExperimentPlan` 中记录 `implementation_intent`。这提供弱归因但不消耗额外候选 slot，样本量大。
- **完整 AnalogyHypothesis + matched control（P3 启用）：** 只对 Context Agent 有高信心且能产出具体 `relation_mapping` 的迁移假设使用。预期每 run 中 analogy 实验占比不超过 30%。

轻量 tracking 的数据可以回溯性地识别"哪些 inspiration 来源与候选成功相关"，为 P3 选择值得做完整 matched-control 验证的假设提供信号。

### 6.7 `AnalogyResult`

```json
{
  "schema": "openhyra-analogy-result.v1",
  "analogy_hypothesis_id": "analogy_0017",
  "guided_record_id": "sol_0048",
  "control_record_id": "sol_0049",
  "guided_delta": 0.006,
  "control_delta": 0.001,
  "transfer_gain": 0.005,
  "transfer_gain_standard_error": 0.002,
  "predicted_slice_effect": 0.007,
  "prediction_direction_correct": true,
  "verdict": "transfer_supported"
}
```

`verdict` 只能是：

- `transfer_supported`；
- `transfer_refuted`；
- `inconclusive`；
- `invalid_control`；
- `execution_failed`。

单次 `transfer_supported` 不升级为通用规则；它只支持该 source、target、intervention、协议和问题切片下的关系。

### 6.8 `ExperimentPlan`

`ExperimentPlan` 是 Context Agent 和 Proposal Agent 之间的唯一正式接口。

```json
{
  "schema": "openhyra-experiment-plan.v1",
  "id": "plan_0048",
  "action": "continue",
  "target_island_epoch_id": "island_02_epoch_03",
  "generation_operator": "analogy_transfer",
  "parent_ids": ["sol_0031"],
  "inspiration_ids": ["sol_0019"],
  "analogy_hypothesis_id": "analogy_0017",
  "implementation_intent": "add log-moneyness as one normalized MLP input",
  "negative_constraints": [
    "do_not_change_hidden_width",
    "do_not_increase_training_paths",
    "do_not_modify_evaluator_owned_stopping_rule"
  ],
  "success_criterion": "paired_high_volatility_slice_lcb_gt_0",
  "budget": {
    "candidate_count": 2,
    "sandbox_seconds_per_cell": 60,
    "max_artifact_bytes": 8388608
  }
}
```

`expected_gain` 和 `confidence` 继续记录，但只有当 scheduler 明确消费这些字段时才能影响预算。未被任何组件消费的字段必须标记为 telemetry，不能在文档中称为控制逻辑。

---

## 7. 候选程序与可信协议

### 7.1 “所有算法生成代码”的精确定义

v5 要求所有新搜索候选的生成结果包含代码，但不要求 evaluator 执行任意候选推理代码。

候选代码的作用是：

```text
公开输入数据 + 公共实例参数 + 候选随机种子
                      ↓
               candidate train.py
                      ↓
          严格 schema 的冻结策略工件
```

这使传统规则、解析近似、数值方法和神经网络都能进入程序搜索：

- 显式规则可以由 `train.py` 直接编译为 typed expression artifact；
- Ridge 或随机特征可以拟合并导出线性权重；
- MLP 可以训练并导出 dense weights；
- 混合算法可以导出受限 Policy Graph；
- 未来 Transformer 只能导出受支持的 sequence artifact，不能把 PyTorch 模型对象直接交给 evaluator。

### 7.2 协议能力矩阵

| 协议 | v5 阶段 | 候选工件 | Trusted Runner | 说明 |
|---|---|---|---|---|
| `bermudan-lsmc-feature-ir.v1` | 已有兼容 | `feature_program.json` | evaluator Feature IR | 历史基线，不作为新系统唯一搜索面 |
| `continuation-linear.v1` | P1 | normalization + per-step coefficients | Linear Runner | 覆盖 Ridge/显式线性组合 |
| `openhyra-policy-spec.v1` | P1 | normalization + per-step MLP weights | MLP Runner | 现有 MLP continuation wire format |
| `continuation-residual-hybrid.v1` | P1 | linear branch + MLP residual | Hybrid Runner | 第一种正式混合协议 |
| `continuation-expression.v1` | P2 | bounded typed expression IR | Expression Runner | 支持解析/规则型 continuation |
| `sequence-continuation.v1` | 延后 | causal sequence graph + weights | 未实现 | 仅在非 Markov/path-dependent 任务出现后评审 |
| `direct-stop-policy.v1` | 延后 | stop logits/probability | 未实现 | 无 continuation 时不能直接复用现有 dual proxy |

任何协议只有在以下条件全部满足后才能从“实验性”升级为“可搜索”：

1. schema 已冻结；
2. trusted loader 对未知字段 fail closed；
3. runner 通过确定性、无状态、批次/顺序一致性、有界和有限性测试；
4. 训练工件与候选源码、实例、种子、运行时和输入 bundle 建立 hash 链；
5. public evaluator 和 private audit 均有独立入口；
6. 旧 Feature IR 回归结果不受影响。

### 7.3 生成粒度

Proposal Agent 默认不从零生成整份算法，而是使用以下 operator：

| Operator | parent | inspiration | 允许动作 |
|---|---:|---:|---|
| `local_mutation` | 1 | 0 | 修改一个明确机制或超参数 |
| `ablation` | 1 | 0 | 删除一个组件，验证必要性 |
| `repair` | 1 | 0 | 只修实现错误，不改变假设 |
| `analogy_transfer` | 1 | 1-2 | 把源机制映射到目标算法 |
| `composition` | 1-2 | 0-2 | 使用受限组合算子形成混合算法 |
| `restart_from_skeleton` | 0 | 1-2 | 在岛重置或新协议启动时从固定 skeleton 生成 |

默认每次候选只允许一个主要 intervention。多处无关改动会破坏归因，应由静态 diff analyzer 拒绝或标记为 `compound_edit`。

### 7.4 代码 prompt

Proposal Agent 只接收：

- `ExperimentPlan`；
- 精确 protocol 和 artifact 规范；
- 一个完整 parent 源码；
- 最多两个 inspiration 的相关函数或 diff；
- 失败约束和 sandbox API；
- candidate seed。

Proposal Agent 默认不接收全部 Experience Bank、private audit、无关岛的完整代码或未经转义的候选日志。

### 7.5 静态预检和修复

候选进入完整评估前执行：

1. bundle 文件集合、路径和大小检查；
2. Python 语法解析；
3. manifest/schema 检查；
4. import allowlist 和明显危险调用检查；
5. 与 `ExperimentPlan` 的 diff 范围检查；
6. 单实例 sandbox probe；
7. 冻结工件 trusted loader 检查。

静态预检是质量过滤，不是安全边界。安全仍由 sandbox、文件系统限制、进程/资源限制和 evaluator 的 data-only 边界承担。

Repair 只接收结构化错误：

- failure layer；
- error category；
- 精确 schema path；
- stderr 的受限尾部；
- 原始 ExperimentPlan；
- 失败源码和 diff。

Repair 不重新运行 Context Agent，不改变原假设；若必须改变机制或预算，则创建新的 ExperimentPlan。

---

## 8. Verifier、Evaluator 与行为采集

### 8.1 三层数据分离

| 数据层 | 用途 | 是否回流 EB | 是否供 Context 使用 |
|---|---|---:|---:|
| Public Search | 排序、候选选择和在线反馈 | 是 | 是 |
| Extended Dev / Behavior Probe | 行为分析、条件切片和相似性 | 是 | 是 |
| Private Audit | 冻结 Top-K 的一次性验收 | 否，独立报告 | 否 |

private audit 的候选列表、seed、实例、单项结果、总分和失败信息均不得进入搜索 Experience Bank。系统可以在运行结束后记录“audit 已完成”的外部状态，但不得暴露任何能影响下一轮搜索的结果。

### 8.2 单 cell 评估流程

```text
冻结 AlgorithmBundle
    ↓
evaluator 生成 per-instance 训练输入
    ↓
新建隔离 sandbox → 执行 train.py
    ↓
可信 loader 验证 FrozenPolicyArtifact
    ↓
Trusted Runner 加载并冻结
    ↓
独立 pricing / dual / behavior probe 随机流
    ↓
score + per-instance metrics + BehaviorProfile
    ↓
ExperimentEvent commit
```

每个 `(candidate, instance, repeat)` 使用独立 sandbox，不允许跨 cell 持久化候选状态。

### 8.3 Behavior Probe Suite v1

`bermudan-behavior-probe.v1` 使用公开、固定、版本化的八个开发实例。每个实例产生四类 descriptor：

1. 相对 frozen baseline 的标准化 per-instance improvement；
2. exercise rate / stopping boundary summary；
3. continuation calibration 或 reference consistency proxy；
4. loss distribution 的 CVaR 与 seed stability。

因此 v1 基础行为向量至少为 32 维，而不是仅使用 4-8 个最终分数做相关性判断。敏感度、资源和解释信号作为附加字段，不进入第一版 behavior cell key。

v1 的 probe suite 使用固定的八个开发实例。为防止候选系统性地向这些实例过拟合，每个 major phase 结束后 evaluator 应扩展 probe suite（增加新实例），旧实例保留但在行为分析中降权。此外，probe suite 中保留至少两个 held-out 实例不用于 BehaviorProfile 反馈，仅用于验证 profile 的外推稳定性。

### 8.4 VaR/CVaR 与尾部指标

所有风险统计以预先定义的 loss 为对象。v1 使用：

```text
loss(path) = -(
  discounted_payoff_candidate(path)
  - discounted_payoff_frozen_baseline(path)
)
```

- `VaR_95`：loss 的经验 95% 分位数；
- `CVaR_95`：超过 `VaR_95` 的 loss 均值；
- 所有候选使用相同 public probe paths；
- 报告有效路径数和 bootstrap 或重复随机流下的不确定性；
- VaR/CVaR 只描述策略尾部行为，不证明算法内部机制。
- 当 frozen baseline 在特定实例上的绝对表现低于预设阈值时，该实例的 VaR/CVaR 计算需标注 `baseline_weak`，分析时分层报告，避免 baseline 弱势掩盖候选差异。

### 8.5 行为扰动

可信 Behavior Profiler 可执行以下 bounded probes：

- moneyness 小幅变化；
- volatility、correlation、time-to-maturity 分层；
- 状态尺度变换；
- 资产置换对称性；
- 同一输入的批次拆分与调用顺序变化；
- 相同算法在不同训练 seed 下的稳定性。

所有 probe 由 evaluator 构造，候选不得得知未来 private audit 条件。

### 8.6 失败与早停记录

以下状态均产生 `ExperimentEvent`：

- `ok`；
- `early_stopped`；
- `static_rejected`；
- `artifact_rejected`；
- `timeout`；
- `oom`；
- `violation`；
- `runtime_error`；
- `cancelled`。

失败事件记录结构化 `failure_layer`、`error_category`、资源使用和 parent lineage。原始候选日志只作为不可信 artifact 保存，Context 读取前必须裁剪、转义并带安全提示。

---

## 9. Experience Bank：事实账本与派生视图

### 9.1 逻辑层次

Experience Bank 分为四层：

1. **Immutable Ledger**：发生了什么；
2. **Artifact Store**：源代码、权重、probe 和训练轨迹；
3. **Deterministic Projections**：岛、行为单元、谱系、近邻和统计；
4. **Versioned Annotations**：语义标签、机制推断、类比假设和研究规则。

### 9.2 物理布局

```text
eb/
├── records.jsonl                       # 现有 append-only 核心记录
├── events/
│   ├── experiment_events.jsonl         # 可信实验事实
│   ├── plan_events.jsonl               # 冻结 ExperimentPlan
│   ├── annotation_events.jsonl         # LLM 推断及其 provenance
│   └── analogy_results.jsonl            # 类比干预结果
├── objects/                             # content-addressed artifacts
│   └── sha256/<prefix>/<digest>/...
└── derived/
    ├── population_islands.v1.json
    ├── behavior_cells.v1.json
    ├── lineage_graph.v1.json
    ├── semantic_index.v1.json
    ├── analogy_graph.v1.json
    ├── evidence_projection.v1.json
    └── context_views/
```

`records.jsonl` 继续作为兼容入口。新事件可以通过 record id 与旧记录关联，不要求一次性迁移所有历史记录。

### 9.3 Artifact Store

以下对象使用 SHA-256 内容寻址：

- AlgorithmBundle；
- FrozenPolicyArtifact；
- per-instance metrics；
- BehaviorProfile；
- sandbox runtime metrics；
- candidate diff；
- prompt、response 和 parser payload；
- MechanismCard；
- AnalogyHypothesis 和 matched control spec。

对象一经写入不可修改。同一内容只存储一次；记录通过 digest 引用。

### 9.4 确定性派生视图

确定性视图只能依赖：

- ledger facts；
- content-addressed trusted artifacts；
- 冻结的规则和量化阈值；
- 明确的 schema 版本。

删除 `derived/` 后，重建结果必须位精确一致。若一个视图需要 LLM 判断，它不属于确定性视图，其判断必须先作为 `annotation_event` 写入事实可追溯层。

### 9.5 LLM Annotation Event

```json
{
  "schema": "openhyra-annotation-event.v1",
  "id": "annotation_0082",
  "annotation_type": "mechanism_inference",
  "target_record_ids": ["sol_0048"],
  "evidence_record_ids": ["sol_0019", "sol_0031", "sol_0048"],
  "model": "model-id",
  "backend": "backend-id",
  "prompt_sha256": "...",
  "response_sha256": "...",
  "parser_schema": "openhyra-mechanism-card.v1",
  "created_at": "2026-09-01T00:00:00Z"
}
```

LLM annotation 可以被新事件支持、反驳或替代，但旧 annotation 不删除。

### 9.6 兼容现有记录

历史 Feature IR 记录可以生成有限的 `BehaviorProfile` 和确定性 facts，但缺少的数据必须标记为 `not_observed`，不能用 LLM 补齐成事实。

历史记录默认进入 `legacy_population`，不伪造其原始岛归属。它们可以作为 inspiration 或 baseline，但不参与 v5 岛重置统计，除非通过明确 replay 重新评估。

---

## 10. 独立种群岛与行为单元

### 10.1 岛的定义

v5 的岛是独立演化种群，不是算法家族。默认建立四个岛，每个岛从同一个 frozen baseline 或同一个 protocol skeleton 开始，但使用不同的 proposal sampling seed 独立演化。

一个候选可以是 Ridge、MLP、表达式方法或混合算法；其算法类别不决定岛归属。候选产生于哪个岛，就写入该 island epoch。

### 10.2 Island Epoch

岛在每次 reset 后进入新 epoch：

```json
{
  "schema": "openhyra-island-epoch.v1",
  "island_id": "island_02",
  "epoch": 3,
  "seed_record_ids": ["sol_0031"],
  "started_after_context_round": 40,
  "proposal_seed": 230041,
  "status": "active"
}
```

旧 epoch 的记录保留。reset 只是停止从旧种群采样，不删除任何实验。

### 10.3 岛内选择

默认 prompt 从同一岛采样两个 inspiration：

1. 先从行为单元中采样，兼顾单元最优分数与稀缺性；
2. 再在该单元内采样候选，偏向高分、较短代码和较小工件；
3. sampled programs 按分数从低到高排列，帮助 LLM 看出改进方向；
4. 生成的新候选回到同一 island epoch。

`analogy_transfer` 是例外：target parent 来自目标岛，source inspiration 可以来自其他岛。跨岛来源必须写入 `inspiration_ids` 和 `AnalogyHypothesis`。

### 10.4 岛重置与迁移

v1 使用可复现的轮次调度，不使用墙钟四小时：

- 每完成 10 个 Context rounds 执行一次 island review；
- 按每岛 best public score 排序；
- 淘汰表现最差的一半 active epochs；
- 每个被淘汰岛从存活岛中均匀采样一个岛，并复制其最优候选作为新 epoch seed；
- 若最优分数并列，优先选择更早产生、代码更短的候选；
- reset 和 seed 选择写入 ledger。

以上超参数（4 岛、10 rounds/review、淘汰一半）是初始默认值，不是经验证的最优设计。选择依据：4 岛 × ~50 candidates/island ≈ 200 total，匹配当前单次 run 的预算规模；10 rounds 给每岛约 10 次独立探索后再做淘汰判断；淘汰一半是 FunSearch 原始方案的简化。这些参数必须在 P-1 replay 中做敏感性分析，P2 实现时冻结到 run manifest。H4（种群多样性）的结论需报告对这些超参数的敏感性。

该机制首先作为 FunSearch-style baseline 使用。只有实验表明其他调度显著更好，才能增加 UCB 或 MAP-Elites 变体。

### 10.5 行为单元

行为单元位于岛内，但 cell key 与 island id 无关。v1 使用两个量化 descriptor：

1. aggregate public improvement bucket；
2. tail loss CVaR bucket。

```text
behavior_cell_key = (
  performance_bucket,
  tail_risk_bucket
)
```

v1 限制为 2D 的原因：当前实验规模（~200 candidates/run）下，4D cell key 会导致大多数 cell 为空或单例，多样性维护在极度稀疏的 cell 空间中没有实际效果。当候选数量稳定超过 500 后，可扩展到 3-4 维（增加 OTM robustness 和 exercise aggressiveness）。cell key 的维度是实验参数，不是固定设计。

bucket 边界在一次 run 开始前冻结并写入 run manifest。不得根据本次 run 的最终结果事后调整边界。

完整 32+ 维行为向量用于近邻检索和分析；cell key 只用于保存多样性，不能替代完整 profile。

### 10.6 禁止的岛操作

v1 禁止：

- 因 `runner_type`、activation 或 LLM 标签不同而自动分岛；
- 因得分相关性高而合并岛；
- 因得分相关性低而分裂岛；
- 让 LLM 直接修改岛成员；
- 删除低分候选记录；
- 用 private audit 结果重排岛。

---

## 11. 语义记忆与 Analogy Graph

### 11.1 语义标签

语义标签用于检索和解释，不控制岛归属。标签示例：

- `linear_regression`；
- `symbolic_feature`；
- `neural_residual`；
- `importance_sampling`；
- `robust_loss`；
- `monotonic_constraint`；
- `high_volatility_specialist`；
- `seed_unstable`。

每个标签必须说明来源：

- `deterministic`：由代码或 manifest 解析；
- `trusted_observation`：由 BehaviorProfile 得到；
- `llm_inference`：由 annotation event 产生。

### 11.2 语义规则

语义规则不使用“出现三次即 confirmed”的固定状态机。每条规则记录：

- 适用范围；
- 预测对象；
- 支持记录；
- 反驳记录；
- 谱系独立组；
- 实例独立组；
- 最近一次校准时间；
- 当前 evidence status。

`evidence_status` 为：

- `hypothesis`；
- `supported_in_scope`；
- `mixed_evidence`；
- `refuted_in_scope`；
- `outdated_protocol`。

### 11.3 Analogy Graph

Analogy Graph 是跨算法关系的独立投影：

```text
node: ExperimentRecord / Mechanism / Intervention / BehaviorSlice
edge:
  structurally_related
  behaviorally_near
  complementary_on_slice
  analogy_hypothesized
  transfer_supported
  transfer_refuted
  composed_from
  counterexample_to
```

每条边引用产生它的 annotation、AnalogyHypothesis 或 AnalogyResult。图本身不创造新事实。

### 11.4 从相似到类比的升级条件

一个关系只有经过以下路径才能升级：

```text
语义或行为线索
    ↓
analogy_hypothesized
    ↓  冻结映射、预测、falsifier 和 matched control
执行 guided candidate + control
    ↓
transfer_supported / transfer_refuted / inconclusive
```

`behaviorally_near` 永远不能自动升级为 `transfer_supported`。

### 11.5 神经网络解释证据

神经网络解释可以包括：

- 输入消融；
- 受控 counterfactual；
- 梯度或局部敏感度；
- symbolic surrogate；
- hidden representation probe；
- 对称性和单调性测试。

其中只有输入/输出干预产生的 observable behavior 可以进入 trusted observation。梯度解释、surrogate 和自然语言解释进入 LLM/analysis annotation，并必须附带稳定性检查。

---

## 12. Context Agent 与 Proposal Agent

### 12.1 职责分离

Context Agent 回答：

- 当前知道什么？
- 哪些证据相互冲突？
- 哪个行为区域或机制尚未探索？
- 哪个类比值得通过什么实验检验？
- 需要什么 parent、control 和预算？

Proposal Agent 回答：

- 如何在协议和负约束内把 ExperimentPlan 实现成代码？
- 如何最小化无关改动？
- 如何通过 schema 和 sandbox 预检？

Context Agent 不写代码；Proposal Agent 不修改研究假设和成功标准。

### 12.2 Context 管道

#### Stage 0：Evidence Builder（确定性）

输入：ledger、BehaviorProfile、岛、行为单元、lineage、failure taxonomy。

输出：`PortfolioPacket`、`AnalysisPacket` 候选集合和数据完整性诊断。

LLM 调用：0。

#### Stage 1：Analyst

输入：全局 portfolio、目标岛摘要、top/frontier/failure/counterexample。

输出：带证据 ID 的观察、冲突、未知项和下一步信息价值排序。

禁止：直接写代码；把 LLM 推断写成事实；用 private audit。

#### Stage 2：Analogy Hypothesizer

输入：Stage 1、MechanismCard、目标 parent、候选 source、反例。

输出：0-2 个 `AnalogyHypothesis`，或说明当前证据不足以做类比并选择普通 mutation/ablation。

**默认路径是 fallback 到普通 mutation/ablation。** Analogy 是例外，不是常规操作。只有当 Stage 2 能产出足够具体的 `relation_mapping`（明确的 source_role、target_role 和非空的 shared_relation）时，才输出 AnalogyHypothesis。如果 LLM 只能写出 vague 的映射关系（如 "both use normalization"），必须 fallback。

要求：明确 non-correspondence、falsifier 和 matched control。

#### Stage 3：Experiment Planner

输入：Stage 1-2、island schedule、预算、活跃方向和 stop evidence。

输出：严格 schema 的 `ExperimentPlan`。

实现：优先使用确定性 scheduler，LLM 只能在允许范围内选择实验意图。

### 12.3 Analyst 输出

```json
{
  "schema": "openhyra-analysis-report.v1",
  "observations": [
    {
      "claim": "hybrid residual candidates improve high-volatility cells but are seed-unstable",
      "evidence_record_ids": ["sol_0041", "sol_0044", "sol_0046"],
      "evidence_type": "trusted_behavior",
      "scope": "high_volatility_public_dev"
    }
  ],
  "conflicts": [],
  "unknowns": ["whether log-moneyness transfers to plain MLP"],
  "recommended_information_gain_targets": ["analogy_0017"]
}
```

所有 observation 必须引用记录。没有记录支持的内容只能进入 `speculations`，不能进入 `observations`。

### 12.4 Proposal 生成与多候选

一个 ExperimentPlan 可以请求：

- 一个 guided candidate；
- 一个 matched control；
- 一个必要的 implementation repair；
- 普通搜索时最多四个同方向但随机 seed 不同的候选。

guided candidate 和 control 必须由同一 Proposal model、相同最大 token、相同 sandbox/evaluator budget 生成和评估。生成顺序随机化，避免第一个候选获得系统性优势。

### 12.5 停止权

Context Agent 可以请求 `action=stop`，但 Stop Controller 继续独立审核：

- 完成 context rounds；
- 最近窗口有效候选数；
- best-so-far 改善；
- 重复率和失败率；
- behavior coverage；
- 尚未完成的 preregistered AnalogyHypothesis；
- 预算剩余。

存在未执行的高信息价值 matched-control 实验时，Stop Controller 默认拒绝停止，除非资源预算不足。

---

## 13. Verifier 回流与 Context 检索预算

### 13.1 回流数据分层

| 数据 | 保存方式 | Context 默认读取 | Proposal 默认读取 |
|---|---|---|---|
| ID、parent、inspiration、hash、schema | ledger 全量 | 摘要 | 精确 parent/inspiration |
| status、score、CI/SE、per-instance | trusted artifact | 结构化表 | 仅成功标准相关字段 |
| BehaviorProfile | 完整 object + 摘要 | 目标切片和近邻 | 不直接读取全量 |
| runtime、memory、artifact size | trusted artifact | 预算和失败分析 | 约束摘要 |
| candidate source/diff | content-addressed | 默认不读完整代码 | parent 全文 + inspiration diff |
| MechanismCard | versioned object | 目标候选和 source | 实现意图相关部分 |
| AnalogyHypothesis/Result | event + graph | 全量目标关系 | 当前计划 |
| raw log | untrusted artifact | 裁剪和转义后按需 | repair 时读取 |
| private audit | 独立审计目录 | 永不读取 | 永不读取 |

### 13.2 检索包

#### `PortfolioPacket`

始终包含：

- 四个 active island epochs 的规模、best、最近改善和覆盖；
- 全局 best 和 frozen baseline；
- 每岛一个 best、一个 novelty frontier 和一个 informative failure；
- 尚未完成的 analogy/control 对；
- 预算和 stop diagnostics。

初始软预算：4,000 tokens。

实现硬上限：16,000 characters。

#### `AnalysisPacket`

包含：

- PortfolioPacket；
- 目标岛最近 10 个摘要；
- 最多 6 个代表性 BehaviorProfile 摘要；
- 最多 4 条语义规则及其支持/反驳；
- 最多 3 个失败模式；
- 至少 1 个反例（若存在）。

初始软预算：12,000 tokens。

实现硬上限：48,000 characters。

#### `AnalogyPacket`

包含：

- 一个 target parent；
- 最多 4 个 source candidates；
- 最多 2 个 counterexamples；
- 对应 MechanismCard 和 BehaviorProfile 的相关切片；
- 已有 transfer-supported/refuted edges；
- 允许的组合算子和 matched-control 模板。

初始软预算：20,000 tokens。

实现硬上限：80,000 characters。

#### `ProposalPacket`

包含：

- 一个完整 parent source；
- 最多两个 inspiration 的函数级 diff；
- ExperimentPlan；
- protocol、artifact 和 sandbox 文档；
- 负约束和 candidate seed。

初始软预算：16,000 tokens。

实现硬上限：64,000 characters。

### 13.3 检索原则

- token/character 上限是安全上限，不是应填满的配额；
- 优先按证据价值和当前假设检索，不按时间简单截断；
- Context 看行为和证据，Proposal 看代码和约束；
- 一个记录可以在多个 packet 中出现，但完整代码只进入 ProposalPacket；
- packet 记录生成规则、selected record ids、版本和 hash；
- 检索策略必须作为实验变量进行消融。

---

## 14. 混合算法与组合算子

### 14.1 设计原则

“各取所长”必须被翻译为受限组合算子。系统不接受只有自然语言描述、无法说明数据流和控制变量的“混合模型”。

### 14.2 v1 允许的组合算子

| 算子 | 定义 | 主要用途 | 关键控制 |
|---|---|---|---|
| `feature_augment` | 给目标模型增加源算法使用的输入表示 | 迁移归纳偏置 | 匹配输入维度/参数控制 |
| `residualize` | `C = C_base + R_nn` | 保留传统基线并学习残差 | base 冻结与否必须声明 |
| `convex_gate` | `C = g*C_a + (1-g)*C_b`，`g∈[0,1]` | 条件性组合 | gate 受 trusted runner 约束 |
| `ensemble_mean` | 多分支确定性均值 | 降低 seed variance | 等计算 control |
| `distill_to_linear` | 用复杂 teacher 生成公开训练 target，再拟合简单 student | 可解释压缩 | teacher 数据来源和泄漏检查 |
| `constraint_wrap` | 对输出施加单调、对称或边界约束 | 注入金融结构 | 约束由 evaluator-owned operator 执行 |

P1 只实现 `feature_augment` 和 `residualize`。其他算子在已有迁移证据和独立 Runner 评审后加入。

### 14.3 第一种正式混合协议

`continuation-residual-hybrid.v1`：

```text
continuation(state, time)
  = linear_symbolic_branch(state, time)
  + bounded_mlp_residual(state, time)
```

约束：

- 两个分支共享 evaluator-owned normalization contract；
- 输出统一折现到 `t=0`；
- residual 分支输出裁剪范围固定；
- runner 按固定 float64 顺序计算；
- candidate artifact 同时声明线性和 MLP 权重；
- evaluator 继续独占 `immediate >= continuation` 的停止决策。

### 14.4 混合增益的控制

任何混合候选必须至少匹配一个 control：

- 参数量匹配；或
- 训练时间匹配；或
- 推理成本匹配；

主分析使用预注册的首要 matching 维度，其余作为敏感性分析。混合模型比单模型更好不能自动解释为“成功类比”，除非对应 `AnalogyHypothesis` 的预测切片和机制干预也成立。

---

## 15. 科学验证方案

### 15.1 主要研究假设

**H1：类比迁移。** 在 matched parent、runner、训练预算和编辑幅度下，analogy-guided intervention 的平均 `TransferGain` 大于 0。

**H2：搜索效率。** 完整 v5 系统在固定 evaluator calls 下的 normalized best-so-far AUC 高于 FunSearch-style code evolution baseline。

**H3：行为表征。** BehaviorProfile + MechanismCard 对迁移成功的预测能力高于代码语义标签或最终分数近邻。

**H4：种群多样性。** 独立岛 + behavior cells 相对单种群减少行为覆盖坍缩，且不显著降低短期 best score。

**H5：置信度校准。** Context Agent 对 analogy 成功概率的预测在 held-out interventions 上具有可测的 calibration，而不只是排序能力。

**H6：条件性算法优势。** EB 累积的异构算法实验数据中存在基于合约特征（moneyness、volatility、维度等）的可提取条件性优势模式，且基于该模式的搜索资源分配优于均匀分配。

### 15.2 零假设与可接受负结果

- H1 的零假设：`TransferGain <= 0`；
- H2 的零假设：完整系统的 AUC 不优于 baseline；
- H3 的零假设：复杂表征不优于简单标签/分数；
- H4 的零假设：岛模型只增加复杂度，不增加有效覆盖；
- H5 的零假设：LLM confidence 不可校准。
- H6 的零假设：不存在可稳定提取的条件性优势模式，或基于该模式的分配不优于均匀分配。

任何零结果都必须完整报告，不以更换指标、切片或阈值规避。

### 15.3 实验组

| Arm | 搜索结构 | 记忆/表征 | Analogy |
|---|---|---|---|
| A | 当前单阶段 Context + flat EB | 代表性历史表 | 无 |
| B | FunSearch-style 独立岛 | score + behavior cells | 无 |
| C | B | 语义标签/规则 | 无显式干预 |
| D | B | BehaviorProfile | 近邻检索，无 AnalogyHypothesis |
| E | B | BehaviorProfile + MechanismCard | 完整可干预 Analogy |

可选工程消融不替代以上五组主实验。

**实验优先级。** 如果资源只允许部分 arm，按以下顺序执行：

1. **B vs A**（最先）：验证 FunSearch-style 岛模型本身是否优于现有单阶段搜索。如果 B 不优于 A，后续 arm 的基础设施增益无法归因。
2. **E vs B**（主比较）：验证完整 analogy 框架相对 FunSearch baseline 的增益。
3. **D vs B 和 C vs B**（消融）：分离 BehaviorProfile 和语义标签各自的贡献。

若只能跑两组，跑 B 和 E。

### 15.4 预算匹配

所有组固定：

- 相同 Agent LLM 和版本；
- 相同 temperature、最大输出和 backend；
- 相同 Context rounds；
- 相同 Proposal candidates；
- 相同 evaluator calls；
- 相同 public/dev 实例和随机流；
- 相同 sandbox 时间、内存和 artifact 大小；
- 相同 private audit 规则；
- 单独报告 LLM tokens、墙钟时间和失败重试。

若某组额外调用 LLM，其总 token 和调用成本必须计入成本指标；主效果不得只按 evaluator calls 报告。

### 15.5 主要指标

1. `normalized_best_so_far_auc`：以 evaluator calls 为横轴的最佳分数曲线面积；
2. `calls_to_threshold`：达到预注册阈值所需 evaluator calls；
3. `transfer_gain`：guided candidate delta 减 matched control delta；
4. `positive_transfer_rate` 与 `negative_transfer_rate`；
5. `behavior_coverage`：访问过的非空行为单元比例；
6. `analogy_calibration_error`：预测成功概率与实际成功率的偏差。
7. `conditional_advantage_auc`：基于合约特征的条件性算法分配策略相对均匀分配的 normalized best-so-far AUC 差异。

次要指标：

- wall-clock AUC；
- token-adjusted AUC；
- runtime/parameter-adjusted score；
- duplicate 和 functional redundancy；
- code build success rate；
- repair rate；
- island collapse；
- private audit re-ranking stability。

### 15.6 Matched-control 实验

每个 `AnalogyHypothesis` 至少产生一对：

```text
同一 target parent
    ├── guided: 实施类比干预
    └── control: 等预算、等编辑幅度、机制无关或映射打乱的干预
```

要求：

- 先冻结 hypothesis 和 control spec，再生成代码；
- guided/control 生成顺序随机化；
- 使用同一 evaluator paths；
- 比较预注册切片和总体指标；
- control 无效时结果标记 `invalid_control`，不得算成功；
- implementation failure 与 mechanism refutation 分开统计。

### 15.7 同任务内的 held-out 设计

v5 不要求跨任务，但必须避免只在同一表面上自我验证：

- held-out public/dev problem instances：相同任务族，不同 moneyness、volatility、correlation、dimension；
- held-out algorithm subfamily：某些 analogy 在未用于提出规则的 target runner 上验证；
- surface perturbation：等价变量命名、函数拆分或代码格式变化，检查类比是否只依赖代码表面；
- lineage holdout：source 和 target 不共享最近 parent；
- random seed holdout：proposal 和 training seed 独立。

private audit 不承担 analogy 训练或选择功能，只做最终外部验收。

### 15.8 统计计划

采用两阶段设计：

1. **Pilot**：每组 10 个独立 search runs，只用于估计方差、失败率和最小可检测效应，不做确认性显著结论；
2. **Confirmatory**：根据 Pilot 方差做预注册 power analysis，目标 power 0.8、双侧 alpha 0.05；每组不少于 20 个且不超过 50 个独立 runs。

若估计需要超过 50 runs/组才能检测预设最小实际效应，则停止确认性实验，报告当前资源下不具可辨识性，不降低效应阈值追求显著性。

### 15.8.1 计算预算估算

进入确认性实验前，必须报告：

- 单次 run 的预计 wall-clock 小时数（含 LLM 调用等待）；
- 单次 run 的 LLM API token 成本；
- 全部 arm 的总 wall-clock 和 API 成本；
- 在当前硬件（单机 / 多机）下的并行度和完成时间。

若总成本超过预算，按 §15.3 的实验优先级削减 arm 数量，不降低单 arm 的 run 数量。

分析要求：

- paired 设计优先；
- 报告效应量和置信区间；
- 预先指定主指标和主比较 E vs B；
- 其他 arm pair 使用 Holm 或 FDR 控制多重比较；
- failed runs 按预注册规则纳入，不只分析成功运行；
- 同时报告 evaluator-call、token 和 wall-clock 三种成本尺度；
- exploratory slice 结果明确标注，不能替代主结果。

### 15.9 允许的科学结论

如果 H1-H5 获得支持，可以声称：

> 在固定百慕大最优停时问题族、指定算法协议和匹配预算下，显式行为表征与可证伪的跨算法干预类比提高了 OpenHyra 的搜索效率或迁移质量。

如果 H6 获得支持，可以额外声称：

> 在同一问题族内，异构算法的条件性优势模式可从实验数据中稳定提取，且基于该模式的搜索资源分配优于均匀分配。

不能直接声称：

- 模型获得通用算法理解；
- 基础模型的 attention 被改善；
- 系统具备一般科学发现能力；
- 结果跨金融任务或科学领域成立；
- 行为相似证明内部机制相同。

---

## 16. 安全、完整性与隐私边界

### 16.1 Sandbox 前置门槛

开放候选代码前必须验证：

- Seatbelt 或等价 kernel/container 边界是候选进程的第一层约束；
- 无网络；
- 只允许读取候选源码、该 cell 输入和白名单 runtime；
- 只允许写入该 cell 输出和受控临时目录；
- 不能读取 repository、evaluator、其他 cell、audit request 或历史工件；
- 子进程继承限制，不能留下 detached process；
- CPU、wall-clock、memory、process count、file count 和 total output 有硬上限；
- 每个 cell 使用新 sandbox；
- 候选 cwd 不能劫持 trusted wrapper 的 imports。

若 macOS Seatbelt + supervisor 不能形成可靠的内存、进程和输出硬限制，开放训练轨保持 experimental/default-off，不能用于 private audit。

### 16.2 Artifact 边界

- v1 per-instance MLP artifact 只接受 `normalization.json` 和连续 `step_*.npy`；
- 不接受 `training_meta.json`、manifest 副本、checkpoint、日志、子目录或隐藏文件；
- telemetry 由 supervisor 写入独立 trusted 路径；
- artifact root 和文件拒绝 symlink、hardlink 异常、TOCTOU 变化和特殊文件；
- trusted loader 验证后才实例化 runner；
- runner 不访问候选源码。

### 16.3 Prompt injection

Experience Bank 中的 candidate description、source comments、logs、LLM annotations 和论文摘录均是不可信数据。

Context/Proposal prompt 必须：

- 将这些内容标记为 quoted data；
- 对日志和源码设置边界与长度上限；
- 明确只有 harness 的 protocol 和 ExperimentPlan 是指令；
- 拒绝从候选文本中读取“修改 evaluator”“读取隐藏集”等指令；
- 记录最终 prompt hash 和 selected record ids。

### 16.4 Private Audit

private audit 必须：

- 在 private seed 生成前冻结 Top-K bundle；
- 绑定 run manifest、EB snapshot、record、source、bundle、runtime 和 policy artifact hash；
- 由独立 evaluator request 执行；
- 将结果写入独立 audit directory；
- 不调用 Context、Proposal、semantic consolidation 或 analogy graph update；
- audit 后不恢复本次 run 的搜索。

---

## 17. 可观测性、验收指标与停止线

### 17.1 每轮必须记录

- Context packet ids、hash、字符数和估计 token；
- Agent backend、model、temperature、latency 和失败；
- ExperimentPlan 和 parser verdict；
- Proposal parent/inspiration、operator、diff size 和 build attempts；
- sandbox status、CPU/wall time、peak RSS、process count、output size；
- artifact protocol、hash 和 runner validation；
- evaluator calls、per-instance scores、SE/CI；
- BehaviorProfile version；
- island epoch、behavior cell 和 lineage；
- analogy/control pairing和结果；
- EB commit version。

### 17.2 工程验收

系统级必须满足：

- 同一 ledger 和 artifacts 重建 deterministic projections 位精确一致；
- LLM annotations 不参与 deterministic rebuild；
- Feature IR legacy 评分保持回归一致；
- MLP runner 对单样本、批次拆分和调用顺序位精确一致；
- guided/control lineage 无缺失；
- private audit 零回流；
- 任一 schema/path/hash 错误 fail closed；
- failed candidate 也产生完整、不可变、可检索事件。

### 17.3 研究验收

P3 进入确认性实验前必须满足：

- Pilot 中至少 80% guided/control 对完成有效评估；
- BehaviorProfile 对相同 frozen policy 重复计算一致；
- 迁移预测在执行前冻结；
- matched control 的预算差异在预注册容差内；
- 至少存在 `transfer_supported`、`transfer_refuted` 或 `inconclusive` 三类中的两类，证明系统没有把所有输出强行解释为成功；
- 所有 arm 的 evaluator-call 和 token 统计完整。

### 17.4 停止线

出现以下任一情况，停止扩大搜索空间并回到协议/测量修复：

- private data 或 audit result 回流；
- candidate code 进入 evaluator 进程；
- artifact schema 被未知字段或文件绕过；
- BehaviorProfile 对相同工件不可重复；
- guided/control 无法匹配预算；
- 超过 30% analogy 结果因 invalid control 无法判定；
- 岛模型在 Pilot 中没有增加行为覆盖且显著降低 best-so-far AUC；
- Context 的类比 confidence 无法区分支持和反驳；
- sandbox 无法提供声明的硬隔离。

Transformer 扩展的额外停止线：在 Markov 百慕大任务上若无明确序列信息增益，不得仅为扩大模型类别加入 Transformer runner。

---

## 18. 迁移与实施阶段

### 18.0 最小可发表路径

完整 P-1 → P4 路径需要 15-20 周。如果时间或资源不允许走完全部阶段，以下最小路径可以产出一个 bounded 但可发表的结果：

1. **P-1**（1-2 周）：replay BehaviorProfile，验证 schema 可行性。
2. **P0 + P1 partial**（3-4 周）：统一工件基础设施，让 Feature IR baseline、MLP 和至少一种 hybrid 共存于同一 evaluator 管道。
3. **条件性算法优势分析**（1-2 周）：用已有 EB 数据和新增的 BehaviorProfile per-instance metrics，回答 §3.2 次问题 6——提取基于合约特征的算法优势模式。这不需要完整的 Analogy Graph 或独立种群岛。
4. **简化 A/B 对比**（2-3 周）：只跑 B vs A（FunSearch-style 岛 vs 现有单阶段），或 E vs B（完整 analogy vs FunSearch baseline），用少于 5 组的 arm 获得初步搜索效率结论。

该路径产出的 claim 范围更窄（不包含完整的 H1-H5 验证），但足以支撑一篇有 empirical finding 的论文。完整 P3-P4 路径在时间允许时继续执行。

### P-1：离线 Replay 与测量可行性（1-2 周）

目标：在不改变主循环的情况下验证 BehaviorProfile 和 Analogy schema 是否可用。

- [ ] 冻结 `BehaviorProfile.v1`、`MechanismCard.v1`、`AnalogyHypothesis.v1` schema；
- [ ] 对现有 Feature IR baseline、Ridge 变体和实验性 MLP 生成 replay profile；
- [ ] 建立至少 8 个手工审查的 source-target intervention 对，其中同时包含预期成功和预期失败；
- [ ] 生成 matched controls；
- [ ] 验证 profile 重复性、descriptor 稳定性和 TransferGain 计算；
- [ ] 产出 Pilot variance、失败率和成本估计；
- [ ] 决定是否进入 P0；若无法形成有效 control，则停止 analogy 集成。

### P0：统一现有安全与工件基础设施（2-3 周）

目标：把已有实验性代码变成一致、可追溯但仍 default-off 的公共搜索能力。

- [ ] 保留原 `PRD_CONTEXT_AGENT_V2.md` 和 legacy Feature IR；
- [ ] 将现有 MLP policy artifact 规范设为唯一 v1 wire format；
- [ ] 从 PRD 和代码中删除 per-instance artifact 接受 `training_meta.json` 的冲突描述；
- [ ] 将 training telemetry 移到 trusted supervisor 输出；
- [ ] 接入 per-instance training pipeline 的显式 experimental task protocol；
- [ ] 将 algorithm Top-K freeze/provenance 接入对应 audit entrypoint；
- [ ] 完成 sandbox hardening 和 adversarial acceptance tests；
- [ ] 建立 `present / verified / integrated / default-on` 状态矩阵；
- [ ] legacy Feature IR 全量回归通过。

### P1：代码搜索与第一种混合协议（3-4 周）

目标：新候选统一生成代码，支持 linear、MLP 和 residual hybrid。

- [ ] 定义 `AlgorithmBundle.v1`；
- [ ] Proposal Agent 支持 parent + inspiration + generation operator；
- [ ] 支持函数级 diff 和 compound-edit 检测；
- [ ] 实现 `continuation-linear.v1`；
- [ ] 稳定 `openhyra-policy-spec.v1`（MLP continuation）；
- [ ] 实现 `continuation-residual-hybrid.v1`；
- [ ] 实现 `feature_augment` 和 `residualize`；
- [ ] 完成静态预检、单实例 probe 和结构化 repair；
- [ ] 记录完整 bundle/source/artifact provenance；
- [ ] 运行 50 个候选的工程 smoke search，不做科学结论。

### P2：Experience Bank v5 与 FunSearch-style 岛（3-4 周）

目标：把事实、行为、种群和语义解耦。

- [ ] 增加 event logs 和 content-addressed object store；
- [ ] 实现 deterministic projection rebuild；
- [ ] 实现四个独立 population islands 和 island epochs；
- [ ] 实现 10-round reset/migration；
- [ ] 实现 32+ 维 BehaviorProfile 和四维 behavior cell key；
- [ ] 实现 lineage graph；
- [ ] 实现 semantic annotation events，但不用于岛归属；
- [ ] 实现 legacy record replay/migration；
- [ ] 验证删除 `derived/` 后位精确重建。

### P3：Context Analogy 与定向 Proposal（4-5 周）

目标：形成可证伪的跨算法迁移闭环。

- [ ] 实现 Evidence Builder；
- [ ] 实现 Analyst evidence-bound 输出；
- [ ] 实现 MechanismCard 事实/推断分离；
- [ ] 实现 Analogy Graph；
- [ ] 实现 preregistered AnalogyHypothesis；
- [ ] 实现 matched-control 自动计划；
- [ ] 实现 Portfolio/Analysis/Analogy/Proposal packets；
- [ ] 实现检索 provenance；
- [ ] 运行每组 10 runs 的 Pilot；
- [ ] 依据停止线决定是否进入确认性实验。

### P4：确认性系统实验与封闭科学轨（4-6 周）

目标：验证搜索效率和类比迁移，而不是展示单个最好结果。

- [ ] 预注册主指标、主比较、最小实际效应和 power analysis；
- [ ] 运行 A-E 五组 matched-budget 实验；
- [ ] 报告 evaluator-call、token 和 wall-clock 三种成本；
- [ ] 运行 held-out instance、algorithm subfamily、surface 和 lineage tests；
- [ ] 对冻结 Top-K 执行 private audit；
- [ ] 分析 H1-H5 和零结果；
- [ ] 公开 schema、代码、非私有配置、随机种子和完整失败统计。

### P5：未来协议评审

只有满足任务动机和安全门槛后才启动：

- `continuation-expression.v1`；
- 非 Markov/path-dependent benchmark；
- `sequence-continuation.v1`；
- Decoder-only Transformer artifact；
- direct stop policy 与独立 dual proxy。

P5 每个协议单独形成设计与安全评审，不作为 v5 主实施计划的完成条件。

---

## 19. 维护、模块边界与版本策略

### 19.1 建议模块边界

| 模块 | 单一职责 |
|---|---|
| `eb.py` | append-only core record 和兼容读取 |
| `experience_events.py` | v5 event schemas 和 append writers |
| `object_store.py` | content-addressed immutable objects |
| `behavior_profiler.py` | evaluator-owned public/dev probes |
| `behavior_index.py` | deterministic descriptors、cells 和近邻 |
| `island_scheduler.py` | island epochs、sampling、reset 和 migration |
| `mechanism_cards.py` | facts 与 annotations 组装 |
| `analogy_graph.py` | hypothesis/result edges 和投影 |
| `context_retrieval.py` | packet 构建、预算和 provenance |
| `context_agent.py` | Analyst、Hypothesizer、ExperimentPlan orchestration |
| `proposal_agent.py` | 代码生成、diff、repair |
| `policy_protocols/` | versioned trusted loaders/runners |
| `algorithm_auditing.py` | candidate bundle freeze 和 per-cell provenance |

模块名是建议边界，不要求一次性重命名现有文件。实施时优先保持 public interfaces 稳定，逐步抽离职责。

### 19.2 Schema 版本

- schema 名称携带主版本，例如 `.v1`；
- 同一主版本只允许增加不影响解析的文档说明，不允许增加可接受字段；
- 新字段、新文件、新激活函数、新 runner、新输出语义或新 artifact layout 必须升级 schema；
- reader 可以支持多个历史版本，但每个版本单独 fail closed；
- run manifest 冻结本次 run 允许的 schema 列表。

### 19.3 Derived Index 版本

每个 derived view 保存：

- schema/version；
- source ledger length；
- source event digests；
- projection code hash；
- descriptor/bucket config hash；
- created time。

发现 stale 或 source mismatch 时必须重建，不允许静默继续使用。

### 19.4 LLM 版本漂移

所有 annotation 和 plan 保存 model/backend/prompt/response hash。模型升级后：

- 旧 annotation 保留；
- 新模型不能原地改写旧卡片；
- 需要重分析时追加新 annotation；
- 对比研究冻结模型版本；
- 产品运行可以升级，但必须开启新的 run manifest。

### 19.5 文档状态

实施清单不再只有 checkbox，另外维护状态：

| 状态 | 含义 |
|---|---|
| `planned` | 仅存在 PRD |
| `present` | 代码存在，未形成验收证据 |
| `verified` | 独立测试/审查通过 |
| `integrated` | 已接入真实调用链 |
| `default-off` | 已集成但仅实验性启用 |
| `default-on` | 满足安全和回归门槛，正式启用 |

任何文档不得把 `present` 描述成 `integrated`，也不得把 `verified` 描述成已经产生科学证据。

### 19.6 兼容和回滚

- legacy Feature IR 路径保持独立，可作为回归基线；
- 新 protocol 通过 task config 显式选择；
- v5 derived views 可以完全删除并重建；
- 关闭 v5 Context 后可以回退到现有 `build_inspiration()`；
- 回滚不得删除 v5 events，只停止消费；
- private audit 协议升级必须保留旧 frozen manifest 的可读性。

---

## 20. 风险登记与应对

| 风险 | 严重度 | 触发信号 | 应对 |
|---|---|---|---|
| 语义岛导致搜索隔离 | 高 | 混合候选和跨族迁移减少 | 岛与语义标签正交，禁止按标签归岛 |
| 行为相似被误当机制相同 | 高 | 高相似候选迁移频繁失败 | 只通过 intervention + control 升级关系 |
| LLM 自我解释幻觉 | 高 | MechanismCard 无证据引用 | facts/inference 分区，强制 evidence ids |
| public/dev 过拟合 | 高 | public 提升、private 重排严重 | hidden audit 不回流，held-out instance/surface |
| 混合模型计算优势混淆 | 高 | 参数/时间显著增加 | 参数、时间或推理成本 matched control |
| sandbox 逃逸/资源旁路 | 致命 | 越权读写、detached process | default-off，先修 kernel/container boundary |
| derived view 不可重建 | 高 | 相同 ledger 产生不同 island/cell | 确定性投影和 source hash；LLM 单独存事件 |
| Context token 膨胀 | 中 | 成本上升但搜索无收益 | packet 配额、消融和 token-adjusted AUC |
| 岛重置损失短期最优 | 中 | AUC下降、覆盖无提升 | 保留全局 best，reset 只改采样视图 |
| behavior descriptor 选错 | 中 | cell coverage 高但无搜索收益 | P-1 replay、版本化 descriptor、对照实验 |
| matched control 无效 | 高 | invalid control >30% | 收紧 operator、自动预算检查、停止线 |
| Transformer 变成无意义扩张 | 中 | Markov任务无序列增益 | 延后到非 Markov/path-dependent benchmark |
| schema 过度复杂 | 中 | 大量记录无法完整生成 | 分层对象、最小 required fields、P-1 验证 |
| LLM 代码生成能力不足 | 高 | compound_edit 率 >50% 或 build success rate <60% | 简化 ExperimentPlan 的 intervention 粒度，增加 repair 预算，或 fallback 到模板化生成 |
| probe suite 统计过拟合 | 中 | public probe 提升但 held-out instance 无改善 | 每 major phase 扩展 probe suite，保留 held-out 实例，分层报告 |

---

## 21. 决策记录

### D1：保留旧 PRD，新增系统级 v5

原因：旧 PRD 含有安全边界、per-instance training 和历史讨论价值；直接覆盖会丢失决策谱系。

### D2：岛是独立种群，不是算法类别

原因：语义分岛会阻碍异构算法互相启发；FunSearch-style 岛的主要作用是多样性和局部进化。

### D3：行为单元与语义标签正交

原因：语义描述和实际行为可能不一致，必须同时保留并允许产生反例。

### D4：代码是统一生成表面，不是统一机制表征

原因：训练型模型的行为由代码、数据、随机性、训练过程和权重共同决定。

### D5：Proposal Agent 继续拥有代码生成

原因：Context 应负责研究策略，Proposal 应负责实现；合并会降低独立评估和维护清晰度。

### D6：Analogy 必须通过干预迁移验收

原因：分类、相似度和自然语言解释均不能证明一个机制可迁移。

### D7：第一版只支持 Linear、MLP 和 residual hybrid

原因：足以检验跨算法和混合假设，同时保持 trusted runner 范围可审计。

### D8：Transformer 延后

原因：当前 Markov 任务缺少必需的序列信息，且 trusted sequence runner、安全和成本尚未建立。

### D9：LLM consolidation 不做确定性重建

原因：temperature=0 也不保证跨模型、backend 或时间的位精确结果；应保存 annotation event。

### D10：开放搜索与封闭科学轨继续分离

原因：开放代码可以自行模拟或改变实际样本使用，不能用来识别受控样本效率。

---

## 22. 参考文献

1. Romera-Paredes, B. et al. **Mathematical discoveries from program search with large language models.** Nature 625, 468–475 (2024). https://doi.org/10.1038/s41586-023-06924-6
2. Novikov, A. et al. **AlphaEvolve: A coding agent for scientific and algorithmic discovery.** arXiv:2506.13131 (2025). https://arxiv.org/abs/2506.13131
3. Mouret, J.-B. & Clune, J. **Illuminating search spaces by mapping elites.** arXiv:1504.04909 (2015). https://arxiv.org/abs/1504.04909
4. Longstaff, F. A. & Schwartz, E. S. **Valuing American Options by Simulation: A Simple Least-Squares Approach.** Review of Financial Studies 14(1), 113–147 (2001).
5. Rogers, L. C. G. **Monte Carlo valuation of American options.** Mathematical Finance 12(3), 271–286 (2002).
6. Becker, S., Cheridito, P. & Jentzen, A. **Deep Optimal Stopping.** Journal of Machine Learning Research 20(74), 1–25 (2019).

---

## 文档审阅清单

审阅本 PRD 时，应逐项确认：

- [ ] 是否认可“岛、行为单元、语义标签、Analogy Graph”四者正交；
- [ ] 是否认可所有新候选生成代码、但 evaluator 只执行可信 Runner；
- [ ] 是否认可 Context/Proposal 职责分离；
- [ ] 是否认可第一阶段只支持 Linear、MLP 和 residual hybrid；
- [ ] 是否认可 Analogy 必须配置 matched control；
- [ ] 是否认可 private audit 零回流；
- [ ] 是否认可 P-1 replay 是进入系统改造前的必要门槛；
- [ ] 是否认可系统实验以搜索效率和 TransferGain 为主，而不是单个最高分；
- [ ] 是否认可不从本任务直接外推基础模型通用科学类比能力；
- [ ] 是否认可 v1 behavior cell key 限制为 2D，待规模扩展后再增维；
- [ ] 是否认可 Analogy 分为轻量 tracking（P1）和完整 matched control（P3）两层；
- [ ] 是否认可最小可发表路径（§18.0）作为资源不足时的 fallback 计划。
