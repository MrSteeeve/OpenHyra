# OpenHyra Context Agent v2 — 系统重构方案

> 从特征搜索到算法搜索 — 单 continuation 接口 + per-instance 训练
>
> Status: Draft v4 | Date: 2026-08-29 | Scope: Context Agent, Evaluator, Experience Bank

---

## 目录

1. [概述](#1-概述)
2. [现有系统评估](#2-现有系统评估)
3. [威胁模型与信任边界](#3-威胁模型与信任边界)
4. [候选提交物与可信 Runner](#4-候选提交物与可信-runner)
5. [评估器重构](#5-评估器重构)
6. [Context Agent 改进：三阶段管道](#6-context-agent-改进)
7. [Experience Bank 演进：岛模型 × 三层记忆](#7-experience-bank-演进)
8. [NN 训练基础设施](#8-nn-训练基础设施)
9. [科学定位](#9-科学定位)
10. [实施阶段](#10-实施阶段)
11. [开放问题](#11-开放问题)

---

## 1. 概述

OpenHyra 的搜索空间被 AST IR 锁定在 LSMC 特征函数空间——理论天花板是 Ridge 在特征空间中的线性模型 V(S) ~ w^T * phi(S)。本方案将搜索空间从"特征选择"扩展到"算法选择"，同时保持评估器拥有的信任边界不退化。

**核心设计约束（v1/v2 review 后确立）：**

- **训练自由，推理可信。** 候选提交训练代码 (`train.py`)，在沙箱中对每个合约实例独立训练，可使用任意 Python 逻辑（自定义 loss、curriculum learning、importance sampling 等）。训练完成后导出权重文件。评估器用自有的可信 runner 加载权重做推理。候选代码永远不进入评估器进程。
- **每实例重新训练，搜索的是算法而非模型。** 当前系统对每个合约实例独立拟合 Ridge（特征定义迁移，权重重新产生）。V3 保持此模式：同一份 `train.py` 在每个实例上独立运行，测量的是训练算法的质量，不是特定权重的泛化能力。
- **Primal-dual 对偶有效性不可妥协。** continuation value 必须是确定性的、无状态的固定函数，由可信 runner 保证。
- **公开搜索集与私有审计集的边界不退化。** 私有审计结果不进入 Experience Bank，不回传 Context Agent。
- **架构决策需要论证，实现细节需要消融验证。** 三阶段管道是搜索空间扩展后的架构必然（见 Section 6.1），但每个阶段的具体实现（数据源、prompt 结构、context 分配）需要消融实验验证。

**三种模型的区分（v3 明确）：**

| 模型 | 角色 | 示例 | 谁控制 |
|------|------|------|-------|
| Agent LLM | 分析历史实验（Stage 1）→ 提出假设（Stage 2）→ 编写 train.py（Stage 3） | Claude, GPT | harness |
| 策略模型 | 输入 (state, time, params) → 停止决策 | Ridge, MLP, RandomFeatures | 训练：候选代码；推理：evaluator |
| 金融随机模型 | 模拟价格路径，计算收益 | Black-Scholes GBM | evaluator 独占 |

---

## 2. 现有系统评估

### 2.1 搜索空间过窄（确认）

当前候选只能提交 `feature_program.json`——有界 AST IR，最多 16 特征、128 节点、深度 8。评估器固定 Ridge 回归 (alpha=1e-6)、路径数、所有金融计算。

- 搜索锁死在"LSMC 特征工程"——Longstaff-Schwartz (2001) 后已被充分研究
- 理论天花板：Ridge 在特征空间中是线性模型
- 排除了 Deep Optimal Stopping (Becker et al. 2019) 等纯 NN 方法
- 排除了 NN-LSMC 混合方法（Herrera et al. 2021, Goudenege et al. 2023——见 9.1 文献）
- 无法搜索回归器本身

### 2.2 Context Agent 现状（公正描述）

当前实现 (`context_agent.py`) 并非 PRD v1 描述的那么简陋。实际能力：

**已有的结构化能力：**
- 代表性采样 (`_select_history_records`, line 245)：确定性优先级填充——种子记录 (2)、全局最优 (1)、最近失败 (limit//4)、证据等级、研究配额 (limit//8)、方向多样性 (limit//4)、时间近度填充
- 全局统计计算 (`_history_summary`, line 318)：状态分布、方向计数和频率
- 跨迭代记忆 (`_previous_analysis`, line 400)：读取 analyses/iter_XXXX.json
- 活跃方向注入 (line 503)：避免重复提议
- 停止权外部审核 (line 509)：stop 请求由确定性 Controller 独立审核，LLM 不直接决定停止

**真正缺失的：**
- 实验间关系没有显式建模——聚类和相似度判断完全隐式依赖 LLM 上下文
- 代表性采样策略从未被实验验证——不知道这 80 条记录是否最优
- 经验积累无层次——EB 是 append-only 平面表，120 次实验后信息密度低
- 决策质量无测量——没有消融数据说明当前方案的提升贡献

### 2.3 Experience Bank 现状（公正描述）

EB (`eb.py`) 是 append-only JSONL，核心记录字段：id, parent, score, status, description, path, log_tail, metrics, metadata, created。

**已有的关系元数据**（`harness.py` line 996-1013）：
- `duplicate_of`：SHA256 完全匹配
- `numeric_duplicate_of`：数值结果集合哈希匹配
- `repair_of` + `repair_note`：修复链
- `candidate_seed`：候选种子
- `editable_file_sha256`：可编辑文件哈希
- `context_meta`：Context Agent 决策元数据
- `attempt_index` + `attempt_kind`：尝试序列

**缺失的：** 显式聚类/分组、层次化索引、结构化总结。现有关系是"点对点"（A 是 B 的修复、A 和 B 内容相同），没有"群组"概念。

---

## 3. 威胁模型与信任边界

### 3.1 当前信任模型

当前系统的安全来自候选只提交**数据 IR**：

```
候选 → feature_program.json (纯数据)
                │
                ▼
评估器（可信进程）：解释 IR → 构造特征 → Ridge 回归 → 策略应用 → 对偶构造 → 评分
```

评估器从不执行候选代码。沙箱 (`sandbox.py` line 33) 使用 macOS Seatbelt：
- `(allow default)` + `(deny network*)` + `(deny file-write*)` except sandbox dir
- `(deny file-read* (literal "{evaluator}"))` 阻止读取评估器源码
- 但这不是机密测试集的隔离边界——候选可以读取 sandbox 外的大部分文件

### 3.2 扩展搜索空间的威胁

如果候选提交可执行代码（`policy.py`），评估器需要调用候选的 `should_stop()` 和 `continuation_value()` 函数。这创造了以下攻击面：

| 威胁 | 后果 | 当前 IR 是否存在 |
|------|------|-----------------|
| 候选代码在评估器进程内执行 | 候选可读取/修改评估器状态、测试集、随机种子 | 否（IR 是数据，评估器解释） |
| `should_stop` 区分 outer 和 inner 调用 | 破坏对偶构造的鞅性质，"上界"不再是上界 | 否（确定性 Ridge 预测） |
| `continuation_value` 有副作用或非确定性 | 对偶增量的条件期望不为零，Rogers bound 无效 | 否 |
| 候选通过时间/调用顺序/文件侧信道推断隐藏集 | 对隐藏集过拟合 | 受限（IR 不含逻辑） |
| 候选在评估期间继续训练（更新权重） | 策略不再"冻结"，定价和对偶阶段看到不同策略 | 否 |

### 3.3 设计决策：训练自由 + 推理可信（per-instance）

**候选提交训练代码 (`train.py`) + 推理声明 (`manifest.json`)。evaluator 对每个合约实例独立调用候选沙箱训练，然后用自有可信 runner 加载训练产出的权重做推理。**

```
evaluator 控制的 per-instance 循环：
┌─────────────────────────────────────────────────┐
│ For each (instance, repeat):                     │
│                                                  │
│   evaluator 生成训练路径                          │
│       │  training_paths.npy (该实例专有)          │
│       ▼                                          │
│   候选沙箱（不可信，隔离进程）                      │
│       │  python train.py                         │
│       │  读取 training_paths.npy + instance.json │
│       │  任意训练逻辑（自定义 loss、curriculum 等） │
│       │  导出权重 → policy_artifact/weights.npy   │
│       ▼                                          │
│   ══════ 信任边界：权重文件 ══════                 │
│       │                                          │
│   evaluator 验证权重（维度、有界、有限）             │
│       ▼                                          │
│   可信 runner（evaluator 进程内）                  │
│       │  加载权重 → 确定性推理                     │
│       │  should_stop + approximate_value          │
│       ▼                                          │
│   evaluator：定价 → 对偶构造 → 该实例评分          │
└─────────────────────────────────────────────────┘
聚合所有实例评分 → paired LCB
```

**为什么是 per-instance 训练：**

当前 evaluator 已经是 per-instance 的——同一个 feature_program.json 在每个实例上独立做 Ridge 拟合（evaluator.py line 836）。候选提交的是"用什么特征"（算法/结构），而不是"训练好的权重"（具体模型）。V3 保持此模式：候选提交的是"怎么训练"（train.py），evaluator 在每个实例上独立调用。

这意味着评估的是**训练算法的质量**（给定 N 条路径，能学出多好的策略），而不是特定权重对新合约的泛化能力。这与 PRD 的科学问题直接对应——比较不同算法在不同样本量下的表现。

**候选训练自由度：**

候选的 `train.py` 可以做任何事情，只要最终导出 manifest 声明的 runner_type 所需的权重格式：
- 自定义损失函数（不限于 MSE/Ridge）
- Curriculum learning（先简单样本后困难样本）
- 重要性采样（候选自行计算采样权重）
- 多轮训练 + early stopping
- 集成学习（训练多个子模型，导出集成权重）
- 纯手工规则（直接写权重，不训练）

**计算开销：**

小 MLP 每实例训练 ~5-10 秒。4 实例 × 2 repeat × 4 候选 = 32 次沙箱调用。总计约 3-5 分钟/迭代（当前 Ridge 约 20-30 秒/迭代）。对于搜索空间的巨大扩展，这个开销完全可接受。

---

## 4. 候选提交物与可信 Runner

### 4.1 唯一的 Runner 接口：`continuation`

当前 evaluator 的 `FrozenPolicy` 只暴露一个核心函数（evaluator.py line 559）：

```python
def continuation(self, time_index: int, states: np.ndarray) -> np.ndarray:
    """给定时间步和状态，返回折现到 t=0 的继续价值估计。"""
```

evaluator 自行从中推导停止决策和对偶代理（evaluator.py line 568-572, 644）：

```python
# 停止决策（evaluator 计算，不是候选）
exercise = (immediate > 0.0) & (immediate >= continuation)

# 对偶价值代理（evaluator 计算）
approximate_value = max(immediate, continuation)
```

**V4 的可信 Runner 保持这个设计不变。** Runner 只实现一个函数：

```python
class TrustedRunner:
    def continuation(self, time_index: int, states: np.ndarray,
                     instance: dict) -> np.ndarray:
        """加载候选导出的权重，计算折现到 t=0 的继续价值估计。
        由 evaluator 代码保证：
        - 确定性（相同输入 → 相同输出）
        - 无状态（不依赖调用顺序、次数）
        - 有界（输出裁剪到 [-1e6, 1e6]）
        - 有限（拒绝 NaN/Inf 权重）
        """
        ...
```

停止决策、到期处理、`max(immediate, continuation)` 的对偶代理构造全部由 evaluator 完成。候选/runner 永远不决定何时停止。

**为什么不是双接口：** 如果 `should_stop` 和 `approximate_value` 是两个独立的候选输出，它们可能由不同模型产生，导致下界策略和上界代理不一致。单 `continuation` 接口从根源上消除了这种分歧。

**Deep Optimal Stopping（直接学习停止决策）的处理：** Becker et al. 2019 风格的纯停止决策 NN 不提供数值化的 continuation value，无法直接用于 Rogers 对偶构造。如需支持，应另设协议并说明对偶代理的来源。V4 Phase 1 不支持此类策略。

### 4.2 Per-timestep 权重结构

当前 evaluator 的 `FrozenPolicy` 对每个行权时点维护独立的 `RidgeStep`（evaluator.py line 556）：

```python
@dataclass(frozen=True)
class FrozenPolicy:
    steps: tuple[RidgeStep, ...]  # 每个行权时点（除最后一个）一个模型
```

V4 的权重工件必须对齐此结构。

**权重目录布局：**

```
output/
├── step_000.npy    # 时点 0 的模型权重
├── step_001.npy    # 时点 1 的模型权重
├── ...
├── step_NNN.npy    # 时点 N-2 的模型权重（最后一个时点无模型——到期必须行权）
├── normalization.json  # 每时点的输入归一化统计 {"mean": [...], "scale": [...]}
└── training_meta.json  # 可选：训练过程元数据
```

**权重文件约束：**

| 约束 | 规则 | 验证方 |
|------|------|-------|
| 文件数 | 恰好 n_exercise_times - 1 个 step_*.npy | evaluator |
| dtype | float64 | evaluator (`np.load(allow_pickle=False)`) |
| 维度 | 与 manifest 中声明的网络结构一致 | evaluator |
| 数值 | 所有元素有限 (`np.all(np.isfinite(...))`) | evaluator |
| 大小 | 单个文件 <= 1 MB, 总计 <= 8 MB | evaluator |
| 路径 | 纯文件名，禁止绝对路径和 `..` | evaluator |
| 加载 | `np.load(path, allow_pickle=False)` | evaluator |
| 哈希 | evaluator 计算每个文件的 SHA256，记入 metrics | evaluator |

### 4.3 manifest.json（推理声明）

候选提交前确定，对所有实例共用。

```json
{
  "schema": "openhyra-policy-spec.v1",
  "runner_type": "mlp",
  "inference_config": {
    "input_dim": "n_assets",
    "layers": [64, 32],
    "activation": "tanh",
    "output_dim": 1,
    "output_clip": [-1e6, 1e6]
  },
  "output_semantics": "discounted_continuation_value_t0",
  "normalization": "per_step",
  "weight_pattern": "step_{:03d}.npy"
}
```

**`output_semantics`** 必须声明为以下之一：
- `discounted_continuation_value_t0`：折现到 t=0 的继续价值（与当前 Ridge 一致）
- Runner 根据此声明决定是否需要额外折现调整

### 4.4 两条轨道：开放搜索 vs 封闭科学

**开放搜索轨（正常迭代）：**
- 候选提交 `train.py`，可执行任意 Python
- 候选可以自行模拟额外路径（无法阻止——有 instance.json 就能写 GBM 模拟器）
- 评估指标：固定时间/内存/推理预算下的最终 paired LCB
- 不能称为"样本效率"——因为实际使用的样本量无法控制
- evaluator 记录 `train.py` 的运行时间、内存峰值、输出大小到 metrics

**封闭科学轨（对照实验）：**
- 不使用候选 `train.py`——evaluator 有固定的、经过审计的训练实现
- 候选只提交 manifest.json（模型架构 + 超参数）
- evaluator 用自有训练代码在固定路径上训练，路径数是受控自变量
- 这才能真正度量"给定 N 条路径，哪种架构/超参学出更好的策略"

**为什么必须分两条：** 开放搜索轨中，候选拿到 Black-Scholes 参数就能用 numpy 自行模拟任意多路径。"不提供模拟器 API"不等于"不能模拟"。声称在这种条件下测量"样本效率"是伪命题。

### 4.5 train.py（训练代码，仅开放搜索轨）

候选的训练代码。在沙箱中执行，每个 (instance, repeat) 独立调用一次。

**输入**（evaluator 提供，只读）：
- `training_paths.npy`：该实例的训练路径 `(n_paths, n_steps, n_assets)`
- `instance.json`：合约参数（payoff_type, strike, rate, volatilities, exercise_times 等）
- `payoffs.npy`：各时间步的折现收益 `(n_paths, n_steps)`
- `discount_factors.npy`：折现因子 `(n_steps,)`

**输出**（写入 evaluator 指定的输出目录）：
- `step_*.npy`：按 manifest weight_pattern 命名的权重文件
- `normalization.json`：每时点归一化统计
- `training_meta.json`（可选）

**沙箱约束（必须在 P0 就位，开放 train.py 之前）：**
- 无网络
- 文件写入限于输出目录
- 文件读取限于：候选源码、输入数据、白名单运行时库
- **不得读取**：宿主仓库、其他实例数据、审计请求、历史工件、evaluator 源码
- 超时：60 秒 / 实例
- 内存：1 GB
- 文件数上限、总字节上限

### 4.6 可信 Runner 注册表

| runner_type | `continuation` 逻辑 | per-step 权重含义 | 保证 |
|------------|---------------------|------------------|------|
| `ridge_lsmc` | features.json 定义特征 → 标准化 → 线性预测 | `[intercept, coef_1, ..., coef_k]` | 确定性、有界、向后兼容 |
| `mlp` | 前馈 NN: input → hidden layers → scalar output | 各层 weight matrix + bias vector | 确定性、有界（输出裁剪） |
| `random_features` | `sigma(W_fixed * S) * alpha` | `W_fixed` (evaluator seed 生成) + `alpha` | 确定性、有界 |

Phase 1 实现 `ridge_lsmc`（向后兼容）和 `mlp`。

**Runner 验收标准（替代 v3 的错误标准）：**

v3 用 "10K 次 bound_order_ok > 99.5%" 作为 runner 验收——这是错误对象。Rogers 对偶有效性来自鞅构造的条件中心化（evaluator.py line 654-690），不是经验排序通过率。

正确的 runner 验收测试：

| 测试 | 验证什么 | 通过标准 |
|------|---------|---------|
| 确定性 | 相同 (time_index, states) → 相同输出 | 100% 位精确一致 |
| 无状态 | 改变调用顺序 / 批次拆分 → 相同输出 | 100% 位精确一致 |
| 有界 | 输出 <= output_clip | 所有输出有限且在裁剪范围内 |
| 鞅终值 | martingale_terminal_mean 的 95% CI 覆盖 0 | 多次独立重复下覆盖率 > 90% |
| 已知基准 | 在有解析解的实例上，上下界均覆盖真值 | CI 覆盖真值 |

`raw_bound_order_ok` 仅作为诊断报告，不作为合法性判定。

### 4.7 向后兼容

现有 AST IR 候选 (`feature_program.json`) 不需要 `train.py`。评估器检测到旧格式后，使用内置的 evaluate_features + fit_ridge 流程（即当前逻辑），等效于：

- `runner_type = "ridge_lsmc"`
- `features.json` = 原 IR
- 训练 = evaluator 内置 Ridge（不调用候选沙箱）

Phase 1 评估器同时支持旧格式和新格式。历史数据不作废。

### 4.8 候选类型光谱

| 策略类型 | runner_type | train.py 做什么 | 导出什么 |
|---------|------------|----------------|---------|
| 传统 LSMC | `ridge_lsmc` | 无需 train.py（旧格式兼容） | features.json |
| 自定义特征 + Ridge | `ridge_lsmc` | 构造特征 → Ridge 拟合 → 导出系数 | features.json + weights.npy |
| 自定义回归 + 特征 | `linear` | 任意回归方法（Lasso, ElasticNet 等）→ 导出系数 | weights.npy |
| 浅层 MLP | `mlp` | 手写 MLP 训练 → 导出各层权重 | weights.npy |
| 随机特征网络 | `random_features` | 生成随机隐藏层 → 训练线性输出层 → 导出全部权重 | weights.npy |
| 自定义训练 MLP | `mlp` | Curriculum + Huber loss + early stopping → 导出权重 | weights.npy |
| （未来）ONNX | `onnx` | PyTorch 训练 → torch.onnx.export → 导出 ONNX | model.onnx |

**天花板在哪里：** 搜索空间的限制在 runner 注册表的推理表达能力。`mlp` runner 可以表示任意前馈网络（通过 manifest 声明层结构）。`ridge_lsmc` 可以表示任意 AST 特征的线性组合。未来 `onnx` runner 可以表示几乎任意计算图。天花板很高，远高于当前 AST IR。

**训练自由度没有天花板：** 候选的 `train.py` 可以用任何 Python 方法训练。梯度下降、进化算法、贝叶斯优化、集成学习、课程学习——任何产出正确权重格式的方法都行。

---

## 5. 评估器重构

### 5.1 三层数据分离

当前系统已经区分公开搜索和私有审计（task.json line 23-49）。V2 保持并强化这条边界：

| 层级 | 用途 | 结果去向 | 可迭代 |
|------|------|---------|-------|
| **公开搜索集** (bermudan-public-v1) | 候选排序和选择 | 标量 paired LCB → EB | 是，每轮反馈 |
| **扩展开发集** (bermudan-dev-v1) | 结构分析（按实例/参数配置） | 逐实例标量分数 → EB | 是，但仅搜索阶段 |
| **私有审计集** (bermudan-hidden-v1) | 一次性验收 | 报告 → 人工审阅，不进入 EB | 否，一次性 |

**关键约束：** 扩展开发集的逐实例分数可以进入 EB 供 Context Agent 做结构分析（如"该策略在高相关性下表现差"），但私有审计集的任何结果不得进入 EB，不得回传 Context Agent。

### 5.2 评估流程（per-instance 训练 + 推理）

**开放搜索轨流程：**

```
evaluator 接收候选提交物 (manifest.json + train.py)
    │  验证 manifest 格式、runner_type 已注册
    ▼
For each (instance, repeat) in 公开搜索集:
    │
    ├─ evaluator 生成训练路径: simulate_paths(instance, n_paths, seed)
    │  → training_paths.npy, payoffs.npy, discount_factors.npy, instance.json
    │
    ├─ 调用候选沙箱: python train.py output_dir (隔离进程)
    │  → 候选读取训练数据，执行任意训练逻辑
    │  → 导出 step_*.npy + normalization.json 到输出目录
    │
    ├─ evaluator 验证权重工件:
    │  → 文件数 == n_exercise_times - 1
    │  → np.load(allow_pickle=False), dtype float64
    │  → np.all(np.isfinite(...))
    │  → 维度与 manifest.inference_config 一致
    │  → SHA256 哈希记入 metrics
    │
    ├─ evaluator 可信 runner 加载权重
    │  → runner.continuation(time_index, states, instance) → ndarray
    │
    ├─ evaluator 从 continuation 推导:
    │  → exercise = (immediate > 0) & (immediate >= continuation)
    │  → approximate_value = max(immediate, continuation)
    │
    ├─ evaluator 生成独立定价路径 + 对偶路径
    │  → 计算 lower_bound, upper_bound, paired improvement vs baseline
    │
    └─ 记录该 (instance, repeat) 的评分 + 沙箱 metrics（时间、内存、输出大小）
    
聚合所有 (instance, repeat) → paired_lower_bound_lcb → 最终 score
```

**封闭科学轨流程：**

```
evaluator 接收候选提交物 (manifest.json，无 train.py)
    │  验证 manifest.inference_config
    ▼
For each (instance, repeat, n_paths_budget) in 实验矩阵:
    │
    ├─ evaluator 使用自有训练代码 + 候选声明的架构/超参
    │  → 在受控路径数下训练，训练代码经过审计
    │
    ├─ evaluator 验证 + 加载权重（同上）
    │  → runner.continuation(time_index, states, instance)
    │
    └─ 记录该 (instance, repeat, n_paths) 的评分
    
输出：学习曲线（score vs n_paths），固定预算下跨架构比较
```

**向后兼容：** 旧格式 (feature_program.json) 跳过沙箱调用，evaluator 直接用内置 evaluate_features + fit_ridge（当前逻辑不变）。

### 5.3 评估矩阵（修正 moneyness 标注）

对 **put 期权**：S < K 是 ITM（价内），S > K 是 OTM（价外）。
对 **call 期权**：S > K 是 ITM，S < K 是 OTM。

| 维度 | 配置 | 说明 |
|------|------|------|
| Moneyness (Put) | ATM (S=K), ITM (S=0.8K), OTM (S=1.2K) | 对 Put: S<K 为价内 |
| Moneyness (Call/Max-Call) | ATM (S=K), OTM (S=0.8K), ITM (S=1.2K) | 对 Call: S>K 为价内 |
| 合约类型 | 1d Put, Max-Call (2-5 assets), Basket Put | 维度泛化 |
| 波动率 | 低 (0.1), 中 (0.2), 高 (0.4) | 波动率敏感性 |
| 相关性 | 低 (0.1), 中 (0.5), 高 (0.9) | 多资产相关性影响（仅多资产合约） |

### 5.4 分数体系

| 指标 | 用途 | 方向 | 数据层级 |
|------|------|------|---------|
| `paired_lower_bound_lcb` | 主要排序分数 | max | 公开搜索集 |
| `per_instance_lcb` | Context Agent 结构分析 | max | 扩展开发集 |
| `structural_stability` (CV) | 跨配置稳定性 | min | 扩展开发集 |
| `training_time_s` | 效率参考（不排序） | - | 沙箱记录 |
| `normalized_primal_dual_confidence_gap` | 最终验收 | min | 私有审计集，不入 EB |

**关于 primal-dual gap 的精确表述：**

理论上 E[upper_bound] >= E[lower_bound] 成立（Rogers 2002）。但有限蒙特卡洛样本下 raw gap = upper_mean - lower_mean 可以为负。当前评估器正确地报告 `raw_bound_order_ok` 而不裁剪（evaluator.py line 847）。V2 保持此行为。gap >= 0 是理论期望，不是每次实验的保证。

### 5.5 蒙特卡洛数据策略

**预生成标准训练集：**
- 评估器在每轮开始前模拟标准训练路径集
- 所有候选收到完全相同的训练路径 → 公平比较
- 存储为 numpy `.npy` 格式
- 路径格式：`(n_paths, n_steps, n_assets)` 三维数组

**关于额外模拟（与 Section 4.4 两轨道设计对齐）：**

- **开放搜索轨**：候选 `train.py` 可以用 numpy 自行模拟额外路径（拥有 instance.json 中的 Black-Scholes 参数就足够了）。无法阻止，也不应假装能阻止。evaluator 记录训练时间/内存/输出大小，但不声称度量"样本效率"。
- **封闭科学轨**：无 `train.py`，无法自行模拟。evaluator 控制训练代码和路径数。这是唯一能做样本效率实验的模式。

**OTM 训练数据覆盖问题：**

对 put 期权的 OTM 场景 (S > K)，大部分路径终止于价外，停时决策训练信号稀疏。对 call 期权的 OTM 场景 (S < K) 同理。

OTM 稀疏是 per-instance 的固有挑战，不是跨实例能解决的——每个 instance 独立训练独立模型，instance A 的 ITM 路径不帮助 instance B 的 OTM 学习。应对方式是候选自身的算法选择（如 importance sampling、重新加权、特征工程），这正是算法搜索的意义所在。

---

## 6. Context Agent 改进

### 6.1 设计动机：为什么必须分阶段

当搜索空间从"AST 特征选择"扩展到"训练算法合成"（写 `train.py`），单次 LLM 调用不再可行。原因：

1. **搜索空间类型跳变。** 从组合优化（128 节点 AST 中选特征）变成程序合成（写完整训练代码）。单次调用要同时理解历史模式、提出算法假设、生成正确代码——每一步都需要不同的推理模式。
2. **实验关系语义复杂化。** AST IR 空间中两个实验的差异可以用特征差异描述。算法空间中两个 `train.py` 可能代码完全不同但学到相似策略（不同初始化 + 不同学习率但收敛到相似权重），或代码相似但一个关键细节导致巨大性能差异。
3. **决策维度交互。** 以前只需决定"用什么特征"。现在需同时决定 runner_type、架构、训练算法、损失函数——这些维度之间有交互（tanh + MSE 和 ReLU + Huber 可能需要完全不同的学习率），不适合在一个上下文窗口中隐式处理。

**结论：** 三阶段管道不是"需要消融验证是否值得"的候选改进，而是搜索空间扩展后的架构必然。消融验证的对象是每个阶段的具体实现，不是是否分阶段。

### 6.2 三阶段管道架构

```
EB records + per-instance scores + knowledge index
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Stage 1: 结构分析（Analyst）                      │
│   输入：EB 统计、per-instance 得分矩阵、           │
│         功能相似度聚类、失败模式索引               │
│   输出：结构性洞察列表                             │
│         e.g. "高相关性实例上 MLP 比 Ridge 好 5%， │
│              但 OTM 实例上训练不稳定"              │
│   LLM 调用：1 次，重分析轻决策                    │
└────────────────────┬────────────────────────────┘
                     │ insights: list[StructuralInsight]
                     ▼
┌─────────────────────────────────────────────────┐
│ Stage 2: 假设生成（Hypothesizer）                 │
│   输入：Stage 1 洞察 + 未探索区域 + 活跃方向       │
│   输出：具体的算法假设 + 实验规格                   │
│         e.g. "OTM 不稳定可能因目标函数稀疏，       │
│              试 Huber loss + importance sampling"  │
│   LLM 调用：1 次，重创意轻细节                    │
└────────────────────┬────────────────────────────┘
                     │ hypothesis: AlgorithmHypothesis
                     ▼
┌─────────────────────────────────────────────────┐
│ Stage 3: 代码生成（Synthesizer）                  │
│   输入：Stage 2 假设 + manifest 模板 +             │
│         parent 实验代码（如有）+ 负约束            │
│   输出：train.py + manifest.json                  │
│   LLM 调用：1 次，重实现正确性                    │
└────────────────────┬────────────────────────────┘
                     │ candidate submission
                     ▼
              Proposal Agent 提交
```

**每阶段的 prompt 是独立 context window。** Stage 1 看完整 EB 分析数据但不需要代码细节；Stage 3 看详细的代码模板和 API 文档但不需要全部历史数据。这比单次调用塞入所有信息更高效。

**向后兼容：** 当搜索空间仍为 AST IR（旧格式候选）时，Stage 2 的假设直接是特征组合描述，Stage 3 生成 `feature_program.json` 而非 `train.py`。管道结构不变，内容适配。

**与 Proposal Agent 的关系：** 当前系统中 Context Agent 决定"试什么方向"，Proposal Agent 负责"生成代码"。三阶段管道将代码生成吸收进 Stage 3（Synthesizer），Proposal Agent 退化为提交和格式校验层——接收 Stage 3 的产出，做最终的 schema 合规检查后提交给 evaluator。这是有意的设计：代码生成需要看到 Stage 2 的假设和负约束才能产出有针对性的实现，把它拆到管道外部会丢失上下文。

### 6.3 Stage 1：结构分析

**输入数据：**

| 数据 | 来源 | 用途 |
|------|------|------|
| EB 全局统计 | `_history_summary` | 搜索进度概览 |
| per-instance 得分矩阵 | 扩展开发集 `dev_per_instance_lcb` | 发现实例维度的结构模式 |
| 功能相似度聚类 | 派生索引（见 6.5） | 识别冗余实验和有效变种 |
| 最近 N 次实验摘要 | EB 最近记录 | 当前搜索轨迹 |
| 失败模式索引 | 派生索引 | 已知死胡同 |

**输出格式：**

```python
@dataclass(frozen=True)
class StructuralInsight:
    pattern: str           # 观察到的模式（如"MLP 在高相关性下优于 Ridge"）
    evidence: str          # 支撑证据（实验 ID 和得分差异）
    implication: str       # 对下一步搜索的暗示
    confidence: str        # "strong" | "tentative" | "speculative"
```

**Stage 1 不做决策。** 它只输出观察和模式，不推荐具体实验。这让分析质量可以独立评估——一个洞察是否有数据支撑，与它是否导致好的下一步实验无关。

### 6.4 Stage 2：假设生成

**输入：** Stage 1 的洞察列表 + 未探索区域 + 活跃方向（避免重复）+ 停止信号评估。

**输出格式：**

```python
@dataclass(frozen=True)
class AlgorithmHypothesis:
    # 停止/继续决策（保持现有 ContextDecision 的停止权外部审核机制）
    action: str                      # "continue" | "stop"
    reason: str                      # 停止/继续的理由

    # 假设描述
    hypothesis: str                  # 具体的算法假设（如"Huber loss 缓解 OTM 稀疏"）
    motivation: str                  # 基于哪条 Stage 1 洞察
    expected_gain: float             # >= 0，预期提升幅度
    confidence: float                # [0, 1]

    # 实验规格
    runner_type: str                 # "ridge_lsmc" | "mlp" | "random_features"
    architecture_spec: dict          # 架构参数（层数、宽度、激活函数等）
    training_strategy: str           # 训练策略的自然语言描述
    parent_experiment_id: str | None # 基于哪个实验迭代（代码继承）
    negative_constraints: list[str]  # 明确排除的方向（已知失败）

    # 验证标准
    success_criterion: str           # 如何判断假设成立
    phase: str                       # 搜索阶段标注
```

**Stage 2 做创意决策，不做代码决策。** 它决定"试什么"，不决定"怎么写"。这让假设质量可以独立评估——一个假设是否合理，与代码实现是否正确无关。

### 6.5 Stage 3：代码生成

**输入：** Stage 2 的假设 + manifest 模板 + parent 实验的 `train.py`（如指定了 `parent_experiment_id`）+ 负约束 + API 文档（训练数据格式、权重输出格式、沙箱约束）。

**输出：** `train.py` + `manifest.json`，直接提交给 Proposal Agent。

**Stage 3 的 context window 内容：**
- manifest.json schema 文档（固定，~2K tokens）
- 训练数据格式文档（固定，~1K tokens）
- 沙箱约束文档（固定，~1K tokens）
- parent 实验的 `train.py` 源码（如有，~2-5K tokens）
- Stage 2 的 `AlgorithmHypothesis`（~500 tokens）
- 负约束列表（~500 tokens）

总计 ~7-10K tokens 的精准 context，远小于当前单次调用的 96K 预算。代码生成质量应更高。

### 6.6 实验索引与功能相似度

**语义标签（快速分类）：**

v1 提出的 KnowledgeTree 是单父节点树。v2 改为多标签关系图——一个实验可以属于多个标签（如"Ridge + 多项式特征"同时属于"Ridge 类"和"多项式类"）。

- EB (`records.jsonl`) 保持 append-only JSONL 作为事实账本，不修改
- 关系图是**可重建的派生索引**，有版本号，存在独立文件 (`knowledge_index.json`)
- 从 EB 重建索引是幂等的——删除索引文件后可以重新生成

标签由 LLM 在 EB commit 时增量分配。标签是开放集合，LLM 可以创建新标签。

**功能相似度（补充标签的不足）：**

语义标签基于描述文本，无法捕捉"代码不同但策略相似"的情况。功能相似度基于实验产出：

| 信号 | 计算方式 | 捕捉什么 |
|------|---------|---------|
| per-instance 得分向量相关性 | `corr(score_A, score_B)` 在扩展开发集上 | 两个策略在哪些实例上强/弱的模式是否一致 |
| 权重形态签名 | `weight_shape` + 参数量级分布 | 模型结构相似性 |
| 行权边界重叠度 | 同一批路径上两个策略的停止决策一致率 | 策略行为相似性 |

**per-instance 得分向量相关性是最有价值的信号。** 两个实验如果在所有实例上的得分模式高度相关（>0.9），说明它们学到了本质相似的策略，即使代码完全不同。Context Agent 可以据此避免冗余探索。反之，如果两个实验在某些实例上分数差异很大，这些实例就是区分策略优劣的关键，Stage 1 应重点分析。

**行权边界重叠度的计算开销较大**（需要对同一批路径运行两个 runner），仅在 Stage 1 判定需要深度比较时按需计算，不在每次 commit 时自动执行。

**验收信号（下游指标，不是 ARI）：**

| 指标 | 衡量什么 | 期望方向 |
|------|---------|---------|
| 重复实验率 | 索引是否帮助避免重复 | 下降 |
| 重复失败率 | 索引是否帮助避免已知失败 | 下降 |
| 下一轮提升概率 | 索引是否帮助找到更好方向 | 上升 |
| 方向覆盖率 | 索引是否促进探索 | 上升 |
| Context 构建成本 | 索引是否降低 token 消耗 | 下降 |
| 功能冗余率 | 新实验与已有实验的得分向量相关性 >0.9 的比例 | 下降 |

不使用 HAC ARI 作为校准信号。ARI 只衡量两种聚类是否一致，不衡量分类是否"正确"。两种方法可以稳定地一起犯同样的错误。

### 6.7 Context 压缩策略

三阶段管道的 context 分配与单次调用不同。每阶段只加载该阶段需要的数据。

**Stage 1（Analyst）context 预算：48K chars**

| 内容块 | 预算 | 来源 |
|-------|-----|------|
| 全局概览 | ~5K | EB 统计 + 搜索进度 |
| per-instance 得分矩阵 | ~8K | 扩展开发集，按实验 × 实例列表 |
| 功能相似度聚类摘要 | ~5K | 派生索引中的聚类和相关系数 |
| 最优路径详情 | ~10K | 全局最优实验 + 同组近邻 |
| 最近实验详情 | ~10K | 最近 N 次实验的描述、分数、标签 |
| 失败模式 | ~5K | 已知不可行方向及原因 |
| 保留缓冲 | ~5K | 动态扩展 |

**Stage 2（Hypothesizer）context 预算：24K chars**

| 内容块 | 预算 | 来源 |
|-------|-----|------|
| Stage 1 洞察 | ~5K | `list[StructuralInsight]` |
| 未探索区域 | ~3K | 未涉及的 runner_type / 标签 |
| 活跃方向 | ~3K | 避免重复 |
| 可用 runner 注册表 | ~2K | 架构选项和约束 |
| 最优实验摘要 | ~5K | 当前 top-3 的简要描述 |
| 保留缓冲 | ~6K | 动态扩展 |

**Stage 3（Synthesizer）context 预算：16K chars**

| 内容块 | 预算 | 来源 |
|-------|-----|------|
| Stage 2 假设 | ~1K | `AlgorithmHypothesis` |
| manifest schema | ~2K | 固定文档 |
| 训练数据格式 | ~1K | 固定文档 |
| 沙箱约束 | ~1K | 固定文档 |
| parent 实验 train.py | ~5K | 如有 parent_experiment_id |
| 负约束 | ~1K | 已知失败的代码模式 |
| 保留缓冲 | ~5K | 动态扩展 |

**总 token 成本**：三阶段合计 ~88K chars（三次独立 LLM 调用，不共享 context window）vs 当前 ~96K chars（一次调用）。总成本相当，但每阶段只看该阶段需要的数据，信噪比更高。额外的 API 调用延迟（~2-3 秒/阶段）相对于沙箱训练时间（分钟级）可忽略。

### 6.8 消融验证策略

三阶段架构是确定的设计决策，不做消融。消融验证的对象是每个阶段的具体实现选择：

| 消融实验 | 对照 | 测量指标 |
|---------|------|---------|
| Stage 1：有/无 per-instance 得分矩阵 | 只看聚合分数 vs 看逐实例分数 | 洞察是否更具体、是否关联到实例维度的模式 |
| Stage 1：有/无功能相似度 | 只用语义标签 vs 标签 + 得分向量相关性 | 功能冗余率是否下降 |
| Stage 2：有/无 parent 实验继承 | 每次从头生成 vs 可指定 parent 迭代 | 重复失败率、代码变更幅度 |
| Stage 3：有/无负约束注入 | 不限制 vs 注入已知失败模式 | 重复失败率 |
| 全局：三阶段 vs 单阶段基线 | 现有单次 LLM 调用 | 下一轮 paired LCB 提升概率 |

每项消融跑 50 次迭代作为快速筛选，有信号的用 200 次迭代验证。

### 6.9 扩展 ContextDecision

Stage 2 的 `AlgorithmHypothesis` 替代当前的 `ContextDecision` 作为管道输出。为兼容现有 harness 接口，转换函数将 `AlgorithmHypothesis` 映射到 `ContextDecision`：

```python
@dataclass(frozen=True)
class ContextDecision:
    # 现有字段（保持不变）
    action: str                      # "continue" | "stop"
    analysis: str                    # Stage 1 洞察的摘要
    reason: str
    expected_gain: float             # >= 0
    confidence: float                # [0, 1]
    next_experiment: str | None      # Stage 2 hypothesis 的自然语言描述
    phase: str
    target_claim_id: str | None
    success_criterion: str | None
    # 新增字段
    candidate_type_hint: str         # runner_type 建议: "ridge_lsmc"|"mlp"|"any"
    parent_experiment_id: str | None # 建议基于哪个实验迭代
    negative_constraints: list[str]  # 明确排除的方向（已知失败）
    # 三阶段管道产物
    structural_insights: list[dict]  # Stage 1 洞察（序列化的 StructuralInsight）
    algorithm_hypothesis: dict | None  # Stage 2 假设（序列化的 AlgorithmHypothesis）
```

### 6.10 候选预检与结构化修复

LLM 生成的 `train.py` 出错率远高于选 AST 节点。搜索空间扩展后，如果每个有 bug 的候选都进入完整的 4×2 评估流程，计算浪费严重。

**三层过滤（Stage 3 产出 → 评估器入口）：**

```
Stage 3 产出 train.py + manifest.json
    │
    ▼
Layer 1: 静态检查（不调 LLM，毫秒级）
    │  ast.parse 语法检查
    │  manifest.json schema 合规（validate_policy_manifest）
    │  禁止 import 白名单检查（numpy/json/math/sys/os.path/argparse）
    │  输出目录使用检查（代码中是否引用了 --output 参数并写 step_*.npy）
    │
    ▼  通过 → Layer 2
    │  失败 → 带错误信息回 Stage 3 重试（限 2 次）
    │
Layer 2: 单实例快速探测（1 个 ATM 实例，~10 秒）
    │  沙箱运行 train.py（最简单的合约实例）
    │  检查是否产出了正确格式的权重文件
    │  evaluator 验证权重（load_policy_artifact）
    │
    ▼  通过 → Layer 3
    │  失败 → 结构化修复（见下）
    │
Layer 3: 全量评估（4 实例 × 2 repeat）
```

**Layer 1 的静态检查不是安全边界（安全由沙箱保证），而是快速反馈机制。** 它在 Stage 3 的 context window 还热的时候就捕获明显错误，避免启动沙箱。

**结构化修复（替代当前 repair 链）：**

当前 repair 只给 LLM 一个 `log_tail`（沙箱 stderr）。改为结构化修复上下文：

```python
@dataclass(frozen=True)
class RepairContext:
    original_hypothesis: dict         # Stage 2 的假设（不变，修的是实现不是方向）
    train_py_source: str              # 失败的代码
    failure_layer: str                # "static" | "probe" | "eval"
    error_category: str               # "syntax" | "import" | "shape_mismatch" |
                                      # "nan_weights" | "timeout" | "oom" | "runtime"
    error_message: str                # 具体错误信息
    instance_params: dict | None      # 失败的实例参数（Layer 2+ 才有）
    stderr_tail: str                  # 沙箱 stderr 最后 50 行
```

修复回到 Stage 3（不重跑 Stage 1 + 2），因为假设没变，只是代码实现有 bug。限 2 次修复尝试，超过则：
- 标记最终失败状态 + `error_category` + `error_message`
- 记入 EB（带结构化失败信息，供 Stage 1 的失败模式分析使用）

**与 Stage 1 的耦合：** "MLP 训练超时"和"权重全是 NaN"是完全不同的失败模式。结构化的 `error_category` 让 Stage 1 能做有意义的失败模式聚类——比如发现"所有 ReLU + 大学习率的候选都产出 NaN 权重"，从而在 Stage 2 注入负约束。粗糙的 `log_tail` 做不到这一点。

### 6.11 分级评估

**问题：** 每个候选的完整评估从 ~5 秒（Ridge）增加到 ~3-5 分钟（MLP per-instance 训练）。如果 4 个候选中有 3 个是低质量的，浪费 ~9-15 分钟。

**方案：渐进式评估**

```
候选通过 Layer 1 + 2 后
    │
    ▼
Phase A: 单实例评估（1 个 ATM 实例，最具区分力的实例）
    │  产出 probe_score
    │  比较 probe_score vs baseline_probe_score
    │  如果 probe_score < baseline - 2σ → 早停，不进入 Phase B
    │
    ▼
Phase B: 全量评估（4 实例 × 2 repeat）
    │  产出 paired_lower_bound_lcb
    │
    ▼
记入 EB
```

**早停阈值标定：** Phase A 使用的实例应该是 probe_score 与 full_score 相关性最高的实例。这个相关性可以从 EB 的 `dev_per_instance_lcb` 历史数据中计算——选择与聚合分数相关性最高的那个实例作为 probe 实例。

**早停条件：** `probe_score < best_baseline_probe - 2 * baseline_probe_std`。即候选在 probe 实例上的表现比 baseline 差两个标准差以上。这个条件很保守（只过滤明显差的候选），避免误杀有潜力的候选。

**EB 记录：** 被早停的候选仍然记入 EB，标记 `status: "early_stopped"`，记录 `probe_score` 和 `probe_instance_id`。Stage 1 需要知道这些候选的存在——一个被早停的候选可能暗示了一个有价值但实现不佳的方向。

**计算节省估计：** 假设 50% 的候选会被早停（保守估计），每次迭代从 4×(3-5 分钟) = 12-20 分钟降到 ~8-12 分钟。节省的时间可以用于增加候选数量或增加评估 repeat 数。

---

## 7. Experience Bank 演进

### 7.1 设计哲学：岛模型 × 三层记忆

EB 的核心挑战是：随着实验数量增长（百次迭代），如何让 Context Agent 从历史中高效提取可操作的知识，而不是被原始数据淹没。

**两个设计灵感来源：**

1. **FunSearch 的岛模型**（Romera-Paredes et al., Nature 2023）：维护多个独立种群（islands），每个种群独立进化，定期做岛间迁移。防止搜索坍缩到单一方向。AlphaEvolve（DeepMind, 2025）继承了这一架构。
2. **人类记忆的三层结构**（认知心理学）：情景记忆（具体事件）→ 经由巩固（consolidation）压缩为 → 语义记忆（抽象规则）。ExpeL（Zhao et al., AAAI 2024）验证了"具体例子 + 抽象规则"双层存储优于只存其中一层。

**合成设计：每个岛维护自己的三层记忆。**

- **岛 = 策略家族**（如 Ridge-LSMC、MLP-tanh、Random-Features），是搜索空间的逻辑分区
- **情景记忆** = 该岛的具体实验记录，按新旧分层压缩
- **语义记忆** = 从该岛的实验中提炼出的抽象规则，带置信度标签
- **程序性记忆** = LLM 的预训练知识，EB 不管

### 7.2 不变量

- EB (`records.jsonl`) 保持 append-only JSONL 作为**不可变事实账本**
- 新增字段通过扩展 `metrics` 和 `metadata` 添加，不修改核心 schema
- 岛、记忆层、演化树都是**可重建的派生数据**——删除 `derived/` 目录后可从 `records.jsonl` + `artifacts/` 完全重建
- EB 记录不可变——不得实现删除或修改已 commit 记录的功能

### 7.3 物理存储布局

```
eb/
├── records.jsonl                 # 不可变事实账本（现有，不改）
├── artifacts/                    # 候选源码归档（P1 新增）
│   ├── exp_017/
│   │   ├── manifest.json
│   │   └── train.py
│   └── ...
│
└── derived/                      # 所有派生数据（可从 records.jsonl 重建）
    ├── islands.json              # 岛定义 + 统计元数据 + UCB 优先级
    ├── semantic_memory/          # 每岛的语义记忆
    │   ├── ridge_lsmc.json
    │   ├── mlp_tanh.json
    │   └── ...
    ├── episodic_views/           # 每岛的分层情景记忆视图
    │   ├── ridge_lsmc.json
    │   ├── mlp_tanh.json
    │   └── ...
    ├── evolution_trees.json      # 演化树视图（parent-child 链 + delta）
    └── functional_similarity.json # 得分向量相关性 + 跨岛聚类
```

**`artifacts/` 是源码归档，不是派生数据**——它不能从 `records.jsonl` 重建（JSONL 只存哈希不存代码）。归档后只读，与 EB 的 `editable_file_sha256` 关联验证。Stage 3 读取 parent 实验代码时从这里取。

### 7.4 岛模型

#### 7.4.1 岛的定义

岛是 EB 记录的**逻辑分区**，由 `runner_type` + 功能聚类共同决定。物理上仍是同一个 JSONL，岛是派生视图。

**初始岛划分**（基于 runner_type，自动）：

| 岛 ID | 条件 | 说明 |
|-------|------|------|
| `ridge_lsmc` | `runner_type == "ridge_lsmc"` | 传统 LSMC 特征工程 |
| `mlp_tanh` | `runner_type == "mlp"` && `activation == "tanh"` | tanh MLP |
| `mlp_relu` | `runner_type == "mlp"` && `activation == "relu"` | ReLU MLP |
| `random_features` | `runner_type == "random_features"` | 随机特征网络 |

**岛的分裂**：当一个岛内出现功能相似度 <0.5 的子群（per-instance 得分向量相关性低），该岛可以分裂。分裂由 Stage 1 在分析中建议，由确定性逻辑执行。分裂只影响派生视图，不修改 EB 记录。

**岛的合并**：当两个岛的最优实验功能相似度 >0.9，说明它们实质上在探索同一策略，可以合并。同样只影响派生视图。

#### 7.4.2 岛统计元数据

```json
// islands.json
{
  "version": 42,
  "islands": {
    "mlp_tanh": {
      "experiment_count": 12,
      "best_experiment_id": "exp_031",
      "best_score": 0.018,
      "iterations_since_improvement": 2,
      "total_island_iterations": 8,
      "per_instance_score_pattern": [0.02, 0.01, 0.03, -0.01],
      "stalled": false,
      "created_at_iteration": 15
    },
    "ridge_lsmc": {
      "experiment_count": 45,
      "best_experiment_id": "exp_019",
      "best_score": 0.016,
      "iterations_since_improvement": 8,
      "total_island_iterations": 38,
      "per_instance_score_pattern": [0.015, 0.018, 0.014, 0.017],
      "stalled": true,
      "created_at_iteration": 1
    },
    "random_features": {
      "experiment_count": 0,
      "best_experiment_id": null,
      "best_score": null,
      "iterations_since_improvement": null,
      "total_island_iterations": 0,
      "stalled": false,
      "created_at_iteration": null
    }
  }
}
```

#### 7.4.3 候选名额的岛间分配

每次迭代的候选名额（默认 4 个）在岛之间分配。Stage 2 接收分配建议，可以覆盖但需要给出理由。

**分配公式**（UCB 风格，借鉴 SELA 的 MCTS 探索-利用平衡）：

```
island_priority(i) = normalize(best_score(i))
                   + C * sqrt(ln(total_iterations) / max(1, island_iterations(i)))
                   + E * unexplored_bonus(i)
```

- 第一项：奖励表现好的岛（利用）
- 第二项：奖励被忽视的岛（探索），C 是探索系数（初始值 1.0，可调）
- 第三项：给从未尝试过的岛一个固定奖励 E（初始值 2.0）

**分配规则：**
- 按 `island_priority` 降序排列，前 4 名各分 1 个名额
- 如果某岛标记为 `stalled`（连续 N 轮无提升），其优先级打折（乘 0.5）
- 至少保留 1 个名额给优先级最高的岛（防止全部名额给探索性岛）

**Stage 2 的覆盖权：** Stage 2 可以偏离分配建议（如"这个假设需要同时在 MLP 和 Ridge 上验证，给 MLP 2 个名额"），但覆盖原因会记入 EB 的 `context_meta`，供后续消融分析。

### 7.5 三层记忆

#### 7.5.1 语义记忆（Semantic Memory）

每个岛维护一组抽象规则，由 Stage 1 在分析过程中生成和更新。

```json
// semantic_memory/mlp_tanh.json
{
  "island_id": "mlp_tanh",
  "version": 12,
  "rules": [
    {
      "id": "rule_001",
      "content": "tanh 在 2 层时比 3 层稳定",
      "confidence": "confirmed",
      "confirmation_count": 5,
      "confirming_experiments": ["exp_031", "exp_033", "exp_035", "exp_041", "exp_043"],
      "created_at_iteration": 18,
      "last_confirmed_at_iteration": 32
    },
    {
      "id": "rule_002",
      "content": "学习率 > 0.01 容易产出 NaN 权重",
      "confidence": "confirmed",
      "confirmation_count": 3,
      "confirming_experiments": ["exp_025", "exp_029", "exp_037"],
      "created_at_iteration": 20,
      "last_confirmed_at_iteration": 28
    },
    {
      "id": "rule_003",
      "content": "Huber loss 在 OTM 实例上可能优于 MSE",
      "confidence": "tentative",
      "confirmation_count": 1,
      "confirming_experiments": ["exp_041"],
      "created_at_iteration": 30,
      "last_confirmed_at_iteration": 30
    },
    {
      "id": "rule_004",
      "content": "更宽的网络总是更好",
      "confidence": "refuted",
      "confirmation_count": 2,
      "refuting_experiments": ["exp_039", "exp_045"],
      "created_at_iteration": 22,
      "refuted_at_iteration": 34
    }
  ]
}
```

**置信度状态机：**

```
                 1 次信号
  (new rule) ──────────→ tentative
                              │
              ≥3 次确认       │  被新实验否定
                 ↓            ↓
            confirmed    refuted
                │
                │  被新实验否定
                ↓
            refuted
```

- `tentative`：1-2 次实验支持，Stage 2 可以据此提假设但不应视为定论
- `confirmed`：≥3 次独立实验支持，Stage 2 应视为可靠先验
- `refuted`：被后续实验否定，Stage 2 不应继续基于此规则提假设

**规则生成的时机：** Stage 1 每次运行时，读取该岛的语义记忆 + 新实验数据，输出 `list[StructuralInsight]`。巩固过程（7.6）将洞察转化为规则更新。

#### 7.5.2 情景记忆（Episodic Memory）

每个岛的实验记录按重要性分三层，供不同 Stage 使用：

| 层级 | 保留什么 | 过期策略 | 谁消费 |
|------|---------|---------|-------|
| **核心记忆** | 岛最优实验 + 最近突破实验（分数创新高的实验） | 永不过期 | Stage 1（分析）、Stage 3（parent 代码） |
| **近期记忆** | 最近 N 次实验的摘要（描述 + 标签 + 分数 + error_category + 一句话总结） | 滑动窗口，N = 10 | Stage 1（趋势分析） |
| **远期摘要** | 更早实验的压缩摘要（一段话描述该岛的探索历程） | 每次巩固时重写 | Stage 1（背景理解） |

```json
// episodic_views/mlp_tanh.json
{
  "island_id": "mlp_tanh",
  "version": 12,
  "core_memories": [
    {
      "experiment_id": "exp_031",
      "role": "island_best",
      "description": "tanh [64,32] MSE lr=0.005",
      "score": 0.018,
      "per_instance_lcb": [0.02, 0.01, 0.03, -0.01],
      "iteration": 24,
      "has_artifact": true
    },
    {
      "experiment_id": "exp_041",
      "role": "recent_breakthrough",
      "description": "tanh [64,32] Huber lr=0.003",
      "score": 0.017,
      "per_instance_lcb": [0.015, 0.015, 0.025, 0.005],
      "iteration": 30,
      "note": "OTM 实例从 -0.01 提升到 0.005",
      "has_artifact": true
    }
  ],
  "recent_memories": [
    {
      "experiment_id": "exp_045",
      "description": "tanh [128,64] MSE lr=0.005",
      "score": 0.015,
      "status": "success",
      "iteration": 33,
      "one_liner": "过宽导致过拟合，验证集分数反降"
    },
    {
      "experiment_id": "exp_043",
      "description": "tanh [64,32] MSE lr=0.001",
      "score": 0.016,
      "status": "success",
      "iteration": 32,
      "one_liner": "更小学习率略有提升但不显著"
    }
  ],
  "distant_summary": "迭代 15-24：从 Ridge 基线出发探索 MLP。前 5 次实验确立了 2 层 tanh [64,32] 的基本架构（3 层和 ReLU 被排除——3 层在所有实例上都不如 2 层，ReLU 导致 dead neuron 问题）。学习率从 0.01 逐步降到 0.005 后分数稳定。",
  "evolution_chain": [
    {"id": "exp_023", "score": 0.012, "delta": "从 Ridge 切换到 MLP [32]"},
    {"id": "exp_027", "score": 0.014, "delta": "扩展到 [64,32]"},
    {"id": "exp_031", "score": 0.018, "delta": "调整 lr 到 0.005, 当前最优"},
    {"id": "exp_041", "score": 0.017, "delta": "切换 Huber loss, OTM 改善"}
  ]
}
```

**`evolution_chain`**（AIDE 风格演化链）：从该岛的初始实验到当前最优的 parent-child 路径，每步记录分数变化和关键改动。由 EB 的 `parent` / `repair_of` 字段重建。Stage 3 读 parent 代码时同时看到整条链，理解"走到这一步的过程"。

#### 7.5.3 程序性记忆

程序性记忆 = LLM 的预训练知识（如何写 Python、如何实现反向传播、金融数学知识）。EB 不存储也不管理。

但有一个交叉点：Stage 3 的固定文档（manifest schema、训练数据格式、沙箱约束）可以视为"外显化的程序性记忆"。这些文档不随实验变化，不属于 EB，但 Stage 3 每次都需要。

### 7.6 记忆巩固（Consolidation）

每次迭代结束后（新实验记入 EB 后），运行增量巩固过程。这是一个确定性过程 + 一次轻量 LLM 调用。

```
新实验记入 records.jsonl
    │
    ▼  确定性：岛归属
    新实验按 runner_type + activation 分配到岛
    │
    ▼  确定性：岛统计更新
    更新 islands.json（experiment_count, best_score, stalled 标记等）
    │
    ▼  确定性：功能相似度增量更新
    计算新实验与已有实验的 per-instance 得分向量相关性
    更新 functional_similarity.json
    如果新实验与当前岛的其他成员相关性 <0.5 → 标记为潜在分裂信号
    │
    ▼  确定性：情景记忆分层
    核心记忆：检查新实验是否是岛最优或创新高 → 加入/替换
    近期记忆：滑动窗口更新（最近 10 条）
    远期摘要：如果近期窗口滑出了内容 → 标记为待压缩
    │
    ▼  确定性：演化树更新
    如果新实验有 parent → 追加到 evolution_trees.json
    │
    ▼  LLM 调用（轻量，~2K tokens）
    输入：新实验摘要 + 该岛当前语义记忆
    输出：
      - 新规则候选（如有）
      - 已有规则的确认/否定更新
      - 远期摘要更新（如有待压缩的记录）
    更新 semantic_memory/<island>.json + episodic_views/<island>.json
```

**巩固成本：** 确定性部分 <100ms（纯 numpy 和 JSON 操作）。LLM 调用 ~2K tokens，对比每次迭代的 Stage 1-3 总量 ~88K 是很小的开销。

**巩固的幂等性：** 删除 `derived/` 后，从 `records.jsonl` 重跑全部巩固过程，结果应一致（LLM 调用的非确定性由 temperature=0 缓解，但不保证位精确一致——语义记忆是辅助视图，不是事实账本）。

### 7.7 Stage 1 的 Context 渲染

巩固后，Stage 1 看到的不再是原始 EB 记录，而是按岛组织的三层记忆视图：

```
== 全局概览 ==
总实验: 67 | 活跃岛: 3 | 最优: exp_031 (MLP-tanh, 0.018)
名额分配建议: mlp_tanh ×2 (UCB=1.8), ridge_lsmc ×1 (UCB=0.9), random_features ×1 (UCB=2.3, 未探索)

== Island: mlp_tanh (12 实验, 最优 0.018, 2 轮无提升) ==
语义记忆:
  [confirmed ×5] tanh 在 2 层时比 3 层稳定
  [confirmed ×3] 学习率 > 0.01 容易产出 NaN 权重
  [tentative ×1] Huber loss 在 OTM 实例上可能优于 MSE
  [refuted] 更宽的网络总是更好
核心记忆:
  exp_031: tanh [64,32] MSE lr=0.005, score=0.018, per-instance=[0.02, 0.01, 0.03, -0.01]
  exp_041: tanh [64,32] Huber lr=0.003, score=0.017, OTM 从 -0.01→0.005
演化链: exp_023(0.012) → exp_027(0.014) → exp_031(0.018) → exp_041(0.017)
近期: exp_045(0.015, 过宽过拟合), exp_043(0.016, 小 lr 不显著)
远期: "前 5 次确立 2 层 tanh 架构，排除 3 层和 ReLU"

== Island: ridge_lsmc (45 实验, 最优 0.016, 8 轮无提升 → stalled) ==
语义记忆:
  [confirmed ×8] 多项式特征到 3 阶后收益递减
  [confirmed ×4] log 特征对高波动率有效
  [confirmed ×6] 该岛已趋于饱和
核心记忆:
  exp_019: poly3 + log + cross, score=0.016
远期: "前 30 次从 baseline 0.012 提升到 0.016，主要贡献来自多项式和 log 特征"

== Island: random_features (0 实验, 未探索) ==
无历史数据。文献参考：随机特征网络介于线性和全训练 NN 之间（Section 9.1）。
```

**对比当前 Stage 1 看到的**：80 条原始 EB 记录（充满 hash、时间戳、重复的失败记录、格式各异的 log_tail），信息密度低且无结构。新视图：
- 按策略家族组织（不是按时间排列）
- 抽象规则 + 置信度（不需要 LLM 从原始数据重新发现模式）
- 分层压缩（远期实验只保留一段话摘要）
- 演化链可视化（理解"怎么走到这一步"）

### 7.8 新增 EB 记录字段

**新增 metrics 字段**（写入 records.jsonl，不可变）：

```json
{
  "runner_type": "mlp",
  "weight_shape": [[64, 5], [64], [1, 64], [1]],
  "artifact_format_version": "openhyra-policy-spec.v1",
  "mean_training_seconds_per_instance": 5.3,
  "max_training_seconds_per_instance": 8.1,
  "mean_inference_seconds_per_call": 0.001,
  "training_paths_per_instance": 4096,
  "dev_per_instance_lcb": [0.012, -0.003, 0.008, 0.015],
  "error_category": null,
  "probe_score": 0.013,
  "probe_instance_id": "atm_put_1d"
}
```

`error_category` 和 `probe_score` 来自 Section 6.10-6.11 的预检和分级评估。

**新增 metadata 字段**（写入 records.jsonl，不可变）：

```json
{
  "knowledge_labels": ["mlp", "2-layer", "tanh-activation"],
  "island_id": "mlp_tanh",
  "knowledge_version": 42
}
```

`island_id` 在 commit 时由确定性规则分配（基于 runner_type + inference_config.activation）。

### 7.9 功能相似度

Section 6.6 引入的 per-instance 得分向量相关性存储在 `derived/functional_similarity.json`。

```json
{
  "version": 42,
  "score_correlations": [
    {"pair": ["exp_017", "exp_023"], "corr": 0.94, "same_island": true},
    {"pair": ["exp_023", "exp_031"], "corr": 0.87, "same_island": false}
  ],
  "cross_island_signals": [
    {
      "pair": ["exp_019", "exp_031"],
      "corr": 0.45,
      "note": "Ridge 最优和 MLP 最优的得分模式差异大——不同策略，互补可能"
    }
  ]
}
```

相关性矩阵只存储 >0.5 的同岛配对和所有跨岛 best-vs-best 配对。

**与岛模型的交互：**
- 同岛高相关性（>0.9）→ 冗余信号，Stage 1 应避免重复探索
- 同岛低相关性（<0.5）→ 潜在分裂信号（岛内出现了不同的子策略）
- 跨岛低相关性 → 互补信号（不同策略在不同实例上各有优势，混合可能有价值）

### 7.10 结构化 Commit 流程

```
候选通过预检（Section 6.10）+ 评估（Section 6.11）
    │
    ▼  评估器验证 + 评分
评分结果（score, metrics, per_instance_lcb）
    │
    ▼  确定性：特征提取
runner_type, weight_shape, artifact_version, error_category
    │
    ▼  确定性：岛分配
island_id = assign_island(runner_type, inference_config)
    │
    ▼  LLM 增量分类（标签分配，~1K tokens）
knowledge_labels
    │
    ▼  源码归档
artifacts/exp_NNN/{manifest.json, train.py}
    │
    ▼
EB.commit()  写入完整记录到 records.jsonl
    │
    ▼  巩固过程（Section 7.6）
    ├─ 确定性：岛统计 + 功能相似度 + 情景分层 + 演化树
    └─ LLM：语义记忆更新 + 远期摘要压缩（~2K tokens）
```

---

## 8. NN 训练基础设施

### 8.1 沙箱能力

| 阶段 | 可用库 | 单实例训练时间上限 | 硬件 |
|------|-------|-----------------|------|
| Phase 1 | Python + numpy + scipy | 60s | CPU |
| Phase 2 | + PyTorch (CPU) | 120s | CPU |
| Phase 3 | + MPS / CUDA | 120s | GPU |

注意：训练时间是**单实例**的上限。4 实例 × 2 repeat = 8 次沙箱调用。Phase 1 总训练时间上限 = 8 × 60s = 480s。

**Phase 1 已足够。** 百慕大任务的 NN 规模小——2-3 层 MLP，几百到几千参数。numpy 手写前馈网络 + 反向传播在 CPU 上几秒可训练完成。60 秒的单实例上限非常宽裕。

### 8.2 per-instance 训练流程

候选的 `train.py` 在每个 (instance, repeat) 上被独立调用：

1. evaluator 生成该实例的训练路径 → `training_paths.npy`
2. evaluator 创建隔离沙箱，放入训练数据 + 候选代码
3. 沙箱执行：`python train.py --output /output/`
4. 候选读取训练数据，用任意方法训练，导出权重
5. 沙箱退出，evaluator 读取输出目录
6. evaluator 验证权重格式和数值
7. evaluator 用可信 runner 加载权重 → 推理 + 评分

**每次沙箱调用是独立的——候选无法跨实例传递信息。** 这确保候选代码不能利用 instance A 的训练数据来优化 instance B 的策略。

候选永远不需要实现推理代码。推理由评估器的可信 runner 执行。

### 8.3 预算向量（封闭科学轨专用）

预算向量仅在封闭科学轨有意义——因为只有在这条轨上，evaluator 控制训练代码，路径数才是可控的自变量。

| 资源 | 限制 (per instance) | 目的 |
|------|---------------------|------|
| 训练路径数 | 固定（自变量，如 512~16384） | 样本效率对比 |
| 额外模拟 | 不适用（无候选 train.py） | 训练由 evaluator 执行 |
| 单实例训练时间 | 固定上限（如 60s） | 计算效率对比 |
| 峰值内存 | 固定（如 1GB） | 模型规模约束 |
| 推断时间 | 固定上限（如 0.1s per call） | 部署实用性 |

**开放搜索轨不使用此预算向量。** 开放搜索轨中候选可自行模拟路径，记录训练时间/内存作为 observational metrics，但不作为受控变量。

**关键区分：** "给定 N 条路径，哪种架构学出更好策略"是封闭科学轨回答的问题。"哪种训练算法（含自行模拟）产出最好策略"是开放搜索轨回答的问题。两个问题都有价值，不应混淆。

---

## 9. 科学定位

### 9.1 已有文献（修正 v1 的文献空白声明）

NN 与 LSMC 的混合**并非**完全空白。已有工作：

- ⚠️ **"Neural network regression for Bermudan option pricing"** — 用 NN 替代 LSMC 中的 Ridge 回归，保留 Longstaff-Schwartz 的后向递归框架。*归属待验证：v3 归为 Herrera et al. (2021)，reviewer 指出可能为 Lapeyre & Lelong。*
- ⚠️ **"Optimal Stopping via Randomized Neural Networks"** — 随机特征网络（固定随机隐藏层 + 训练线性输出层），介于纯线性和全训练 NN 之间。*归属待验证：v3 归为 Goudenège et al. (2023)，reviewer 指出可能为 Herrera, Krach, Ruyssen, Teichmann。*
- **Becker et al. (2019)**: "Deep Optimal Stopping" — 纯 NN 直接学习停止决策（注意：不输出 continuation value，与 Section 4.1 的 runner 接口不直接兼容）
- ⚠️ **Lelong (2019)**: v3 描述为"双核 SVM"，reviewer 指出实际为 Wiener chaos expansion。*待查阅原文确认。*

**这些工作的局限（仍然存在的研究空间）：**
- 每项工作固定一种特定混合方式，没有**自动搜索**最优混合结构
- 没有系统的**样本复杂度对比**——哪种混合方式在哪个样本量区间最优？
- 没有跨合约类型的**结构稳定性**分析

### 9.2 核心科学问题（修正后）

> **在固定计算预算下，符号归纳偏置能否改善最优停时策略的样本效率？**

具体来说：给定一族百慕大最优停时问题和固定的训练路径数 n，比较：
- 纯线性模型 (Ridge LSMC) 的 primal-dual gap
- 纯 NN (MLP) 的 primal-dual gap
- 混合模型 (符号特征 + NN 残差) 的 primal-dual gap

在 n 从小到大变化时，三者的相对表现是否存在交叉点？

**注意：这是一个实验问题，不是理论证明。** 结果可能是 H0（混合没有优势），这同样有科学价值——说明符号知识在 NN 足够大时成为多余的归纳偏置。

### 9.3 Deep Hedging 的区分

v1 将 Deep Hedging (Buehler et al. 2019) 列为"被排除的现代方法"。需要澄清：Deep Hedging 解决的是**对冲策略**优化（最小化风险度量），不是**最优停时**问题。两者的数学结构不同：

- 最优停时：max_tau E[payoff(S_tau)] — 选择何时行权
- 对冲优化：min_delta rho(V_T - sum delta_t * dS_t) — 选择对冲比率

Deep Hedging 的创新（在 NN 中编码无套利约束）对最优停时问题不直接适用。不应作为搜索空间扩展的论据。

### 9.4 实验设计

| 变量 | 角色 | 值 |
|------|------|-----|
| 策略类型 | 自变量 | ridge_lsmc / mlp / random_features / 混合 |
| 训练样本量 | 自变量 | 512, 1024, 2048, 4096, 8192, 16384 |
| 合约配置 | 受控变量 | 评估矩阵中的固定配置集 |
| 训练时间上限 | 受控变量 | 固定（如 120s） |
| 随机种子 | 重复变量 | 每配置 x 样本量 x 策略类型跑 5 个种子 |
| Primal-dual gap | 因变量 | 主要指标 |
| 训练收敛时间 | 因变量 | 辅助指标 |

**关键对照：** 封闭科学轨专用。每个策略类型使用完全相同的训练路径（相同种子生成）。evaluator 控制训练代码，候选只声明架构。这样对比出来的才是架构的样本效率。

**实验设计已知问题（需在 P3 前解决）：**
- 策略类型 × 合约配置是交叉因素，当前表格未说明如何处理交互效应
- 5 个种子重复可能不足以检测小效应——P3 开始前应做 power analysis（基于 P1 的方差估计）
- "混合"策略类型定义模糊（符号特征 + NN 残差有多种组合方式），不是一个单一的处理水平
- 论文级实验应报告效应量和置信区间，而非只报 p-value

---

## 10. 实施阶段

**顺序原则：安全边界必须在 train.py 开放之前就位。** P0 同时建立格式规范和沙箱安全——拆成"先格式后安全"会导致在无安全保障的状态下运行任意训练代码的窗口期。

### P0: 格式规范 + 安全边界 + 向后兼容 runner（2-3 周）

**格式与验证：**
- [ ] 定义 `openhyra-policy-spec.v1` schema 和 manifest.json 格式
- [ ] 实现 manifest 验证器（runner_type 注册、per-step 权重格式声明检查）
- [ ] 实现权重工件验证器（文件数 == n_exercise_times - 1、dtype float64、`np.load(allow_pickle=False)`、`np.all(np.isfinite(...))`、大小上限、SHA256 记录）

**向后兼容 Runner：**
- [ ] 实现 `ridge_lsmc` 可信 runner（读取 features.json IR + 回归系数，实现 `continuation(time_index, states, instance)` 接口）
- [ ] 回归测试：现有 AST IR 候选通过新流程评分与旧流程完全一致
- [ ] 在评估器中同时支持旧格式（纯 feature_program.json）和新格式（manifest + 权重工件）

**沙箱安全（开放 train.py 的前置条件）：**
- [ ] 恶意权重测试：NaN/Inf/超大权重 → runner 拒绝
- [ ] 维度篡改测试：权重维度与 manifest 不一致 → 拒绝
- [ ] 跨实例隔离测试：候选写入持久文件 → 下一个实例沙箱看不到
- [ ] 资源逃逸测试：超出内存/时间/文件大小上限 → 进程被杀
- [ ] 文件读取隔离测试：候选尝试读取宿主仓库、evaluator 源码、其他实例数据 → 拒绝
- [ ] `allow_pickle=False` 对所有 `.npy` 加载路径的审计

**Runner 验收测试（替代 v3 的 "bound_order_ok > 99.5%"）：**
- [ ] 确定性测试：相同 (time_index, states) → 位精确一致
- [ ] 无状态测试：改变调用顺序/批次拆分 → 位精确一致
- [ ] 有界测试：所有输出有限且在 output_clip 范围内
- [ ] 鞅终值测试：martingale_terminal_mean 的 95% CI 覆盖 0（多次重复覆盖率 > 90%）
- [ ] 已知基准测试：有解析解的实例上，上下界覆盖真值

**文档：**
- [ ] 信任边界、per-instance 训练流程、三层数据分离规则

### P1: 开放搜索轨——per-instance 沙箱训练（2-3 周）

*前置条件：P0 所有安全测试通过。*

- [ ] 实现 per-instance 沙箱训练调度（evaluator 为每个 instance×repeat 创建独立沙箱，传入训练数据 + instance.json，调用 `python train.py output_dir`，收集权重输出）
- [ ] 实现 `mlp` 可信 runner（前馈网络推理，`continuation` 接口）
- [ ] 实现 `random_features` 可信 runner
- [ ] 修改 Proposal Agent 支持 train.py + manifest.json 编辑
- [ ] 实现候选预检 Layer 1（静态检查：语法、manifest schema、import 白名单）
- [ ] 实现候选预检 Layer 2（单实例快速探测 + 权重验证）
- [ ] 实现结构化修复：RepairContext 数据结构 + Stage 3 重试循环（限 2 次）
- [ ] 实现分级评估：Phase A（单实例 probe）→ 早停判定 → Phase B（全量评估）
- [ ] 端到端验证：候选提交 MLP 训练代码，每实例独立训练，评分正确
- [ ] 沙箱 metrics 记录：训练时间、峰值内存、输出大小、权重哈希
- [ ] 实现源码归档：EB commit 时将 manifest.json + train.py 归档到 `artifacts/exp_NNN/`（P2 Stage 3 读 parent 代码的前置）
- [ ] 验证搜索空间扩展是否有收益：跑 50 次迭代，对比纯 IR 搜索

### P2: 三阶段 Context Agent + EB 岛模型 + 消融验证（4-5 周）

**三阶段管道：**
- [ ] 基线测量：当前单阶段系统 50 次迭代的下游指标（重复率、失败率、提升概率、覆盖率）
- [ ] 实现三阶段管道骨架（Analyst → Hypothesizer → Synthesizer），每阶段独立 LLM 调用
- [ ] 实现 Stage 1 数据准备：per-instance 得分矩阵构建、功能相似度计算（得分向量相关性）
- [ ] 实现 Stage 2 输出格式（AlgorithmHypothesis）→ ContextDecision 转换
- [ ] 实现 Stage 3 的 context 组装（manifest 文档 + parent 代码 + 负约束）

**EB 岛模型 + 三层记忆（Section 7）：**
- [ ] 实现岛分配逻辑：commit 时按 runner_type + activation 自动归岛
- [ ] 实现 `islands.json` 派生视图：岛统计元数据 + UCB 优先级计算
- [ ] 实现语义记忆：Stage 1 输出 → 巩固为 confirmed/tentative/refuted 规则
- [ ] 实现情景记忆分层：核心记忆（岛最优 + 突破）、近期记忆（滑动窗口 N=10）、远期摘要
- [ ] 实现巩固过程：确定性部分（岛统计 + 功能相似度 + 情景分层）+ LLM 轻量调用（语义更新 + 远期压缩）
- [ ] 实现 Stage 1 context 渲染：按岛组织的三层记忆视图（替代原始 EB 记录列表）
- [ ] 实现演化树派生视图：parent-child 链 + 每步 delta
- [ ] 实现 `derived/` 重建验证：删除 derived/ → 从 records.jsonl 重跑巩固 → 结果语义一致

**消融验证：**
- [ ] 消融验证：Stage 1 有/无 per-instance 得分矩阵、有/无功能相似度（各 50 次迭代）
- [ ] 消融验证：Stage 2 有/无 parent 实验继承（50 次迭代）
- [ ] 消融验证：有/无 Layer 1 + 2 预检对代码成功率的影响
- [ ] 消融验证：有/无分级评估早停对总迭代效率的影响
- [ ] 消融验证：岛模型 + 三层记忆 vs 原始 EB 记录（50 次迭代，对比 Stage 1 输出质量和下游搜索效率）
- [ ] 全局对比：三阶段管道 + 岛模型 EB vs 单阶段基线 + 原始 EB（50 次迭代，有信号的用 200 次验证）
- [ ] 扩展开发集逐实例分数回传 Stage 1（搜索集数据，非私有审计）

### P3: 封闭科学轨——对照实验（3-4 周）

- [ ] 实现封闭科学轨评估流程（evaluator 自有训练代码 + 候选架构声明）
- [ ] 固定预算向量（路径数 x 训练时间 x 内存 x 随机种子）
- [ ] 运行对照：ridge_lsmc vs mlp vs random_features vs 混合
- [ ] 每种 x 6 个样本量 x 5 个种子 x 4 个合约配置 = ~480 次评估
- [ ] 分析 gap vs 样本量学习曲线：是否存在交叉点？
- [ ] 分析结构稳定性：哪种策略类型在哪些配置下更稳健？
- [ ] 撰写结果（验证或否定假设，两种结果都有价值）

**TODO（科学实验设计，需在 P3 开始前解决）：**
- 实验矩阵中策略类型和合约配置是混合因素，需要决定使用交叉设计还是分层随机化
- 5 个种子的重复次数是否足够做 paired comparison？需要 power analysis
- "混合"策略类型的定义不精确——符号特征 + NN 残差的具体组合方式需要 P1 实验数据指导

---

## 11. 开放问题

**Q1. Runner 注册表扩展策略**
新 runner 需要什么级别的审查？
*建议：代码 review + Section 4.6 中定义的五项验收测试（确定性、无状态、有界、鞅终值、已知基准）。不使用 bound_order_ok 通过率作为判定标准。*

**Q2. ONNX runner 时间表**
ONNX 可以覆盖几乎所有 NN 架构，但引入 onnxruntime 依赖。
*建议：P1 用手写 runner (mlp, random_features)。如果候选确实需要更复杂架构，P3 后添加 ONNX runner。*

**Q3. 单实例训练预算的精确值**
60s per instance 对 Phase 1 的小 MLP 足够（实际可能只用 5-10 秒）。扩展到更复杂架构时需要调整。
*建议：Phase 1 设 60s/instance。记录每次训练的实际耗时，如果候选普遍触及上限则考虑放宽。*

**Q4. 评估矩阵精确规模**
Phase 1 保持 4 个实例（与现有 public suite 一致），扩展开发集多大？
*建议：Phase 1 开发集 8 个实例（4 公开 + 4 新增）。P3 科学实验需要更完整覆盖，~16 个实例。*

**Q5. 消融实验的样本量**
50 次迭代是否足够检测 Context Agent 改进的效果？
*建议：先跑 50 次作为快速筛选。有信号的改进用 200 次迭代验证。无信号的直接放弃。*

**Q6. Deep Optimal Stopping 支持**
Becker et al. 2019 风格的策略直接学习停止决策（二元 yes/no），不输出数值化的 continuation value。这类策略无法直接适配 Section 4.1 的单 `continuation` 接口，也无法用于 Rogers 对偶构造。
*建议：V4 Phase 1 不支持。如需支持，需另设协议，明确对偶代理的来源（如用另一个模型提供 approximate_value）。*

**Q7. 文献归属待验证**
以下引用在 GPT review 中被指出可能有误，需要人工查阅原文确认：
- "Neural network regression for Bermudan option pricing" — Section 9.1 归属 Herrera et al. (2021)，reviewer 认为应为 Lapeyre & Lelong
- "Optimal Stopping via Randomized Neural Networks" — Section 9.1 归属 Goudenège et al. (2023)，reviewer 认为应为 Herrera, Krach, Ruyssen, Teichmann
- Lelong (2019) 描述为"双核 SVM" — reviewer 认为实际是 Wiener chaos expansion

*这些需要查阅原文后修正。当前标记为 ⚠️ 待验证。*
