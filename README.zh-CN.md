# OpenHyra

![CI](https://github.com/MrSteeeve/OpenHyra/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

[English](README.md) | **中文**

对腾讯混元 **Hyra**（Hunyuan Research Agent，[技术报告](https://hy.tencent.com/research/hyra)、
[官方结果仓库](https://github.com/Tencent-Hunyuan/Hyra-results)）的开源复现。

**第一阶段目标：`sums_diffs`** —— 和差集指数问题：构造有限整数集 $A$，最大化

$$C(A) = \frac{\log\left(|A+A| / |A|\right)}{\log\left(|A-A| / |A|\right)}$$

其中 $A+A = \{a+b : a,b \in A\}$，$A-A = \{a-b : a,b \in A\}$。该任务纯 CPU、
评估客观确定、跨机器可比，是验证研究智能体循环的理想第一战场。本仓库只采用公开的
**SimpleTES sums_diffs v1 协议**：$2 \le |A| \le 512$、元素位于 $[-10^6, 10^6]$、
候选硬超时 180 秒，参考成绩为 **1.144887**。Hyra 发布的大集合 artifact 不满足该协议，
因此不进入本项目成绩表。

## Harness 架构（对齐技术报告）

```
┌───────────────┐   inspirations   ┌────────────────┐   solution    ┌─────────┐
│ Context Agent │ ───────────────► │ Proposal Agent │ ────────────► │ Sandbox │
│  (LLM：读全库    │                  │  ×N workers      │               │ +可信评估器 │
│   写局势分析)    │                  │ (Claude/Codex)   │               └────┬────┘
└──────▲────────┘                  └────────────────┘                     │
       │                     ┌──────────────────┐                         │
       └──────────────────── │ Experience Bank  │ ◄───────────────────────┘
                             └──────────────────┘        results
```

- **Experience Bank**（`eb.py`）：保存每个候选的代码、产物、日志、评估指标；失败、
  crash、violation 和低分候选同样作为独立记录入库；线程安全。
- **Context Agent**（`context_agent.py`）：LLM agent，每轮读全库写 ≤250 词局势分析、
  定下一个实验方向；分析持久化为跨轮记忆；失败自动回退到确定性方向轮换。
- **Proposal Agent**（`proposal_agent.py`）：headless Claude Code 或 Codex CLI 在 draft 中修改
  唯一可编辑文件，写 `PROPOSAL.md`；默认每个 Context 独立生成 4 个候选，
  `--workers N` 时多个提案与评估重叠执行，不做组内 winner 筛选。
- **Sandbox**（`sandbox.py`）：macOS Seatbelt（`sandbox-exec`）隔离——禁网络、
  只允许写沙盒目录；退出码非零即 crash。
- **可信评估**：候选只输出 `solution.json`（集合本身），分数由沙盒外的
  `tasks/<name>/evaluator.py` 精确枚举 $A+A$ 和 $A-A$ 并校验 SimpleTES 约束——候选自报的
  任何数字都不被采信。
- **完整性白名单**：除任务声明的可编辑文件与 `PROPOSAL.md` 外，任何文件增删改
  → 判 `violation`，不进沙盒。
- **工程预检 + 有限修复**：候选入沙盒前先过 AST 静态预检（规则来自实测崩溃模式，
  如未钳制的退火进度、动态空区间 `randrange`；不过者记 `rejected`）；crash/timeout
  的候选获得 `candidate_repair_attempts` 次修复机会（把失败日志作为不可信数据回喂给
  提案后端做最小修复，重跑冻结检查后再评估），全部尝试留痕于 EB metadata。

## 任务插件

```
tasks/sums_diffs/
  task.json        方向(max)、可编辑文件、超时、评估并发、回退方向表、工程 invariants
  TASK.md          给 agent 的任务说明与协议约束
  evaluator.py     可信精确评估器（SimpleTES v1 协议的独立重实现）
  seed_solution/   SimpleTES 官方 17 元素初始集合（C ≈ 1.059793）
runs/<task>/       经验库、draft、沙盒等运行时产物（不入库）
```

## 运行

```bash
# 依赖：Python >= 3.10 + numpy；提案与分析默认用 Claude Code CLI（headless）
python3 harness.py --task sums_diffs --init          # 官方 SimpleTES seed 过可信管线入库
python3 harness.py --task sums_diffs --iterations 5 --workers 2
# 改用 Codex CLI（--model 可省略，使用 Codex 当前默认模型）
python3 harness.py --task sums_diffs --iterations 5 --workers 2 \
  --backend codex --model gpt-5.6-sol
python3 harness.py --task sums_diffs --status
```

`--iterations` 表示 Context 轮数，而不是 EB 新增记录数。默认
`candidates_per_context=4`，因此每轮正常新增 4 条候选记录；可以用
`--candidates-per-context N` 临时覆盖。所有候选结果都会写入 EB，之后的 Context
可以同时利用成功经验、低分反例和失败日志。

**调度协议说明**：Context 允许领先于结果的窗口固定为
`workers + eval_concurrency` 轮。历史实验（`trial_01`–`trial_04` 系列）在旧版
调度器上以 `--max-inflight 1` 的严格串行反馈模式运行；该参数已在当前版本移除，
比较不同 bundle 时请以各自 manifest 记录的调度参数为准。

## 与真 Hyra 的已知差距

- Proposal 模型为 Claude Code 或 Codex CLI（代替内部混元模型）；
- 单机并发规模远小于其信号量控制的 agent 舰队；
- 未实现评估器共进化外层循环（本任务评估器固定，无此需求）；
- Seatbelt 为进程级隔离，弱于容器/VM。

## 路线图

sums_diffs 搜索复现 → `smallest_adder` 官方产物独立审计 → `qubit_routing` 等
更多 AI4Science 任务插件化。
