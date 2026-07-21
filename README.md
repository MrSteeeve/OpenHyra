# OpenHyra

对腾讯混元 **Hyra**（Hunyuan Research Agent，[技术报告](https://hy.tencent.com/research/hyra)、
[官方结果仓库](https://github.com/Tencent-Hunyuan/Hyra-results)）的开源复现。

**第一阶段目标：`sums_diffs`** —— Hyra AI4Science 赛道中的和差集指数问题：构造有限整数集
A，最大化 `C(A) = log(|A+A|/|A|) / log(|A−A|/|A|)`。该任务纯 CPU、评估客观确定、
跨机器可比，是验证研究智能体循环的理想第一战场。此前公开最好结果 1.14489（SimpleTES），
Hyra 报告 **1.15971** —— 本仓库的可信评估器已独立复算其公开 artifact，确认
C(A) = 1.159715。

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

- **Experience Bank**（`eb.py`）：每个 solution 的代码、产物、日志、评估指标；线程安全。
- **Context Agent**（`context_agent.py`）：LLM agent，每轮读全库写 ≤250 词局势分析、
  定下一个实验与出发点；分析持久化为跨轮记忆；失败自动回退到确定性方向轮换。
- **Proposal Agent**（`proposal_agent.py`）：headless Claude Code 或 Codex CLI 在 draft 中修改
  唯一可编辑文件，写 `PROPOSAL.md`；`--workers N` 时多个提案与评估重叠执行。
- **Sandbox**（`sandbox.py`）：macOS seatbelt（`sandbox-exec`）隔离——禁网络、
  只允许写沙盒目录；退出码非零即 crash。
- **可信评估**：候选只输出 `solution.json`（集合本身），分数由沙盒外的
  `tasks/<name>/evaluator.py` 用 FFT 重算并校验约束——候选自报的任何数字都不被采信。
- **完整性白名单**：除任务声明的可编辑文件与 `PROPOSAL.md` 外，任何文件增删改
  → 判 violation，不进沙盒。

## 任务插件

```
tasks/sums_diffs/
  task.json        方向(max)、可编辑文件、超时、评估并发、回退方向表
  TASK.md          给 agent 的任务说明与协议约束
  evaluator.py     可信评估器（FFT 计算 |A+A|、|A−A|，校验元素约束）
  seed_solution/   种子：Conway MSTD 集的无进位乘积构造（C≈1.0344）
runs/<task>/       经验库、draft、沙盒等运行时产物（不入库）
```

## 运行

```bash
# 依赖：python3 + numpy；提案与分析默认用 Claude Code CLI（headless）
python3 harness.py --task sums_diffs --init          # 种子过可信管线入库
python3 harness.py --task sums_diffs --iterations 5 --workers 2
# 改用 Codex CLI + GPT/Codex 模型（--model 可省略，使用 Codex 当前默认模型）
python3 harness.py --task sums_diffs --iterations 5 --workers 2 \
  --backend codex --model gpt-5.6-sol
python3 harness.py --task sums_diffs --status
```

## 与真 Hyra 的已知差距

- Proposal 模型为 Claude Code 或 Codex CLI（代替内部混元模型）；
- 单机并发规模远小于其信号量控制的 agent 舰队；
- 未实现评估器共进化外层循环（本任务评估器固定，无此需求）；
- seatbelt 为进程级隔离，弱于容器/VM。

## 路线图

sums_diffs 搜索复现 → `smallest_adder` 官方产物独立审计 → `qubit_routing` 等
更多 AI4Science 任务插件化。
