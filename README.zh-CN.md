# OpenHyra

![CI](https://github.com/MrSteeeve/OpenHyra/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

[English](README.md) | **中文**

对腾讯混元 **Hyra**（Hunyuan Research Agent）Harness [1] 的开源复现，
并将其用于完成 **sums_diffs** 任务。OpenHyra实现了一个自主研究循环，让模型智能体提出求解器、
沙盒运行、可信评估器打分，每次尝试无论成败都作为经验存入经验库，
供后续轮次利用。

## 任务

构造有限整数集 $A$，最大化和差集指数

$$C(A) = \frac{\log\left(|A+A| \/\ |A|\right)}{\log\left(|A-A| \/\ |A|\right)}$$

其中 $A+A = \{a+b : a,b \in A\}$ ，$A-A = \{a-b : a,b \in A\}$。

对绝大多数集合 $C(A) < 1$（加法可交换，差通常多于和）；
和占优（MSTD）构造能把它推到 1 以上 [4]。

本仓库遵循公开的 **SimpleTES sums_diffs 任务要求** [3]：
$2 \le |A| \le 512$、元素取值在 $[-10^6, 10^6]$ 区间内、候选方案运行时间最长 180 秒，
由沙盒外的可信评估器精确枚举 $A+A$ 与 $A-A$。

## 结果

| 系统 | $C(A)$ | 是否满足 SimpleTES v1 协议 |
|---|---|---|
| 官方种子（17 元素初始构造） | 1.059793 | ✓ |
| **OpenHyra（本仓库）** | **1.111815** | ✓ |
| SimpleTES [3] | 1.144887 | ✓ |
| Hyra [1, 2] | 1.159715 | ✗ （公开结果含 181,131 个元素，超出任务要求范围） |

我们的最优解由 Codex 后端的一次运行（20 轮 Context × 每轮 4 个候选）从官方种子
出发搜得，经可信评估器打分并独立复核：$n = 405$、$|A+A| = 2395$、$|A-A| = 2003$。

## 工作原理

```
┌───────────────┐   inspirations   ┌────────────────┐   solution    ┌─────────┐
│ Context Agent │ ───────────────► │ Proposal Agent │ ────────────► │ Sandbox │
│  (模型读取全库  │                  │  ×N workers    │               │ 可信评估 │
│   撰写局势分析) │                  │ (Claude/Codex) │               └────┬────┘
└──────▲────────┘                  └────────────────┘                     │
       │                     ┌──────────────────┐                         │
       └──────────────────── │ Experience Bank  │ ◄───────────────────────┘
                             └──────────────────┘        results
```

**Experience Bank（经验库）**：每个候选的代码、产物、日志、指标，成功、崩溃、
低分一律作为独立记录入库。

**Context Agent**：LLM 每轮读全库、写一段简短局势分析（持久化为跨轮记忆），
并确定下一个实验方向。

**Proposal Agent**：Claude Code 或 Codex CLI 在隔离 draft 中修改求解器；
每份 Context 简报扇出多个独立候选，提案生成与评估重叠执行。

**沙盒 + 可信评估**：候选在 macOS Seatbelt 下运行，
只输出 `solution.json`，分数在沙盒外重算。完整性白名单拒绝可编辑文件之外的任何改动，
AST 预检在启动前拦截已知崩溃模式，崩溃候选获得有限次 LLM 修复机会并全程留痕。

## 快速开始

```bash
# 依赖：macOS、Python >= 3.10、numpy，以及 Claude Code 或 Codex CLI
python3 harness.py --init                      # 官方种子过可信管线入库
python3 harness.py --iterations 5 --workers 2  # 运行自主循环
python3 harness.py --status                    # 查看经验库
```

## 参考文献

1. Hyra Team. *Hyra: Hunyuan Research Agent* — 技术报告，腾讯，2026。
   <https://hy.tencent.com/research/hyra>
2. Tencent-Hunyuan. *Hyra-results: research artifacts from Hyra.*
   <https://github.com/Tencent-Hunyuan/Hyra-results>
3. *SimpleTES: Evaluation-driven Scaling for Scientific Discovery.*
   arXiv:2604.19341. <https://arxiv.org/abs/2604.19341>
4. G. Martin, K. O'Bryant. *Many sets have more sums than differences.*
   Additive Combinatorics, CRM Proc. Lecture Notes 43, 2007。
   <https://arxiv.org/abs/math/0608131>
