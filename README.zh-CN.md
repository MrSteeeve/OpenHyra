# OpenHyra

![CI](https://github.com/MrSteeeve/OpenHyra/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

[English](README.md) | **中文**

对腾讯混元 **Hyra**（Hunyuan Research Agent）公开 Harness 架构 [1] 的开源、部分复现，
目前用于演示 **sums_diffs** 任务。OpenHyra 实现了一个自主研究循环，让模型智能体提出求解器、
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

| 系统 | $C(A)$ |
|---|---:|
| 官方种子（初始构造） | 1.059793 |
| **OpenHyra 历史运行** | **1.111815**（$n=405$） |
| SimpleTES [3] | 1.144887 |

上表只放同一 SimpleTES v1 协议下的结果。Hyra 公布的 1.159715 artifact [1, 2]
含 181,131 个元素，超过 $|A|\le512$，这里只把它视为跨协议参考，不放入比较表。

OpenHyra 集合来自 Codex 后端的一次历史运行（20 轮 Context × 每轮 4 个候选），
经可信评估器打分并独立复核：$n=405$、$|A+A|=2395$、$|A-A|=2003$。
该实验早于当前“所有结果独立入库”和“repair 不可变”语义：每轮只保存了 winner artifact，
其他候选只留下摘要。集合及独立 verifier 已作为明确标注的
[legacy artifact](artifacts/sums_diffs/openhyra-1.111814562869239-legacy/) 发布；
当前 Harness 尚未重跑产生新的主结果。

## 工作原理

```
┌───────────────┐   inspirations   ┌────────────────┐   solution    ┌─────────┐
│ Context Agent │ ───────────────► │ Proposal Agent │ ────────────► │ Sandbox │
│  (模型读取摘要  │                  │  ×N workers    │               │ 可信评估 │
│   和近期记录)   │                  │ (Claude/Codex) │               └────┬────┘
└──────▲────────┘                  └────────────────┘                     │
       │                     ┌──────────────────┐                         │
       └──────────────────── │ Experience Bank  │ ◄───────────────────────┘
                             └──────────────────┘        results
```

**Experience Bank（经验库）**：每个候选的代码、产物、日志、指标，成功、崩溃、
低分一律作为独立记录入库。

**Context Agent**：LLM 每轮读取所有记录的结构化摘要、近期日志、近期失败和当前最佳实现，
写一段简短局势分析（持久化为跨轮记忆），并确定下一个实验方向。它尚不能检索任意历史
源码目录或 artifact。

**Proposal Agent**：Claude Code 或 Codex CLI 在独立 draft 目录中按后端权限修改求解器；
这些目录用于组织和校验改动，并不是 OpenHyra 统一提供的 OS 安全边界。
每份 Context 简报扇出多个独立候选，提案生成与评估重叠执行。

**沙盒 + 可信评估**：候选在 macOS Seatbelt 下运行，网络被禁止、写入被限制在沙盒内；
宿主机大多数文件仍可读，因此这是写入约束，不是机密性沙盒。候选 `solution.json`
必须是有大小上限、单硬链接的普通文件，Harness 将其复制到候选不可写的可信目录后再评分。
完整性白名单拒绝可编辑文件之外的改动，AST 预检拦截已知崩溃模式；失败和修复尝试分别
作为不可变 EB 记录，并由 `repair_of` 连接。

每个 run 在 `--init` 时冻结代码、任务、evaluator、模型、并发、资源限制和 seed 到
`run_manifest.json`。续跑发现影响结果的 provenance 漂移会拒绝执行；进程锁禁止两个
Harness 同时写同一 `run-id`。

## 快速开始

```bash
# 依赖：macOS、Python >= 3.10、numpy，以及 Claude Code 或 Codex CLI
python3 harness.py --run-id demo --init --workers 2
python3 harness.py --run-id demo --iterations 5 --workers 2
python3 harness.py --run-id demo --status
python3 harness.py --run-id demo --export-bundle bundles/demo
```

初始化和续跑时应传入相同的 `--backend`、`--model`、`--workers`、候选数和 trial seed；
如需改变这些设置，应新建 `--run-id`。

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
