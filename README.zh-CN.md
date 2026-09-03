# OpenHyra

![CI](https://github.com/MrSteeeve/OpenHyra/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

[English](README.md) | **中文**

对腾讯混元 **Hyra**（Hunyuan Research Agent）公开 Harness 架构 [1] 的开源、部分复现，
目前用于演示**百慕大期权最优停止**任务。OpenHyra 实现了一个自主研究循环，让模型智能体提出求解器、
沙盒运行、可信评估器打分，每次尝试无论成败都作为经验存入经验库，
供后续轮次利用。

## 任务

### 百慕大期权最优停止

OpenHyra 现在提供两条并行的百慕大任务。历史的
`bermudan_optimal_stopping` 仍搜索有界、类型化的特征表达式程序，算法固定为 Ridge
LSMC；新增的 `bermudan_python_search` 搜索 AlgorithmBundle：候选自己的 `train.py`
在每个实例和 repeat 上于独立训练沙盒中运行，并依据 `manifest.json` 生成纯数据型的
continuation 工件。当前注册了 MLP、仿射线性和有界表达式三类 runner，因此纯逻辑解和
神经网络解可以走同一条可信评估链路。

MLP 与线性工件直接返回以时间零计价的贴现货币值。表达式工件保留历史 Feature IR
终端在当前行权时点的执行价归一化语义；可信 runner 在交给行权与评分前统一执行
`strike * exp(-rate * t)` 的换算。

两条任务中的风险中性模型、路径模拟、合约、贴现、因果行权、原始—对偶审计、计算预算
和统计量都由评估器控制。Python 只开放在逐实例训练阶段，定价评估器不会导入候选代码。
公开搜索使用相互独立的拟合路径与定价路径，并让候选和冻结基线共享随机数；分数是相对
基线、按执行价归一化的下界改善之保守下置信值。

最终验收是与搜索分离的一次性动作：Harness 先校验并冻结去重后的 Top-K 规范化工件，
然后才生成新的私有种子，让所有冻结工件在同一隐藏实例集上接受审计。审计使用条件中心化
的嵌套鞅计算原始—对偶置信 Gap。结果只写入 `final_audit.json`，不会回流经验库或下一轮
搜索；公开 termination 仅携带种子承诺，导出已完成的审计记录后才可用其中种子独立复现。

特征任务继续作为兼容基线，Python 任务则是新增的算法搜索入口。只有在数据型工件及其
源码 bundle 冻结后才生成隐藏种子，从而封闭相关反馈通道；公开分数仍不是定理，也不是
生产价格。完整协议、金融模型、评分规则和结论边界见[特征任务说明](tasks/bermudan_optimal_stopping/TASK.md)
与 [Python 任务说明](tasks/bermudan_python_search/TASK.md)。

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
作为不可变 EB 记录，并由 `repair_of` 连接。Proposal 源码会在最终白名单检查、预检与
执行前封存；EB 只由父进程控制的源码快照和可信 evaluator 输出组装，不再从仍可写的
draft 入库。导出 bundle 前还会重新核对源码、solution、evidence 与可编辑文件哈希，
并写出同一批已验证字节。

**研究晋级**：有限集合仍是可执行探针，也是排行榜分数的唯一来源。Context 现在会在
`construct / falsify / formalize / repair_formalization` 阶段间推进，并分别维护数值
frontier 与研究 frontier；相同有限集合上的证明进展不会再被数值去重吞掉。可信反例或
形式化失败会触发一次独立 research revision，并从父进程封存的源码快照继续；这里的
可信反例包括 obligation、claim 或 standalone certificate 的重算失败，形式化失败包括
`rejected` 与 `infrastructure_error`。

形式化提交中的每个 `proofs[]` 条目只含 claim ID 和 Lean proof term。任务自带的可信 specification 固定 import、
定理类型构造与公理白名单；候选给出有理目标，evaluator 归一化后由父进程 wrapper 绑定。
隔离 runner 必须先单独编译候选 wrapper，再在独立进程和输出通道中运行可信声明检查与
axiom audit；候选编译输出不会作为审计证据。全部通过后 claim 才会成为
`formal_checked`。默认没有进程内或非隔离降级路径：未配置 runner 时状态为
`unavailable`。运行者可显式传入
`--formal-runner /绝对路径/runner`；其路径、字节和 SHA-256 会写入 provenance，并在
每次证明调用前重检。runner 还必须证明输出完整未截断，并对实际 Lean 二进制、toolchain、
Mathlib commit/tree 与不可变执行环境给出指纹；探测和证明阶段的指纹必须完全一致。
Lean 4.26.0 和 Mathlib revision 已随任务 specification 固定。仓库当前提供 fail-closed
客户端协议，但不附带可直接运行的隔离 runner 或离线 Mathlib 镜像。

证明闭环要求同一条 EB 记录、同一个规范化有理目标上同时完成
`universal_upper_bound`、`approximating_family`、`supremum_eq` 和
`nonattainment`，并且该记录的 claim、obligation、certificate 可信反例数都为零。
若某条定理通过但同一记录仍有反例，该定理可以单独为 `formal_checked`，总状态则为
`formal_checked_with_refutation`，不能通过停止门。目前这套机制并不表示 OpenHyra
已独立发现或证明 \(c=2\)；只有实际通过该 gate 的 run artifact 才能支持这种结论。

每个 run 在 `--init` 时冻结代码、任务、evaluator、模型、并发、资源限制、seed 和停止策略到
`run_manifest.json`。续跑发现影响结果的 provenance 漂移会拒绝执行；进程锁禁止两个
Harness 同时写同一 `run-id`。

### 受 Harness 审核的 Agent 停止

主动停止需要显式启用 `--agent-stop`，`--iterations` 仍然是不可突破的轮数上限。
启用后，Context 轮次改为顺序决策，保证每次判断都看到上一轮完整入库的结果；同一轮中的
多个候选仍然并行生成和评估。Context 的 `stop` 只是请求，默认只有同时满足以下证据才接受：
至少完成 6 轮、连续 4 轮没有超过 `0.0001` 的有效提升、最近 4 轮至少有 4 个成功候选。
Context JSON 无效或模型调用失败时一律继续。每次带 `--iterations` 的 pipeline
调用都会把终止原因和证据写入 `termination.json`；若接受 Agent 停止，还会保存 Agent
原始判断与 Harness 审核结论。导出 bundle 时该文件一并保存。

```bash
python3 harness.py --run-id guarded --init --workers 2 --agent-stop
python3 harness.py --run-id guarded --iterations 20 --workers 2 --agent-stop
```

可以通过 `--min-contexts-before-stop`、`--stop-patience`、`--stop-min-delta`、
`--stop-recent-window` 和 `--stop-min-successful-candidates` 调整审核条件；初始化和续跑必须
使用相同设置。

## 快速开始

```bash
# 依赖：macOS、Python >= 3.10、numpy，以及 Claude Code 或 Codex CLI

# 百慕大特征搜索，以及一次性冻结 Top-K 审计。
python3 harness.py --run-id bermudan-demo --init --workers 2 --trial-seed 1729
python3 harness.py --run-id bermudan-demo --iterations 5 --workers 2 --trial-seed 1729
python3 harness.py --run-id bermudan-demo --final-audit
python3 harness.py --run-id bermudan-demo --status
python3 harness.py --run-id bermudan-demo --export-bundle bundles/bermudan-demo

# Python AlgorithmBundle 搜索（MLP/线性/表达式 runner）。
python3 harness.py --task bermudan_python_search --run-id bermudan-python-demo \
  --init --workers 1 --trial-seed 1729
python3 harness.py --task bermudan_python_search --run-id bermudan-python-demo \
  --iterations 5 --workers 1 --trial-seed 1729
python3 harness.py --task bermudan_python_search --run-id bermudan-python-demo \
  --final-audit
```

初始化和续跑时应传入相同的 `--backend`、`--model`、`--workers`、候选数、trial seed
和停止策略；如需改变这些设置，应新建 `--run-id`。

## 参考文献

1. Hyra Team. *Hyra: Hunyuan Research Agent* — 技术报告，腾讯，2026。
   <https://hy.tencent.com/research/hyra>
2. Tencent-Hunyuan. *Hyra-results: research artifacts from Hyra.*
   <https://github.com/Tencent-Hunyuan/Hyra-results>
