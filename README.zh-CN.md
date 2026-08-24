# OpenHyra

![CI](https://github.com/MrSteeeve/OpenHyra/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

[English](README.md) | **中文**

对腾讯混元 **Hyra**（Hunyuan Research Agent）公开 Harness 架构 [1] 的开源、部分复现，
目前用于演示 **sums_diffs** 与**百慕大期权最优停止**任务。OpenHyra 实现了一个自主研究循环，让模型智能体提出求解器、
沙盒运行、可信评估器打分，每次尝试无论成败都作为经验存入经验库，
供后续轮次利用。

## 任务

### 和差集搜索

构造有限整数集 $A$，最大化和差集指数

$$C(A) = \frac{\log\left(|A+A| \/\ |A|\right)}{\log\left(|A-A| \/\ |A|\right)}$$

其中 $A+A = \{a+b : a,b \in A\}$ ，$A-A = \{a-b : a,b \in A\}$。

对绝大多数集合 $C(A) < 1$（加法可交换，差通常多于和）；
和占优（MSTD）构造能把它推到 1 以上 [4]。

当前任务接受任意满足 $|A|\ge2$ 的显式有限集合，元素取值仍须位于
$[-10^6,10^6]$，但不再设置集合大小的固定上限。候选方案运行时间最长 180 秒，
由沙盒外的可信评估器精确枚举 $A+A$ 与 $A-A$；artifact 大小、评估时间和评估内存
构成实际资源边界。

`solution.json` 还可以携带唯一的 current `openhyra-research` schema；没有
`v1`/`v2` 并行协议。构造必须是类型化的 positional digit product，并显式给出有限
检查层和白名单 obligation。可信评估器另行生成 `evidence.json`：通过重算的 obligation
标为 `bounded_checked`，失败项标为 `refuted` 并保存最小反例；关联 claim 只能成为
带界或反例证据的索引，但不会因此获得可信数学状态；没有受信 implication rule 时，
自然语言 claim 仍是 `unverified`。这些字段绝不改变数值分数，有限检查也不会被冒充为
渐近证明。候选输入字段见 [任务说明](tasks/sums_diffs/TASK.md)。

### 百慕大期权最优停止

第二个任务搜索一个有界、类型化的特征表达式程序，固定算法为 Ridge LSMC，金融环境由
可信评估器控制的风险中性 Black--Scholes 模型给出。候选既不提交价格，也不提交可执行
Python；路径模拟、合约、贴现、回归、因果行权、计算预算和统计量全部归评估器所有。
公开搜索使用相互独立的拟合路径与定价路径，并让候选和冻结基线共享随机数；分数是相对
基线、按执行价归一化的下界改善之保守下置信值。

最终验收是与搜索分离的一次性动作：Harness 先校验并冻结去重后的 Top-K 规范化工件，
然后才生成新的私有种子，让所有冻结工件在同一隐藏实例集上接受审计。审计使用条件中心化
的嵌套鞅计算原始—对偶置信 Gap。结果只写入 `final_audit.json`，不会回流经验库或下一轮
搜索；公开 termination 仅携带种子承诺，导出已完成的审计记录后才可用其中种子独立复现。

当前版本刻意限定为“Phase 1 特征搜索 + Phase 4 验收审计”，尚未开放任意策略、Python
或完整算法搜索。数据型工件冻结后才生成隐藏种子，封闭了当前任务所需的反馈通道；这并不
意味着现有写入约束沙盒已成为可安全运行任意候选代码的机密性边界。完整 IR、金融协议、
评分规则和结论边界见[百慕大任务说明](tasks/bermudan_optimal_stopping/TASK.md)。

## 结果

| 系统 | $C(A)$ |
|---|---:|
| 官方种子（初始构造） | 1.059793 |
| **OpenHyra 历史运行** | **1.111815**（$n=405$） |
| SimpleTES [3] | 1.144887 |

这些是历史参考点，不是当前同协议排行榜：OpenHyra 历史运行和 SimpleTES 结果产生于
限制集合大小的设置，而当前任务已经取消固定的集合大小上限。Hyra 公布的 artifact
[1, 2] 尚未通过当前可信评估器及其资源边界重新运行，因此不加入表格。

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
至少完成 6 轮、连续 4 轮没有超过 `0.0001` 的有效提升、最近 4 轮至少有 4 个成功候选；
对 sums_diffs 还必须完成上述四类同目标形式证明。Context JSON 无效、模型调用失败、
runner 缺失、存在可信反例或证明不完整时一律继续。每次带 `--iterations` 的 pipeline
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
python3 harness.py --run-id demo --init --workers 2
python3 harness.py --run-id demo --iterations 5 --workers 2
python3 harness.py --run-id demo --status
python3 harness.py --run-id demo --export-bundle bundles/demo

# 百慕大特征搜索，以及一次性冻结 Top-K 审计。
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-demo \
  --init --workers 2 --trial-seed 1729
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-demo \
  --iterations 5 --workers 2 --trial-seed 1729
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-demo \
  --final-audit
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-demo \
  --export-bundle bundles/bermudan-demo

# 形式化 run 在初始化和续跑时必须使用同一个可信隔离 runner。
python3 harness.py --run-id formal --init \
  --formal-runner /absolute/path/to/openhyra-formal-runner
```

初始化和续跑时应传入相同的 `--backend`、`--model`、`--workers`、候选数、trial seed
和停止策略；如需改变这些设置，应新建 `--run-id`。

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
