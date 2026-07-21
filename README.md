# OpenHyra

对腾讯混元 **Hyra**（Hunyuan Research Agent，[技术报告](https://hy.tencent.com/research/hyra)、
[官方结果仓库](https://github.com/Tencent-Hunyuan/Hyra-results)）的开源复现，
目前实现了其 **NanoChat AutoResearch** 任务（基于
[karpathy/autoresearch](https://github.com/karpathy/autoresearch)），
可在一台 Apple Silicon Mac 上端到端运行。

## Hyra 是什么

Hyra 是一个面向"性能导向研究任务"的自主研究智能体。其 Harness 是一个异步的
生产者-消费者循环：

- **Experience Bank（经验库）**：存储每个历史 solution 的代码、产物、日志与评估分数；
- **Context Agent**：持续从经验库中整合出多样化的"灵感"上下文（历史尝试、最优方案、
  探索方向提示），推入任务队列；
- **Proposal Agent**（多个并发）：从队列领取灵感，写出新的 solution
  （一个以 `solve.sh` 为入口的文件夹），在独立沙盒中运行、评分，结果提交回经验库；
- 循环直至主动退出或预算耗尽，返回历史最优方案。对未提供评估器的任务，
  Hyra 还会在外层循环中让评估器与 solution 共同进化（本复现未实现这一层）。

## 仓库结构

```
┌───────────────┐   inspirations   ┌────────────────┐   solution    ┌─────────┐
│ Context Agent │ ───────────────► │ Proposal Agent │ ────────────► │ Sandbox │
└──────▲────────┘                  └────────────────┘               └────┬────┘
       │                    ┌──────────────────┐                         │
       └─────────────────── │ Experience Bank  │ ◄───────────────────────┘
                            └──────────────────┘        results

eb.py             经验库（solution 文件夹 + 分数 + 日志 + 诊断指标）
context_agent.py  LLM Context Agent：每轮读全库写局势分析、定下一步方向与 parent
                  （分析存 eb/analyses/ 作为跨轮记忆；调用失败回退到方向轮换表）
proposal_agent.py 提案（headless LLM CLI 修改 train.py）
sandbox.py        沙盒运行 solve.sh，优先读 solution.json（对齐官方产物格式）
harness.py        主循环（单 GPU 下退化为串行；冻结文件校验防 reward hacking）
seed_solution/    种子 solution（solve.sh + train.py + prepare.py）
eb/ drafts/ sandboxes/   运行时产物（不入库）
```

任务环境 **autoresearch 不在本仓库内**：它是
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) 的独立 clone，
默认路径 `~/GitHub/autoresearch`（可用环境变量 `OPENHYRA_AUTORESEARCH` 覆盖），
harness 只引用它的 Python 环境和数据缓存。

## 任务环境（autoresearch）搭建指引

NanoChat AutoResearch 任务定义在 karpathy/autoresearch：agent 只能修改
`train.py`，训练固定 5 分钟墙钟预算，以 `val_bpb`（validation bits per byte，
越低越好）为唯一指标；`prepare.py`（数据、tokenizer、评估）冻结不可改。

```bash
# 1. 获取任务环境
git clone https://github.com/karpathy/autoresearch.git ~/GitHub/autoresearch
cd ~/GitHub/autoresearch

# 2. Apple Silicon 移植（上游只支持 NVIDIA GPU；按其 README 的小算力指南适配）：
#    - pyproject.toml：去掉 CUDA-only 的 kernels 依赖与 cu128 torch 源
#    - train.py：SDPA 替代 FlashAttention-3（滑窗用 banded mask）；
#      非 CUDA 跳过 torch.compile；MPS eager 下优化器标量用 Python float；
#      设备自适应（cuda/mps/cpu）；DEPTH 4、TOTAL_BATCH_SIZE 2^14、WINDOW_PATTERN "L"
#    - prepare.py：MAX_SEQ_LEN 512、EVAL_TOKENS 0.5M、dataloader/eval 设备自适应
#    （5 分钟时间预算与评估协议保持不变）

# 3. 安装依赖、下载数据（2 个训练 shard + 固定验证 shard）、训练 tokenizer
uv sync
uv run prepare.py --num-shards 2

# 4. 手动跑一次，验证环境并拿到种子分数
uv run train.py > run.log 2>&1
grep "^val_bpb:" run.log
```

## 运行 OpenHyra

```bash
# 用上一步的分数初始化经验库
python3 harness.py --seed seed_solution --seed-score <上一步的val_bpb> --seed-desc "baseline"

# 自主迭代（每轮 ≈ 提案 2-3 分钟 + 训练评估 ~7 分钟）
python3 harness.py --iterations 5

# 查看经验库
python3 harness.py --status
```

## 复现要点

- **任务协议不变**：只允许修改 `train.py`；`prepare.py`（tokenizer、dataloader、
  时间预算、`evaluate_bpb` 指标）冻结不可动。
- **反 reward hacking**：技术报告提到 NanoChat 上出现过泄漏未来 token 的
  hack，本复现将因果注意力与冻结评估作为硬约束写入 Proposal Agent 的任务说明。
- **solution 格式对齐官方产物**：与 Hyra-results 一致，每个 solution 是带
  `solve.sh` 入口的自包含文件夹。

## 改进方向

- 移植 Hyra 公开 solution 中平台无关的组件（n-gram 查表、induction 机制、
  分组学习率等，见官方结果仓库）；
- 提升 MPS 吞吐（更大 micro-batch、减少每步 host 开销、尝试 MLX 后端）；
- 更多 harness 迭代与更丰富的灵感上下文；
- 在 CUDA 机器上恢复上游完整设置以与公开数字对齐。
