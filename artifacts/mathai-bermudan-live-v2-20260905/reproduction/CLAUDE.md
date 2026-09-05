# Workflow

模型分工（写死，不要偏离）：

### Claude Opus = 顶层大脑

- **顶层设计**：架构决策、方案选型、tradeoff 判断、研究方向制定
- **任务拆解与路由**：把设计方案拆成可执行的子任务，分派给 Codex
- **最终检查**：Codex 产出的代码/测试必须经过 Opus 审阅确认后才算完成
- **综合结论**：多个 Codex 角色返回结果后，由 Opus 做最终裁决

Opus 不写大段实现代码，但设计和把关是 Opus 的职责，不可下放。

### Codex = 执行层

通过 `codex:codex-rescue` subagent 调用（GPT-5.6 Sol），角色由 prompt 区分：

| 角色 | 调用方式 | 职责 |
|------|---------|------|
| **Implementer** | `codex:codex-rescue` | 代码生成、重构、scaffold、重复性改动 |
| **Reviewer** | `codex:codex-rescue` | 独立审查（fresh context）、bug 检测、一致性检查 |
| **Tester** | `codex:codex-rescue` | 测试生成、覆盖率验证、回归测试 |

### Fallback

`.claude/agents/deep-reasoner.md` 和 `fast-worker.md` 仅在 Codex 不可用时启用，走 Claude 模型替代。

## 调用规范

1. Opus 做顶层设计，产出方案后拆解为子任务路由给 Codex。
2. 每个 Codex 调用的 prompt 必须以角色名开头（如 `[Implementer]`、`[Tester]`），并包含完整上下文（文件路径、已有决策、约束条件）。
3. 标准流程：Opus 设计 → Codex Implementer → Codex Reviewer + Tester（可并行）→ Opus 最终确认。
4. 简单任务可跳过设计阶段，直接 Implementer → Tester → Opus 确认。
5. 高风险改动由 Opus 亲自审查 Codex diff，不依赖 Codex Reviewer 独立放行。

## 项目领域规则

- OpenHyra 是 Hyra 研究型 agent harness 的开源复现。核心循环：Context Agent → Proposal Agent → Sandbox + Evaluator → Experience Bank。
- sandbox.py 中的 Seatbelt 配置是安全边界，修改前必须经过 Reviewer 审查。
- `harness.py` 中的 provenance（run manifest、hash 链）是完整性保障，任何改动必须保持 hash 链闭合。
- 新增 task 插件必须遵循 `tasks/*/task.json` schema 和 `evaluator.py` 接口。
- 所有 EB commit 记录不可变——不得实现删除/修改已 commit 记录的功能。
- 测试跑 `pytest tests/` 即可，不需要 LLM 后端或 macOS sandbox。

## 技术栈

- Python 3.10+，仅依赖 numpy（核心）+ pytest（dev）
- 无 web 框架、无数据库，纯文件系统状态
- macOS Seatbelt sandbox（仅生产运行需要）
- LLM 后端：Claude Code CLI / Codex CLI，通过 `llm_backend.py` 抽象
