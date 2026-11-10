# 四组模型 API 接口（2026-10-09）

股票专用 API 与四模型比较入口现默认以结构发现为主目标，输出机制档案和冻结机制组合；按模型／需求分别保存。收益作为辅助经济评价，旧收益选优可显式使用 `--research-goal return`。见 [结构发现 Harness](../../docs/stochopia_harness_discovery.md)。

本接口接在既有股票需求 RSI 搜索入口上。三组使用官方 API，Nemotron 使用 OpenRouter；模型只生成 DSL 合约，评价、定价和搜索规则仍由现有代码执行。每组的候选、父代、反馈、经验库和结果独立保存。当前接入没有训练模型，没有新增安全隔离、hash、CLI 模型后端或模型管理框架。

## 模型与已确认决策

| 配置组名 | 实际请求 model | 路由／格式 | 凭据环境变量 |
|---|---|---|---|
| `claude-haiku-5.5` | `claude-haiku-5-5` | Anthropic 官方 `/v1/messages` | `ANTHROPIC_API_KEY` |
| `deepseek-v4.1-flash` | `deepseek-flash` | DeepSeek 官方 `/v1/chat/completions` | `DEEPSEEK_API_KEY` |
| `step-5-preview` | `step-5-preview` | StepFun 官方 `/v1/chat/completions` | `STEPFUN_API_KEY` |
| `nemotron-3-super` | `nvidia/nemotron-3-super-120b-a12b` | OpenRouter `/api/v1/chat/completions` | `OPENROUTER_API_KEY` |

用户已选择 Nemotron 3 Super，按激活参数量考虑；推理口径为各模型官方默认，记录实际用量。所有预设均省略 temperature、top_p、top_k 和推理强度字段，保留供应商默认行为。Haiku 5.5 使用原生 Messages 格式：顶层 `system`、用户 `messages`、`x-api-key` 和 `anthropic-version`；仅提取最终 text 块作为合约，thinking 块仍保存在原始响应中。

规模信息的边界：

| 模型 | 总参数（B） | 激活参数（B） | 依据 |
|---|---:|---|---|
| Haiku 5.5 | 未核实 | 未核实 | 官方公开资料未核实到规模，配置保存 `null` |
| DeepSeek V4.1 Flash | 552 | 输入 8／输出 16 | 官方发布说明；编码器与解码器不同 |
| Step 5 Preview | 未核实 | 未核实 | 官方公开资料未核实到规模，配置保存 `null` |
| Nemotron 3 Super | 120 | 12 | OpenRouter 模型页 |

因此可以先运行四组比较，但不能称为“四组已严格匹配参数规模”。激活参数相近也不等于相同 FLOPs、推理 token 或成本。本轮只统一候选调用数，不统一输出 token 预算；不同 tokenizer 和默认推理方式仍不同。缺失的推理 token 用 `null` 表示，不能当作零。

官方依据：[Haiku 5.5](https://platform.claude.com/docs/en/models/haiku-5-5/overview)、[Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create)、[DeepSeek V4.1 Flash](https://api-docs.deepseek.com/news/news260910/)、[DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、[Step 5 Preview](https://platform.stepfun.com/docs/zh/guides/models/step-5-preview)、[Nemotron 3 Super](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b/providers)、[OpenRouter 供应商路由](https://openrouter.ai/docs/guides/routing/provider-selection)。StepFun 的官方模型列表另经本机已配置凭据只读查询，返回 HTTP 200 并包含 `step-5-preview`。

## 运行

从 OpenHyra 根目录执行。使用相邻 Stochopia 已安装的虚拟环境；httpx、python-dotenv 为现有依赖。默认读取 `../Stochopia/.env`，已有 shell 环境变量优先。用户已在本机 `/Users/williamqian/.config/stochopia/model-api.env` 填写四组凭据，运行时用 `--env-file` 指定该文件；助手不查看其内容。配置、计划和结果仅记录凭据变量名和是否存在，不保存密钥。

先显示计划，无模型请求、无实验结果写入：

```sh
../Stochopia/.venv/bin/python experiments/run_stochopia_model_comparison.py
```

计划默认为四组 × 四份需求 × 两轮 × 每轮两个候选：最多 **64 次生成调用**。用户已要求去掉研究代码额外设置的 4000 输出 token 上限：DeepSeek、StepFun、OpenRouter 默认省略 `max_tokens`；Anthropic 必须提供此字段，按官方模型查询的最大值设置为 128000。这不会强迫模型用满额度。DeepSeek 官方当前默认思考模式的输出上限为 64K；其他供应商与模型仍有自身限制。因此本配置没有统一输出 token／货币上限，实际用量按模型分别记录；计划的 `planned_max_output_tokens` 为 `null`，不表示供应商允许无限输出。`max_calls=4` 是每组、每份需求的上限，手写种子不占 API 次数。需要复现旧设置可显式添加 `--max-tokens 4000`。

非流式生成可能耗时较长，默认取消客户端 120 秒响应读取超时，连接／写入超时仍为 120 秒；这避免把思考较久直接归为网络失败，没有自动重试。四组继续使用官方默认推理，不能把它称为等输出预算或等计算量。

实际执行（会调用付费 API，需要全体所选模型的凭据，使用新的输出目录）：

```sh
../Stochopia/.venv/bin/python experiments/run_stochopia_model_comparison.py --env-file /Users/williamqian/.config/stochopia/model-api.env --execute --output artifacts/equity-four-models-20261009
```

只运行一组、一个需求，先做少量联通验证：

```sh
../Stochopia/.venv/bin/python experiments/run_stochopia_model_comparison.py --models deepseek-v4.1-flash --demands upside-protected --rounds 1 --per-round 1 --max-calls 1 --execute --output artifacts/equity-deepseek-smoke
```

原单任务入口也可直接使用新预设：

```sh
../Stochopia/.venv/bin/python experiments/run_stochopia_structuring.py --generator api --model claude-haiku-5.5 --demand two-way-protected --output artifacts/equity-single-model
```

四组预设在 `api_models.json`。需要显式设置某模型的推理开关时，可复制配置文件、填写该组供应商支持的 `request_options`，通过比较入口的 `--api-config` 或单任务入口的 `--models` 选择；当前默认配置不主动改变推理。此前 YAML 预设仍可用，但需显式指定 `--models ../Stochopia/config/models.yaml` 和该文件中的组名。

每个候选一次 HTTP 请求，无工具、自动重试、修复、遇截断后自动升额或替换模型。HTTP 错误、拒答、截断和解析失败均占一次额度，作为失败候选保存。调用次数预算不足在执行前报错；缺所选组凭据时，在任何模型调用前报错。OpenRouter 配置 `allow_fallbacks=false`；没有固定上游供应商，默认选择仍可能随请求变化，返回的实际供应商会记录，未返回则未知。

## 独立结果与分析

```text
output/
  plan.json                      # 组名、路由、参数信息和预算
  index.json                     # 各组结果文件的路径，无混合分数
  claude-haiku-5.5/
    model_summary.json           # 本组各需求分别列出
    upside-protected/
      run_config.json
      generation.jsonl
      experience.jsonl
      experience_bank/
      candidates/
      selected.json
      held_out.json
      summary.json
    ...其他需求
  deepseek-v4.1-flash/...
  step-5-preview/...
  nemotron-3-super/...
```

每个模型／需求对重新创建生成器、调用额度、discovery 和 EB。相同需求使用相同种子、系统提示、第一轮请求、价格模拟参数、评价规则和候选额度；每轮只读本组已完成开发评价。日志路径、模型组名与机器耗时不进入设计提示，原始记录仍保留这些信息。留出仅在本组开发选择结束后评价，不回流后续生成。没有跨模型父代、共享经验或汇总排名；比较入口的共同文件只是配置和路径索引。

`generation.jsonl` 保存完整生成请求、最终文本、完整供应商响应、UTC 调用时间、请求／返回模型、实际供应商（若有）、请求 ID、finish reason、时长、HTTP 状态和原始 usage。统一列出输入、输出、总 token 和可获取的推理 token。`summary.json` 只汇总已报告 token，并附各字段报告调用数；没有报告的字段为 `null`，部分失败用量可能未知。DeepSeek alias 与 Preview 名称无法保证权重永久不变，即使请求 model 相同；记录响应信息并不等于固定了模型版本。

v2 经济评价器继续报告满足约束后的期望净收益；当前发现模式不按该收益跨机制选唯一赢家，按冻结档案逐一评价代表结构。接口保留结构、参数、父代和开发／留出表现，供按模型分别分析。**收益提升、参数变化、结构创新是不同结论**；本轮未擅自新增复杂度奖励或创新评分。正式报告创新能力还需确定结构创新的分析口径、重复试验数和预算，不能把手写种子或模拟 HTTP 测试当成模型设计成果。

## 本轮实现与验证范围

改动：`generator.py`、`api_config.py`、`api_models.json`、`.env.api.example`、`API.md`、`TASK.md`，两个 `experiments/run_stochopia*.py`，以及 `tests/test_stochopia_structuring.py`、`tests/test_stochopia_api.py`。Stochopia 只补进展文档；环境、定价、结算、搜索核心及冻结论文工件未改。

验证命令：

```sh
../Stochopia/.venv/bin/python -m pytest tests/test_stochopia_api.py tests/test_stochopia_structuring.py tests/test_algorithm_discovery.py tests/test_feedback.py tests/test_experience_events.py tests/test_generic_task_surfaces.py -q
```

覆盖四家请求格式、完整响应与用量、拒答／截断／连接失败额度、计划不调用模型、缺凭据预检、相同初始请求、各组后续反馈与独立经验库。测试使用模拟 HTTP 响应；接口初始交付时真实生成 POST 次数为 0，官方 StepFun 模型列表完成了真实只读查询。后续运行与修正见下节，不把模拟 HTTP 测试当作模型创新。

上述相关测试 **57 项通过（48.02 秒）**。默认 CLI 计划已实际执行并保存至 `artifacts/stochopia-equity-demands-20261009/api-interface-plan.json`，记录四组与四份需求、64 次最多调用以及本机缺失凭据状态；此文件没有生成结果或密钥。

## 真实运行中的输出上限修正

最初的 4000 上限由实现者作为流程验收输出预算设置，未经过默认推理模型的真实验证。首轮 Haiku 收到的 7 次响应均 HTTP 200、`stop_reason=max_tokens`，每次输出 4000、思考 4000、最终文本 0。用户要求优先解决并移除额外上限后，中断第 8 个在途请求；该请求用量未知。旧轮单独保存于 `artifacts/stochopia-equity-demands-20261009/api-four-models-v1`，标记为调用配置诊断，排除创新比较。

[Anthropic 官方说明](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost)明确 `max_tokens` 包含思考和最终文本。认证 GET `/v1/models/claude-haiku-5-5` 返回 `max_tokens=128000`；[DeepSeek 官方文档](https://api-docs.deepseek.com/api/create-chat-completion/)说明可省略该字段；[Step 5 文档](https://platform.stepfun.com/docs/zh/guides/models/step-5-preview)说明 `max_tokens` 默认 INF，但模型仍有输出长度限制；[OpenRouter](https://openrouter.ai/docs/api_reference/overview)也将其列为可选。修正仅移除额外 token 上限，保持输入、评分与默认推理方式。

同一份原始输入的 Haiku 单次对照正常返回 `end_turn` 与可解析的合约 JSON，实际输出 **2621 token，思考 2128 token**，耗时 14.28 秒。它说明完整输出已在这个请求上得到验证，不保证每次输出都结束或合约都满足需求。工件在 `artifacts/stochopia-equity-demands-20261009/output-policy-pilot`，不混入正式比较。

修正后相关测试 **61 项通过（48.64 秒）**。用户另外明确批准完整重跑四组共 **新增 64 次**，结果目录 `artifacts/stochopia-equity-demands-20261009/api-four-models-v2`；调用记录与各组结果分别保存。旧诊断和对照请求的用量单独记账。

## 后续真实运行决策与独立批次

以下是本次运行的实验配置，未回写成所有将来实验的默认值：

- **Haiku**：上述正式批次已完成 16 次。独立分析与 130 条主轨迹回放已完成，文件位于 `api-four-models-v2/claude-haiku-5.5/analysis.md` 与 `replay_check.json`。
- **DeepSeek**：省略输出上限时，首次响应将默认 65536 token 全部用于思考，未产出合约。用户批准一次最大输出 393216 的对照，成功返回；随后批准用相同默认推理、显式 `--max-tokens 393216` 新跑完整 16 次，目录 `api-four-models-v2-deepseek`。该批次 16 次全部 HTTP200/stop，原规则下 8 个模型候选达标，主轨迹独立重放 116/116 通过。默认上限诊断与对照不混入新批次。
- **Step**：省略 `max_tokens`，保留官方默认推理。一次响应达到模型服务最大输出 64000，最终文本为空。用户明确选择继续当前 16 次批次，不降低推理强度；目录 `api-four-models-v2-step`。有人工暂停的请求时长包含暂停，不能作为纯推理延迟。供应商返回 `reasoning_tokens=0`，但实际含大量 `reasoning_content`，应同时报告这个不一致，不能解读成没有思考。
- **Nemotron / DekaLLM**：原批次返回 8 次，其中 5 次 HTTP429，错误归因于上游共享池。原批次终止并独立保留于 `api-four-models-v2-nemotron`，不能把限流视为模型能力失败。
- **Nemotron / DeepInfra**：用户批准新 16 次批次，固定 `provider.only=["deepinfra"]`、`allow_fallbacks=false`，保持默认推理，并按该上游最大输出设置 16384。配置副本保存在工件根目录 `api_models.nemotron-deepinfra.v3.json`，新组名 `nemotron-3-super-deepinfra`。首笔仍返回 HTTP429，上游报告 `engine_overloaded` 和 `upstream_provider_shared_pool`；用户明确结束本次运行，保留于 `api-four-models-v3-nemotron-deepinfra`，不继续调用或换路由。

DekaLLM 为 FP8、最大输出 235929；DeepInfra 为 BF16、最大输出 16384，来自 [OpenRouter 官方 endpoints](https://openrouter.ai/api/v1/models/nvidia/nemotron-3-super-120b-a12b/endpoints)。精度与输出边界不同，不能合并两批候选、用量或成绩。已返回失败请求与中断在途请求分开记录；中断请求用量未知，不按零计算。

本轮实时状态见工件根目录 `live_run_state.json`；逐模型报告和阶段报告只覆盖已落盘记录。模型概要在每份需求完成后增量写入，文件存在不代表四份需求全部完成。

DeepSeek 两个无条件到期支付候选实际最低支付为 0.85，按相同成交价的本金损失上界约 14.87%，但适配器把所有 pay 视作可能缺席而按零下界拒绝。用户已批准在原批次结束后做最小下界修正、所有模型已有工件统一离线重评，不新增 API 调用；原始成绩和修正后的离线选择分别保存。当前准备文件为工件根目录 `proposed_payment_floor_fix.patch`、`proposed_payment_floor_tests.patch`、`revalue_payment_floor.py`，尚未修改运行中的评价器。
