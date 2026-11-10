# Stochopia 股票需求驱动的合约设计实验

股票专用入口与四模型比较入口现默认使用 **结构发现模式**，主结果为可行机制档案和多机制留出评价，见 [结构发现 Harness](../../docs/stochopia_harness_discovery.md)。经济回报仍由 v2 评价器计算，作为辅助指标。通用 `harness.py` 加载的 `task.json` 标为旧收益控制，不支持机制组合选择。实现、评分选择、运行方法和验收边界见 [Harness v2 说明](../../docs/stochopia_harness_v2.md)。以下 v1 描述与命令保留作为历史协议；复现旧需求需显式添加 `--demand-protocol v1`，支付下界局部修正的影响见新版说明。

## 当前研究任务（2026-10-09 已确认）

目标是让 RSI 架构生成、评价和改进新产品。输入浓缩为可检验的股票需求，输出是 DSL 现金流合约；不建立客户画像、谈判或多智能体博弈。当前只用一个股票／股指现货序列。以后增加其他标的，不在本轮实现外汇、利率、集中持股融资或多币种账户。

四份需求统一实际投入 100 万元、期限 3 个月，按相同 Q 公允价值成交：

| demand_id | 支付目标 | 最终本金损失上限 |
|---|---|---|
| `upside-protected` | 指定温和回撤情景收回投入，上涨 10% 情景净收益至少 1% | 0% |
| `upside-risk20` | 同上 | 20% |
| `two-way-protected` | 指定上涨 10% 和下跌 10% 情景净收益均至少 1% | 0% |
| `two-way-risk20` | 同上 | 20% |

这些金额、期限、收益和亏损阈值是用户确认的合成科研夹具，不是银行客户原始委托。1% 是整段三个月的净收益要求，不是年化。具体公开锚点见 `equity_demands.py` 和生成请求的 `demand.anchor_requirements`：回撤任务检查横盘、-5%、-10% 的给定完整月度路径及 +10% 路径；双向任务检查两条正负 10% 路径。有限锚点不能证明任意温和回撤路径都满足需求。

保本是需求输入的选择：保本任务要求合约在到期／自动终止时的保证支付下界覆盖实际购买成本；允许非保本不意味着强迫设计出会亏本金的产品。保护水平改变后重新计算 Q 价格，不手填票息或发行折价。不能把 `funded` 的面额归还直接当作实际投入资金的保证。亏损上限使用现有 DSL 保守支付下界；这会拒绝部分区间估计无法证明的可行组合，不宣称任意 DSL 的完整验证。

不同候选的名义本金由评价器按 Q 单位价格归一化，使实际购买对价为 100 万元（浮点误差处理保留不足一分钱的未投资现金）。模型不能通过扩大名义本金提高分数。相同结构与参数、同一归一化名义本金用于所有场景；每个场景的初始 Q 输入相同。中途自动终止得到的现金保持现金，不再投资或追加利息。

评分 `equity-demand-expected-return.v1`：先满足公开支付锚点和最终亏损上限，再要求所有开发场景通过环境约束、成交一次且完成回放，才获得可排序分数。分数是固定投入预算下的期望净收益率；同时报告期望净利润（元）。当前 P 分布是上涨、横盘、温和下跌、强涨、深跌、回升、冲高回落七个合成场景，各占 1/7；开发和留出各七条路径，分开保存。它是有限压力情景分布，不是市场校准的真实收益预测；Q 仍使用原 GBM 估值，不能用这些 P 权重定价。

复用 Stochopia 的机械账户与公开接口：内部兼容字段不构成客户画像或额外风险效用。旧 `max_loss_pct` 混合中途价格压力损失，因此该兼容字段保持宽松；本任务的最终本金损失上限由需求评价器独立严格检查，不把到期保本误解成中途价格不会下跌。现有发行、对冲风险、账户和独立 replay 保持原行为。

RSI 每轮接收完整需求、公共市场条件、父代和上一轮开发反馈，自由组合允许的 DSL 原语。历史产品名称只作参考，不限定为模板选择。手写控制合约用于验证需求可行性和比较；离线生成器仍是预设夹具，不代表模型生成或自我改进。实际 API 结果需要另行运行并报告结构、参数、开发／留出收益及与控制合约的差异，不能只凭新 JSON 或参数改动宣称结构创新。

首轮不做定期收入需求：真正期间现金收入需补中途付息，不能拿到期累计支付替代。已接入 Haiku 5.5、DeepSeek V4.1 Flash、Step 5 Preview 官方 API 与 OpenRouter Nemotron 3 Super，按用户选择使用各模型默认推理；各组结果独立。入口、规模披露边界与运行预算见 [API.md](API.md)。

运行单份需求（从 OpenHyra 根目录）：

```sh
../Stochopia/.venv/bin/python experiments/run_stochopia_structuring.py --generator offline --demand upside-protected --paths 4096 --output artifacts/equity-upside-protected
../Stochopia/.venv/bin/python experiments/run_stochopia_structuring.py --generator api --demand two-way-risk20 --model deepseek-v4.1-flash --rounds 2 --per-round 2 --max-calls 4 --output artifacts/equity-two-way-risk20-api
```

`task.json` 默认是 `upside-protected`。专用入口支持四个 demand_id，分别搜索和选择；每个任务使用相同模型和调用额度。旧流程复现需显式使用 `--demand pipeline`。目前未新增安全隔离、hash 机制、训练模型或改动搜索核心。

公开结构参考：[UBS 障碍反向可转债](https://www.ubs.com/ch/en/services/guide/investments/articles/barrier-reverse-convertibles.html)、[Leonteq 历史 Twin-Win 说明](https://structuredproducts-ch.leonteq.com/news/underlying-in-focus/underlying-temenos-twin-win?language_id=1)、[UBS Bonus Outperformance 示例](https://microsites.ubs.com/spfinder/en/products-detail/equity-bonus-outperformace-certificate.html)。参考结构不等于当前需求参数来自真实客户。

## 已完成的旧流程实验 v1（保留复现）

这是合成数据上的流程实验，不是正式金融 benchmark。候选统一为 **固定 DSL 合约工件**；暂不生成交互策略程序。每个工件在所有场景使用相同结构、参数和名义本金。学习模型以后替换生成器接口，不进入环境内部。

成交规则沿用已经确认的 DSL 实验约定：Q 模型的未来现金流公允现值就是购买对价，无发行加价；名义本金是支付尺度。报价、客户接受、账户、静态月度对冲、自动终止与结算由 Stochopia 执行。候选不能提供价格或分数。

候选文件是 `contract.json`，格式为 `{"protocol":"stochopia-structuring-search.v1","contract":{...}}`。`contract` 使用 `stochopia.dsl-contract.monthly.v1` 的 JSON AST。单标的、整数月观察网格、期限 1–3 个月；支持既有 DSL 实验的终值、平均、运行最值、区间累计、条件支付与自动终止。完整支持边界见 Stochopia 的 DSL 适配报告。未增加主动行权或成交后调仓。

固定交互策略先询问 `capital`、`return_hurdle`、`loss_tolerance`，再报价一次、提交一次，之后持有到终止／到期。报价或提交失败就推进时间，不自动修改候选。模型输入由实际公开 observation、这些询问产生的 history、action schema 和先前开发反馈构成。候选不接收未来市场路径或任务 manifest。

主目标 `accepted-client-return.v1`：所有开发场景合法、无无效动作且各接受一次成交才有可排序分数；分数是既有 `client_flow_adjusted_return` 的场景均值，即外部资金流调整后的客户净清算财富变化／初始客户现金。全部奖励向量、约束失败、风险占用、询问和报价成本同时保留。它没有纳入客户风险效用或发行商利润；规模可能受到该目标驱动，不据此宣称最优产品。另有 `acceptance-rate.v1` 作为单纯流程目标，由实验配置决定，候选不能更改。

公开例子 1 个，开发场景 2 个，最终留出场景 2 个，均为手工合成三个月路径。它们故意共享当前信息而有不同后续走势。开发反馈可以用于下一轮修改；同轮兄弟候选使用相同父代与反馈。留出只在完成开发选择后评价一次，不进入生成请求、经验训练行、父代选择或后续反馈。留出按流程隔离，沿用已有记录／回放；没有新增加密、sandbox 或防篡改设施。

推荐运行 `experiments/run_stochopia_structuring.py`：复用 `AlgorithmDiscoveryLoop`、`DiscoveryLedger` 和 `ExperienceBank`，这是新增任务的 API／离线入口。原有 `harness.py` 的 CLI 模型后端没有改动。本任务同时提供它可加载的 `task.json`、数据提交用 `solve.sh` 和标准 JSON evaluator CLI；首轮验收运行的是专用入口，不声称已验证通用 CLI 搜索或 macOS sandbox。

生成接口 `ContractGenerator.generate(request) -> Generation` 接收公开条件、父代、开发反馈、目标、轮次和槽位，返回工件、原始输出、用量、时长和错误。离线夹具与 OpenAI-compatible API 已实现。API 每候选一次请求、无工具、无自动重试、无截断升额；截断、网络失败、解析失败也计入调用额度并记录为失败候选。输入和供应商返回的 tokens 单独记账；无法预知实际货币账单。

API 的原始输出采用 `contract_sexpr`、`params`、`notional`、`rationale`，由生成适配器转成 AST 工件；语义校验留给 evaluator。以后模型可直接返回相同工件，不需要训练或扩展一个通用模型框架。

主要输出：`run_config.json`、`generation.jsonl`、`experience.jsonl`、`discovery.jsonl`、`experience_bank/`、每场景的 trajectory/evaluation、冻结的 `selected.json` 与单独的 `held_out.json`。训练尚未发生；`experience.jsonl` 只有开发数据，公开种子单独保存，留出结果不自动导出为训练样本。

复现命令（从 OpenHyra 目录执行；使用已安装 Stochopia 和 StructurerDSL 的相邻虚拟环境）：

```sh
../Stochopia/.venv/bin/python experiments/run_stochopia_structuring.py --generator offline --demand pipeline --output artifacts/stochopia-structuring-offline --paths 4096
../Stochopia/.venv/bin/python experiments/run_stochopia_structuring.py --generator api --demand pipeline --models ../Stochopia/config/models.yaml --model deepseek-v4-flash --rounds 2 --per-round 2 --max-calls 4 --max-tokens 4000 --output artifacts/stochopia-structuring-api
```

API 默认读取本任务 `api_models.json` 与现有 Stochopia `.env`，不保存凭据；历史 YAML 配置需显式指定。正式比较需预先固定模型、提示信息、调用次数、输出上限、价格模拟路径和种子、交互额度、开发／留出定义及评分口径。API 能显式控制这些预算，但本身不保证公平，也不保证供应商模型保持不变。四组规模未全部核实，默认推理也不等于等计算量，见 [API.md](API.md)。
