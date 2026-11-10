# Stochopia 市场世界模型：逐期前推搜索任务

协议：`openhyra-stochopia-world-model.v1`（评分协议 `stochopia.market-world-model-forecast.v1`，实现在 Stochopia
`stochopia/research/forecast_protocol.py`）。日期：2026-11-08。

## 1. 这个循环要完成的系统

维护并不断修订一个**中证 500 / 中证 1000 的 P/Q 联合概率世界模型**，使它对真实未来的预测越来越好。

- **认知**：一个完整的 Python 程序 `world_model.py`。它就是模型对市场的"理论"：波动怎么演化、冲击分布、
  两个指数的相依、IV 与基差怎么变、P 侧状态与 Q 侧报价怎么联动。
- **外部环境**：真实市场。P 侧是指数随后 21 个交易日的真实路径；Q 侧（用户选定的 B）是 21 个交易日后真实
  观测到的报价：中证 1000 MO 期权 30 天 ATM IV、IC / IM 期货隐含 carry。
- **更新**分两层：
  - 内层：每个预测起点，程序只用截至该日的数据重新拟合，即参数随外部数据更新。
  - 外层：OpenHyra 循环让基座模型（API）读评分反馈和经验记录，修改程序结构，即理论随外部反馈更新。

为什么不直接以"产品设计"为目标：设计目标没有唯一正确的定义，也没有外部真值；预测有 proper scoring rule
和真实结果可对。衍生品设计的相关性通过 **payoff 探针**进入评分：用 StructurerDSL 写的 6 个路径依赖结构
（上涨 digital、0.95 下触、1.03 上触、ATM 亚式看涨、0.97–1.03 区间累计、21 日迷你雪球 KO 1.02 / KI 0.93）。
模型必须把这些 payoff 的分布也预测准。将来在 `StochopiaDSLEnv` 里做产品设计时，可以把通过审计的世界模型作为
P 侧生成器（尚未接线，见第 7 节）。

## 2. 候选程序接口

```
python world_model.py --input IN --output OUT
```

- `IN/history.npz`：截至起点（含当日）的**固定长度**尾部窗口，最后一行是起点：
  - `returns (504,2)`：日对数收益；
  - `rate_3m (504,)`：91 天连续 SHIBOR；
  - `carry (126,2)`：IC / IM 91 天期货隐含 carry；
  - `iv_1m (126,)`：MO 期权 30 天 ATM IV。
  - 无报价处为 NaN。不提供日期和指数点位。窗口长度固定，输入长度不会暴露起点在面板里的位置。
- `IN/query.json`：`horizon=21`、`n_paths=256`、`seed`。
- `OUT/forecast.npz`：只能包含两个数组：
  - `returns (256,21,2)`：联合样本路径，|r| ≤ 0.5；
  - `q (256,3)`：[CSI1000 IV, CSI500 carry, CSI1000 carry] 在第 21 个交易日的样本。

只允许 numpy / scipy。每个起点单独运行一次，在 Seatbelt 训练沙箱里执行（`sandbox.run_training_sandbox`）：
- 断网；
- 只能写自己的输出目录与临时目录；
- 读不到面板文件或其他起点的数据；
- 单次 20 秒、1 GB。

因为每个进程只拿到截至自己起点的前缀，前视在构造上不可能发生。

## 3. 评分（可信端，候选自报的指标一律不用）

- 每个目标在每个起点算样本 CRPS，二值探针退化为 Brier 分数。
- 对照参考模型（EWMA(0.94) 高斯 GBM + 滚动相关；Q 侧随机游走）计算 skill：
  `1 − Σ CRPS_候选 / Σ CRPS_参考`。
- 四个族：`return`、`volatility`、`payoff_probe`、`q_quote`。族内对目标取平均，总分再对族取平均。
  0 表示与参考持平，越高越好。
- 计分目标集合在每个 split 内固定：观测数不足 3 的目标不计入点估计，也不计入任何 bootstrap 抽样。
  候选在有观测的目标上 CRPS 不是有限值，直接判为违规，不能通过"弃权"让某个族消失。
  Q 样本限定在 IV ∈ (0, 3]、|carry| < 2。
- 置信下界：对起点做循环 moving-block bootstrap（块长 3，1000 次），取 5% 分位作为 `score_lcb_95`。
- CRPS 用 n² 形式（V 统计量），二值时恰好等于 Brier 分数。它比无偏版本多出 E|X−X'|/(2n)，对欠离散的
  集合略有利；n = 256 时影响很小。
- 反馈内容：各目标 skill、校准（80% 中心区间覆盖率、PIT 偏斜、二值的预测概率对实现频率）、按起点处
  过去 21 日已实现波动三分位的切片。不含日期，也不含逐起点数值。

## 4. 数据划分

- 起点：从第 504 个交易日开始（保证窗口完整），每 21 个交易日一个，要求后面还有完整的 21 天可供验证。
- **dev**：起点 ≤ 2023-06-30，共 75 个（2017-01-25 至 2023-06-20）。只有 dev 进入循环反馈。
  最后一个 dev 起点的结果窗口延伸到 2023-07-21，也就是第一个 test 起点，两者的结果不重叠。
- **test**：起点 > 2023-06-30，共 28 个（2023-07 至 2025-11）。搜索结束后，只对选出的模型和 seed 各评一次。
  注意：2024–2025 在本项目里已经用过（见 Stochopia Q 侧 bundle 的 split_policy），所以 test 不是私有未见集。
- Q 侧 IV 与 IM carry 从 2022-07-22 才有，dev 里只有约 12 个起点有这两个目标。

## 5. 循环

`experiments/run_stochopia_world_model.py`，基于 `algorithm_discovery.AlgorithmDiscoveryLoop`：

1. 每轮冻结当前最优程序作为父代。每个槽位轮换一个修订方向：波动动态、尾部与相依、Q 侧动态、联合状态。
2. 生成器收到的请求包括：父代程序全文、父代 dev 反馈、最近 12 条经验。它返回完整程序，并声明三项：
   - 机制；
   - 预测哪些目标或族会改进；
   - 什么结果会证伪这个机制。
3. 可信评测在 dev 上打分，并核对声明的预测是否兑现（`prediction_check.hit_rate`）。核对结果写进经验，
   下一轮可见。
4. 结束后按 dev 最高分选出模型，对它和 seed 做一次 test 审计（`final_audit.json`）。

产物：`generations.jsonl`（原始回复与用量）、`experience.jsonl`、`ledger.jsonl`、`programs/*.py`、
`summary.json`、`final_audit.json`。

### 运行

在 OpenHyra 目录下运行：

```
# 离线：手写 fixture，不调用模型，只验证管线
../Stochopia/.venv/bin/python experiments/run_stochopia_world_model.py --generator offline \
    --rounds 2 --per-round 2 --output artifacts/stochopia-world-model-<date>/offline

# API：预设名取自 tasks/stochopia_structuring/api_models.json；不加 --execute 只打印调用计划
../Stochopia/.venv/bin/python experiments/run_stochopia_world_model.py --generator deepseek-v4.1-flash \
    --rounds 3 --per-round 2 --output artifacts/stochopia-world-model-<date>/deepseek --execute
```

面板需要先在 Stochopia 里构建：`.venv/bin/python -m stochopia.research.market_panel`。

## 6. 已知局限

- **样本少**：dev 只有 75 个月频起点，test 只有 28 个，单次提升很难显著，应以 LCB 和 test 一起判断。
- **预训练知识与日期反推**：基座模型知道 2015–2025 年的 A 股行情。已做的防护：
  - 固定长度窗口，长度不泄露起点位置；
  - 提示词里没有年份。

  仍然存在的风险：
  - Q 窗口内若出现从 NaN 变为有报价（2022-07-22 MO/IM 上市），模型可以据此推出日期，影响约 6 个 dev 起点；
  - 程序可能凭收益形态认出具体行情（如 2024-02、2024-09、2025-04），进而硬编码历史事件。

  这些风险目前没有检测。计划中的对照是在打乱区块的合成面板上复评同一程序，看行为是否依赖具体位置。
- **IV 只有 ATM**：没有完整曲面；CSI500 的 IV（510500 ETF 期权）本地只有抽样日，未使用。
- **参考模型和 seed 程序语义相同**：seed 的分数恒为 0，是一致性检查。
- 离线 fixture 是手写的固定程序，它们的分数不代表任何模型能力。

## 7. 下一步（未实现）

- ~~把通过 test 审计的世界模型接成 Stochopia 的 P 侧生成器~~：2026-11-08 已完成，`r02s01` 进入 Stochopia
  `stochopia/research/world_models`，生成器为 `stochopia.research.world_model_generator`，处于草稿层
  （development-only；层级定义见 StructurerDSL `docs/SPEC.md` 11.2），详见 Stochopia `docs/research/2026-11-08-market-world-model.md`。下一步是在它的分布下
  设计产品（OpenHyra 设计循环），再在真实路径上回看。
- 允许程序在起点之间保存状态（在线学习），做法是把前一起点的输出状态传给下一起点，仍然只能传过去的信息。
- 做匹配对照：冻结世界模型的智能体对比可更新世界模型的智能体，同样的模型、同样的调用预算。
