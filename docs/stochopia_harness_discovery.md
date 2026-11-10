# 股票合约的结构发现 Harness

## 研究目标与入口

研究问题是模型能否在 Harness 与验证环境支持下发现不同的可行现金流机制，以及这些机制如何跨轮形成、修正与重组。期望净收益继续报告，但不再跨机制选唯一赢家。没有任意“新颖性分数”或复杂度奖励。

股票专用 `experiments/run_stochopia_structuring.py` 与四模型比较入口，在 v2 股票任务上默认 `research_goal="discovery"`；显式 `--research-goal return` 复现旧收益搜索。函数的旧 v1 默认与 pipeline 保持收益控制。通用 `harness.py` 使用的 `task.json` 标为 `legacy-economic-control`：其通用标量循环没有机制组合选择能力，结构研究应使用专用入口，不能把通用最高分选择冒称发现模式。

研究目标标识为 `feasible-mechanism-discovery.v1`，与经济评价目标 `equity-demand-expected-return.v2` 分开。此次没有改变股票需求、P/Q 参数、公允价规则、账户或本金损失门槛，没有改正式 benchmark v3。

## 从单一赢家改为机制档案

1. 候选仍先通过 DSL、公开需求约束与环境回放。失败候选保留反馈，不能作为可行发现。
2. 将程序解释为支付规则描述：支付／自动终止时间、条件、支付方向与金额表达式的数据来源和组合关系。忽略数值参数、正向尺度及产品名字；对可交换组合排序。
3. 每个候选在相同的 13 条公开探针上实际执行，保留每月的单位名义现金流。它们包括终值网格和两组同终值、不同历史的路径。去除各支付月的共同固定偏移，并消除一个全局正尺度；支付月位置仍单独比较。这样纯杠杆放大、发行价格差或等价重写不能凭差距变大创建机制。
4. 相同规则描述作为同一机制的参数变体；描述不同但公开执行无差异的候选也不自动增加机制。不同描述且具有执行差异才成为另一组，并保存具体对照路径、支付月及双方现金流。
5. 每一组都保留成员和首次差异证据。组内可用经济收益选择代表；组间没有收益淘汰。下一轮父代来自获得提案机会较少的机制，探索机会不由最高收益垄断。

档案不宣称完全的语义判等。有限探针会漏掉未触及的条件，规则描述也可能合并重要的参数重排，或拆分事实上等价的机制。所有候选及反馈保留，分组边界可复查；不能把自动计数组成论文里的“市场新产品数量”。13 条只是发现诊断，不是完整风险检查或需求证明。

参考档案在生成前冻结：手写种子与现有四族标准产品的规则描述。没有额外求解其最优参数，不增加 API 次数。与参考描述匹配者标记为 `reference-profile`；不匹配者标记为 `unmatched-profile-hypothesis`。后者仅是相对这一有限参考库的待审假设，亚式等已知市场结构也可能不在该库中，绝不能称市场首创。终值支付可以有机制差异，使用路径原语也不自动证明创新。

## 提案与跨轮反馈

提案轮换“探索机制”“改进当前机制”“重组机制”。不要求一个新机制击败其他机制的均值。需求约束保持不变，模型仍需说明具体行为差异与经济代价，而非只给新名称。所有操作仍在现有 DSL 内完成。

同一轮的开发反馈和可用档案在兄弟候选评价前冻结；父代名额可以分别分配给不同的已评价机制。生成只使用公开探针和开发信息，不给实际 P 样本或留出信息。独立采样对照使用同一发现目标、提案轮换与调用额度，提示中没有档案、父代或历史反馈，最终仍通过同一档案分类和组合评价。

`--initial-archive 路径` 支持从保存的代表结构开始下一批。只接受相同模型组、相同需求的档案；导入的是合约工件，重新做开发评价，不沿用旧分数或读取留出结果。机制已获得的提案名额会继承，避免反复从种子抢占名额。它是结构组合的热启动，不是精确恢复随机进程或模型参数学习。没有跨模型共享经验。

## 冻结与评价

结束生成后，冻结所有机制代表并逐一做留出评价，不只测最高收益合约。留出结果不用于更换代表、重新分组或下一次提示。

主要输出：

- `reference_mechanisms.json`：冻结的有限参考库。
- `mechanism_archive.json`：机制假设、成员、首次执行对照、代表工件和分配过的父代名额。
- `selected.json`：`selection_kind="mechanism-portfolio"`，没有唯一发现赢家。
- `held_out.json`：冻结代表逐一评价的结果；留出失败也保留。
- `summary.json`：可行机制组数、带公开对照且未匹配参考的组数、留出可行组数；收益与价格信息归入 `economic_reference`，仅作辅助比较。
- `search_history.json`：机制数量随轮次变化及父代分配。

评分路径和代表回放总数按整个留出组合另行记账。结构多的模型会产生更多留出计算，因此相同 API 次数不等于相同总计算量；比较时同时报告调用、token、评价成本及机制质量，不混合模型结果。

这验证的是反馈指导的结构工件搜索；没有训练新模型、语言生长、Harness 自修改或新增安全/hash 机制。不能直接声称实现了递归自我改进。

## 离线运行

```sh
# OpenHyra 根目录；不加载凭据，不调用 API。
../Stochopia/.venv/bin/python experiments/run_stochopia_structuring.py --generator offline --demand upside-risk20 --rounds 3 --per-round 2 --paths 1024 --output artifacts/discovery-offline
../Stochopia/.venv/bin/python experiments/run_stochopia_structuring.py --generator offline --demand upside-risk20 --rounds 3 --per-round 2 --paths 1024 --search-mode independent --output artifacts/discovery-independent
# 在同模型／需求的档案上继续；仍重新评价导入工件。
../Stochopia/.venv/bin/python experiments/run_stochopia_structuring.py --generator offline --demand upside-risk20 --initial-archive artifacts/discovery-offline/mechanism_archive.json --output artifacts/discovery-next
```

默认离线生成器是手写夹具，不能把它的机制数量当作模型创新。小 Q/P 批次仅验工程流程，不证明精确定价或研究有效性。真实 API 的下一批调用需要新的预算授权，Step 和 Nemotron 的暂停／结束决定保持有效。

## 本次验证

回归覆盖低收益机制保留与获得后续名额、纯杠杆与等价重写不增加机制、仅有未触发条件不增加机制、终值机制和现金流时间差异的真实执行对照、已知分层结构标注、失败候选不入档案、全部代表留出、独立对照、同模型档案热启动和留出不回流。金融规则与原 API 接口回归同时运行。

实际离线报价—提交—结算—回放、独立对照及热启动记录保存在 `artifacts/stochopia-harness-discovery-verification/`；验收汇总见该目录 `verification.md`。

## 受控路径研究的新入口（2026-10-12）

上述v2是历史结构发现协议。新的路径需求、优化目录、理论前沿和配对效力
预检见 `docs/stochopia_path_research_protocol.md`，入口为
`experiments/run_stochopia_path_research.py`。默认离线，不调用API。不能将
旧v2的机制组数解释为新协议的经济优势或统计创新率。
