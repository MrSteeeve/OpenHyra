# 数据字段与表之间的关系

| 文件 | 一行代表什么 | 主要字段 |
|---|---|---|
| main_results_summary.csv | 一种方法，三个 seed 的描述性汇总 | L/K、U/K、raw gap/K、gap score 的均值与跨 seed SD |
| main_results_by_seed.csv | 一种方法在一个 seed 的最终测试结果 | method、seed、score、source_json、program_path |
| program_evaluations.csv | 一个冻结程序在一种评价面板上的结果，或一个未成功生成的尝试 | stage、key、status、score、failure_reason、source_digest |
| evaluation_cells.csv | 一个程序、一个合约实例、一次重复 | lower、upper、对应 SE、strike、raw_gap_normalized、confidence_gap_normalized |
| original_search_attempts.csv | 一次原始模型生成尝试 | round、operator、public_score、control_score、失败与成本 |
| original_search_trajectory.csv | 原始搜索起点或一次尝试结束后的状态 | cumulative_model_seconds、incumbent_key、public_incumbent_score、calibration_incumbent_gap |
| original_selected_incumbents.csv | 原始公开分数选中的最终策略，允许保留初始参考 | selection_rule、method、key |
| feedback_attempts.csv | 一次预先分配的新反馈尝试 | arm、operator、calibration_gap、holdout_gap、parent_source_digest、失败与调用成本 |
| feedback_pairs.csv | 同一 seed × 操作符下的两组反馈对照 | parent gap、两组有效 gap、相对父程序 reduction |
| selected_incumbents.csv | 新校准面板选出的最终策略，允许保留父程序 | selected_operator、selection_rule、key |
| model_calls.csv | 一次真实模型调用 | experiment、method、role、wall_seconds、returncode、tokens_reported |
| cost_and_reliability.csv | 一条实验流程的全部调用汇总 | attempts、valid_candidates、failed_attempts、缺失 token 记录数 |
| infrastructure_recovery.csv | 首次运行的一个尝试及其恢复资格 | eligible、original_failure、record_path、record_sha256 |
| architecture_comparison.csv | 原始或新增实验的一条流程设计 | context_agent、parent、history、selection、test_feedback |
| stopping_behavior.csv | 一次重复下某个行权日期的实际停时占比 | mode、actor、instance_id、repeat、exercise_date_index、stop_fraction |

stage 的 legacy_audit 是原始隐藏面板；calibration 是供本轮选择和反馈使用的面板；holdout 是本轮独立最终测试。key 标识冻结源码与 seed 的组合，source_json 定位到原始完整结果，路径相对数据包根目录。

程序评价表包含失败生成记录以保留分母；并非每一行都实际执行过数值评价。status=error 的 score 为空，不应当作零 gap。只有显式标注 effective_gap 的操作性比较将不可用提案替换为父程序。

lower_normalized 与 upper_normalized 是各实例先除以各自 strike、再求均值。confidence_gap_normalized 是归一化置信 gap 的普通均值；score 还加了 0.25 倍的 90 分位数，二者不可互换。*_sd 为三个 seed 之间的样本标准差，不是标准误或置信区间。

CSV 保留约 12 位有效数字，Excel 保存数值而非格式化字符串。程序源码与原始 JSON 是进一步精确复核的依据。unknown / 空值表示未记录或不适用；不自动补为零。所有参考算法的数值评价耗时是实际 evaluator wall time，包含共同上界计算；fit_seconds 仅为候选训练耗时。
