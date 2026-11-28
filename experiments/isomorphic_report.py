"""Read the frozen M3 evidence and render a five-table study report, without evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import csv
import json
from math import isfinite
from pathlib import Path
import re
from statistics import fmean
from typing import Any

Json = dict[str, Any]
ENVIRONMENTS = ("env_alpha", "env_beta")
ARMS = ("d_only", "true_z_d", "shuffled_z_d")
TRACKS = tuple((env, arm) for env in ENVIRONMENTS for arm in ARMS)
Z_KEYS = ("sign_acf_sum_1_32", "flow_vr_32_4", "trade_to_top1_depth", "midprice_shift_rate", "cancel_to_limit_ratio",
          "zero_wait_prob", "interarrival_cv2", "book_imbalance_bias", "flow_sign_bias")
METRICS = ("val_nll_total", "dt_nll", "counterfactual_impact_mae", "exhausted_replicate_rate", "memory_gap_64")
METRIC_NAMES = ("Δ总 NLL", "Δdt NLL", "ΔCF MAE (tick)", "Δ耗尽率", "ΔG_mem(64)")
ALIGNED_IDS = {"env_alpha": {"C04", "C05", "C06"}, "env_beta": {"C03", "C08"}}
VERDICTS = ("SUPPORTED", "REFUTED", "INCONCLUSIVE", "NOT_EVALUABLE")
OMITTED = "Regime statistics omitted in this control condition; rely on model residuals and structural boundaries."
LABELS = re.compile(r"(?i)(?<![A-Za-z0-9])(?:w[26]|thin|thick|hawkes|long)(?![A-Za-z0-9])")


# --- Frozen evidence and auditable denominators ---


def _load(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path.name}")
    return value


def _journal(path: Path) -> list[Json]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _number(value: object) -> str:
    if type(value) in (int, float) and isfinite(value):
        return format(value, ".6g")
    return "NA"


def _fmt(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows]])


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)" if denominator else "NA (0 denominator)"


def _boolean(value: bool) -> str:
    return "通过" if value else "未通过"


def _pair(value: object) -> str:
    """Candidate-minus-reference effect; no interval is invented for unsupported days."""
    if not isinstance(value, Mapping):
        return "NA（无配对记录）"
    estimate = _number(value.get("estimate"))
    interval = value.get("interval")
    band = "[" + ", ".join(_number(x) for x in interval) + "]" if interval is not None else "无区间"
    days = f"{value.get('paired_days', '?')}/{value.get('registered_days', '?')} 日"
    draws = f"{value.get('bootstrap_valid', '?')}/{value.get('bootstrap_total', '?')} 抽样"
    return f"{estimate} {band} ({days}; {draws})"


def _z(block: Mapping[str, Any]) -> str:
    lines = ["metric | median | IQR"]
    for key in Z_KEYS:
        value = block[key]
        lines.append(f"{key} | {_number(value['median'])} | {_number(value['intraday_iqr'])}")
    return "\n".join(lines)


def _z_block(messages: list[Json]) -> tuple[str, list[Json]]:
    """Remove exactly the treatment block; retain every other message character."""
    start = "[Macro Data Regime Z]\n"
    end = "\n\n[Model Residuals D]"
    blocks: list[str] = []
    stripped = []
    for message in messages:
        content = message["content"]
        if start in content:
            previous, remainder = content.split(start, 1)
            block, following = remainder.split(end, 1)
            blocks.append(block)
            content = previous + start + end + following
        stripped.append({**message, "content": content})
    if len(blocks) != 1:
        raise ValueError("Each prompt must contain exactly one registered Z block")
    return blocks[0], stripped


def _expect_z(baseline: Json, env: str, arm: str) -> str:
    if arm == "d_only":
        return OMITTED
    source_env = env if arm == "true_z_d" else ENVIRONMENTS[1 - ENVIRONMENTS.index(env)]
    return _z(baseline[source_env]["z"])


def _trial_rows(tracks: Mapping[tuple[str, str], list[Json]], env: str, arm: str) -> list[Json]:
    return tracks[(env, arm)][1:]


def _valid_evaluation(row: Json) -> bool:
    return row.get("evaluation", {}).get("status") == "VALID"


def _parse_valid(row: Json) -> bool:
    prediction = row.get("prediction", {})
    return {"target_metric", "expected_direction", "selected_card_ids", "regime_hypothesis"}.issubset(prediction)


def _primary_match(row: Json, env: str) -> bool:
    cards = row.get("selected_card_ids", [])
    return bool(cards and row.get("prediction") and cards[0] in ALIGNED_IDS[env]
                and (env != "env_beta" or "C06" not in cards)
                and row.get("outcome") not in ("INVALID_PROPOSAL", "PROVIDER_ERROR", "INTERRUPTED_CALL"))


def _prediction_counts(rows: Sequence[tuple[str, Json]]) -> tuple[int, ...]:
    valid = sum(_parse_valid(row) for _, row in rows)
    aligned = sum(_primary_match(row, env) for env, row in rows)
    verdicts = Counter(row.get("prediction_verdict", "NOT_EVALUABLE") for _, row in rows)
    return (len(rows), valid, aligned, *(verdicts[key] for key in VERDICTS))


def _usage(entries: Sequence[Json]) -> str:
    inputs = [e.get("input_tokens") for e in entries]
    outputs = [e.get("output_tokens") for e in entries]
    known_in = sum(x for x in inputs if type(x) is int)
    known_out = sum(x for x in outputs if type(x) is int)
    missing = sum(type(a) is not int or type(b) is not int for a, b in zip(inputs, outputs, strict=True))
    return f"已知输入 {known_in} / 输出 {known_out}；{missing} 个回执用量不完整"


def _track_holdout(holdout: Json, freeze: Json, env: str, arm: str) -> Json:
    digest = freeze["trajectories"][f"{env}/{arm}"]["final_sha256"]
    return holdout[env][digest]


# --- Table 1: controlled information delivery ---


def _table_control(baseline: Json, tracks: Mapping[tuple[str, str], list[Json]], calls: list[Json]) -> tuple[str, bool]:
    data_rows = [[key, *[f"{_number(baseline[env]['z'][key]['median'])} / {_number(baseline[env]['z'][key]['intraday_iqr'])}"
                         for env in ENVIRONMENTS]] for key in Z_KEYS]
    before = []
    overall = True
    for env in ENVIRONMENTS:
        first = [_trial_rows(tracks, env, arm)[0] for arm in ARMS]
        first_z = [_z_block(row["prompt"]) for row in first]
        non_z = all(item[1] == first_z[0][1] for item in first_z[1:])
        ids = [row["active_card_ids"] for row in first]
        same_cards = all(row == ids[0] for row in ids[1:])
        seed = [tracks[(env, arm)][0] for arm in ARMS]
        same_seed = all(row["source_sha256"] == seed[0]["source_sha256"] and
                        row["evaluation"] == seed[0]["evaluation"] for row in seed[1:])
        same_seed = same_seed and seed[0]["evaluation"] == baseline[env]["evaluation"]
        for arm in ARMS:
            rows = _trial_rows(tracks, env, arm)
            track_calls = [r for r in calls if str(r["call_id"]).startswith(f"{env}/{arm}/")]
            reserved = [r for r in track_calls if r["event"] == "reserved"]
            completed = [r for r in track_calls if r["event"] == "completed"]
            expected_z = _expect_z(baseline, env, arm)
            match_z = all(_z_block(row["prompt"])[0] == expected_z for row in rows)
            forbidden = any(LABELS.search(msg["content"]) for row in rows for msg in row["prompt"])
            tokens = [row["prompt_tokens"] for row in rows]
            actual = [row["input_tokens"] for row in completed if type(row.get("input_tokens")) is int]
            cap = max(tokens) <= 2800 and (not actual or max(actual) <= 2800)
            correct_calls = len(reserved) == len(rows) and len({r["call_id"] for r in reserved}) == len(rows)
            receipt_match = all(any(r["call_id"] == f"{env}/{arm}/{row['cand_id']}" and r["messages"] == row["prompt"]
                                    for r in reserved) for row in rows)
            passed = all((non_z, same_cards, same_seed, match_z, not forbidden, cap, correct_calls, receipt_match))
            overall = overall and passed
            before.append([env, arm, ", ".join(ids[0]), _boolean(non_z and same_cards and same_seed), _boolean(match_z),
                           _boolean(not forbidden and receipt_match), f"{len(reserved)} / {len(completed)}",
                           f"{fmean(tokens):.1f} / {max(tokens)}", f"{fmean(actual):.1f} / {max(actual)}" if actual else "NA",
                           _fraction(sum(_valid_evaluation(row) for row in rows), len(rows)), _boolean(passed)])
    data = _table(("训练数据属性（median / IQR）", *ENVIRONMENTS), data_rows)
    audit = _table(("环境", "臂", "首轮竞争短卡", "首轮同源同 D 同卡 / 非 Z 字符一致", "Z 值", "无环境标签 / 请求回执一致",
                    "正式预约 / 完成回执", "预估输入 tokens 均 / 最大", "实际输入 tokens 均 / 最大", "契约通过/轮数", "审计"), before)
    return data + "\n\n真正 Z 臂使用本环境列，置换臂使用另一列，D-only 区块固定为空值说明；只在训练期计算 Z。\n\n" + audit, overall


# --- Tables 2–3: independent prediction and selection ledgers ---


def _prediction_row(label: str, rows: Sequence[tuple[str, Json]]) -> list[object]:
    n, valid, aligned, supported, refuted, uncertain, unavailable = _prediction_counts(rows)
    contra = sum(env == "env_beta" and "C06" in row.get("selected_card_ids", []) for env, row in rows)
    beta_n = sum(env == "env_beta" for env, _ in rows)
    return [label, n, valid, _fraction(aligned, n), _fraction(aligned, valid), f"{supported}/{refuted}/{uncertain}/{unavailable}",
            _fraction(supported, n), _fraction(supported, supported + refuted), _fraction(contra, beta_n)]


def _table_prediction(tracks: Mapping[tuple[str, str], list[Json]]) -> str:
    rows = []
    for env, arm in TRACKS:
        trials = [(env, row) for row in _trial_rows(tracks, env, arm)]
        rows.append(_prediction_row(f"{env}/{arm} · 首轮", trials[:1]))
        rows.append(_prediction_row(f"{env}/{arm} · 全轮", trials))
    for arm in ARMS:
        all_rows = [(env, row) for env in ENVIRONMENTS for row in _trial_rows(tracks, env, arm)]
        first_rows = [(env, _trial_rows(tracks, env, arm)[0]) for env in ENVIRONMENTS]
        rows.append(_prediction_row(f"{arm} · 两环境首轮合计", first_rows))
        rows.append(_prediction_row(f"{arm} · 两环境全轮合计", all_rows))
    return _table(("比较（描述性）", "尝试数", "解析有效", "主卡匹配/全尝试", "主卡匹配/解析有效",
                   "方向支持/证伪/不定/未测", "支持/全尝试", "支持/可判方向", "反向边界 C06/β 尝试"), rows)


def _prediction_boundary(tracks: Mapping[tuple[str, str], list[Json]]) -> str:
    first = {arm: sum(_primary_match(_trial_rows(tracks, env, arm)[0], env) for env in ENVIRONMENTS)
             for arm in ARMS}
    all_rows = [(env, row) for env, arm in TRACKS for row in _trial_rows(tracks, env, arm)]
    n, valid, aligned, supported, refuted, uncertain, unavailable = _prediction_counts(all_rows)
    first_text = "；".join(f"{arm}={first[arm]}/2" for arm in ARMS)
    conclusion = "首轮各臂无可区分的主卡声明差异。" if len(set(first.values())) == 1 else "首轮存在描述性主卡声明差异。"
    return (f"首轮匹配：{first_text}。{conclusion} 全部 {n} 次尝试中 {valid} 次解析有效、{aligned} 次主卡声明匹配、"
            f"{supported} 次严格方向支持、{refuted} 次严格反向、{uncertain} 次统计不定、{unavailable} 次不可评估。"
            "声明匹配与实际方向支持是两个不同读数；不能将前者代替后者。")


def _regret_summary(summary: Json, key: str) -> object:
    value = summary.get("regret", {}).get(key)
    if value is not None:
        return value
    return summary.get("trajectories", {}).get(key, {}).get("regret")


def _regret_text(value: object) -> str:
    if not isinstance(value, Mapping):
        return _number(value)
    estimate = _number(value.get("estimate"))
    support = value.get("paired_days", "?")
    pool = value.get("oracle_models", "?")
    reason = value.get("reason")
    if reason:
        return f"{estimate}; {support}/12 共同日；Oracle {pool} 个模型；{reason}"
    return (f"{estimate} tick；{support}/12 共同日；Oracle {pool} 个模型；"
            f"final {_number(value.get('final_mae'))} / oracle {_number(value.get('oracle_mae'))}；"
            f"oracle SHA {str(value.get('best_sha256', 'NA'))[:12]}")


def _table_selection(tracks: Mapping[tuple[str, str], list[Json]], freeze: Json, summary: Json) -> str:
    rows = []
    for env, arm in TRACKS:
        key = f"{env}/{arm}"
        trials = _trial_rows(tracks, env, arm)
        counts = Counter(row["outcome"] for row in trials)
        other = ", ".join(f"{name}={n}" for name, n in sorted(counts.items())
                          if name not in ("PROMOTED", "STEPPING_STONE", "REJECTED", "DUPLICATE_REJECTED")) or "0"
        selected = freeze["trajectories"][key]
        rows.append([env, arm, sum(_valid_evaluation(row) for row in trials), counts["PROMOTED"], counts["STEPPING_STONE"],
                     counts["DUPLICATE_REJECTED"], counts["REJECTED"], other, selected["final_id"],
                     selected["final_sha256"][:12], _regret_text(_regret_summary(summary, key))])
    return _table(("环境", "臂", "有效评估", "晋升", "踏脚石", "去重", "落选", "其他结果", "最终父模型", "源码 SHA 前缀",
                   "留出选择后悔值 / 共同日支持"), rows)


def _regret_boundary(summary: Json) -> str:
    trivial = []
    for env, arm in TRACKS:
        key = f"{env}/{arm}"
        regret = _regret_summary(summary, key)
        if isinstance(regret, Mapping) and regret.get("estimate") == 0 and regret.get("oracle_models") == 1:
            trivial.append(key)
    if trivial:
        return (f"{', '.join(trivial)} 的零 Regret 仅因 Oracle 集合只有 B0 而成立（无有效备选），"
                "不构成架构选择能力的正面证据。任何零 Regret 都只相对各自实际评价的候选池，而非相对未知全局最优。")
    return "零 Regret 只相对该轨迹实际评价的候选池，而非相对未知全局最优。"


def _same_choices(freeze: Json) -> str:
    identical = [env for env in ENVIRONMENTS
                 if len({freeze["trajectories"][f"{env}/{arm}"]["final_sha256"] for arm in ARMS}) == 1]
    if not identical:
        return "两个环境均未出现三臂完全相同的冻结源码。"
    return (f"三臂在 {', '.join(identical)} 选择了完全相同的冻结源码；此处可测读数的零差来自同一模型选择，"
            "不等价于完整的模型或信息处理等效性检验。")


# --- Post hoc diagnostics of non-evaluable proposals (no new executions) ---

FAILURE_TYPES = (
    ("BOOK_SIDE_LEVEL_SHAPE", "盘口双侧十档误当总共十档"),
    ("OTHER_TENSOR_SHAPE", "其他张量形状不匹配"),
    ("OTHER_GATE", "其他门禁失败"),
    ("RESERVED_LABEL", "保留标签协议拒绝"),
    ("OTHER_PROTOCOL", "其他格式/提案协议失败"),
    ("DUPLICATE", "源码去重"),
    ("PROMPT_CAPACITY", "下一轮上下文容量拒绝"),
    ("REQUEST_FAILURE", "请求错误或中断"),
    ("OTHER_UNEVALUATED", "其他未通过评估"),
)


def _failure_type(row: Json) -> str | None:
    """Classify a frozen unsuccessful evaluation from its outcome/error/source receipt.

    Only the aggregate label is rendered. Raw traceback and source stay in the
    original ledger, not in a log summary or new prompt.
    """
    if _valid_evaluation(row):
        return None
    outcome = row.get("outcome")
    error = str(row.get("error") or row.get("evaluation", {}).get("error") or "")
    if outcome == "GATE_REJECTED":
        if re.search(r"shape .* is invalid for input of size", error, re.IGNORECASE):
            # The observed failures reshape a two-side × ten-level book to
            # only ten elements, sometimes expressed as two × five or ten × one.
            source_lines = [line for line in row.get("source", "").splitlines() if "book_shares" in line]
            if any("reshape(" in line and re.search(r",\s*(?:10|2,\s*5|10,\s*1)\s*\)", line)
                   for line in source_lines):
                return "BOOK_SIDE_LEVEL_SHAPE"
            return "OTHER_TENSOR_SHAPE"
        if re.search(r"size mismatch|dimension .*match|shapes cannot|shape mismatch", error, re.IGNORECASE):
            return "OTHER_TENSOR_SHAPE"
        return "OTHER_GATE"
    if outcome == "INVALID_PROPOSAL":
        return "RESERVED_LABEL" if "reserved environment label" in error else "OTHER_PROTOCOL"
    if outcome == "DUPLICATE_REJECTED":
        return "DUPLICATE"
    if outcome == "PROMPT_CAPACITY_REJECTED":
        return "PROMPT_CAPACITY"
    if outcome in ("PROVIDER_ERROR", "INTERRUPTED_CALL"):
        return "REQUEST_FAILURE"
    return "OTHER_UNEVALUATED"


def _failure_audit(tracks: Mapping[tuple[str, str], list[Json]]) -> str:
    counts = {}
    for env, arm in TRACKS:
        counts[(env, arm)] = Counter(_failure_type(row) for row in _trial_rows(tracks, env, arm))
    observed = [(key, label) for key, label in FAILURE_TYPES if any(count[key] for count in counts.values())]
    rows = []
    for env, arm in TRACKS:
        count = counts[(env, arm)]
        rows.append([env, arm, len(_trial_rows(tracks, env, arm)), count[None],
                     *[count[key] for key, _ in observed]])
    audit = _table(("环境", "臂", "尝试", "有效评估", *[label for _, label in observed]), rows)
    shape_counts = {env: sum(counts[(env, arm)]["BOOK_SIDE_LEVEL_SHAPE"] for arm in ARMS) for env in ENVIRONMENTS}
    initial = {env: _trial_rows(tracks, env, ARMS[0])[0]["active_card_ids"] for env in ENVIRONMENTS}
    exposed = {env: sum("C06" in row.get("active_card_ids", [])
                       for arm in ARMS for row in _trial_rows(tracks, env, arm)) for env in ENVIRONMENTS}
    exposure_boundary = ("特别是 α 首轮没有展示 C06，所以这轮并不直接检验 M2-H3 的自激强度态规律。"
                         if "C06" not in initial["env_alpha"] else
                         "即使展示 C06，该声明对照也不等同于直接检验 M2-H3 的固定算子因果规律。")
    return "\n\n".join([
        "**事后失败分型（仅按封存回执分类，不改变任何选择、判定或结果）**", audit,
        f"盘口双侧十档形状错误：α {shape_counts['env_alpha']}/18，β {shape_counts['env_beta']}/18。"
        "该类别要求门禁回执明确报告非法 reshape，且候选源文显示直接将 `book_shares` 变形为总共十个元素。"
        "其他门禁与协议失败分开计数，避免把同一类错误反复出现误记为多个独立机制反例。",
        "**本轮实现限制。** 公共 `StepObservation.book_shares` 和 `book_ticks` 的步进形状是 `[B, 2, 10]`，"
        "序列形状是 `[B, T, 2, 10]`（两侧各十档）；冻结提示词列出了字段，却没有明确这一形状。"
        "冻结的 L0 谱系只展示结果状态及客观 Diff，没有将具体门禁错误文本交给下一轮，"
        "因而反复的同形状错误缺乏直接可用反馈。该工程瓶颈降低了闭环能实际评估的改动数，"
        "不能把这些失败认作某一结构机制的反事实证伪，也不能只归因于 Z 处理。此次保持原试验锁，不补提案或重跑。",
        f"**先验可达性边界。** 起始竞争卡：α {', '.join(initial['env_alpha'])}；β {', '.join(initial['env_beta'])}。"
        f"全轮 C06 展示次数：α {exposed['env_alpha']}/18，β {exposed['env_beta']}/18。"
        "这是仅由 D 唤醒、与 Z 无关的竞争集合，不同环境可有不同入围卡；表 2 的声明匹配率须解释为"
        "条件于实际展示集合，而不是模型有机会任意选择十张卡。"
        + exposure_boundary + "即便某臂声明匹配更多，也不能用它代替有效候选的实际结构及留出读数。",
    ])


# --- Table 4: unseen holdout effects and historical reference ---


def _table_holdout(holdout: Json, freeze: Json) -> str:
    rows = []
    for env, arm in TRACKS:
        item = _track_holdout(holdout, freeze, env, arm)
        levels = item["evaluation"].get("aggregate", {})
        pair = item["against_B0"]
        daily = item["evaluation"].get("daily", [])
        missing = sum(row.get("counterfactual_impact_mae") is None for row in daily)
        rows.append([env, arm, *[_pair(pair.get(key)) for key in METRICS],
                     _number(levels.get("exhausted_replicate_rate")), _number(levels.get("available_quote_fraction")),
                     f"{missing}/{len(daily)}"])
    return _table(("环境", "臂", *METRIC_NAMES, "最终耗尽率", "最终报价支持率", "CF 缺失日"), rows)


def _absolute_holdout(holdout: Json, freeze: Json, tracks: Mapping[tuple[str, str], list[Json]]) -> str:
    rows = []
    for env in ENVIRONMENTS:
        seed = tracks[(env, ARMS[0])][0]["source_sha256"]
        model_items = [("B0", holdout[env][seed])]
        model_items.extend((arm, _track_holdout(holdout, freeze, env, arm)) for arm in ARMS)
        for label, item in model_items:
            evaluation = item["evaluation"]
            aggregate = evaluation.get("aggregate", {})
            daily = evaluation.get("daily", [])
            support = sum(row.get("counterfactual_impact_mae") is not None for row in daily)
            rows.append([env, label, *[_number(aggregate.get(key)) for key in (*METRICS, "available_quote_fraction")],
                         f"{support}/{len(daily)}"])
    return _table(("环境", "M3 未见模型", "总 NLL", "dt NLL", "CF MAE (tick)", "耗尽率", "G_mem(64)",
                   "报价支持率", "CF 可用日（聚合种子）"), rows)


def _reference_table(path: Path) -> str:
    """M2 values are historical level comparisons, not newly paired holdout estimates."""
    if not path.exists():
        return "M2 参考文件未提供；不能计算跨阶段水平比较。"
    profiles = {"W6_thin_hawkes": ("env_alpha", "E3"), "W2_thick_long": ("env_beta", "E1")}
    models: dict[tuple[str, str], list[Json]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            profile = profiles.get(row["world"])
            if profile is not None and row["experiment"] == profile[1]:
                model = row["model"].replace("hawkes", "self-exciting-intensity")
                models.setdefault((profile[0], model), []).append(row)
    daily_path = path.with_name("daily_results.json")
    daily = _load(daily_path) if daily_path.is_file() else {}
    rows = []
    for (env, model), runs in sorted(models.items()):
        values = []
        for metric in (*METRICS, "available_quote_fraction"):
            finite = [float(row[metric]) for row in runs if row.get(metric) not in (None, "", "None", "nan")]
            finite = [value for value in finite if isfinite(value)]
            values.append(f"{_number(fmean(finite)) if finite else 'NA'} ({len(finite)}/{len(runs)} seeds)")
        days = [day for run in runs for day in daily.get(run["job_id"], [])]
        supported = sum(day.get("counterfactual_impact_mae") is not None for day in days)
        rows.append([env, model, *values, f"{supported}/{len(days)}" if days else "NA"])
    return _table(("对应环境", "M2 参考模型", "总 NLL", "dt NLL", "CF MAE (tick)", "耗尽率", "G_mem(64)",
                   "报价支持率", "CF 可用拟合日（种子×日）"), rows)


# --- Table 5: pairwise control contrasts and evidential boundaries ---


def _pair_decision(metrics: Json) -> str:
    interval = metrics.get("counterfactual_impact_mae", {}).get("interval")
    nll = metrics.get("val_nll_total", {}).get("estimate")
    exhaust = metrics.get("exhausted_replicate_rate", {}).get("estimate")
    if interval is None:
        return "INCONCLUSIVE（CF 不可定）"
    if interval[1] < 0 and nll is not None and exhaust is not None and nll <= .005 and exhaust <= .05:
        return "SUPPORTED（满足护栏）"
    if interval[0] > 0:
        return "REFUTED（CF 恶化）"
    return "INCONCLUSIVE（区间或护栏）"


def _table_conclusion(summary: Json) -> str:
    data = summary["cross_arm_effects"]
    rows = []
    for env in ENVIRONMENTS:
        for arm in ("d_only", "shuffled_z_d"):
            metrics = data[f"{env}/true_z_d-vs-{arm}"]
            rows.append([env, f"true_z_d − {arm}", *[_pair(metrics.get(key)) for key in METRICS],
                         _pair_decision(metrics)])
    return _table(("环境", "固定终局模型比较", *METRIC_NAMES, "预注册判定"), rows)


def _verify_frozen_receipts(tracks: Mapping[tuple[str, str], list[Json]], freeze: Json, holdout: Json) -> None:
    """Verify final choices are last promotions and unseen evaluations match approved hashes."""
    for env in ENVIRONMENTS:
        approved = set(freeze["approved"][env])
        actual = set(holdout[env])
        if actual != approved:
            raise ValueError("Unseen evaluation roster differs from the approved source set")
        evaluated = set()
        for arm in ARMS:
            rows = tracks[(env, arm)]
            for row in rows:
                if row.get("evaluation", {}).get("status") == "VALID":
                    digest = sha256(row["source"].encode()).hexdigest()
                    if digest != row["source_sha256"]:
                        raise ValueError("Source digest differs from the frozen evaluation receipt")
                    evaluated.add(digest)
            final = [row for row in rows if row["outcome"] in ("SEED", "PROMOTED")][-1]
            selected = freeze["trajectories"][f"{env}/{arm}"]
            if final["cand_id"] != selected["final_id"] or final["source_sha256"] != selected["final_sha256"]:
                raise ValueError("Frozen choice differs from final development promotion")
        if approved != evaluated:
            raise ValueError("Approved unseen oracle roster differs from valid development candidate hashes")


def render(artifact: Path) -> str:
    """Render only after the six trajectories and independent holdout are finalized."""
    registration = _load(artifact / "registration.json")
    baseline = _load(artifact / "baseline.json")
    freeze = _load(artifact / "freeze.json")
    holdout = _load(artifact / "holdout.json")
    study = _load(artifact / "study_summary.json")
    calls = _journal(artifact / "api_calls.jsonl")
    handshake = _load(artifact / "handshake.json")
    tracks = {(env, arm): _journal(artifact / "trajectories" / env / arm / "ledger.jsonl") for env, arm in TRACKS}
    reservations = [row for row in calls if row["event"] == "reserved"]
    completions = [row for row in calls if row["event"] == "completed"]
    if not freeze.get("approved") or any(len(rows) != 7 or rows[0]["outcome"] != "SEED" for rows in tracks.values()):
        raise ValueError("Final report requires all six registered six-round trajectories and an approved freeze")
    if len(reservations) != 36 or len({row["call_id"] for row in reservations}) != 36:
        raise ValueError("Final formal report requires exactly 36 unique registered proposal slots")
    _verify_frozen_receipts(tracks, freeze, holdout)
    control, audit = _table_control(baseline, tracks, calls)
    known_completion_ids = {row["call_id"] for row in completions}
    absent = len({row["call_id"] for row in reservations} - known_completion_ids)
    model = registration.get("model", registration.get("model_name", "NA"))
    status = study.get("status", study.get("decision", "未提供冻结判定"))
    thresholds = registration.get("promotion", {})
    known_input = sum(row["input_tokens"] for row in completions if row.get("input_tokens") is not None)
    known_output = sum(row["output_tokens"] for row in completions if row.get("output_tokens") is not None)
    unknown_usage = sum(row.get("input_tokens") is None or row.get("output_tokens") is None for row in completions) + absent
    if (study["formal_calls"], study["formal_input_tokens"], study["formal_output_tokens"], study["unknown_usage_calls"]) != (36, known_input, known_output, unknown_usage):
        raise ValueError("Summary token/accounting counts disagree with the formal call receipts")
    alignment = {arm: sum(_primary_match(row, env) for env in ENVIRONMENTS for row in _trial_rows(tracks, env, arm))
                 for arm in ARMS}
    if alignment != study["alignment_counts"]:
        raise ValueError("Summary alignment counts disagree with the preregistered declaration rule")
    protocol = registration.get("data_protocol", {})
    m2 = artifact.parent / "p2-m2-synthetic" / "experiment_results.csv"
    return "\n".join([
        "# P2-M3 · 同构结构发现对照实验", "",
        "本报告生成器只读取冻结后的数值及实验账本；生成报告本身未调用在线模型、未重新训练。",
        f"- 代码锁：Stochopia `{registration.get('stochopia_commit', 'NA')}`；OpenHyra `{registration.get('openhyra_commit', 'NA')}`。",
        f"- 数据/训练注册：训练 {protocol.get('train_days', 'NA')} 日 / 开发 {protocol.get('dev_days', 'NA')} 日 / 未见留出 {protocol.get('holdout_days', 'NA')} 日；"
        f"训练初值 {protocol.get('training_seeds', 'NA')}；模型 K={protocol.get('model_k', 'NA')} / 真值 K={protocol.get('truth_k', 'NA')}；"
        f"训练预算 `{json.dumps(protocol.get('fit', {}), ensure_ascii=False, sort_keys=True)}`。",
        f"- 单供应商模型：`{model}`；每环境三臂各六轮，正式预约 **{len(reservations)}/36** 次。完成回执 {len(completions)}；无终止回执 {absent}（已消耗预约，不补测）。",
        f"- 正式 API token：{_usage(completions)}；已知总计 {known_input + known_output}；另有 {absent} 个未返回回执的未知用量。",
        f"- 封存后独立留出模型回执：{sum(len(models) for models in holdout.values())} 个；"
        f"有效 {sum(item['evaluation'].get('status') == 'VALID' for models in holdout.values() for item in models.values())} 个（同环境同源码只评价一次）。",
        f"- 独立连通性自检：{handshake.get('status', handshake.get('outcome', '已记录'))}；"
        f"输入 {_number(handshake.get('input_tokens'))} / 输出 {_number(handshake.get('output_tokens'))} tokens（不计入正式 36 次）。",
        f"- 总体墙钟秒：{_number(study.get('wall_seconds', study.get('elapsed_seconds')))}。注册决策阈值：`{json.dumps(thresholds, ensure_ascii=False, sort_keys=True)}`。",
        "- 每个世界仅一条轨迹/臂，六轮由各自父模型递进，轮次非独立样本；三训练种子在日内聚合，按独立交易日配对重抽 2,000 次。"
        "区间描述冻结模型在该留出数据的响应，不是跨研究轨迹的独立因果显著性。", "",
        "## 表 1 · 信息同构、配额与零标签审计", "", control, "",
        f"总体合规审计：**{_boolean(audit)}**。完整首轮非 Z 消息逐字符比较，首轮父模型和竞争短卡逐值比较；"
        "预算分开报告封存的输入估计及供应商实际输入 token，不以输出 token 代替输入。", "",
        "## 表 2 · 账本一：先验短卡选择与事前方向预测", "", _table_prediction(tracks), "",
        "环境匹配主卡的预注册规则：α 为 C04/C05/C06，β 为 C03/C08 且不引用 C06；β 中引用 C06 单独计数。"
        "主卡指模型 JSON 声明的第一个被展示短卡，不代表已证明生成代码实现了该机理。解析有效但卡不在当轮展示集合中的声明不计匹配。"
        "方向支持/证伪要求 95% 配对区间位于零的一侧；无区间或跨零为不定，未完成评估为未测。"
        "同时报告全尝试分母与可判方向分母，避免只统计成功解析/训练的调用。", "", _prediction_boundary(tracks), "",
        "## 表 3 · 账本二：开发轨迹与独立留出集 Regret", "", _table_selection(tracks, freeze, study), "",
        "最终模型由开发账本封存；留出 Oracle 仅在封存后评价同轨迹有效候选及 B0。Regret 比较只能使用声明的共同有效日，"
        "缺失不能以零效应填充；Oracle 是事后选择基线，不允许反哺提案。", "",
        _regret_boundary(study), "", _same_choices(freeze), "", _failure_audit(tracks), "",
        "## 表 4 · 账本三：未见留出集配对效应", "", _table_holdout(holdout, freeze), "",
        "Δ=最终模型−固定 B0；每单元列出估计、95% 区间、共同配对日及有效 bootstrap 次数。负 CF MAE、NLL 与耗尽差是改善；"
        "G_mem 是历史依赖读数，不能把绝对增大直接视为性能改善。", "",
        "**M3 冻结模型未见水平值**（包括各环境 B0，用于与历史结果比较）：", "",
        _absolute_holdout(holdout, freeze, tracks), "",
        "**M2 冻结手工探针水平参照**（旧评估窗；不是此次 holdout 的配对实验）：", "", _reference_table(m2), "",
        "M2 按原始每个训练种子的历史水平值再均值，CF 可用拟合日为种子×日期计数；M3 先聚合重复训练、"
        "再按共同物理日统计。两项无法直接作为同日配对效应，特别是缺失 CF 较多时，不能把不同支持样本的水平差当作改进。", "",
        "## 表 5 · 信息臂结论与下一轮实盘准备边界", "", _table_conclusion(study), "",
        f"- 结构声明匹配（12 次分母/臂）：{', '.join(f'{arm}={alignment[arm]}/12' for arm in ARMS)}；严格两侧比较规则：{_boolean(study['alignment_supported'])}。",
        f"- 双环境双控制臂留出支持：{_boolean(study['holdout_supported'])}；全四组 CF 恶化判据：{'成立' if study['holdout_refuted'] else '不成立'}。",
        f"- 冻结判定规则：{registration.get('controlled_comparison_rule', 'NA')}。", "",
        f"预注册总体结论：**{status}**。只在真实 Z 同时较 D-only 和置换 Z 有区分证据时才支持可用数据先验，"
        "不得把真实 Z 臂自身涨分视作信息因果效应。未达支持判据不等于证明等效或证明无效；"
        "报价缺失须保留为不可定；不能外推到真实市场。", "",
        "此次只执行合成对照及冻结后一次复验；后续实盘研究未启动。原始提示、回复、配对结果、选择过程及 SHA256 校验见 [OpenHyra M3 封存证据](https://github.com/MrSteeeve/OpenHyra/tree/world-model/artifacts/p2-m3-isomorphic-control)。", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(args.artifact), encoding="utf-8")


if __name__ == "__main__":
    main()
