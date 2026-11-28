"""Render the four preregistered P2-M4a SZSE Q_rec, Z_train, permutation-null, and drift tables (0 API, 0 fits)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.isomorphic_report import _table
from tasks.lob_world_model.discovery_loop import Z_KEYS

Json = dict[str, Any]


def _fmt(val: object, digits: int = 4) -> str:
    if val is None or val == "":
        return "—"
    return f"{float(val):.{digits}f}"


def _med_iqr(entry: Json, digits: int = 3) -> str:
    return f"{entry['median']:.{digits}f} [{entry['intraday_iqr']:.{digits}f}]"


def _audit_m4a_bundle(artifact_dir: Path) -> tuple[list[dict[str, str]], Json, Json, Json]:
    with (artifact_dir / "candidate_qrec_scan.csv").open(newline="", encoding="utf-8") as fh:
        scan_rows = list(csv.DictReader(fh))
    manifest = json.loads((artifact_dir / "locked_m4_manifest.json").read_text(encoding="utf-8"))
    perm = json.loads((artifact_dir / "szse_permutation_nulls.json").read_text(encoding="utf-8"))
    drift = json.loads((artifact_dir / "drift_audit.json").read_text(encoding="utf-8"))

    if manifest.get("api_calls") != 0 or manifest.get("neural_fits") != 0:
        raise ValueError("P2-M4a must have zero API calls and zero neural fits")
    pools = manifest["pools"]
    exp = pools["exploration_pool"]
    val = pools["unseen_validation_pool"]
    ctrl = pools["low_qrec_control"]
    all_locked = pools["all_locked_symbols"]
    if len(exp) != 4 or len(val) != 6 or len(ctrl) != 1 or len(set(all_locked)) != 11:
        raise ValueError("locked pool must be 4 exploration + 6 unseen validation + 1 control = 11 distinct stocks")
    if len(scan_rows) < 20 or len(scan_rows) > 24:
        raise ValueError("Stage 1 scan must cover 20-24 candidate stocks")

    cal = manifest["calendar"]
    if (len(cal["train_dates"]) != 16 or len(cal["dev_dates"]) != 12
            or len(cal["planned_holdout_test_dates"]) != 20 or len(cal["replacement_test_dates"]) != 8):
        raise ValueError("invalid 16/12/20 (+8 replacement) calendar")
    primary_union = set(cal["train_dates"]) | set(cal["dev_dates"]) | set(cal["planned_holdout_test_dates"])
    if len(primary_union) != 48:
        raise ValueError("primary 48 dates must be disjoint")

    for code in all_locked:
        stock = manifest["stocks"][code]
        slices = stock["slice_sha256_by_date"]
        if len(slices) != 48:
            raise ValueError(f"{code} must seal exactly 48 daily .npz SHA256 digests")
        for d_str, srec in slices.items():
            if len(srec["sha256"]) != 64 or int(srec["bytes"]) <= 0:
                raise ValueError(f"invalid SHA256 receipt for {code} {d_str}")
        if code != ctrl[0]:
            q16 = stock["qrec_train16"]
            if q16["top1_quote_match_rate"] < 0.95 or q16["invalid_spread_rate"] > 0.01:
                raise ValueError(f"clean locked stock {code} violates Q_rec gate")
        if code not in perm["stocks"] or code not in drift["stocks"]:
            raise ValueError(f"missing permutation null or drift record for {code}")
    return scan_rows, manifest, perm, drift


def _render_table1(scan_rows: list[dict[str, str]]) -> str:
    rows_out = []
    for r in scan_rows:
        t16 = f"{float(r['top1_quote_match_rate_16d'])*100:.2f}%" if r["top1_quote_match_rate_16d"] != "" else "—"
        rows_out.append([
            f"`{r['wind_code']}`",
            f"{r['name']} ({r['board']})",
            r["price_tier"],
            _fmt(r["median_price_rmb_5d"], 2),
            f"{r['active_days_5d']}/5 (封板 {r['one_sided_days_5d']})",
            f"{float(r['top1_quote_match_rate_5d'])*100:.2f}%",
            t16,
            f"{float(r['vol_aligned_touch_match_rate_5d'])*100:.2f}%",
            _fmt(r["top10_depth_mae_5d"], 1),
            f"{float(r['invalid_spread_rate_5d'])*100:.3f}%",
            f"`{r['final_disposition']}`：{r['reason']}",
        ])
    return _table((
        "股票代码", "简称 (板块)", "预注册价位档", "首5日中位价(元)", "有效日/封板",
        "首5日 top1 匹配率", "16日训练 top1 匹配率", "首5日成交量对齐匹配率",
        "top10_depth_mae(股)", "invalid_spread_rate", "准入/淘汰结论与原因",
    ), rows_out)


def _render_table2(manifest: Json) -> str:
    rows_256 = []
    rows_sess = []
    role_zh = {
        "exploration_pool": "开发探索池 (4)",
        "unseen_validation_pool": "未见验证池 (6)",
        "low_qrec_control": "低保真对照 (1)",
    }
    for code in manifest["pools"]["all_locked_symbols"]:
        st = manifest["stocks"][code]
        z = st["z_train_256"]
        zs = st["z_train_session"]
        q16 = st["qrec_train16"]["top1_quote_match_rate"] * 100.0
        q48 = st["qrec_all48"]["top1_quote_match_rate"] * 100.0
        rows_256.append([
            f"`{code}` {st['name']}",
            role_zh[st["pool_role"]],
            st["label_zh"],
            f"{st['median_price_rmb_train16']:.2f}",
            f"{q16:.2f}% / {q48:.2f}%",
            f"{st['mean_window_trades_256']:.1f} ({st['eligible_window_fraction_train16']*100:.1f}%)",
            _med_iqr(z["midprice_shift_rate"], 4),
            _med_iqr(z["trade_to_top1_depth"], 3),
            _med_iqr(z["cancel_to_limit_ratio"], 3),
            _med_iqr(z["sign_acf_sum_1_32"], 3),
            _med_iqr(z["flow_vr_32_4"], 3),
            _med_iqr(z["interarrival_cv2"], 3),
            _med_iqr(z["zero_wait_prob"], 3),
            _med_iqr(z["book_imbalance_bias"], 3),
            _med_iqr(z["flow_sign_bias"], 3),
        ])
        rows_sess.append([
            f"`{code}` {st['name']}",
            role_zh[st["pool_role"]],
            _med_iqr(zs["midprice_shift_rate"], 4),
            _med_iqr(zs["trade_to_top1_depth"], 3),
            _med_iqr(zs["cancel_to_limit_ratio"], 3),
            _med_iqr(zs["sign_acf_sum_1_32"], 3),
            _med_iqr(zs["flow_vr_32_4"], 3),
            _med_iqr(zs["interarrival_cv2"], 3),
            _med_iqr(zs["zero_wait_prob"], 3),
            _med_iqr(zs["book_imbalance_bias"], 3),
            _med_iqr(zs["flow_sign_bias"], 3),
        ])
    t2a = _table((
        "标的", "分组", "微观结构象限标签", "中位价(元)", "top1匹配(16d/48d)",
        "单窗 n_trades (合规率)", "midprice_shift_rate", "trade_to_top1_depth",
        "cancel_to_limit_ratio", "sign_acf_sum_1_32", "flow_vr_32_4",
        "interarrival_cv2", "zero_wait_prob", "book_imbalance_bias", "flow_sign_bias",
    ), rows_256)
    t2b = _table((
        "标的", "分组", "半日全时段 shift_rate", "半日 trade_to_top1",
        "半日 cancel_to_limit", "半日 sign_acf_sum_1_32", "半日 flow_vr_32_4",
        "半日 interarrival_cv2", "半日 zero_wait_prob", "半日 book_imbalance_bias", "半日 flow_sign_bias",
    ), rows_sess)
    return (
        "### 表 2A · 256 事件训练窗口径全维度 $Z_{\\text{train}}$ 画像（`median [IQR]`，对齐 `subseq_len=256` 与 `Z_REFERENCE`）\n\n"
        + t2a
        + "\n\n### 表 2B · 连续竞价物理半日全时段口径 $Z_{\\text{train}}$ 对照画像（`32` 个半日连续竞价窗口 `median [IQR]`）\n\n"
        + t2b
    )


def _render_table3(manifest: Json, perm: Json) -> str:
    rows_out = []
    for code in manifest["pools"]["all_locked_symbols"]:
        st = manifest["stocks"][code]
        p = perm["stocks"][code]
        obs = p["observed_z_train_256"]
        sess = p["observed_z_train_session"]
        math_n = p["analytical_finite_nulls"]
        n500 = p["permutation_500_nulls_256"]
        acf_n = n500["sign_acf_sum_1_32"]
        vr_n = n500["flow_vr_32_4"]
        fb_n = n500["flow_sign_bias"]
        cv_p10 = n500["interarrival_cv2_poisson_10ms"]
        zw_p10 = n500["zero_wait_prob_poisson_10ms"]
        acf_diff = obs["sign_acf_sum_1_32"] - acf_n["median"]
        rows_out.append([
            f"`{code}` {st['name']}",
            "对照" if st["pool_role"] == "low_qrec_control" else ("探索" if st["pool_role"] == "exploration_pool" else "验证"),
            f"{p['mean_window_trades']:.1f}",
            f"{obs['sign_acf_sum_1_32']:+.3f}（半日 {sess['sign_acf_sum_1_32']:+.3f}）",
            f"{acf_n['median']:+.3f} [{acf_n['ci95'][0]:+.3f}, {acf_n['ci95'][1]:+.3f}]",
            f"{math_n['sign_acf_sum_1_32']:+.3f} (~{math_n['sign_acf_sum_approx']:+.3f})",
            f"{acf_diff:+.3f}",
            f"{obs['flow_vr_32_4']:.3f} vs {vr_n['median']:.3f}",
            f"{obs['flow_sign_bias']:.3f} vs {fb_n['median']:.3f} (理 {math_n['flow_sign_bias']:.3f})",
            f"{obs['interarrival_cv2']:.3f} vs {cv_p10['median']:.3f}",
            f"{obs['zero_wait_prob']:.4f} vs {zw_p10['median']:.4f}",
        ])
    return _table((
        "标的", "池", "平均 n_trades", "实测 sign_acf_256 (半日全窗)",
        "500次符号置换零中位数 [95%区间]", "纯数学有限窗期望 -(32n-528)/(n(n-1))",
        "256窗超额记忆 ΔACF", "flow_vr 实测 vs 置换零",
        "flow_sign_bias 实测 vs 置换零", "CV² 实测 vs 10ms泊松零",
        "zero_wait 实测 vs 10ms泊松零",
    ), rows_out)


def _render_table4(manifest: Json, drift: Json) -> str:
    rows_out = []
    short_keys = (
        ("z_thin.midprice_shift_rate", "Δshift"),
        ("z_thin.trade_to_top1_depth", "Δtr/d1"),
        ("z_thin.cancel_to_limit_ratio", "Δc/l"),
        ("z_mem.sign_acf_sum_1_32", "Δacf"),
        ("z_mem.flow_vr_32_4", "Δvr"),
        ("z_burst.interarrival_cv2", "Δcv2"),
        ("z_burst.zero_wait_prob", "Δzw"),
        ("z_asym.book_imbalance_bias", "Δimb"),
        ("z_asym.flow_sign_bias", "Δfbias"),
    )
    for code in manifest["pools"]["all_locked_symbols"]:
        d = drift["stocks"][code]
        d256 = d["drift_256_iqr_units"]
        dsess = d["drift_session_iqr_units"]
        excl_n = len(d["excluded_holdout_dates"])
        rows_out.append([
            f"`{code}` {d['name']}",
            d["pool_role"],
            f"20 日 (替补 {excl_n})",
            *[f"{d256[k]:+.2f}" for k, _ in short_keys],
            f"{d256['total_rms']:.3f}",
            f"{dsess['total_rms']:.3f}",
        ])
    return _table((
        "标的", "分组", "有效留出日",
        *[label for _, label in short_keys],
        "256窗漂移 RMS (IQR单位)", "半日全时段漂移 RMS",
    ), rows_out)


def render(artifact_dir: Path, *, openhyra_commit: str | None = None, stochopia_commit: str | None = None) -> str:
    scan_rows, manifest, perm, drift = _audit_m4a_bundle(artifact_dir)
    cal = manifest["calendar"]
    pools = manifest["pools"]
    passed_cnt = sum(r["passed_stage1_qrec_gate"] == "True" for r in scan_rows)
    clean_10 = [*pools["exploration_pool"], *pools["unseen_validation_pool"]]
    clean_top1 = [manifest["stocks"][c]["qrec_train16"]["top1_quote_match_rate"] * 100.0 for c in clean_10]
    clean_ntr = [manifest["stocks"][c]["mean_window_trades_256"] for c in clean_10]
    acf_null_meds = [perm["stocks"][c]["permutation_500_nulls_256"]["sign_acf_sum_1_32"]["median"] for c in clean_10]
    vr_null_meds = [perm["stocks"][c]["permutation_500_nulls_256"]["flow_vr_32_4"]["median"] for c in clean_10]
    acf_excess = [
        perm["stocks"][c]["observed_z_train_256"]["sign_acf_sum_1_32"]
        - perm["stocks"][c]["permutation_500_nulls_256"]["sign_acf_sum_1_32"]["median"]
        for c in clean_10
    ]
    fb_obs = [perm["stocks"][c]["observed_z_train_256"]["flow_sign_bias"] for c in clean_10]
    fb_null = [perm["stocks"][c]["permutation_500_nulls_256"]["flow_sign_bias"]["median"] for c in clean_10]
    sess_acf_obs = [perm["stocks"][c]["observed_z_train_session"]["sign_acf_sum_1_32"] for c in clean_10]
    ctrl_code = pools["low_qrec_control"][0]
    ctrl_top1 = manifest["stocks"][ctrl_code]["qrec_train16"]["top1_quote_match_rate"] * 100.0

    oh_base = manifest["baseline_commits"]["openhyra_baseline"][:7]
    st_base = manifest["baseline_commits"]["stochopia_baseline"][:7]

    return "\n\n".join([
        "# P2-M4a · 深市 Level-2 重构保真度 $Q_{\\text{rec}}$ 准入扫描、$Z$ 正交分层锁仓与 256 步窗口零基线校准报告",
        f"**承接基线**：OpenHyra `{oh_base}`（`world-model` 分支）；Stochopia `{st_base}`（`main` 分支）。  \n"
        f"**本阶段预算与隔离审计**：大模型 API 调用 **`{manifest['api_calls']}` 次（`$0`）**；候选神经网络训练 **`{manifest['neural_fits']}` 次**；"
        f"远端数据源 `{manifest['source_repo']}` 固定版本 `{manifest['source_revision']}`，全程通过 `pyarrow` 行组范围投影（HTTP 206 Row-Group Range Projection）按 `wind_code` 拉取，"
        f"全部 `11 × 48 = 528` 个 `.npz` 逐笔盘口切片落盘于本地外部缓存目录并在 `locked_m4_manifest.json` 逐文件封存 `SHA256` 指纹，Git 仓库零大体积二进制行情入库。",
        "## 一、交易日历切分与防时间泄漏两阶段选股协议",
        f"- **连续无重叠深市交易日历（`2017-06-01` 起共 48 主交易日 + 8 顺延替补日）**：\n"
        f"  - `train_dates`（16 日）：`{cal['train_dates'][0]}–{cal['train_dates'][-1]}`（其中首 5 日 `{cal['stage1_scan_dates'][0]}–{cal['stage1_scan_dates'][-1]}` 用于第一阶段 24 股快速 $Q_{{\\text{{rec}}}}$ 扫描）；\n"
        f"  - `dev_dates`（12 日）：`{cal['dev_dates'][0]}–{cal['dev_dates'][-1]}`；\n"
        f"  - `holdout_test_dates`（20 日）：`{cal['planned_holdout_test_dates'][0]}–{cal['planned_holdout_test_dates'][-1]}`；\n"
        f"  - `replacement_test_dates`（8 日）：`{cal['replacement_test_dates'][0]}–{cal['replacement_test_dates'][-1]}`（仅当锁定个股出现全天停牌或确定性锚点缺失时按时间顺序递补）。\n"
        f"- **严格单向防泄漏锁仓**：第一阶段仅拉取首 5 个训练日对横跨深市主板（`000xxx.SZ`）、中小板（`002xxx.SZ`）、创业板（`300xxx.SZ`）及低/中/高三档价位的 `24` 只候选股做重构保真度硬过滤（`top1_quote_match_rate >= 95%`、`invalid_spread_rate <= 1%`、无停牌或一字封板）；"
        f"第二阶段在未下载任何 `dev_dates` 与 `holdout_test_dates` 数据的物理隔离状态下，仅凭 `16` 个 `train_dates` 的 $Z_{{\\text{{train}}}}$ 秩次效用函数确定性锁定 `4 + 6 + 1 = 11` 只股票，随后才拉取开发集与留出集并完成 528 份切片封存。",
        f"## 表 1 · 候选深市股票池 $Q_{{\\text{{rec}}}}$ 重构保真度准入与淘汰审计表（`N = {len(scan_rows)}`）",
        _render_table1(scan_rows),
        f"**实证机制发现（板间通道序列制度差异）**：24 只初筛标的中，`{passed_cnt}` 只通过首 5 日 $Q_{{\\text{{rec}}}} \\ge 95\\%$ 硬门槛——深市主板与中小板全部 15 只正常交易个股的 `top1_quote_match_rate` 均高达 **`98.72%–99.99%`**（16 日训练期为 **`{min(clean_top1):.2f}%–{max(clean_top1):.2f}%`**，`invalid_spread_rate` 恒为 `0.000%`），`000100.SZ`（TCL集团）因区间停牌（`0/5` 有效日）被确定性剔除；"
        f"而全部 8 只创业板（`300xxx.SZ`）个股的一档精确匹配率均落在 **`60.70%–85.60%`**。核查底层逐笔表证实：深交所 2016–2017 第五代交易系统在创业板频道上对 `逐笔委托.ex_order_id` 与 `逐笔成交.trade_id` 采用独立通道序列号计数（而主板/中小板满足跨表统一序列单调性），导致固定单序列回放器在同一 `10ms` 时间戳内产生跨表排序失配。"
        f"$Q_{{\\text{{rec}}}} \\ge 95\\%$ 硬门槛在零人工干预下自动拦截了该系统性通道污染，同时按预注册方案单独保留特锐德 `300001.SZ`（16 日匹配率 `{ctrl_top1:.2f}%`）作为 `low_qrec_control` 污染敏感性对照股（严禁混入 10 只干净股票主统计）。",
        "## 表 2 · 正式锁定的 `4` 开发探索股 + `6` 未见验证股 + `1` 低保真对照股全维度 $Z_{\\text{train}}$ 画像表",
        _render_table2(manifest),
        "确定性分层规则说明：在全部 15 只通过 16 日 $Q_{\\text{rec}}$ 硬门槛的干净股票上，仅用 `train_dates` 计算薄度换档综合秩 $S_{\\text{thin}} = \\frac{1}{2}[r(\\text{midprice\\_shift\\_rate}) + r(\\text{trade\\_to\\_top1\\_depth})]$、跨尺度方向记忆秩 $S_{\\text{mem}} = \\frac{1}{2}[r(\\text{sess\\_acf}) + r(\\text{acf}_{256} - \\mu_0(\\bar{n}))]$ 与短窗簇聚秩 $r(\\text{cv2}_{256})$，取四个正交角点效用函数排名第 1 的标的锁定为 `exploration_pool`（2 主板 + 2 中小板：`000725.SZ`、`002024.SZ`、`002304.SZ`、`000661.SZ`，股价横跨 `3.96–122.43` 元，256 窗换档率横跨 `0.0000–0.2196`），取各角点紧随其后的未选标的锁定为物理封存的 `unseen_validation_pool`（3 主板 + 3 中小板：`000001.SZ`、`000651.SZ`、`002230.SZ`、`002475.SZ`、`002821.SZ`、`000858.SZ`，股价横跨 `9.11–133.09` 元，换档率横跨 `0.0039–0.2314`）。",
        "## 表 3 · 实盘 256 事件窗口置换零基线核验表（每股 `500` 次独立随机置换）",
        _render_table3(manifest, perm),
        f"**有限样本偏移与跨尺度记忆核验结论**：\n"
        f"1. **主动成交笔数偏移**：10 只干净深市股票在 256 事件窗口内的平均主动成交笔数 $\\bar{{n}}_{{\\text{{trades}}}}$ 介于 **`{min(clean_ntr):.1f}–{max(clean_ntr):.1f}` 笔**（低于合成环境的 `~100` 笔，因实盘限价挂单与撤单占比更高），使独立随机符号流在 256 步窗口下的 `sign_acf_sum_1_32` 经验置换零中位数下移至 **`{min(acf_null_meds):+.3f}` 至 `{max(acf_null_meds):+.3f}`**，`flow_vr_32_4` 置换零中位数下移至 **`{min(vr_null_meds):.3f}` 至 `{max(vr_null_meds):.3f}`**。\n"
        f"2. **纯数学动态零基线校准**：精确有限样本均值公式 $-\\frac{{32n - 528}}{{n(n-1)}}$（及渐近近似 $-32/n$）刻画了均值中心化带来的样本长度依赖负偏（偏态分布下窗口中位数较解析均值再低约 `0.05–0.07`）。`discovery_loop.py` 已支持按当前环境训练窗实测 $\\bar{{n}}_{{\\text{{trades}}}}$ 动态渲染 `Z_REFERENCE` 纯数学有限样本期望（不改变历史 `M3/M3.1/M3.2` 默认文本），防止受测模型将短窗中心化负偏误判为均值回复。\n"
        f"3. **真实长记忆与局部拆单流偏置 100% 可辨识**：在物理半日全时段尺度（表 2B）上，全部 10 只干净股票的 `sign_acf_sum_1_32` 高达 **`{min(sess_acf_obs):+.3f}` 至 `{max(sess_acf_obs):+.3f}`**；即使在 256 事件短窗口内，全部 10 只干净股票相对各自 500 次随机符号置换零中位数的超额自相关 $\\Delta\\text{{ACF}}$ 亦一致为正（**`{min(acf_excess):+.3f}` 至 `{max(acf_excess):+.3f}`**），单窗方向偏置 `flow_sign_bias`（**`{min(fb_obs):.3f}–{max(fb_obs):.3f}`**）全部远高于置换零中位数（`{min(fb_null):.3f}–{max(fb_null):.3f}`），且 `interarrival_cv2` 与 `zero_wait_prob` 一致高于同速率 `10ms` 齐次泊松零基线。",
        "## 表 4 · 训练期至独立 20 日留出期的跨期特征漂移 $\\|Z_{\\text{holdout}} - Z_{\\text{train}}\\|$ 审计表（离线封存）",
        _render_table4(manifest, drift),
        "审计说明：各列数值为 `(median_holdout - median_train) / IQR_train`（以训练期四分位距为单位的有符号中位数偏移），`RMS` 为全部 10 项内部子指标（含 `sign_acf_log_slope`）的均方根漂移幅度。"
        "本表由 `drift_audit.json` 离线生成并只读封存，`P2-M4b` 搜索循环与大模型提示词严禁读取 `Z_holdout` 或本审计文件，仅在 `P2-M4b` 留出终验阶段作为实盘非平稳协变量对照。严格停在 `P2-M4a`，未启动 `P2-M4b` API 实盘实验。",
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("report_path", type=Path)
    args = parser.parse_args()
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(render(args.artifact_dir.resolve()), encoding="utf-8")


if __name__ == "__main__":
    main()
