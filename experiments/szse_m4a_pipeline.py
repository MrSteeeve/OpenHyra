"""Reproducible P2-M4a SZSE Level-2 Q_rec scan, train-only Z stratification, 48-day seal, and 256-step permutation null check (0 API calls, 0 fits)."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import importlib
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from stochopia.common import atomic_write_json
from stochopia.data import SZSEDaySlice, load_day_slice, reconstruct_szse_day, save_day_slice
from stochopia.market.windows import CHILDREN, calibrate_market, load_market_days, select_windows
from stochopia.measure.data_properties import compute_property_drift, measure_data_properties
from stochopia.neural.lob_training import encode_day
from tasks.lob_world_model.discovery_loop import Z_KEYS, finite_window_nulls
from tasks.lob_world_model.holdout_data import REGISTERED_M4_HOLDOUT_SCHEMA

Json = dict[str, Any]
SOURCE_REPO = "venvoo/china-a-share-l2-level2-limit-order-book-tick-data"
SOURCE_REVISION = "1faacb93f0c07e72f49659c6879008f96782c238"
PERMUTATION_SEED = 20261128
N_PERMUTATIONS = 500

# Contiguous SZSE calendar starting 2017-06-01 (16 train + 12 dev + 20 holdout + 8 replacement = 56 days)
TRAIN_DATES: tuple[int, ...] = (
    20170601, 20170602, 20170605, 20170606, 20170607, 20170608, 20170609, 20170612,
    20170613, 20170614, 20170615, 20170616, 20170619, 20170620, 20170621, 20170622,
)
DEV_DATES: tuple[int, ...] = (
    20170623, 20170626, 20170627, 20170628, 20170629, 20170630,
    20170703, 20170704, 20170705, 20170706, 20170707, 20170710,
)
HOLDOUT_TEST_DATES: tuple[int, ...] = (
    20170711, 20170712, 20170713, 20170714, 20170717, 20170718, 20170719, 20170720, 20170721, 20170724,
    20170725, 20170726, 20170727, 20170728, 20170731, 20170801, 20170802, 20170803, 20170804, 20170807,
)
REPLACEMENT_TEST_DATES: tuple[int, ...] = (
    20170808, 20170809, 20170810, 20170811, 20170814, 20170815, 20170816, 20170817,
)
STAGE1_SCAN_DATES: tuple[int, ...] = TRAIN_DATES[:5]
LOW_QREC_CONTROL = "300001.SZ"

# Ex-ante 24-stock candidate roster (8 Main Board + 8 SME + 8 ChiNext across 3 price tiers)
CANDIDATE_ROSTER: tuple[tuple[str, str, str, str], ...] = (
    ("000001.SZ", "平安银行", "深市主板", "low_3_15"),
    ("000725.SZ", "京东方A", "深市主板", "low_3_15"),
    ("000100.SZ", "TCL集团", "深市主板", "low_3_15"),
    ("000333.SZ", "美的集团", "深市主板", "mid_20_60"),
    ("000651.SZ", "格力电器", "深市主板", "mid_20_60"),
    ("000858.SZ", "五粮液", "深市主板", "mid_20_60"),
    ("000538.SZ", "云南白药", "深市主板", "high_70_200"),
    ("000661.SZ", "长春高新", "深市主板", "high_70_200"),
    ("002024.SZ", "苏宁云商", "中小板", "low_3_15"),
    ("002027.SZ", "分众传媒", "中小板", "low_3_15"),
    ("002415.SZ", "海康威视", "中小板", "mid_20_60"),
    ("002475.SZ", "立讯精密", "中小板", "mid_20_60"),
    ("002230.SZ", "科大讯飞", "中小板", "mid_20_60"),
    ("002594.SZ", "比亚迪", "中小板", "mid_20_60"),
    ("002304.SZ", "洋河股份", "中小板", "high_70_200"),
    ("002821.SZ", "凯莱英", "中小板", "high_70_200"),
    ("300001.SZ", "特锐德", "创业板", "low_3_15"),
    ("300059.SZ", "东方财富", "创业板", "low_3_15"),
    ("300017.SZ", "网宿科技", "创业板", "low_3_15"),
    ("300015.SZ", "爱尔眼科", "创业板", "mid_20_60"),
    ("300124.SZ", "汇川技术", "创业板", "mid_20_60"),
    ("300136.SZ", "信维通信", "创业板", "mid_20_60"),
    ("300618.SZ", "寒锐钴业", "创业板", "high_70_200"),
    ("300628.SZ", "亿联网络", "创业板", "high_70_200"),
)


def _git_head(repo_dir: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_fetcher(market_root: Path):
    sys.path.insert(0, str(market_root))
    try:
        return importlib.import_module("fetch_fast_m2")
    finally:
        sys.path.pop(0)


def _flatten_z(prop: Json) -> dict[str, dict[str, float]]:
    flat = {k: v for dim in prop.values() for k, v in dim.items()}
    return {k: flat[k] for k in Z_KEYS}


def slice_windows_256(days: list[SZSEDaySlice]) -> tuple[list[dict[str, Any]], list[int]]:
    windows: list[dict[str, Any]] = []
    trade_counts: list[int] = []
    for day in days:
        for s0, s1 in day.session_bounds:
            for w0 in range(s0, s1 - 255, 256):
                act = day.action_type[w0:w0 + 256]
                ntr = int(np.sum((act == 2) | (act == 3)))
                trade_counts.append(ntr)
                if ntr >= 33:
                    windows.append({
                        "action_type": act,
                        "dt_ms": day.dt_ms[w0:w0 + 256],
                        "rel_ticks": day.rel_ticks[w0:w0 + 256],
                        "volume_shares": day.volume_shares[w0:w0 + 256],
                        "pre_book_ticks": day.pre_book_ticks[w0:w0 + 256],
                        "pre_book_shares": day.pre_book_shares[w0:w0 + 256],
                        "session_bounds": ((0, 256),),
                    })
    return windows, trade_counts


def _scan_days_qrec(raw_root: Path, code: str, dates: tuple[int, ...]) -> tuple[list[SZSEDaySlice], Json]:
    snaps = vol_snaps = active_days = one_sided_days = 0
    match_sum = vol_match_sum = depth_mae_sum = log_mae_sum = inv_sum = 0.0
    mids: list[float] = []
    events_list: list[int] = []
    slices: list[SZSEDaySlice] = []
    for d in dates:
        day_dir = raw_root / str(d)
        q = pq.read_table(
            day_dir / "行情.parquet",
            columns=["wind_code", "time", "bid_px1", "ask_px1"],
            filters=[("wind_code", "==", code)],
        )
        if len(q) == 0:
            continue
        t = q["time"].to_numpy()
        in_sess = ((t >= 93000000) & (t < 113000000)) | ((t >= 130000000) & (t < 145700000))
        b1 = q["bid_px1"].fill_null(0).to_numpy()[in_sess]
        a1 = q["ask_px1"].fill_null(0).to_numpy()[in_sess]
        one_sided_frac = float(np.mean((b1 <= 0) | (a1 <= 0))) if len(b1) else 1.0
        if one_sided_frac > 0.20:
            one_sided_days += 1
        sl, rep = reconstruct_szse_day(day_dir, code)
        if rep is None or rep.n_snapshots == 0 or any(stop <= start for start, stop in sl.session_bounds):
            continue
        active_days += 1
        slices.append(sl)
        n_s = rep.n_snapshots
        snaps += n_s
        match_sum += rep.top1_quote_match_rate * n_s
        depth_mae_sum += rep.top10_depth_mae * n_s
        log_mae_sum += rep.top10_log_vol_mae * n_s
        inv_sum += rep.invalid_spread_rate * n_s
        if rep.n_volume_aligned_snapshots > 0:
            vol_snaps += rep.n_volume_aligned_snapshots
            vol_match_sum += rep.vol_aligned_touch_match_rate * rep.n_volume_aligned_snapshots
        mids.append(float(np.median(sl.mid_ticks_x2) / 200.0))
        events_list.append(len(sl.seq_id))
    top1 = match_sum / snaps if snaps else 0.0
    vol_m = vol_match_sum / vol_snaps if vol_snaps else 0.0
    d_mae = depth_mae_sum / snaps if snaps else 0.0
    l_mae = log_mae_sum / snaps if snaps else 0.0
    inv_r = inv_sum / snaps if snaps else 1.0
    med_px = float(np.median(mids)) if mids else 0.0
    return slices, {
        "active_days": active_days,
        "total_days": len(dates),
        "one_sided_days": one_sided_days,
        "n_snapshots": snaps,
        "n_vol_aligned_snapshots": vol_snaps,
        "median_price_rmb": round(med_px, 4),
        "top1_quote_match_rate": round(top1, 6),
        "vol_aligned_touch_match_rate": round(vol_m, 6),
        "top10_depth_mae": round(d_mae, 4),
        "top10_log_vol_mae": round(l_mae, 6),
        "invalid_spread_rate": round(inv_r, 6),
        "mean_daily_events": round(float(np.mean(events_list)), 1) if events_list else 0.0,
    }


def _ranks(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    out = np.empty_like(arr)
    out[order] = np.linspace(0.0, 1.0, len(arr)) if len(arr) > 1 else 0.5
    return out


def select_stratified_pool(train16_profiles: dict[str, Json]) -> Json:
    """Deterministically select 4 exploration + 6 unseen validation + 1 low-Q_rec control using ONLY train_dates Z."""
    clean_codes = sorted(
        code for code, row in train16_profiles.items()
        if row["eligible_16d"] and code != LOW_QREC_CONTROL
    )
    if len(clean_codes) < 10:
        raise ValueError(f"insufficient clean stocks passing 16-day Q_rec gate: {len(clean_codes)}")

    r_shift = _ranks([train16_profiles[c]["z_train_256_flat"]["midprice_shift_rate"]["median"] for c in clean_codes])
    r_trd1 = _ranks([train16_profiles[c]["z_train_256_flat"]["trade_to_top1_depth"]["median"] for c in clean_codes])
    s_thin = 0.5 * (r_shift + r_trd1)

    r_sess_acf = _ranks([train16_profiles[c]["z_train_session_flat"]["sign_acf_sum_1_32"]["median"] for c in clean_codes])
    r_exc_acf256 = _ranks([
        train16_profiles[c]["z_train_256_flat"]["sign_acf_sum_1_32"]["median"]
        - finite_window_nulls(train16_profiles[c]["mean_window_trades"])["sign_acf_sum_1_32"]
        for c in clean_codes
    ])
    s_mem = 0.5 * (r_sess_acf + r_exc_acf256)
    r_cv2_256 = _ranks([train16_profiles[c]["z_train_256_flat"]["interarrival_cv2"]["median"] for c in clean_codes])

    idx = {c: i for i, c in enumerate(clean_codes)}
    u_q1 = (1.0 - s_thin) + r_sess_acf
    u_q2 = (1.0 - s_thin) + 0.5 * (1.0 - s_mem) + 0.5 * (1.0 - r_cv2_256)
    u_q3 = s_thin + r_cv2_256 - 0.3 * s_mem
    u_q4 = s_thin + r_exc_acf256
    u_low_burst = (1.0 - r_cv2_256) + 0.5 * (1.0 - s_mem)

    chosen: set[str] = set()

    def pick_top(scores: np.ndarray, k: int = 1) -> list[str]:
        ordered = [clean_codes[i] for i in np.argsort(-scores, kind="mergesort") if clean_codes[i] not in chosen]
        picked = ordered[:k]
        chosen.update(picked)
        return picked

    # Step A: Lock 4 exploration corners (1 per physical quadrant)
    exp_q1 = pick_top(u_q1, 1)[0]
    exp_q2 = pick_top(u_q2, 1)[0]
    exp_q3 = pick_top(u_q3, 1)[0]
    exp_q4 = pick_top(u_q4, 1)[0]
    exploration = [
        (exp_q1, "Q1_thick_locked_long_memory", "低换档锁价厚盘 + 长方向记忆"),
        (exp_q2, "Q2_thick_low_memory_baseline", "低换档厚盘 + 低记忆近泊松基准"),
        (exp_q3, "Q3_thin_active_high_clustering", "高换档活跃薄盘 + 强时间簇聚"),
        (exp_q4, "Q4_thin_high_shift_short_persistence", "高换档极薄盘 + 强短窗方向持续"),
    ]

    # Step B: Lock 6 unseen validation peers spanning the same 4 quadrants (2 Q1 + 1 Q2/low-burst + 2 Q3 + 1 Q4)
    val_q1 = pick_top(u_q1, 2)
    val_q2 = pick_top(u_low_burst, 1)
    val_q3 = pick_top(u_q3, 2)
    val_q4 = pick_top(u_q4, 1)
    validation = [
        (val_q1[0], "Q1_thick_locked_long_memory", "低换档厚盘 + 长方向记忆（未见验证）"),
        (val_q1[1], "Q1_thick_mid_short_persistence", "中低换档厚盘 + 强短窗记忆与同刻簇聚（未见验证）"),
        (val_q2[0], "Q2_mid_low_clustering_baseline", "中换档 + 最低时间簇聚近泊松基准（未见验证）"),
        (val_q3[0], "Q3_thin_high_clustering_low_mem", "中高换档薄盘 + 高簇聚低记忆对照（未见验证）"),
        (val_q3[1], "Q3_ultra_thin_high_shift_low_mem", "最高换档极薄盘 + 低长记忆反事实对照（未见验证）"),
        (val_q4[0], "Q4_mid_high_shift_short_persistence", "中高换档薄盘 + 强短窗方向持续（未见验证）"),
    ]

    scores_dump = {
        c: {
            "s_thin": round(float(s_thin[idx[c]]), 4),
            "s_mem": round(float(s_mem[idx[c]]), 4),
            "r_sess_acf": round(float(r_sess_acf[idx[c]]), 4),
            "r_exc_acf256": round(float(r_exc_acf256[idx[c]]), 4),
            "r_cv2_256": round(float(r_cv2_256[idx[c]]), 4),
            "u_q1": round(float(u_q1[idx[c]]), 4),
            "u_q2": round(float(u_q2[idx[c]]), 4),
            "u_q3": round(float(u_q3[idx[c]]), 4),
            "u_q4": round(float(u_q4[idx[c]]), 4),
        }
        for c in clean_codes
    }
    return {
        "exploration_pool": [{"wind_code": c, "quadrant": q, "label_zh": lbl} for c, q, lbl in exploration],
        "unseen_validation_pool": [{"wind_code": c, "quadrant": q, "label_zh": lbl} for c, q, lbl in validation],
        "low_qrec_control": [{
            "wind_code": LOW_QREC_CONTROL,
            "quadrant": "CONTROL_low_qrec_contaminated",
            "label_zh": "低重构保真度敏感性对照（创业板通道失配 ~83.7%）",
        }],
        "clean_rank_scores": scores_dump,
    }


def _reconstruct_and_check_day(
    raw_dir: Path, cache_dir: Path, code: str, date: int, horizon_ms: int | None,
) -> tuple[SZSEDaySlice, Json, list[Json]]:
    try:
        day, report = reconstruct_szse_day(raw_dir / str(date), code)
    except ValueError as exc:
        if str(exc).startswith(f"No quotes for {code} "):
            raise RuntimeError(f"no_quotes:{code}") from exc
        raise
    if (report is None or report.n_snapshots == 0 or day.wind_code != code
            or day.trading_date != date or len(day.session_bounds) != 2
            or any(stop <= start for start, stop in day.session_bounds)):
        raise RuntimeError(f"incomplete_day:{code}")
    anchors: list[Json] = []
    if horizon_ms is not None:
        try:
            windows = select_windows([encode_day(day)], horizon_ms)
        except ValueError as exc:
            if str(exc).startswith(f"Incomplete dev window for {code} {date} "):
                raise RuntimeError(f"incomplete_window:{code}") from exc
            raise
        if len(windows) != 2:
            raise RuntimeError(f"incomplete_window:{code}")
        for w in windows:
            bid, ask = map(int, day.pre_book_ticks[w.start, :, 0])
            bid_sh, ask_sh = map(int, day.pre_book_shares[w.start, :, 0])
            if not (0 < bid < ask and bid_sh > 0 and ask_sh > 0):
                raise RuntimeError(f"invalid_anchor:{code}:session-{w.session}")
            anchors.append({
                "symbol": code, "date": date, "session": w.session,
                "seq_id": int(day.seq_id[w.start]), "spread_ticks": ask - bid,
            })
    npz_path = cache_dir / str(date) / f"{code}.npz"
    save_day_slice(npz_path, day)
    readings = {k: None if isinstance(v, float) and not isfinite(v) else v for k, v in asdict(report).items()}
    rec = {
        "date": date, "symbol": code, "n_events": len(day.seq_id),
        "npz_sha256": _sha256_file(npz_path), "npz_bytes": npz_path.stat().st_size,
        **readings,
    }
    return day, rec, anchors


def run_permutation_nulls(
    days_by_code: dict[str, list[SZSEDaySlice]], locked_codes: list[str], seed: int = PERMUTATION_SEED,
) -> Json:
    rng = np.random.default_rng(seed)
    results: dict[str, Json] = {}
    for code in locked_codes:
        days = days_by_code[code]
        wins256, ntr_all = slice_windows_256(days)
        obs_z256 = _flatten_z(measure_data_properties(wins256))
        obs_zsess = _flatten_z(measure_data_properties(days))
        ntr_elig = [int(np.sum((w["action_type"] == 2) | (w["action_type"] == 3))) for w in wins256]
        mean_ntr = float(np.mean(ntr_elig))
        math_nulls = finite_window_nulls(mean_ntr)

        # Global aggressive sign pool across the 16 training days of this stock
        all_signs_list = []
        for d in days:
            agg = (d.action_type == 2) | (d.action_type == 3)
            all_signs_list.append(1 - 2 * (d.action_type[agg] % 2).astype(np.int8))
        sign_pool = np.concatenate(all_signs_list)
        total_agg = len(sign_pool)

        # Pre-generate 500 global permutations of the stock's training sign stream
        # Use independent index shuffles over the eligible windows' concatenated trade positions
        total_win_trades = sum(ntr_elig)
        perm_signs_bank = np.empty((N_PERMUTATIONS, total_win_trades), dtype=np.float64)
        for b in range(N_PERMUTATIONS):
            perm = rng.permutation(total_agg)[:total_win_trades]
            perm_signs_bank[b] = sign_pool[perm]

        n_win = len(wins256)
        acf_mat = np.empty((N_PERMUTATIONS, n_win), dtype=np.float64)
        vr_mat = np.empty((N_PERMUTATIONS, n_win), dtype=np.float64)
        fbias_mat = np.empty((N_PERMUTATIONS, n_win), dtype=np.float64)
        cv2_pois10_mat = np.empty((N_PERMUTATIONS, n_win), dtype=np.float64)
        zw_pois10_mat = np.empty((N_PERMUTATIONS, n_win), dtype=np.float64)
        cv2_exp_mat = np.empty((N_PERMUTATIONS, n_win), dtype=np.float64)

        # Session wait pools for empirical wait-shuffling null
        sess_waits = [d.dt_ms[s0:s1].astype(np.float64) for d in days for s0, s1 in d.session_bounds if s1 - s0 >= 256]
        all_waits = np.concatenate(sess_waits)
        cv2_shuf_mat = np.empty((N_PERMUTATIONS, n_win), dtype=np.float64)
        zw_shuf_mat = np.empty((N_PERMUTATIONS, n_win), dtype=np.float64)

        offset = 0
        for w_idx, w in enumerate(wins256):
            nw = ntr_elig[w_idx]
            s_block = perm_signs_bank[:, offset:offset + nw]
            offset += nw
            centered = s_block - s_block.mean(axis=1, keepdims=True)
            variance = np.einsum("bi,bi->b", centered, centered)
            cov_sum = np.zeros(N_PERMUTATIONS, dtype=np.float64)
            for lag in range(1, 33):
                cov_sum += np.einsum("bi,bi->b", centered[:, lag:], centered[:, :-lag])
            acf_mat[:, w_idx] = np.where(variance > 1e-8, cov_sum / np.maximum(variance, 1e-8), 0.0)

            sums32 = np.lib.stride_tricks.sliding_window_view(s_block, 32, axis=1).sum(axis=2)
            sums4 = np.lib.stride_tricks.sliding_window_view(s_block, 4, axis=1).sum(axis=2)
            v32 = np.var(sums32, axis=1) / 32.0
            v4 = np.var(sums4, axis=1) / 4.0
            vr_mat[:, w_idx] = v32 / np.maximum(v4, 1e-8)
            fbias_mat[:, w_idx] = np.abs(s_block.mean(axis=1))

            mean_dt = max(float(w["dt_ms"].mean()), 1.0)
            exp_waits = rng.exponential(mean_dt, size=(N_PERMUTATIONS, 256))
            cv2_exp_mat[:, w_idx] = np.var(exp_waits, axis=1) / (np.mean(exp_waits, axis=1) + 1e-9) ** 2

            # Homogeneous Poisson process observed on SZSE's 10ms clock grid
            t_cum = np.cumsum(exp_waits, axis=1)
            t_grid = np.floor(t_cum / 10.0) * 10.0
            dt_10ms = np.diff(t_grid, axis=1, prepend=0.0)
            cv2_pois10_mat[:, w_idx] = np.var(dt_10ms, axis=1) / (np.mean(dt_10ms, axis=1) + 1e-9) ** 2
            zw_pois10_mat[:, w_idx] = np.mean(dt_10ms == 0.0, axis=1)

        # Global wait permutation across training sessions (500 draws)
        need_waits = n_win * 256
        for b in range(N_PERMUTATIONS):
            idx_w = rng.integers(0, len(all_waits), size=(n_win, 256))
            w_shuf = all_waits[idx_w]
            cv2_shuf_mat[b] = np.var(w_shuf, axis=1) / (np.mean(w_shuf, axis=1) + 1e-9) ** 2
            zw_shuf_mat[b] = np.mean(w_shuf == 0.0, axis=1)

        def _null_stats(mat: np.ndarray) -> Json:
            win_medians = np.median(mat, axis=1)
            q025, q50, q975 = np.quantile(win_medians, (0.025, 0.5, 0.975))
            return {
                "median": round(float(q50), 6),
                "ci95": [round(float(q025), 6), round(float(q975), 6)],
            }

        results[code] = {
            "n_windows_256_eligible": n_win,
            "n_windows_256_total": len(ntr_all),
            "eligible_window_fraction": round(n_win / max(len(ntr_all), 1), 6),
            "mean_window_trades": round(mean_ntr, 2),
            "median_window_trades": round(float(np.median(ntr_elig)), 2),
            "observed_z_train_256": {k: round(obs_z256[k]["median"], 6) for k in Z_KEYS},
            "observed_z_train_session": {k: round(obs_zsess[k]["median"], 6) for k in Z_KEYS},
            "analytical_finite_nulls": {k: round(v, 6) for k, v in math_nulls.items()},
            "permutation_500_nulls_256": {
                "sign_acf_sum_1_32": _null_stats(acf_mat),
                "flow_vr_32_4": _null_stats(vr_mat),
                "flow_sign_bias": _null_stats(fbias_mat),
                "interarrival_cv2_continuous_exp": _null_stats(cv2_exp_mat),
                "interarrival_cv2_poisson_10ms": _null_stats(cv2_pois10_mat),
                "zero_wait_prob_poisson_10ms": _null_stats(zw_pois10_mat),
                "interarrival_cv2_global_wait_shuffle": _null_stats(cv2_shuf_mat),
                "zero_wait_prob_global_wait_shuffle": _null_stats(zw_shuf_mat),
            },
        }
        p500 = results[code]["permutation_500_nulls_256"]
        print(
            f"[perm500] {code} n_tr={mean_ntr:5.1f} | "
            f"acf256 obs={obs_z256['sign_acf_sum_1_32']['median']:+.3f} vs null={p500['sign_acf_sum_1_32']['median']:+.3f} (math={math_nulls['sign_acf_sum_1_32']:+.3f}) | "
            f"vr256 obs={obs_z256['flow_vr_32_4']['median']:.3f} vs null={p500['flow_vr_32_4']['median']:.3f} | "
            f"fbias obs={obs_z256['flow_sign_bias']['median']:.3f} vs null={p500['flow_sign_bias']['median']:.3f} | "
            f"cv2 obs={obs_z256['interarrival_cv2']['median']:.3f} vs pois10={p500['interarrival_cv2_poisson_10ms']['median']:.3f} | "
            f"zw obs={obs_z256['zero_wait_prob']['median']:.4f} vs pois10={p500['zero_wait_prob_poisson_10ms']['median']:.4f}",
            flush=True,
        )
    return {
        "seed": seed,
        "num_permutations": N_PERMUTATIONS,
        "window_events": 256,
        "stocks": results,
    }


def run_lock_materialize_and_audit(market_root: Path, artifact_dir: Path, stochopia_root: Path) -> None:
    dl = _load_fetcher(market_root)
    raw_dir = market_root / "raw"
    cache_dir = market_root / "cache"
    stage1_data = json.loads((market_root / "stage1_scan_24.json").read_text(encoding="utf-8"))
    train16_profiles = json.loads((market_root / "stage2_train16_profiles.json").read_text(encoding="utf-8"))

    # 1. Deterministically lock 4 + 6 + 1 solely from 16-day train_dates Z
    strat = select_stratified_pool(train16_profiles)
    exp_codes = [x["wind_code"] for x in strat["exploration_pool"]]
    val_codes = [x["wind_code"] for x in strat["unseen_validation_pool"]]
    ctrl_codes = [x["wind_code"] for x in strat["low_qrec_control"]]
    locked_codes = [*exp_codes, *val_codes, *ctrl_codes]
    role_map: dict[str, tuple[str, str, str]] = {}
    for item in strat["exploration_pool"]:
        role_map[item["wind_code"]] = ("exploration_pool", item["quadrant"], item["label_zh"])
    for item in strat["unseen_validation_pool"]:
        role_map[item["wind_code"]] = ("unseen_validation_pool", item["quadrant"], item["label_zh"])
    for item in strat["low_qrec_control"]:
        role_map[item["wind_code"]] = ("low_qrec_control", item["quadrant"], item["label_zh"])

    print("Locked 4 exploration:", exp_codes)
    print("Locked 6 unseen validation:", val_codes)
    print("Locked 1 low-Q_rec control:", ctrl_codes)

    # 2. Materialize train_dates .npz cache & calibrate horizons for the 11 locked stocks
    train_days_by_code: dict[str, list[SZSEDaySlice]] = {c: [] for c in locked_codes}
    slice_receipts: list[Json] = []
    anchors_all: list[Json] = []
    for d in TRAIN_DATES:
        for c in locked_codes:
            day_sl, rec, _ = _reconstruct_and_check_day(raw_dir, cache_dir, c, d, horizon_ms=None)
            rec["split"] = "train"
            slice_receipts.append(rec)
            train_days_by_code[c].append(day_sl)

    horizons: dict[str, int] = {}
    for c in locked_codes:
        fitted = load_market_days(market_root, c, list(TRAIN_DATES))
        horizons[c] = 2 * CHILDREN * int(calibrate_market(market_root, fitted, c)["spacing_ms"])
        # Verify train_dates also have valid anchors
        for d_enc in fitted:
            wins = select_windows([d_enc], horizons[c])
            if len(wins) != 2:
                raise RuntimeError(f"train anchor check failed: {c} {d_enc.day.trading_date}")

    # 3. Fetch dev_dates (12 days) for the 11 locked stocks and materialize .npz cache
    dl.fetch_days(DEV_DATES, locked_codes, revision=SOURCE_REVISION, raw_subdir="raw")
    dev_excluded: dict[str, list[Json]] = {}
    # Check if any locked stock has a suspension on DEV_DATES
    for d in DEV_DATES:
        for c in locked_codes:
            day_sl, rec, anc = _reconstruct_and_check_day(raw_dir, cache_dir, c, d, horizon_ms=horizons[c])
            rec["split"] = "dev"
            slice_receipts.append(rec)
            anchors_all.extend(anc)

    # 4. Fetch holdout_test_dates (20 days) with per-stock chronological replacement from REPLACEMENT_TEST_DATES if needed
    dl.fetch_days(HOLDOUT_TEST_DATES, locked_codes, revision=SOURCE_REVISION, raw_subdir="raw")
    holdout_days_by_code: dict[str, list[SZSEDaySlice]] = {c: [] for c in locked_codes}
    effective_holdout_by_code: dict[str, list[int]] = {}
    excluded_holdout_by_code: dict[str, list[Json]] = {}

    for c in locked_codes:
        accepted_dates: list[int] = []
        excluded_rows: list[Json] = []
        rep_idx = 0
        candidate_stream = list(HOLDOUT_TEST_DATES)
        i = 0
        while len(accepted_dates) < len(HOLDOUT_TEST_DATES):
            if i >= len(candidate_stream):
                if rep_idx >= len(REPLACEMENT_TEST_DATES):
                    raise RuntimeError(f"replacement pool exhausted for {c}: {len(accepted_dates)}/20")
                next_rep = REPLACEMENT_TEST_DATES[rep_idx]
                rep_idx += 1
                dl.fetch_days([next_rep], locked_codes, revision=SOURCE_REVISION, raw_subdir="raw")
                candidate_stream.append(next_rep)
            d = candidate_stream[i]
            i += 1
            try:
                day_sl, rec, anc = _reconstruct_and_check_day(raw_dir, cache_dir, c, d, horizon_ms=horizons[c])
            except RuntimeError as exc:
                excluded_rows.append({"date": d, "reason": str(exc)})
                print(f"[holdout replacement] {c} {d} excluded ({exc}); advancing replacement pool", flush=True)
            else:
                rec["split"] = "holdout_test"
                slice_receipts.append(rec)
                anchors_all.extend(anc)
                accepted_dates.append(d)
                holdout_days_by_code[c].append(day_sl)
        effective_holdout_by_code[c] = accepted_dates
        excluded_holdout_by_code[c] = excluded_rows

    # Write local market_root/manifest.json compatible with holdout_data.py
    market_manifest = {
        "schema": REGISTERED_M4_HOLDOUT_SCHEMA,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "symbols": locked_codes,
        "train_dates": list(TRAIN_DATES),
        "dev_dates": list(DEV_DATES),
        "test_dates": list(HOLDOUT_TEST_DATES),
        "replacement_test_dates": list(REPLACEMENT_TEST_DATES),
        "embargo_dates": [],
    }
    atomic_write_json(market_root / "manifest.json", market_manifest)

    # 5. Offline property drift audit: compute_property_drift(Z_train, Z_holdout) (never read by discovery loop)
    drift_audit: dict[str, Json] = {}
    for c in locked_codes:
        z_tr_256 = train16_profiles[c]["z_train_256"]
        z_tr_sess = train16_profiles[c]["z_train_session"]
        ho_wins256, ho_ntr = slice_windows_256(holdout_days_by_code[c])
        z_ho_256 = measure_data_properties(ho_wins256)
        z_ho_sess = measure_data_properties(holdout_days_by_code[c])
        drift_256 = compute_property_drift(z_tr_256, z_ho_256)
        drift_sess = compute_property_drift(z_tr_sess, z_ho_sess)
        role, quad, lbl = role_map[c]
        drift_audit[c] = {
            "name": train16_profiles[c]["name"],
            "board": train16_profiles[c]["board"],
            "pool_role": role,
            "quadrant": quad,
            "effective_holdout_dates": effective_holdout_by_code[c],
            "excluded_holdout_dates": excluded_holdout_by_code[c],
            "holdout_n_windows_256": len(ho_wins256),
            "holdout_mean_window_trades": round(float(np.mean([x for x in ho_ntr if x >= 33])), 2),
            "z_holdout_256_flat": _flatten_z(z_ho_256),
            "z_holdout_session_flat": _flatten_z(z_ho_sess),
            "drift_256_iqr_units": {k: round(v, 6) for k, v in drift_256.items()},
            "drift_session_iqr_units": {k: round(v, 6) for k, v in drift_sess.items()},
        }
        print(
            f"[drift] {c} {train16_profiles[c]['name']:<4} ({role:<22}) "
            f"excl={len(excluded_holdout_by_code[c])} | "
            f"256_rms={drift_256['total_rms']:.3f} sess_rms={drift_sess['total_rms']:.3f}",
            flush=True,
        )

    # 6. Run 500-draw permutation null calibration on train_dates (10 clean + 1 control)
    perm_nulls = run_permutation_nulls(train_days_by_code, locked_codes, seed=PERMUTATION_SEED)

    # 7. Write sealed P2-M4a artifacts in OpenHyra/artifacts/p2-m4a-szse-stratification/
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 7a. candidate_qrec_scan.csv
    csv_path = artifact_dir / "candidate_qrec_scan.csv"
    csv_fields = [
        "wind_code", "name", "board", "price_tier", "median_price_rmb_5d",
        "active_days_5d", "one_sided_days_5d", "n_snapshots_5d",
        "top1_quote_match_rate_5d", "vol_aligned_touch_match_rate_5d",
        "top10_depth_mae_5d", "top10_log_vol_mae_5d", "invalid_spread_rate_5d",
        "mean_daily_events_5d", "passed_stage1_qrec_gate",
        "top1_quote_match_rate_16d", "invalid_spread_rate_16d",
        "final_disposition", "pool_role", "quadrant", "reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for row in stage1_data["rows"]:
            code = row["wind_code"]
            t16 = train16_profiles.get(code, {}).get("qrec_16d", {})
            if code in role_map:
                role, quad, lbl = role_map[code]
                disp = "LOCKED_CONTROL" if role == "low_qrec_control" else "LOCKED_CLEAN"
                reason = lbl
            elif not row["passed_stage1_qrec_gate"]:
                role, quad = "rejected", "none"
                disp = row["stage1_status"]
                if disp == "REJECTED_SUSPENDED":
                    reason = f"首5日连续停牌（有效交易日 {row['active_days']}/5）"
                elif disp == "REJECTED_LIMIT_LOCKUP":
                    reason = f"首5日存在单边涨跌停封板（{row['one_sided_days']} 日）"
                else:
                    reason = (
                        f"首5日一档匹配率 {row['top1_quote_match_rate']*100:.2f}% < 95% 硬门槛"
                        "（深市创业板 2017 委托/成交双通道独立序列号在同 10ms 内跨通道排序失配）"
                    )
            else:
                role, quad = "eligible_unselected", "clean_reserve"
                disp = "ELIGIBLE_RESERVE"
                reason = "通过 Q_rec >= 95% 硬门槛（16日均有效），同象限已有跨度更极端代表股入选"
            writer.writerow({
                "wind_code": code,
                "name": row["name"],
                "board": row["board"],
                "price_tier": row["price_tier"],
                "median_price_rmb_5d": row["median_price_rmb"],
                "active_days_5d": row["active_days"],
                "one_sided_days_5d": row["one_sided_days"],
                "n_snapshots_5d": row["n_snapshots"],
                "top1_quote_match_rate_5d": row["top1_quote_match_rate"],
                "vol_aligned_touch_match_rate_5d": row["vol_aligned_touch_match_rate"],
                "top10_depth_mae_5d": row["top10_depth_mae"],
                "top10_log_vol_mae_5d": row["top10_log_vol_mae"],
                "invalid_spread_rate_5d": row["invalid_spread_rate"],
                "mean_daily_events_5d": row["mean_daily_events"],
                "passed_stage1_qrec_gate": row["passed_stage1_qrec_gate"],
                "top1_quote_match_rate_16d": t16.get("top1_quote_match_rate", ""),
                "invalid_spread_rate_16d": t16.get("invalid_spread_rate", ""),
                "final_disposition": disp,
                "pool_role": role,
                "quadrant": quad,
                "reason": reason,
            })

    # 7b. Compute 48-day pooled Q_rec per locked stock and assemble locked_m4_manifest.json
    by_stock_slices: dict[str, list[Json]] = {c: [] for c in locked_codes}
    for rec in slice_receipts:
        by_stock_slices[rec["symbol"]].append(rec)
    for c in locked_codes:
        by_stock_slices[c].sort(key=lambda r: (r["split"], r["date"]))
        if len(by_stock_slices[c]) != 48:
            raise RuntimeError(f"expected 48 slices for {c}, got {len(by_stock_slices[c])}")

    locked_profiles: dict[str, Json] = {}
    for c in locked_codes:
        role, quad, lbl = role_map[c]
        p16 = train16_profiles[c]
        recs48 = by_stock_slices[c]
        tot_snaps_48 = sum(int(r["n_snapshots"]) for r in recs48)
        top1_48 = sum(float(r["top1_quote_match_rate"]) * int(r["n_snapshots"]) for r in recs48) / tot_snaps_48
        dmae_48 = sum(float(r["top10_depth_mae"]) * int(r["n_snapshots"]) for r in recs48) / tot_snaps_48
        inv_48 = sum(float(r["invalid_spread_rate"]) * int(r["n_snapshots"]) for r in recs48) / tot_snaps_48
        locked_profiles[c] = {
            "wind_code": c,
            "name": p16["name"],
            "board": p16["board"],
            "price_tier": p16["price_tier"],
            "pool_role": role,
            "quadrant": quad,
            "label_zh": lbl,
            "median_price_rmb_train16": p16["qrec_16d"]["median_price_rmb"],
            "calibrated_horizon_ms": horizons[c],
            "qrec_train16": p16["qrec_16d"],
            "qrec_all48": {
                "n_days": 48,
                "n_snapshots": tot_snaps_48,
                "top1_quote_match_rate": round(top1_48, 6),
                "top10_depth_mae": round(dmae_48, 4),
                "invalid_spread_rate": round(inv_48, 6),
            },
            "mean_window_trades_256": p16["mean_window_trades"],
            "median_window_trades_256": p16["median_window_trades"],
            "n_windows_256_eligible_train16": p16["n_windows_256_eligible"],
            "eligible_window_fraction_train16": p16["eligible_window_fraction"],
            "z_train_256": {
                **p16["z_train_256_flat"],
                "mean_window_trades": p16["mean_window_trades"],
            },
            "z_train_session": p16["z_train_session_flat"],
            "effective_holdout_test_dates": effective_holdout_by_code[c],
            "excluded_holdout_test_dates": excluded_holdout_by_code[c],
            "slice_sha256_by_date": {
                str(r["date"]): {
                    "split": r["split"],
                    "sha256": r["npz_sha256"],
                    "bytes": r["npz_bytes"],
                    "n_events": r["n_events"],
                    "n_snapshots": r["n_snapshots"],
                    "top1_quote_match_rate": r["top1_quote_match_rate"],
                    "invalid_spread_rate": r["invalid_spread_rate"],
                }
                for r in sorted(recs48, key=lambda x: int(x["date"]))
            },
        }

    manifest_out = {
        "schema": REGISTERED_M4_HOLDOUT_SCHEMA,
        "milestone": "P2-M4a",
        "api_calls": 0,
        "neural_fits": 0,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "baseline_commits": {
            "openhyra_baseline": "5fce212aecca40d8fec5cccdc5ef23f4fd811d47",
            "stochopia_baseline": "2533584477cda21b24c3f2b7972df741d9762c04",
        },
        "calendar": {
            "stage1_scan_dates": list(STAGE1_SCAN_DATES),
            "train_dates": list(TRAIN_DATES),
            "dev_dates": list(DEV_DATES),
            "planned_holdout_test_dates": list(HOLDOUT_TEST_DATES),
            "replacement_test_dates": list(REPLACEMENT_TEST_DATES),
            "total_primary_days": 48,
        },
        "qrec_gate": {
            "min_top1_quote_match_rate": 0.95,
            "max_invalid_spread_rate": 0.01,
            "max_one_sided_quote_fraction_per_day": 0.20,
            "require_zero_suspensions_in_train": True,
        },
        "pools": {
            "exploration_pool": exp_codes,
            "unseen_validation_pool": val_codes,
            "low_qrec_control": ctrl_codes,
            "all_locked_symbols": locked_codes,
        },
        "stratification": strat,
        "stocks": locked_profiles,
    }
    atomic_write_json(artifact_dir / "locked_m4_manifest.json", manifest_out)
    atomic_write_json(
        artifact_dir / "drift_audit.json",
        {
            "note": "Offline covariate audit only. Discovery loop and LLM prompts MUST NEVER read z_holdout or drift_audit.json.",
            "train_dates": list(TRAIN_DATES),
            "planned_holdout_test_dates": list(HOLDOUT_TEST_DATES),
            "stocks": drift_audit,
        },
    )
    atomic_write_json(artifact_dir / "szse_permutation_nulls.json", perm_nulls)
    print(f"All P2-M4a JSON/CSV artifacts written to {artifact_dir} (528 .npz slices sealed).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("train16", "lock_and_seal"))
    parser.add_argument("--market-root", type=Path, default=Path("/Users/williamqian/MarketData/Stochopia-SZSE-P2-M4"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "p2-m4a-szse-stratification",
    )
    parser.add_argument(
        "--stochopia-root",
        type=Path,
        default=Path("/Users/williamqian/GitHub/Stochopia-harness-v21"),
    )
    args = parser.parse_args()
    if args.step == "train16":
        run_stage1_and_train16(args.market_root.resolve())
    elif args.step == "lock_and_seal":
        run_lock_materialize_and_audit(
            args.market_root.resolve(), args.artifact_dir.resolve(), args.stochopia_root.resolve(),
        )


if __name__ == "__main__":
    main()
