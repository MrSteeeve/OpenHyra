"""Offline schema, anti-leakage audit, and 4-table rendering tests for P2-M4a SZSE stratification."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from experiments.szse_m4a_report import render
from tasks.lob_world_model.discovery_loop import Z_KEYS, finite_window_nulls


def _dummy_z(shift: float, acf: float, cv2: float) -> dict:
    base = {k: {"median": 0.1, "intraday_iqr": 0.05} for k in Z_KEYS}
    base["midprice_shift_rate"] = {"median": shift, "intraday_iqr": 0.02}
    base["sign_acf_sum_1_32"] = {"median": acf, "intraday_iqr": 0.10}
    base["interarrival_cv2"] = {"median": cv2, "intraday_iqr": 0.15}
    return base


def test_render_m4a_validates_4_6_1_pool_48_slices_and_renders_four_tables(tmp_path: Path) -> None:
    art = tmp_path / "p2-m4a-szse-stratification"
    art.mkdir()
    exp = ["000725.SZ", "002024.SZ", "002304.SZ", "000661.SZ"]
    val = ["000001.SZ", "000651.SZ", "002230.SZ", "002475.SZ", "002821.SZ", "000858.SZ"]
    ctrl = ["300001.SZ"]
    locked = [*exp, *val, *ctrl]
    extra = [f"3000{i:02d}.SZ" for i in range(10, 20)]

    train_dates = list(range(20170601, 20170617))
    dev_dates = list(range(20170620, 20170632))
    holdout_dates = list(range(20170701, 20170721))
    repl_dates = list(range(20170801, 20170809))
    all_48 = [*train_dates, *dev_dates, *holdout_dates]

    csv_fields = [
        "wind_code", "name", "board", "price_tier", "median_price_rmb_5d",
        "active_days_5d", "one_sided_days_5d", "n_snapshots_5d",
        "top1_quote_match_rate_5d", "vol_aligned_touch_match_rate_5d",
        "top10_depth_mae_5d", "top10_log_vol_mae_5d", "invalid_spread_rate_5d",
        "mean_daily_events_5d", "passed_stage1_qrec_gate",
        "top1_quote_match_rate_16d", "invalid_spread_rate_16d",
        "final_disposition", "pool_role", "quadrant", "reason",
    ]
    with (art / "candidate_qrec_scan.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields)
        w.writeheader()
        for c in [*locked, *extra]:
            is_clean = c in exp or c in val
            w.writerow({
                "wind_code": c, "name": c[:6], "board": "主板", "price_tier": "mid_20_60",
                "median_price_rmb_5d": 25.0, "active_days_5d": 5, "one_sided_days_5d": 0,
                "n_snapshots_5d": 20000, "top1_quote_match_rate_5d": 0.995 if is_clean else 0.80,
                "vol_aligned_touch_match_rate_5d": 0.95, "top10_depth_mae_5d": 50.0,
                "top10_log_vol_mae_5d": 0.02, "invalid_spread_rate_5d": 0.0,
                "mean_daily_events_5d": 20000.0, "passed_stage1_qrec_gate": is_clean,
                "top1_quote_match_rate_16d": 0.995 if is_clean else (0.83 if c == "300001.SZ" else ""),
                "invalid_spread_rate_16d": 0.0 if c in locked else "",
                "final_disposition": "LOCKED_CLEAN" if is_clean else ("LOCKED_CONTROL" if c == "300001.SZ" else "REJECTED_LOW_QREC"),
                "pool_role": "exploration_pool" if c in exp else ("unseen_validation_pool" if c in val else ("low_qrec_control" if c == "300001.SZ" else "rejected")),
                "quadrant": "Q1", "reason": "ok",
            })

    stocks_manifest = {}
    stocks_perm = {}
    stocks_drift = {}
    for idx, c in enumerate(locked):
        role = "exploration_pool" if c in exp else ("unseen_validation_pool" if c in val else "low_qrec_control")
        top1 = 0.837 if c == "300001.SZ" else 0.995
        z256 = _dummy_z(0.01 * idx, -0.30, 1.10)
        zsess = _dummy_z(0.01 * idx, +1.80, 1.55)
        stocks_manifest[c] = {
            "wind_code": c, "name": c[:6], "board": "主板", "price_tier": "mid_20_60",
            "pool_role": role, "quadrant": "Q1", "label_zh": "测试象限",
            "median_price_rmb_train16": 25.0, "calibrated_horizon_ms": 32000,
            "qrec_train16": {"top1_quote_match_rate": top1, "invalid_spread_rate": 0.0},
            "qrec_all48": {"n_days": 48, "n_snapshots": 96000, "top1_quote_match_rate": top1,
                           "top10_depth_mae": 40.0, "invalid_spread_rate": 0.0},
            "mean_window_trades_256": 75.0, "median_window_trades_256": 74.0,
            "n_windows_256_eligible_train16": 800, "eligible_window_fraction_train16": 0.99,
            "z_train_256": {**z256, "mean_window_trades": 75.0},
            "z_train_session": zsess,
            "effective_holdout_test_dates": holdout_dates,
            "excluded_holdout_test_dates": [],
            "slice_sha256_by_date": {
                str(d): {
                    "split": "train" if d in train_dates else ("dev" if d in dev_dates else "holdout_test"),
                    "sha256": "ab" * 32, "bytes": 1024, "n_events": 10000, "n_snapshots": 2000,
                    "top1_quote_match_rate": top1, "invalid_spread_rate": 0.0,
                }
                for d in all_48
            },
        }
        stocks_perm[c] = {
            "n_windows_256_eligible": 800, "n_windows_256_total": 808,
            "eligible_window_fraction": 0.99, "mean_window_trades": 75.0, "median_window_trades": 74.0,
            "observed_z_train_256": {k: z256[k]["median"] for k in Z_KEYS},
            "observed_z_train_session": {k: zsess[k]["median"] for k in Z_KEYS},
            "analytical_finite_nulls": finite_window_nulls(75.0),
            "permutation_500_nulls_256": {
                "sign_acf_sum_1_32": {"median": -0.34, "ci95": [-0.36, -0.32]},
                "flow_vr_32_4": {"median": 0.45, "ci95": [0.42, 0.48]},
                "flow_sign_bias": {"median": 0.092, "ci95": [0.088, 0.096]},
                "interarrival_cv2_continuous_exp": {"median": 0.96, "ci95": [0.94, 0.98]},
                "interarrival_cv2_poisson_10ms": {"median": 0.99, "ci95": [0.97, 1.01]},
                "zero_wait_prob_poisson_10ms": {"median": 0.015, "ci95": [0.013, 0.017]},
                "interarrival_cv2_global_wait_shuffle": {"median": 1.30, "ci95": [1.25, 1.35]},
                "zero_wait_prob_global_wait_shuffle": {"median": 0.030, "ci95": [0.028, 0.032]},
            },
        }
        drift_vals = {
            "z_mem.sign_acf_sum_1_32": 0.1, "z_mem.sign_acf_log_slope": 0.0, "z_mem.flow_vr_32_4": -0.1,
            "z_thin.trade_to_top1_depth": 0.2, "z_thin.midprice_shift_rate": 0.15, "z_thin.cancel_to_limit_ratio": -0.05,
            "z_burst.zero_wait_prob": 0.05, "z_burst.interarrival_cv2": 0.12,
            "z_asym.book_imbalance_bias": -0.08, "z_asym.flow_sign_bias": 0.04,
            "total_rms": 0.11,
        }
        stocks_drift[c] = {
            "name": c[:6], "board": "主板", "pool_role": role, "quadrant": "Q1",
            "effective_holdout_dates": holdout_dates, "excluded_holdout_dates": [],
            "holdout_n_windows_256": 1000, "holdout_mean_window_trades": 76.0,
            "z_holdout_256_flat": z256, "z_holdout_session_flat": zsess,
            "drift_256_iqr_units": drift_vals, "drift_session_iqr_units": drift_vals,
        }

    (art / "locked_m4_manifest.json").write_text(json.dumps({
        "schema": "stochopia-p2-m4-v1", "milestone": "P2-M4a", "api_calls": 0, "neural_fits": 0,
        "source_repo": "venvoo/test", "source_revision": "rev",
        "baseline_commits": {"openhyra_baseline": "5fce212", "stochopia_baseline": "2533584"},
        "calendar": {
            "stage1_scan_dates": train_dates[:5], "train_dates": train_dates,
            "dev_dates": dev_dates, "planned_holdout_test_dates": holdout_dates,
            "replacement_test_dates": repl_dates, "total_primary_days": 48,
        },
        "pools": {"exploration_pool": exp, "unseen_validation_pool": val,
                  "low_qrec_control": ctrl, "all_locked_symbols": locked},
        "stocks": stocks_manifest,
    }), encoding="utf-8")
    (art / "szse_permutation_nulls.json").write_text(json.dumps({
        "seed": 1, "num_permutations": 500, "window_events": 256, "stocks": stocks_perm,
    }), encoding="utf-8")
    (art / "drift_audit.json").write_text(json.dumps({
        "train_dates": train_dates, "planned_holdout_test_dates": holdout_dates, "stocks": stocks_drift,
    }), encoding="utf-8")

    md = render(art)
    for heading in ("## 表 1", "## 表 2", "## 表 3", "## 表 4"):
        assert heading in md
    assert "`300001.SZ`" in md and "`000725.SZ`" in md

    # Reject clean stock violating Q_rec >= 0.95
    bad = json.loads((art / "locked_m4_manifest.json").read_text(encoding="utf-8"))
    bad["stocks"]["000725.SZ"]["qrec_train16"]["top1_quote_match_rate"] = 0.90
    (art / "locked_m4_manifest.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="violates Q_rec gate"):
        render(art)
