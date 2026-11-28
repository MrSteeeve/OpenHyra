"""Equal-seed, equal-day synthetic metrics and paired cluster resampling."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import numpy as np
from experiments.synthetic_tau_z.design import BOOTSTRAP_SEED, N_BOOTSTRAP, SEEDS, summarize_days
from stochopia.measure.impact import identify_law

Json = dict[str, Any]
C2 = "c2_bootstrap_pass_rate"
SCALAR_METRICS = (
    "val_nll_total", "dt_nll", "event_nll", "level_nll", "size_nll", "counterfactual_impact_mae",
    "stylized_impact_mae", "post_stop_relaxation_delta", "visible_minus_blind_auc",
    "exhausted_replicate_rate", "sign_acf_abs_error", "memory_gap_1", "memory_gap_4",
    "memory_gap_16", "memory_gap_64", "available_quote_fraction", "acf_support_lags",
    "zero_wait_rate_error", "interarrival_cv2_error", "interarrival_cv2_rel_error", "free_abs_drift_ticks",
    "zero_wait_rate_generated", "zero_wait_rate_historical", "interarrival_cv2_generated", "interarrival_cv2_historical",
)
METRICS = (*SCALAR_METRICS, C2)


def _number(value: object) -> float | None:
    return float(value) if type(value) in (int, float) and isfinite(value) else None


def pool_training_seeds(panels: Sequence[Sequence[Json]]) -> list[Json]:
    """Average each day's three deterministic training seed fits; all must be measurable."""
    if len(panels) != len(SEEDS) or not panels[0]:
        raise ValueError("one nonempty date panel is required per registered training seed")
    daily = []
    for rows in zip(*panels, strict=True):
        day = rows[0]["trading_date"]
        if any(row["trading_date"] != day for row in rows):
            raise ValueError("seed panels have mismatched dates")
        output = {"trading_date": day}
        for metric in SCALAR_METRICS:
            values = [_number(row.get(metric)) for row in rows]
            output[metric] = None if None in values else float(np.mean(values))
        children = [row.get("c2_child_ticks") for row in rows]
        output["c2_child_ticks"] = None if None in children else np.mean(children, axis=0).tolist()
        daily.append(output)
    return daily


def _draws(n_days: int) -> np.ndarray:
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n_days, size=(N_BOOTSTRAP, n_days))


def _child_panel(daily: Sequence[Json]) -> np.ndarray:
    values = [row["c2_child_ticks"] if row["c2_child_ticks"] is not None else [None] * 4 for row in daily]
    panel = np.asarray(values, dtype=float)
    if panel.shape != (len(daily), 4):
        raise ValueError("C2 requires exactly four registered child marks per independent day")
    return panel


def _c2_draws(panel: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Re-estimate the law after each shared day draw (never bootstrap seeds or sessions)."""
    result = np.full(len(indices), np.nan)
    for i, draw in enumerate(indices):
        paths = panel[draw]
        paths = paths[np.isfinite(paths).all(axis=1)]
        if len(paths) >= 2:
            result[i] = float(identify_law(paths).identified)
    return result


def _bootstrap_summary(samples: np.ndarray, support: int, n_days: int) -> Json:
    finite = samples[np.isfinite(samples)]
    estimate = None if len(finite) < 1800 or support < 6 else float(finite.mean())
    return {
        "estimate": estimate,
        "interval": None if estimate is None else np.quantile(finite, (0.025, 0.975)).tolist(),
        "paired_days": support, "registered_days": n_days,
        "bootstrap_valid": len(finite), "bootstrap_total": N_BOOTSTRAP,
        "estimand": "day-bootstrap C2 admission frequency; interval of draw-wise indicators",
    }


def summarize_levels(daily: Sequence[Json]) -> tuple[dict[str, float | None], dict[str, Json]]:
    """Report level means and exact support; missing quotes stay unavailable."""
    support = {metric: summarize_days([row.get(metric) for row in daily]) for metric in SCALAR_METRICS}
    panel = _child_panel(daily)
    complete = int(np.isfinite(panel).all(axis=1).sum())
    support[C2] = _bootstrap_summary(_c2_draws(panel, _draws(len(daily))), complete, len(daily))
    aggregate = {metric: values["estimate"] for metric, values in support.items()}
    return aggregate, support


def paired_summary(candidate: Json, parent: Json) -> dict[str, Json]:
    """Candidate-minus-parent effects; use identical physical days and bootstrap draws."""
    if candidate.get("split") != parent.get("split"):
        raise ValueError("paired levels must originate from the same registered split")
    if candidate.get("env_id") != parent.get("env_id"):
        raise ValueError("paired levels must originate from the same registered environment")
    first, second = candidate["daily"], parent["daily"]
    dates = [row["trading_date"] for row in first]
    if dates != [row["trading_date"] for row in second] or not dates:
        raise ValueError("paired levels require identical independent-day calendars")
    result = {}
    for metric in SCALAR_METRICS:
        differences = []
        for left, right in zip(first, second, strict=True):
            a, b = _number(left.get(metric)), _number(right.get(metric))
            differences.append(None if a is None or b is None else a - b)
        result[metric] = {**summarize_days(differences), "daily": differences}
    children_a, children_b = _child_panel(first), _child_panel(second)
    common = np.isfinite(children_a).all(axis=1) & np.isfinite(children_b).all(axis=1)
    children_a[~common] = np.nan
    children_b[~common] = np.nan
    draws = _draws(len(dates))
    result[C2] = _bootstrap_summary(_c2_draws(children_a, draws) - _c2_draws(children_b, draws),
                                    int(common.sum()), len(dates))
    return result
