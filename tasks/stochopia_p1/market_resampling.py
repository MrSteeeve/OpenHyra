"""Aligned trading-day bootstrap for P1 and P2 market measurements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import isfinite
from numbers import Real

import numpy as np

from experiments.p1_m3_evidence import M4_MARKET_METRICS, P2_MARKET_METRICS
from experiments.p1_m4_market_stats import MarketClusters
from experiments.p2_m1_quote_support import common_quote_panels
from stochopia.measure.impact import C2_PASS_METRIC, CONTINUOUS_IMPACT_METRICS, P1_METRIC_SCHEMA, P2_METRIC_SCHEMA

from .store import IMPACT, EvaluationRequest, Json, PairedEvaluation


BOOTSTRAP_RESAMPLES = 2000
MIN_PAIRED_DAYS = 6
MIN_VALID_DRAWS = 1800


def _finite(value: object) -> float | None:
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
        return None
    return float(value)


def metric_delta(parent: object, candidate: object) -> float | None:
    before = _finite(parent)
    after = _finite(candidate)
    if before is None or after is None:
        return None
    return after - before


# --- Aligned cluster bootstrap ---


def _summarize_draws(
    parent: np.ndarray, candidate: np.ndarray, *, point_parent: float | None,
    point_candidate: float | None, point_delta: float | None,
    paired_days: int, registered_days: int, sampled_deltas: np.ndarray | None = None,
) -> Json:
    """Take 2.5/5/10/90/95/97.5% quantiles of finite paired [draw] differences.

    A difference is candidate minus parent on the same draw of trading days.
    Marginal intervals use the same retained draws as the difference interval.
    At least max(6, floor(registered_days/2)) jointly supported original
    dates and 1,800 finite paired draws are required. Undefined points
    or insufficient support leave every quantile unavailable.
    """
    differences = candidate - parent if sampled_deltas is None else sampled_deltas
    valid = np.isfinite(differences) & np.isfinite(parent) & np.isfinite(candidate)
    paired_valid = int(valid.sum())
    ci = None
    p05 = None
    p10 = None
    p90 = None
    p95 = None
    parent_ci = None
    candidate_ci = None
    reason = None
    min_days = max(MIN_PAIRED_DAYS, registered_days // 2)
    if paired_days < min_days:
        reason = f"fewer than {min_days} paired trading days"
    elif paired_valid < MIN_VALID_DRAWS:
        reason = f"fewer than {MIN_VALID_DRAWS} finite paired bootstrap draws"
    elif point_delta is None:
        reason = "model readout undefined on original day panel"
    else:
        quantiles = np.quantile(differences[valid], (0.025, 0.05, 0.10, 0.90, 0.95, 0.975))
        ci = [float(quantiles[0]), float(quantiles[5])]
        p05 = float(quantiles[1])
        p10 = float(quantiles[2])
        p90 = float(quantiles[3])
        p95 = float(quantiles[4])
        parent_ci = np.quantile(parent[valid], (0.025, 0.975)).tolist()
        candidate_ci = np.quantile(candidate[valid], (0.025, 0.975)).tolist()
    return {
        "estimate": point_delta,
        "interval": ci,
        "p025": None if ci is None else ci[0],
        "p05": p05,
        "p10": p10,
        "p90": p90,
        "p95": p95,
        "p975": None if ci is None else ci[1],
        "parent_estimate": point_parent,
        "parent_interval": parent_ci,
        "candidate_estimate": point_candidate,
        "candidate_interval": candidate_ci,
        "paired_days": paired_days,
        "registered_days": registered_days,
        "parent_bootstrap_valid": int(np.isfinite(parent).sum()),
        "candidate_bootstrap_valid": int(np.isfinite(candidate).sum()),
        "bootstrap_valid": paired_valid,
        "bootstrap_total": len(differences),
        "reason": reason,
    }


def _mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else None


def _scalar_readout(
    pair: PairedEvaluation, days: tuple[int, ...], draws: np.ndarray, metric: str,
) -> Json:
    """Bootstrap equal-day means for scalar day readouts such as NLL.

    Point estimates average only jointly observed original days. For each
    [draw, day] resample, both arms retain the same jointly observed dates.
    A draw containing no observed pair has no estimate.
    """
    parent = np.array([_finite(pair.parent_rows[day].get(metric)) for day in days], dtype=float)
    candidate = np.array([_finite(pair.candidate_rows[day].get(metric)) for day in days], dtype=float)
    differences = candidate - parent
    valid = np.isfinite(differences)
    sampled = valid[draws]
    counts = sampled.sum(axis=1)

    def samples(values: np.ndarray) -> np.ndarray:
        total = np.where(sampled, values[draws], 0.0).sum(axis=1)
        return np.divide(total, counts, out=np.full(len(draws), np.nan), where=counts > 0)

    return _summarize_draws(
        samples(parent),
        samples(candidate),
        point_parent=_mean(parent[valid]),
        point_candidate=_mean(candidate[valid]),
        point_delta=_mean(differences),
        paired_days=int(valid.sum()),
        registered_days=len(days),
        sampled_deltas=samples(differences),
    )


def _p2_moments(summary: Json, parent: np.ndarray, candidate: np.ndarray) -> None:
    """Add empirical bootstrap means and median, using only finite paired draws."""
    valid = np.isfinite(parent) & np.isfinite(candidate)
    before = parent[valid]
    after = candidate[valid]
    difference = after - before
    summary["mean"] = _mean(difference)
    summary["p50"] = float(np.median(difference)) if len(difference) else None
    summary["parent_bootstrap_mean"] = _mean(before)
    summary["candidate_bootstrap_mean"] = _mean(after)


def _p2_conditional_score(
    ordinary: Json, parent_score: np.ndarray, candidate_score: np.ndarray,
    parent_c2: np.ndarray, candidate_c2: np.ndarray,
) -> Json:
    """Describe finite score differences conditional on C2 pass for both arms.

    Quantiles are empirical conditional bootstrap quantiles, not a confidence
    interval for an unconditional score. They may be available even when the
    original-panel score is undefined or fewer than 1800 draws qualify.
    The ordinary score readout keeps its historical inferential support gate.
    """
    joint = (parent_c2 == 1.0) & (candidate_c2 == 1.0)
    finite = joint & np.isfinite(parent_score) & np.isfinite(candidate_score)
    before = parent_score[finite]
    after = candidate_score[finite]
    differences = after - before
    result = dict(ordinary)
    keys = ("p025", "p05", "p10", "p50", "p90", "p95", "p975")
    for key in keys:
        result[key] = None
    result["interval"] = None
    result["parent_interval"] = None
    result["candidate_interval"] = None
    if len(differences):
        quantiles = np.quantile(differences, (0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975))
        for key, val in zip(keys, quantiles, strict=True):
            result[key] = float(val)
        result["interval"] = [result["p025"], result["p975"]]
        result["parent_interval"] = np.quantile(before, (0.025, 0.975)).tolist()
        result["candidate_interval"] = np.quantile(after, (0.025, 0.975)).tolist()
    _p2_moments(result, before, after)
    result["joint_c2_pass_draws"] = int(joint.sum())
    result["joint_c2_pass_share"] = float(joint.mean()) if len(joint) else None
    result["conditional_score_draws"] = int(finite.sum())
    result["bootstrap_valid"] = int(finite.sum())
    result["estimand"] = "empirical score difference distribution conditional on independent C2 pass in both arms"
    result["reason"] = (
        "no finite jointly admitted score draw" if not len(differences)
        else "conditional quantiles only; not an unconditional confidence interval"
    )
    return result


def _market_readouts(
    pair: PairedEvaluation, dates: tuple[int, ...], draws: np.ndarray, metrics: tuple[str, ...],
) -> Json:
    if pair.parent_market is None or pair.candidate_market is None:
        raise ValueError("repooling requires both models' registered market day clusters")
    parent = MarketClusters.from_payload(pair.parent_market)
    candidate = MarketClusters.from_payload(pair.candidate_market)
    size = len(dates)
    if parent.days != size or candidate.days != size:
        raise ValueError("market day clusters must align with the registered evaluation dates")
    if parent.metric_schema_version != candidate.metric_schema_version:
        raise ValueError("paired market metrics require the same schema")
    p2 = parent.metric_schema_version == P2_METRIC_SCHEMA
    if p2:
        if (
            parent.num_replicates != candidate.num_replicates
            or parent.paths_per_day != candidate.paths_per_day
            or parent.sigma_ticks != candidate.sigma_ticks
            or parent.daily_shares != candidate.daily_shares
            or parent.planned_child_shares != candidate.planned_child_shares
            or not np.array_equal(parent.grid_ms, candidate.grid_ms)
            or not np.array_equal(parent.schedule, candidate.schedule)
        ):
            raise ValueError("P2 paired market metrics require the same replica and reference protocol")
        # Each pooled P2 report contains the independent C2 probability
        # diagnostic and the conditional score distribution.
        metrics = tuple(dict.fromkeys((*metrics, C2_PASS_METRIC, IMPACT)))
        if (parent.continuous_panel is None) != (candidate.continuous_panel is None):
            raise ValueError("paired models must share the same available-quote protocol")
        if parent.continuous_panel is not None and candidate.continuous_panel is not None:
            left, right = common_quote_panels(parent.continuous_panel, candidate.continuous_panel)
            parent = replace(parent, continuous_panel=left)
            candidate = replace(candidate, continuous_panel=right)

    parent_point = parent.estimate(np.arange(size))
    parent_samples = parent.resample(draws, metrics)
    identical = pair.parent_market is pair.candidate_market
    if identical:
        candidate_point = parent_point
        candidate_samples = parent_samples
    else:
        candidate_point = candidate.estimate(np.arange(size))
        candidate_samples = candidate.resample(draws, metrics)

    summaries = {}
    for metric in metrics:
        before = _finite(parent_point[metric])
        after = _finite(candidate_point[metric])
        supported = parent.day_support(metric) & candidate.day_support(metric)
        support_count = int(supported.sum())
        mark_support = None
        if metric in CONTINUOUS_IMPACT_METRICS and parent.continuous_panel is not None:
            mark_support = parent.continuous_mark_support(metric) & candidate.continuous_mark_support(metric)
            support_count = int(mark_support.sum(axis=0).min())
        summaries[metric] = _summarize_draws(
            parent_samples[metric],
            candidate_samples[metric],
            point_parent=before,
            point_candidate=after,
            point_delta=metric_delta(before, after),
            paired_days=support_count,
            registered_days=size,
        )
        if mark_support is not None:
            summaries[metric]["support_definition"] = "minimum common available trading-day count across required marks"
            summaries[metric]["complete_common_days"] = int(mark_support.all(axis=1).sum())
            summaries[metric]["common_days_by_mark"] = mark_support.sum(axis=0).tolist()
        if p2:
            _p2_moments(summaries[metric], parent_samples[metric], candidate_samples[metric])
        if metric == IMPACT:
            estimand = "pooled impact score conditional on original-panel C2 admission"
            if p2:
                estimand = "pooled impact score with independent C2 readmission in every draw"
            summaries[metric]["bootstrap_support"] = {
                "estimand": estimand,
                "parent": parent.score_support(dates, parent_samples[metric]),
                "candidate": candidate.score_support(dates, candidate_samples[metric]),
            }
    if p2:
        parent_c2 = parent_samples[C2_PASS_METRIC]
        candidate_c2 = candidate_samples[C2_PASS_METRIC]
        joint = (parent_c2 == 1.0) & (candidate_c2 == 1.0)
        joint_share = float(joint.mean()) if len(joint) else None
        c2_summary = summaries[C2_PASS_METRIC]
        c2_summary["joint_c2_pass_share"] = joint_share
        c2_summary["joint_c2_pass_draws"] = int(joint.sum())
        c2_summary["bootstrap_probability_delta"] = c2_summary["mean"]
        c2_summary["parent_complete_execution_days"] = int(np.isfinite(parent.paths["child_visible"]).all(axis=(1, 2)).sum())
        c2_summary["candidate_complete_execution_days"] = int(np.isfinite(candidate.paths["child_visible"]).all(axis=(1, 2)).sum())
        c2_summary["estimand"] = "paired C2 indicator differences; bootstrap mean is pass probability difference"
        summaries["delta_c2_pass_prob"] = dict(c2_summary)
        summaries["joint_c2_pass_share"] = joint_share
        summaries["delta_lob_impact_score"] = _p2_conditional_score(
            summaries[IMPACT], parent_samples[IMPACT], candidate_samples[IMPACT], parent_c2, candidate_c2,
        )
    return summaries


def paired_readouts(pair: PairedEvaluation, request: EvaluationRequest, metrics: Sequence[str]) -> Json:
    """Use one 2,000 by day draw matrix across both models and every readout.

    Each draw samples as many days as the registered panel with replacement.
    The 2.5% and 97.5% quantiles form a two-sided interval. The 5% quantile is
    the one-sided lower endpoint. The composite is computed on the full
    sample for its point estimate, and recomputed from pooled physical
    inputs on every draw conditional on original-panel C2 admission. No
    session or training seed is a bootstrap unit. The original support gate
    counts dates where both arms can contribute to the requested readout.
    P2 payloads instead independently judge C2 in all draws, then report
    marginal pass probabilities, paired indicator differences, jointly
    admitted score draws and independent continuous-response intervals.
    """
    pair.validate(request)
    days = request.eval_dates
    if len(days) >= 2:
        rng = np.random.default_rng(request.rollout_seed)
        draws = rng.integers(0, len(days), size=(BOOTSTRAP_RESAMPLES, len(days)))
    else:
        draws = np.empty((0, len(days)), dtype=int)
    requested = tuple(dict.fromkeys(metrics))
    pooled = ()
    if pair.parent_market is not None or pair.candidate_market is not None:
        schema = pair.parent_market.get("metric_schema_version", P1_METRIC_SCHEMA) if pair.parent_market else P1_METRIC_SCHEMA
        market_metrics = P2_MARKET_METRICS if schema == P2_METRIC_SCHEMA else M4_MARKET_METRICS
        pooled = tuple(metric for metric in requested if metric in market_metrics)
        if schema == P2_METRIC_SCHEMA:
            if "delta_c2_pass_prob" in requested or "joint_c2_pass_share" in requested:
                pooled = tuple(dict.fromkeys((*pooled, C2_PASS_METRIC)))
            if "delta_lob_impact_score" in requested:
                pooled = tuple(dict.fromkeys((*pooled, IMPACT)))
    summaries = _market_readouts(pair, days, draws, pooled) if pooled else {}
    for metric in requested:
        if metric not in summaries:
            summaries[metric] = _scalar_readout(pair, days, draws, metric)
    return summaries
