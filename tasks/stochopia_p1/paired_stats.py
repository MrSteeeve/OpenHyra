"""Align trading-day draws and apply the lexicographic P1 validator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real

import numpy as np

from experiments.p1_m3_evidence import M4_MARKET_METRICS
from experiments.p1_m4_market_stats import MarketClusters

from .protocol import StructuredPrediction
from .store import IMPACT, NLL, EvaluationRequest, Json, PairedEvaluation


VERIFIED_GAIN = "VERIFIED_GAIN"
UNVERIFIED_GAIN = "UNVERIFIED_GAIN"
GUARDRAIL_VIOLATION = "GUARDRAIL_VIOLATION"
REFUTED = "REFUTED"
INCONCLUSIVE = "INCONCLUSIVE"
EVALUATION_FAILED = "evaluation_failed"

BOOTSTRAP_RESAMPLES = 2000
MIN_PAIRED_DAYS = 6
MIN_VALID_DRAWS = 1800
NLL_GUARDRAIL = 0.05
DRIFT_LIMIT = 20.0
MIN_GAIN = 0.50


def predictions_from_payload(payload: Json) -> tuple[StructuredPrediction, ...]:
    return tuple(StructuredPrediction(**row) for row in payload["predictions"])


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


def _market_readouts(
    pair: PairedEvaluation, dates: tuple[int, ...], draws: np.ndarray, metrics: tuple[str, ...],
) -> dict[str, Json]:
    if pair.parent_market is None or pair.candidate_market is None:
        raise ValueError("repooling requires both models' registered market day clusters")
    parent = MarketClusters.from_payload(pair.parent_market)
    candidate = MarketClusters.from_payload(pair.candidate_market)
    size = len(dates)
    if parent.days != size or candidate.days != size:
        raise ValueError("market day clusters must align with the registered evaluation dates")

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
        summaries[metric] = _summarize_draws(
            parent_samples[metric],
            candidate_samples[metric],
            point_parent=before,
            point_candidate=after,
            point_delta=metric_delta(before, after),
            paired_days=int(supported.sum()),
            registered_days=size,
        )
        if metric == IMPACT:
            summaries[metric]["bootstrap_support"] = {
                "estimand": "pooled impact score conditional on original-panel C2 admission",
                "parent": parent.score_support(dates, parent_samples[metric]),
                "candidate": candidate.score_support(dates, candidate_samples[metric]),
            }
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
        pooled = tuple(metric for metric in requested if metric in M4_MARKET_METRICS)
    summaries = _market_readouts(pair, days, draws, pooled) if pooled else {}
    for metric in requested:
        if metric not in summaries:
            summaries[metric] = _scalar_readout(pair, days, draws, metric)
    return summaries


# --- Ordered validation layers ---


def _judge_v2(prediction: StructuredPrediction, interval: list[float] | None) -> str:
    """Test directional, magnitude and equivalent forecasts on candidate minus parent.

    Inputs are two closed intervals in the same metric units. Containment
    supports the registered forecast, disjoint intervals refute it, and
    overlap or a missing measured interval is inconclusive. The prediction
    contract requires directional ranges to have the correct sign, and
    equivalent ranges to lie within their preregistered equivalence bands.
    """
    if interval is None:
        return INCONCLUSIVE
    low, high = interval
    target_low, target_high = prediction.interval
    if target_low <= low <= high <= target_high:
        return "SUPPORTED"
    if high < target_low or target_high < low:
        return REFUTED
    return INCONCLUSIVE


def _judge(prediction: StructuredPrediction, summary: Mapping[str, object]) -> str:
    """Judge a frozen M5 forecast against a scalar paired bootstrap summary.

    For equivalence, p05 and p95 must fit inside the prespecified band
    (TOST at alpha=.05). Opposite 95% intervals refute equivalence.
    For an increase, delta must reach the positive target's lower bound
    and p10 must exceed zero; decrease mirrors that rule using p90.
    A point on or across zero, or a 95% interval missing the near-zero
    target bound, refutes the corresponding directional forecast.
    Undefined intervals or required quantiles are inconclusive.
    """
    if summary.get("interval") is None:
        return INCONCLUSIVE
    estimate = _finite(summary.get("estimate"))
    p025 = _finite(summary.get("p025"))
    p05 = _finite(summary.get("p05"))
    p10 = _finite(summary.get("p10"))
    p90 = _finite(summary.get("p90"))
    p95 = _finite(summary.get("p95"))
    p975 = _finite(summary.get("p975"))
    if any(value is None for value in (estimate, p025, p05, p10, p90, p95, p975)):
        return INCONCLUSIVE

    target_low, target_high = prediction.interval
    if prediction.direction == "equivalent":
        band_low, band_high = prediction.equivalence_band
        if band_low <= p05 <= p95 <= band_high:
            return "SUPPORTED"
        if p975 < band_low or p025 > band_high:
            return REFUTED
    elif prediction.direction == "increase":
        if estimate <= 0 or p975 < target_low:
            return REFUTED
        if estimate >= target_low and p10 > 0:
            return "SUPPORTED"
    else:
        if estimate >= 0 or p025 > target_high:
            return REFUTED
        if estimate <= target_high and p90 < 0:
            return "SUPPORTED"
    return INCONCLUSIVE


def _prediction_verdict(prediction: StructuredPrediction, summary: Json, *, analysis_version: str) -> Json:
    """Keep directional support and quantitative interval containment separate."""
    estimate = _finite(summary.get("estimate"))
    low, high = prediction.interval
    if analysis_version == "m5":
        status = _judge(prediction, summary)
    else:
        status = _judge_v2(prediction, summary["interval"])
    return {
        "role": prediction.role,
        "metric": prediction.metric,
        "status": status,
        "point_available": estimate is not None,
        "point_in_interval": estimate is not None and low <= estimate <= high,
    }


def _finite_seed_nll(evidence: Mapping[str, object], seeds: tuple[int, ...]) -> bool:
    losses = evidence.get("train_seed_nll")
    if not isinstance(losses, Mapping):
        return False
    expected = {str(seed) for seed in seeds}
    return {str(seed) for seed in losses} == expected and all(_finite(value) is not None for value in losses.values())


def _guardrails(pair: PairedEvaluation, request: EvaluationRequest, readouts: Json) -> Json:
    """Check finite per-seed losses, paired NLL and absolute drift/depletion bounds.

    NLL uses the equal-day mean candidate-minus-parent difference. Exhaustion
    counts windows with any empty side across the registered seed repetitions.
    Drift uses the mean absolute terminal displacement. A parent at 20 ticks
    or higher sets its own level as the strict upper bound for the candidate.
    Missing measurements cannot pass a guard.
    """
    nll = readouts[NLL]
    nll_delta = nll["estimate"]
    all_seed_nll = _finite_seed_nll(pair.parent_evidence, request.train_seeds) and _finite_seed_nll(
        pair.candidate_evidence, request.train_seeds,
    )
    nll_in_budget = nll_delta is not None and (
        nll_delta <= NLL_GUARDRAIL or np.isclose(nll_delta, NLL_GUARDRAIL, rtol=0, atol=1e-12)
    )
    nll_passed = all_seed_nll and nll["paired_days"] == len(request.eval_dates) and nll_in_budget
    parent_exhausted = pair.parent_evidence.get("exhausted_windows")
    candidate_exhausted = pair.candidate_evidence.get("exhausted_windows")
    exhausted_delta = metric_delta(parent_exhausted, candidate_exhausted)
    exhausted_passed = exhausted_delta is not None and exhausted_delta <= 0
    drift = readouts["free_abs_drift_ticks"]
    parent_drift = drift["parent_estimate"]
    candidate_drift = drift["candidate_estimate"]
    drift_passed = parent_drift is not None and candidate_drift is not None and (
        candidate_drift < (DRIFT_LIMIT if parent_drift < DRIFT_LIMIT else parent_drift)
    )
    return {
        "finite_nll_by_seed": all_seed_nll,
        "nll_passed": bool(nll_passed),
        "delta_val_nll_total": nll_delta,
        "exhaustion_passed": exhausted_passed,
        "delta_exhausted_windows": exhausted_delta,
        "drift_passed": drift_passed,
        "parent_free_abs_drift_ticks": parent_drift,
        "child_free_abs_drift_ticks": candidate_drift,
        "delta_free_abs_drift_ticks": metric_delta(parent_drift, candidate_drift),
    }


def assess(
    pair: PairedEvaluation, request: EvaluationRequest, predictions: tuple[StructuredPrediction, ...],
) -> Json:
    """Assess candidates that have already passed the zero-data program gate."""
    if tuple(prediction.role for prediction in predictions) != ("primary", "guardrail"):
        raise ValueError("assessment requires one primary and one guardrail prediction in order")
    metrics = [NLL, IMPACT, "free_abs_drift_ticks"]
    metrics.extend(p.metric for p in predictions)
    readouts = paired_readouts(pair, request, metrics)
    verdicts = []
    for prediction in predictions:
        verdicts.append(_prediction_verdict(
            prediction, readouts[prediction.metric], analysis_version=request.analysis_version,
        ))
    guardrails = _guardrails(pair, request, readouts)
    guards_passed = all(guardrails[key] for key in ("nll_passed", "exhaustion_passed", "drift_passed"))
    gain = readouts[IMPACT]
    score_delta = gain["estimate"]
    p05 = gain["p05"]
    p10 = gain["p10"]
    lower = p10 if request.analysis_version == "m5" else p05
    has_benchmark_gain = guards_passed and score_delta is not None and score_delta >= MIN_GAIN
    has_benchmark_gain = has_benchmark_gain and lower is not None and lower > 0
    forecasts_passed = verdicts[0]["status"] == "SUPPORTED" and verdicts[1]["status"] != REFUTED
    if not guards_passed:
        status = GUARDRAIL_VIOLATION
    elif not has_benchmark_gain:
        status = REFUTED
    elif forecasts_passed or request.group == "no_llm":
        status = VERIFIED_GAIN
    else:
        status = UNVERIFIED_GAIN
    return {
        "verdict": status,
        "prediction_verdicts": verdicts,
        "readouts": readouts,
        "guardrails": guardrails,
        "has_benchmark_gain": has_benchmark_gain,
        "mean_delta_score": score_delta,
        "p05_delta_score": p05,
        "p10_delta_score": p10,
        "nll_guardrail_passed": guardrails["nll_passed"],
        "aggregate_delta": {metric: readouts[metric]["estimate"] for metric in (NLL, IMPACT)},
    }


def failed_evaluation(predictions: tuple[StructuredPrediction, ...], reason: str) -> Json:
    return {
        "verdict": EVALUATION_FAILED,
        "error": reason,
        "prediction_verdicts": [
            {"role": p.role, "metric": p.metric, "status": INCONCLUSIVE,
             "point_available": False, "point_in_interval": False} for p in predictions
        ],
        "training_budget_consumed": True,
    }
