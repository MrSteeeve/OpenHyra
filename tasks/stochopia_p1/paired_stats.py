"""Align trading-day draws and apply the lexicographic P1 validator."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# Re-export the established statistical API from its implementation module.
from .market_resampling import (
    BOOTSTRAP_RESAMPLES, MIN_PAIRED_DAYS, MIN_VALID_DRAWS,
    _finite, _summarize_draws, metric_delta, paired_readouts,
)

from .protocol import StructuredPrediction
from .store import IMPACT, NLL, EvaluationRequest, Json, PairedEvaluation


VERIFIED_GAIN = "VERIFIED_GAIN"
UNVERIFIED_GAIN = "UNVERIFIED_GAIN"
GUARDRAIL_VIOLATION = "GUARDRAIL_VIOLATION"
REFUTED = "REFUTED"
INCONCLUSIVE = "INCONCLUSIVE"
EVALUATION_FAILED = "evaluation_failed"

NLL_GUARDRAIL = 0.05
DRIFT_LIMIT = 20.0
MIN_GAIN = 0.50


def predictions_from_payload(payload: Json) -> tuple[StructuredPrediction, ...]:
    return tuple(StructuredPrediction(**row) for row in payload["predictions"])


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
