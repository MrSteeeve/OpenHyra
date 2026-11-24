"""Pair trading-day readouts and assess preregistered mechanism predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real

import numpy as np

from .protocol import StructuredPrediction
from .store import IMPACT, NLL, EvaluationRequest, Json, MetricRule, PairedEvaluation


VERIFIED_GAIN = "VERIFIED_GAIN"
UNVERIFIED_GAIN = "UNVERIFIED_GAIN"
REFUTED = "REFUTED"
INCONCLUSIVE = "INCONCLUSIVE"
EVALUATION_FAILED = "evaluation_failed"
BOOTSTRAP_RESAMPLES = 2000


def predictions_from_payload(payload: Json) -> tuple[StructuredPrediction, ...]:
    return tuple(StructuredPrediction(**row) for row in payload["predictions"])


# --- Paired inference and promotion ---


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


def paired_readouts(pair: PairedEvaluation, request: EvaluationRequest, metrics: Sequence[str]) -> Json:
    """Bootstrap the mean of the candidate-minus-parent differences by day.

    Each metric has one paired scalar per registered day. Each of 2,000
    resamples draws as many days as the full panel with replacement. The 2.5%
    and 97.5% quantiles of their means define the interval. The same draw
    matrix serves all metrics. Missing pairs or fewer than two registered
    days prevent an interval; the estimate uses the available pairs.
    """
    pair.validate(request)
    days = request.eval_dates
    rng = np.random.default_rng(request.rollout_seed)
    draws = rng.integers(0, len(days), size=(BOOTSTRAP_RESAMPLES, len(days)))
    summaries: Json = {}
    for metric in dict.fromkeys(metrics):
        values = [
            metric_delta(pair.parent_rows[day].get(metric), pair.candidate_rows[day].get(metric))
            for day in days
        ]
        observed = np.array([value for value in values if value is not None])
        interval = None
        reason = None
        if len(observed) == len(days) and len(days) >= 2:
            interval = np.quantile(observed[draws].mean(axis=1), (0.025, 0.975)).tolist()
        elif len(observed) != len(days):
            reason = "missing paired day outcomes"
        else:
            reason = "fewer than two trading days"
        summaries[metric] = {
            "estimate": float(observed.mean()) if len(observed) else None,
            "interval": interval, "paired_days": len(observed), "registered_days": len(days), "reason": reason,
        }
    return summaries


def _judge(prediction: StructuredPrediction, interval: list[float] | None) -> str:
    """Compare observed and predicted closed intervals for the paired mean.

    Containment supports the forecast, disjoint intervals refute it, and
    overlap or a missing observed interval is inconclusive. Prediction
    validation enforces the sign and, when applicable, the equivalence band.
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


def assess(
    pair: PairedEvaluation, request: EvaluationRequest, predictions: tuple[StructuredPrediction, ...],
    rules: Mapping[str, MetricRule], nll_guardrail: float,
) -> Json:
    """Require a guarded, significant improvement and supported mechanism for promotion."""
    core = predictions[0]
    rule = rules[core.metric]
    metrics = [NLL, IMPACT, *(prediction.metric for prediction in predictions)]
    readouts = paired_readouts(pair, request, metrics)
    verdicts = [{"metric": p.metric, "status": _judge(p, readouts[p.metric]["interval"])} for p in predictions]
    measured = readouts[core.metric]["interval"]
    improvement = False
    harm = False
    if measured is not None:
        low, high = measured
        if rule.direction == "increase":
            improvement = low > rule.min_effect
            harm = high < -rule.min_effect
        else:
            improvement = high < -rule.min_effect
            harm = low > rule.min_effect
    nll_interval = readouts[NLL]["interval"]
    guard = nll_interval is not None and nll_interval[1] <= nll_guardrail
    aggregate = {
        metric: metric_delta(pair.parent_evidence.get(metric), pair.candidate_evidence.get(metric))
        for metric in (NLL, IMPACT)
    }
    score_up = (aggregate[NLL] or 0.0) < 0 or (aggregate[IMPACT] or 0.0) > 0
    if improvement and guard and verdicts[0]["status"] == "SUPPORTED":
        status = VERIFIED_GAIN
    elif improvement or score_up:
        status = UNVERIFIED_GAIN
    elif harm or verdicts[0]["status"] == REFUTED or (nll_interval is not None and nll_interval[0] > nll_guardrail):
        status = REFUTED
    else:
        status = INCONCLUSIVE
    return {"verdict": status, "prediction_verdicts": verdicts, "readouts": readouts,
            "aggregate_delta": aggregate, "nll_guardrail_passed": guard}


def failed_evaluation(predictions: tuple[StructuredPrediction, ...], reason: str) -> Json:
    return {
        "verdict": EVALUATION_FAILED,
        "error": reason,
        "prediction_verdicts": [{"metric": p.metric, "status": INCONCLUSIVE} for p in predictions],
        "training_budget_consumed": True,
    }
