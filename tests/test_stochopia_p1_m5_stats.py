"""M5 retains distinct tests for directional support and interval calibration."""

from dataclasses import replace

import numpy as np
import pytest

from tasks.stochopia_p1.paired_stats import (
    INCONCLUSIVE, REFUTED, VERIFIED_GAIN, _judge, _prediction_verdict, _summarize_draws, assess,
)
from tasks.stochopia_p1.prompt import M5_MIN_EFFECTS, forecast_targets
from tasks.stochopia_p1.protocol import ProposalValidationError, StructuredPrediction, validate_forecasts
from tasks.stochopia_p1.store import IMPACT, NLL, EvaluationRequest, ModelRef, PairedEvaluation


def _summary(
    estimate: float, p025: float, p05: float, p10: float,
    p90: float, p95: float, p975: float,
) -> dict[str, object]:
    return {
        "estimate": estimate, "interval": [p025, p975],
        "p025": p025, "p05": p05, "p10": p10, "p90": p90, "p95": p95, "p975": p975,
    }


def test_directional_support_does_not_require_exact_interval_containment() -> None:
    prediction = StructuredPrediction("lob_impact_score", "increase", (1, 2), role="primary")
    summary = _summary(3, -1, -0.2, 0.1, 4, 5, 6)
    row = _prediction_verdict(prediction, summary, analysis_version="m5")
    assert row["status"] == "SUPPORTED"
    assert row["point_available"] and not row["point_in_interval"]
    assert _judge(prediction, _summary(0, -1, -0.5, -0.1, 1, 2, 3)) == REFUTED
    assert _judge(prediction, _summary(0.8, 0.2, 0.3, 0.4, 0.9, 1.1, 1.2)) == INCONCLUSIVE
    assert _judge(prediction, _summary(0.6, 0.1, 0.2, 0.3, 0.8, 0.9, 0.99)) == REFUTED


def test_negative_direction_uses_the_near_zero_upper_target_bound() -> None:
    prediction = StructuredPrediction("end_over_peak_ratio", "decrease", (-0.5, -0.08), role="primary")
    outcome = _summary(-0.8, -1.0, -0.9, -0.8, -0.02, 0.1, 0.2)
    assert _judge(prediction, outcome) == "SUPPORTED"
    assert not _prediction_verdict(prediction, outcome, analysis_version="m5")["point_in_interval"]
    assert _judge(prediction, _summary(-0.04, -0.09, -0.08, -0.07, -0.01, 0.0, 0.02)) == INCONCLUSIVE
    assert _judge(prediction, _summary(-0.04, -0.07, -0.06, -0.05, -0.01, 0.0, 0.01)) == REFUTED
    assert _judge(prediction, _summary(0.01, -0.2, -0.1, -0.02, 0.1, 0.2, 0.3)) == REFUTED


def test_equivalence_checks_the_tost_band_and_95_percent_refutation() -> None:
    prediction = StructuredPrediction("val_nll_total", "equivalent", (-0.02, 0.02),
                                      role="guardrail", equivalence_band=(-0.05, 0.05))
    assert _judge(prediction, _summary(0.01, -0.06, -0.05, -0.04, 0.04, 0.05, 0.06)) == "SUPPORTED"
    assert _judge(prediction, _summary(0.07, 0.051, 0.055, 0.056, 0.08, 0.09, 0.10)) == REFUTED
    assert _judge(prediction, _summary(0.04, -0.02, 0, 0.01, 0.06, 0.07, 0.08)) == INCONCLUSIVE
    assert _judge(prediction, {"interval": None, "estimate": -1}) == INCONCLUSIVE


@pytest.mark.parametrize(("registered", "valid", "minimum"), [(12, 6, 6), (24, 12, 12), (20, 10, 10)])
def test_support_threshold_and_all_clamped_quantiles(registered: int, valid: int, minimum: int) -> None:
    parent = np.zeros(2000)
    candidate = np.linspace(0.5, 4.5, 2000)
    quantiles = (0.025, 0.05, 0.1, 0.90, 0.95, 0.975)
    names = ("p025", "p05", "p10", "p90", "p95", "p975")
    supported = _summarize_draws(parent, candidate, point_parent=0.0, point_candidate=2.0,
                                 point_delta=2.0, paired_days=valid, registered_days=registered)
    assert [supported[key] for key in names] == pytest.approx(np.quantile(candidate, quantiles))
    failed = _summarize_draws(parent, candidate, point_parent=0.0, point_candidate=2.0,
                              point_delta=2.0, paired_days=minimum - 1, registered_days=registered)
    assert all(failed[key] is None for key in names)
    assert failed["reason"] == f"fewer than {minimum} paired trading days"


def test_m5_prompt_and_validation_enforce_registered_minimum_magnitude() -> None:
    guard = StructuredPrediction("val_nll_total", "equivalent", (-0.05, 0.05),
                                 role="guardrail", equivalence_band=(-0.05, 0.05))
    for metric, minimum in M5_MIN_EFFECTS.items():
        primary = StructuredPrediction(metric, "increase", (minimum / 2, minimum * 2), role="primary")
        if metric == "val_nll_total":
            primary = StructuredPrediction("lob_impact_score", "increase", (0.5, 1), role="primary")
            weak_guard = StructuredPrediction(metric, "decrease", (-1, -minimum / 2), role="guardrail")
            with pytest.raises(ProposalValidationError, match="magnitude"):
                validate_forecasts("metrics", (primary, weak_guard), analysis_version="m5")
            continue
        context = forecast_targets("evidence", (metric,), analysis_version="m5")
        assert f">= {minimum:g}" in context
        with pytest.raises(ProposalValidationError, match="magnitude"):
            validate_forecasts("evidence", (primary, guard), (metric,), analysis_version="m5")
        primary = replace(primary, interval=(minimum, minimum * 2))
        validate_forecasts("evidence", (primary, guard), (metric,), analysis_version="m5")


def test_m5_promotion_uses_p10_and_preserves_legacy_p05_decision() -> None:
    days = tuple(range(1, 25))
    seeds = (103, 137, 211)
    parent = {day: {IMPACT: 30.0, NLL: 8.0, "free_abs_drift_ticks": 2.0} for day in days}
    candidate = {}
    for day, row in parent.items():
        delta = -4.5 if day <= 8 else 4.0
        candidate[day] = {**row, IMPACT: 30.0 + delta}
    evidence = {"exhausted_windows": 0, "train_seed_nll": {str(seed): 8.0 for seed in seeds}}
    pair = PairedEvaluation(parent, candidate, evidence, evidence)
    request = EvaluationRequest("fixture", "evidence_gpt", ModelRef("p", "parent"),
                                ModelRef("c", "candidate"), (-1,), days, seeds, 12300, "dev", "m5")
    primary = StructuredPrediction(IMPACT, "increase", (0.5, 1.0), role="primary")
    guard = StructuredPrediction(NLL, "equivalent", (-0.05, 0.05), role="guardrail",
                                 equivalence_band=(-0.05, 0.05))
    result = assess(pair, request, (primary, guard))
    assert result["verdict"] == VERIFIED_GAIN
    assert result["p05_delta_score"] == -0.25
    assert result["p10_delta_score"] == pytest.approx(0.1041666667)
    assert result["prediction_verdicts"][0]["status"] == "SUPPORTED"
    assert not result["prediction_verdicts"][0]["point_in_interval"]
    assert assess(pair, replace(request, analysis_version="m4"), (primary, guard))["verdict"] == REFUTED
