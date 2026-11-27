"""Check M4 promotion rules and day-clustered nonlinear score inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest

from stochopia.market.szse_eval import MarketClusters

from tasks.lob_world_model.paired_stats import (
    BOOTSTRAP_RESAMPLES,
    GUARDRAIL_VIOLATION,
    INCONCLUSIVE,
    REFUTED,
    UNVERIFIED_GAIN,
    VERIFIED_GAIN,
    _judge as _judge_quantile,
    _judge_v2 as _judge,
    _prediction_verdict,
    _summarize_draws,
    assess,
    paired_readouts,
)
from tasks.lob_world_model.contracts import StructuredPrediction
from tasks.lob_world_model.store import IMPACT, NLL, EvaluationRequest, ModelRef, PairedEvaluation


DAYS = (20170307, 20170308, 20170309, 20170310, 20170313, 20170314)
SEEDS = (103, 137, 211)
DRIFT = "free_abs_drift_ticks"


def _request(group: str = "evidence_gpt", days: tuple[int, ...] = DAYS) -> EvaluationRequest:
    return EvaluationRequest(
        "trial", group, ModelRef("parent", "parent source"), ModelRef("candidate", "candidate source"),
        (20170103,), days, SEEDS, 12300, "dev",
    )


def _paired(
    *, deltas: Sequence[float] = (1.0, 1.0), nll_deltas: Sequence[float] = (-0.015, 0.025),
    parent_drift: float = 2.0, candidate_drift: float = 2.0, exhausted: int = 1,
    seed_nll: float | None = 8.0, days: tuple[int, ...] = DAYS,
) -> PairedEvaluation:
    parent_rows = {}
    candidate_rows = {}
    for index, day in enumerate(days):
        # Shared shifts depend on the day, so independent-arm bootstrap would be wrong.
        score = 36.0 + 10 * index
        nll = 8.0 + index
        parent_rows[day] = {NLL: nll, IMPACT: score, DRIFT: parent_drift}
        candidate_rows[day] = {
            NLL: nll + nll_deltas[index % len(nll_deltas)],
            IMPACT: score + deltas[index % len(deltas)],
            DRIFT: candidate_drift,
        }
    parent = {"train_seed_nll": {str(seed): 8.0 for seed in SEEDS}, "exhausted_windows": 1}
    child = {"train_seed_nll": {"103": 8.0, "137": 8.0, "211": seed_nll}, "exhausted_windows": exhausted}
    return PairedEvaluation(parent_rows, candidate_rows, parent, child)


def _predictions(*, primary: tuple[float, float] = (0.75, 1.25)) -> tuple[StructuredPrediction, ...]:
    return (
        StructuredPrediction(IMPACT, "increase", primary, role="primary"),
        StructuredPrediction(NLL, "equivalent", (-0.01, 0.01), role="guardrail", equivalence_band=(-0.01, 0.01)),
    )


def test_supported_primary_and_inconclusive_guardrail_can_promote() -> None:
    assessment = assess(_paired(), _request(), _predictions())
    assert assessment["verdict"] == VERIFIED_GAIN
    assert [p["status"] for p in assessment["prediction_verdicts"]] == ["SUPPORTED", INCONCLUSIVE]
    assert assessment["readouts"][IMPACT]["interval"] == [1.0, 1.0]
    assert assessment["p05_delta_score"] == 1.0


@pytest.mark.parametrize(
    ("pair", "failed_guard"),
    [
        (_paired(nll_deltas=(0.06, 0.06)), "nll_passed"),
        (_paired(seed_nll=None), "nll_passed"),
        (_paired(exhausted=2), "exhaustion_passed"),
        (_paired(candidate_drift=20.0), "drift_passed"),
        (_paired(parent_drift=21.0, candidate_drift=21.0), "drift_passed"),
    ],
)
def test_second_layer_vetoes_any_guardrail_failure(pair: PairedEvaluation, failed_guard: str) -> None:
    assessment = assess(pair, _request(), _predictions())
    assert assessment["verdict"] == GUARDRAIL_VIOLATION
    assert assessment["guardrails"][failed_guard] is False
    assert assessment["has_benchmark_gain"] is False


def test_high_parent_drift_requires_strict_improvement_and_nll_boundary_is_inclusive() -> None:
    pair = _paired(parent_drift=25.0, candidate_drift=24.0, nll_deltas=(0.05, 0.05))
    assessment = assess(pair, _request(), _predictions())
    assert assessment["guardrails"]["nll_passed"]
    assert assessment["guardrails"]["drift_passed"]
    assert assessment["verdict"] == UNVERIFIED_GAIN
    assert assessment["prediction_verdicts"][1]["status"] == REFUTED


def test_gain_requires_both_magnitude_and_positive_p05() -> None:
    for deltas in ((0.49, 0.49), (-1.2, 2.3), (0.0, 0.0, 0.0, 0.0, 0.0, 3.0)):
        result = assess(_paired(deltas=deltas), _request(), _predictions())
        assert result["verdict"] == REFUTED
        assert result["has_benchmark_gain"] is False
    result = assess(_paired(deltas=(0.50, 0.50)), _request(), _predictions(primary=(0.49, 0.51)))
    assert result["verdict"] == VERIFIED_GAIN


def test_no_llm_climbs_with_identical_guards_and_gain_without_forecast_filter() -> None:
    forecasts = _predictions(primary=(2.0, 3.0))
    pair = _paired(nll_deltas=(0.025, 0.025))
    assert assess(pair, _request(), forecasts)["verdict"] == UNVERIFIED_GAIN
    result = assess(pair, _request("no_llm"), forecasts)
    assert result["verdict"] == VERIFIED_GAIN
    assert [p["status"] for p in result["prediction_verdicts"]] == [REFUTED, REFUTED]
    assert assess(_paired(exhausted=2), _request("no_llm"), forecasts)["verdict"] == GUARDRAIL_VIOLATION


def _market(ratios: Sequence[float]) -> dict[str, object]:
    peak = 0.75
    marks = peak * np.sqrt(np.arange(1, 4) / 3)
    child = [np.stack((marks, -marks)).tolist() for _ in ratios]
    path = []
    for ratio in ratios:
        post = np.linspace(peak, peak * ratio, 4)[1:]
        row = np.r_[0.0, marks, post]
        path.append(np.stack((row, -row)).tolist())
    tape = {"events": [[10, 10]] * len(ratios), "drift_ticks": [[0.0, 0.0]] * len(ratios),
            "flow_acf": [[(1 / np.sqrt(np.arange(1, 9))).tolist()] * 2] * len(ratios),
            "price_msd": [[[1.0, 2.0, 4.0, 8.0]] * 2] * len(ratios)}
    return {
        "visible_mid": path,
        "blind_mid": (np.asarray(path) * 0.25).tolist(),
        "child_visible": child,
        "child_blind": (np.asarray(child) * 0.25).tolist(),
        "child_filled": np.ones((len(ratios), 2, 3)).tolist(),
        "paths_per_day": 2,
        "sigma_ticks": 1.0,
        "daily_shares": 3.0,
        "schedule": [False, True, True, True, False, False, False],
        "generated_tapes": tape,
        "historical_tapes": tape,
        "l1_depth_errors": [0.0] * len(ratios),
    }


def _score(ratio: np.ndarray | float) -> np.ndarray | float:
    # Amplitude and exponent are strictly in band in this construction.
    error = np.maximum.reduce([0.5 - ratio, np.zeros_like(ratio), ratio - 2 / 3]) / (1 / 3)
    return 100 / (1 + 0.35 * error)


def test_paired_market_bootstrap_rescores_paths_with_shared_days() -> None:
    request = _request()
    base = _paired()
    parent_ratios = np.array([0.60, 1.00] * 3)
    candidate_ratios = np.array([0.50, 0.74] * 3)
    pair = PairedEvaluation(
        base.parent_rows, base.candidate_rows, base.parent_evidence, base.candidate_evidence,
        parent_market=_market(parent_ratios), candidate_market=_market(candidate_ratios),
    )
    readouts = paired_readouts(pair, request, (IMPACT, "end_over_peak_ratio"))
    score = readouts[IMPACT]
    size = len(DAYS)
    draws = np.random.default_rng(request.rollout_seed).integers(0, size, size=(BOOTSTRAP_RESAMPLES, size))
    parent = _score(parent_ratios[draws].mean(axis=1))
    candidate = _score(candidate_ratios[draws].mean(axis=1))
    assert score["estimate"] == pytest.approx(_score(candidate_ratios.mean()) - _score(parent_ratios.mean()))
    assert score["estimate"] != pytest.approx((_score(candidate_ratios) - _score(parent_ratios)).mean())
    assert score["interval"] == pytest.approx(np.quantile(candidate - parent, [0.025, 0.975]))
    assert score["p05"] == pytest.approx(np.quantile(candidate - parent, 0.05))
    assert score["parent_interval"] == pytest.approx(np.quantile(parent, [0.025, 0.975]))
    assert score["candidate_interval"] == pytest.approx(np.quantile(candidate, [0.025, 0.975]))
    assert score["paired_days"] == size
    assert score["bootstrap_valid"] == BOOTSTRAP_RESAMPLES
    assert score["bootstrap_support"]["parent"] == {
        "incomplete_execution_dates": [], "incomplete_recovery_dates": [],
        "undefined_draw_count": 0, "undefined_draw_indices": [],
    }
    assert readouts["end_over_peak_ratio"]["estimate"] == pytest.approx(-0.18)


def test_percentiles_and_verdict_use_1982_jointly_finite_draws() -> None:
    parent = np.linspace(1.0, 5.0, BOOTSTRAP_RESAMPLES)
    candidate = parent + np.linspace(-0.4, -0.2, BOOTSTRAP_RESAMPLES)
    parent[:11] = np.nan
    candidate[-7:] = np.nan
    common = slice(11, -7)
    expected = candidate[common] - parent[common]
    result = _summarize_draws(
        parent, candidate, point_parent=3.0, point_candidate=2.7, point_delta=-0.3,
        paired_days=9, registered_days=12,
    )
    assert result["bootstrap_valid"] == 1982
    assert result["reason"] is None
    assert result["interval"] == pytest.approx(np.quantile(expected, [0.025, 0.975]))
    assert result["p05"] == pytest.approx(np.quantile(expected, 0.05))
    assert result["parent_interval"] == pytest.approx(np.quantile(parent[common], [0.025, 0.975]))
    assert result["candidate_interval"] == pytest.approx(np.quantile(candidate[common], [0.025, 0.975]))
    supported = StructuredPrediction("trade_sign_acf_exp_gap", "decrease", (-0.41, -0.19), role="primary")
    refuted = StructuredPrediction("trade_sign_acf_exp_gap", "decrease", (-1.1, -0.8), role="primary")
    assert _judge(supported, result["interval"]) == "SUPPORTED"
    assert _judge(refuted, result["interval"]) == REFUTED


@pytest.mark.parametrize(("paired_days", "valid_draws", "reason"), [
    (5, 2000, "fewer than 6 paired trading days"),
    (6, 1799, "fewer than 1800 finite paired bootstrap draws"),
    (6, 1800, None),
])
def test_bootstrap_support_thresholds(paired_days: int, valid_draws: int, reason: str | None) -> None:
    parent = np.zeros(BOOTSTRAP_RESAMPLES)
    candidate = np.ones(BOOTSTRAP_RESAMPLES)
    candidate[valid_draws:] = np.nan
    result = _summarize_draws(
        parent, candidate, point_parent=0.0, point_candidate=1.0, point_delta=1.0,
        paired_days=paired_days, registered_days=12,
    )
    assert result["bootstrap_valid"] == valid_draws
    assert result["reason"] == reason
    if reason is None:
        assert result["interval"] == [1.0, 1.0]
        assert result["p05"] == 1.0
    else:
        assert result["interval"] is None
        assert result["p05"] is None
        assert _judge(_predictions()[0], result["interval"]) == INCONCLUSIVE


def test_pooled_support_requires_the_same_dates_for_both_arms() -> None:
    days = (*DAYS, 20170315, 20170316)
    parent = _market([0.6] * len(days))
    candidate = _market([0.7] * len(days))
    for index in (0, 1):
        for side in parent["visible_mid"][index]:
            side[-1] = None
    for index in (2, 3):
        for side in candidate["visible_mid"][index]:
            side[-1] = None
    base = _paired(days=days)
    pair = PairedEvaluation(
        base.parent_rows, base.candidate_rows, base.parent_evidence, base.candidate_evidence,
        parent_market=parent, candidate_market=candidate,
    )
    metric = "end_over_peak_ratio"
    assert MarketClusters.from_payload(parent).day_support(metric).sum() == 6
    assert MarketClusters.from_payload(candidate).day_support(metric).sum() == 6
    result = paired_readouts(pair, _request(days=days), (metric,))[metric]
    assert result["estimate"] == pytest.approx(0.1)
    assert result["paired_days"] == 4
    assert result["bootstrap_valid"] >= 1800
    assert result["interval"] is None


def test_scalar_bootstrap_retains_observed_pairs_on_an_incomplete_panel() -> None:
    days = (*DAYS, 20170315, 20170316)
    parent = {day: {NLL: float(index)} for index, day in enumerate(days)}
    candidate = {day: {NLL: float(index + 1)} for index, day in enumerate(days)}
    parent[days[0]][NLL] = None
    candidate[days[-1]][NLL] = None
    result = paired_readouts(PairedEvaluation(parent, candidate, {}, {}), _request(days=days), (NLL,))[NLL]
    assert result["paired_days"] == 6
    assert result["interval"] == [1.0, 1.0]
    assert result["estimate"] == result["candidate_estimate"] - result["parent_estimate"]


def test_bootstrap_conditions_on_c2_admission_of_original_panel() -> None:
    payload = _market([0.6] * 4)
    scale = np.array([0.05, 1.0, 1.0, 1.0]) / 0.75
    for field in ("visible_mid", "blind_mid", "child_visible", "child_blind"):
        payload[field] = (np.asarray(payload[field]) * scale[:, None, None]).tolist()
    clusters = MarketClusters.from_payload(payload)
    weak_draw = np.array([[0, 0, 1, 1]])
    assert clusters.estimate(np.arange(4))[IMPACT] is not None
    # With two low-strength dates, this draw alone fails the 3-SE C2 threshold.
    assert clusters.estimate(weak_draw[0])[IMPACT] is None
    # Its nonlinear score contributes after full-panel admission.
    samples = clusters.resample(weak_draw, (IMPACT,))[IMPACT]
    assert samples[0] == pytest.approx(100.0)
    # A cohort initially failing C2 stays ineligible across its bootstrap.
    for field in ("visible_mid", "blind_mid", "child_visible", "child_blind"):
        payload[field] = np.asarray(payload[field])[weak_draw[0]].tolist()
    ineligible = MarketClusters.from_payload(payload)
    assert ineligible.estimate(np.arange(4))[IMPACT] is None
    assert np.isnan(ineligible.resample(np.array([[2, 2, 2, 2]]), (IMPACT,))[IMPACT][0])


# --- Directional bootstrap and equivalence semantics ---

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
    assert _judge_quantile(prediction, _summary(0, -1, -0.5, -0.1, 1, 2, 3)) == REFUTED
    assert _judge_quantile(prediction, _summary(0.8, 0.2, 0.3, 0.4, 0.9, 1.1, 1.2)) == INCONCLUSIVE
    assert _judge_quantile(prediction, _summary(0.6, 0.1, 0.2, 0.3, 0.8, 0.9, 0.99)) == REFUTED


def test_negative_direction_uses_the_near_zero_upper_target_bound() -> None:
    prediction = StructuredPrediction("end_over_peak_ratio", "decrease", (-0.5, -0.08), role="primary")
    outcome = _summary(-0.8, -1.0, -0.9, -0.8, -0.02, 0.1, 0.2)
    assert _judge_quantile(prediction, outcome) == "SUPPORTED"
    assert not _prediction_verdict(prediction, outcome, analysis_version="m5")["point_in_interval"]
    assert _judge_quantile(prediction, _summary(-0.04, -0.09, -0.08, -0.07, -0.01, 0.0, 0.02)) == INCONCLUSIVE
    assert _judge_quantile(prediction, _summary(-0.04, -0.07, -0.06, -0.05, -0.01, 0.0, 0.01)) == REFUTED
    assert _judge_quantile(prediction, _summary(0.01, -0.2, -0.1, -0.02, 0.1, 0.2, 0.3)) == REFUTED


def test_equivalence_checks_the_tost_band_and_95_percent_refutation() -> None:
    prediction = StructuredPrediction("val_nll_total", "equivalent", (-0.02, 0.02),
                                      role="guardrail", equivalence_band=(-0.05, 0.05))
    assert _judge_quantile(prediction, _summary(0.01, -0.06, -0.05, -0.04, 0.04, 0.05, 0.06)) == "SUPPORTED"
    assert _judge_quantile(prediction, _summary(0.07, 0.051, 0.055, 0.056, 0.08, 0.09, 0.10)) == REFUTED
    assert _judge_quantile(prediction, _summary(0.04, -0.02, 0, 0.01, 0.06, 0.07, 0.08)) == INCONCLUSIVE
    assert _judge_quantile(prediction, {"interval": None, "estimate": -1}) == INCONCLUSIVE


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
