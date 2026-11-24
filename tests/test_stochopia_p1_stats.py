"""Check M4 promotion rules and day-clustered nonlinear score inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from experiments.p1_m4_market_stats import MarketClusters

from tasks.stochopia_p1.paired_stats import (
    BOOTSTRAP_RESAMPLES,
    GUARDRAIL_VIOLATION,
    INCONCLUSIVE,
    REFUTED,
    UNVERIFIED_GAIN,
    VERIFIED_GAIN,
    _bootstrap_bounds,
    assess,
    paired_readouts,
)
from tasks.stochopia_p1.protocol import StructuredPrediction
from tasks.stochopia_p1.store import IMPACT, NLL, EvaluationRequest, ModelRef, PairedEvaluation


DAYS = (20170307, 20170308)
SEEDS = (103, 137, 211)
DRIFT = "free_abs_drift_ticks"


def _request(group: str = "evidence_gpt") -> EvaluationRequest:
    return EvaluationRequest(
        "trial", group, ModelRef("parent", "parent source"), ModelRef("candidate", "candidate source"),
        (20170103,), DAYS, SEEDS, 12300, "dev",
    )


def _paired(
    *, deltas: Sequence[float] = (1.0, 1.0), nll_deltas: Sequence[float] = (-0.015, 0.025),
    parent_drift: float = 2.0, candidate_drift: float = 2.0, exhausted: int = 1,
    seed_nll: float | None = 8.0,
) -> PairedEvaluation:
    parent_rows = {}
    candidate_rows = {}
    for index, day in enumerate(DAYS):
        # Shared shifts depend on the day, so independent-arm bootstrap would be wrong.
        score = 36.0 + 10 * index
        nll = 8.0 + index
        parent_rows[day] = {NLL: nll, IMPACT: score, DRIFT: parent_drift}
        candidate_rows[day] = {NLL: nll + nll_deltas[index], IMPACT: score + deltas[index], DRIFT: candidate_drift}
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
    for deltas in ((0.49, 0.49), (-0.2, 1.3), (0.0, 1.0)):
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
    parent_ratios = np.array([0.60, 1.00])
    candidate_ratios = np.array([0.50, 0.74])
    pair = PairedEvaluation(
        base.parent_rows, base.candidate_rows, base.parent_evidence, base.candidate_evidence,
        parent_market=_market(parent_ratios), candidate_market=_market(candidate_ratios),
    )
    readouts = paired_readouts(pair, request, (IMPACT, "end_over_peak_ratio"))
    score = readouts[IMPACT]
    draws = np.random.default_rng(request.rollout_seed).integers(0, 2, size=(BOOTSTRAP_RESAMPLES, 2))
    parent = _score(parent_ratios[draws].mean(axis=1))
    candidate = _score(candidate_ratios[draws].mean(axis=1))
    assert score["estimate"] == pytest.approx(_score(candidate_ratios.mean()) - _score(parent_ratios.mean()))
    assert score["estimate"] != pytest.approx((_score(candidate_ratios) - _score(parent_ratios)).mean())
    assert score["interval"] == pytest.approx(np.quantile(candidate - parent, [0.025, 0.975]))
    assert score["p05"] == pytest.approx(np.quantile(candidate - parent, 0.05))
    assert score["parent_interval"] == pytest.approx(np.quantile(parent, [0.025, 0.975]))
    assert score["candidate_interval"] == pytest.approx(np.quantile(candidate, [0.025, 0.975]))
    assert score["bootstrap_valid"] == BOOTSTRAP_RESAMPLES
    assert score["bootstrap_support"]["parent"] == {
        "incomplete_execution_dates": [], "incomplete_recovery_dates": [],
        "undefined_draw_count": 0, "undefined_draw_indices": [],
    }
    assert readouts["end_over_peak_ratio"]["estimate"] == pytest.approx(-0.18)


def test_undefined_score_draws_cannot_fake_positive_lower_bound() -> None:
    draws = np.ones(2000)
    draws[:101] = np.nan
    interval, p05, valid = _bootstrap_bounds(draws, estimate=1.0, bounds=(-100.0, 100.0))
    assert interval == [-100.0, 100.0]
    assert p05 == -100.0
    assert valid == 1899
    assert _bootstrap_bounds(draws, estimate=1.0) == (None, None, 1899)


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
