"""P2 rebootstrap independently adjudicates C2 on shared trading-day draws."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from experiments.p1_m4_market_stats import MarketClusters
from stochopia.measure.impact import P2_METRIC_SCHEMA, reference_impact_curve
from tasks.stochopia_p1.paired_stats import BOOTSTRAP_RESAMPLES, paired_readouts
from tasks.stochopia_p1.store import IMPACT, EvaluationRequest, ModelRef, PairedEvaluation


DAYS = (20170307, 20170308, 20170309, 20170310, 20170313, 20170314)
GRID = np.arange(13)
SCHEDULE = np.isin(GRID, (2, 4, 6, 8))
PARENT_MULTIPLIERS = (-4, -4, 0, 2, 5, 7)
CANDIDATE_MULTIPLIERS = (1, 1, 1, 1, 1, 1)
CONTINUOUS = (
    "impact_curve_mae", "post_stop_relaxation_delta",
    "bounded_relaxation_ratio", "visible_minus_blind_auc",
)


def _payload(multipliers: Sequence[float], replicas: int = 1) -> dict[str, object]:
    reference = reference_impact_curve(GRID, 8, scale_ticks=10.0, participation=0.25, permanent_fraction=0.0)
    visible = np.asarray(multipliers)[:, None] * reference
    paths = np.stack((visible, -visible), axis=1)
    tape = {
        "events": [[10, 10]] * len(multipliers),
        "drift_ticks": [[0.0, 0.0]] * len(multipliers),
        "flow_acf": [[(1 / np.sqrt(np.arange(1, 9))).tolist()] * 2] * len(multipliers),
        "price_msd": [[[1.0, 2.0, 4.0, 8.0]] * 2] * len(multipliers),
    }
    return {
        "visible_mid": paths.tolist(),
        "blind_mid": (paths / 3).tolist(),
        "child_visible": paths[:, :, SCHEDULE].tolist(),
        "child_blind": (paths[:, :, SCHEDULE] / 3).tolist(),
        "child_filled": np.full((len(multipliers), 2, int(SCHEDULE.sum())), 100).tolist(),
        "paths_per_day": 2 * replicas,
        "sigma_ticks": 10.0,
        "daily_shares": 1600.0,
        "schedule": SCHEDULE.tolist(),
        "generated_tapes": tape,
        "historical_tapes": tape,
        "l1_depth_errors": [0.0] * len(multipliers),
        "metric_schema_version": P2_METRIC_SCHEMA,
        "grid_ms": GRID.tolist(),
        "planned_child_shares": 100.0,
        "num_replicates": replicas,
    }


def _request() -> EvaluationRequest:
    return EvaluationRequest(
        "p2", "evidence_gpt", ModelRef("parent", "parent source"), ModelRef("candidate", "candidate source"),
        (20170103,), DAYS, (103, 137, 211), 12300, "dev", "m5",
    )


def test_c2_is_independently_readjudicated_not_original_panel_conditioned() -> None:
    payload = _payload(PARENT_MULTIPLIERS)
    clusters = MarketClusters.from_payload(payload)
    assert clusters.estimate(np.arange(6))["c2_pass_prob"] == 0.0
    draws = np.array([[0] * 6, [3] * 6, list(range(6))])
    samples = clusters.resample(draws, ("c2_pass_prob", *CONTINUOUS, IMPACT))
    np.testing.assert_array_equal(samples["c2_pass_prob"], [0.0, 1.0, 0.0])
    assert np.isfinite(samples[IMPACT][1])
    assert np.isnan(samples[IMPACT][0]) and np.isnan(samples[IMPACT][2])
    for metric in CONTINUOUS:
        assert np.isfinite(samples[metric]).all()


@pytest.mark.parametrize("replicas", (1, 4, 8))
def test_cluster_day_count_is_independent_of_replica_count(replicas: int) -> None:
    clusters = MarketClusters.from_payload(_payload(PARENT_MULTIPLIERS, replicas))
    assert clusters.days == len(DAYS)
    assert len(clusters.day_support("impact_curve_mae")) == len(DAYS)
    assert clusters.day_support("impact_curve_mae").all()


def test_bootstrap_probability_and_continuous_intervals_have_full_support() -> None:
    rows = {day: {} for day in DAYS}
    request = _request()
    pair = PairedEvaluation(
        rows, rows, {}, {},
        _payload(PARENT_MULTIPLIERS),
        _payload(CANDIDATE_MULTIPLIERS),
    )
    readings = paired_readouts(pair, request, (*CONTINUOUS, IMPACT))
    c2 = readings["delta_c2_pass_prob"]
    assert c2["parent_estimate"] == 0.0
    assert c2["candidate_estimate"] == 1.0
    assert c2["estimate"] == 1.0
    assert 0 < c2["parent_bootstrap_mean"] < 1
    assert c2["candidate_bootstrap_mean"] == 1.0
    assert c2["mean"] == pytest.approx(1.0 - c2["parent_bootstrap_mean"])
    assert c2["bootstrap_valid"] == BOOTSTRAP_RESAMPLES
    # Candidate always passes; independent parent admission determines the joint share.
    assert readings["joint_c2_pass_share"] == c2["parent_bootstrap_mean"]
    assert readings[IMPACT]["interval"] is None  # no original-panel eligible parent score
    conditional = readings["delta_lob_impact_score"]
    assert conditional["conditional_score_draws"] > 0
    assert conditional["interval"] is not None
    assert conditional["joint_c2_pass_share"] == readings["joint_c2_pass_share"]
    for metric in CONTINUOUS:
        reading = readings[metric]
        for field in ("estimate", "mean", "p025", "p05", "p10", "p50", "p90", "p95", "p975"):
            assert np.isfinite(reading[field]), (metric, field)
        assert reading["bootstrap_valid"] == BOOTSTRAP_RESAMPLES
        assert reading["registered_days"] == len(DAYS)


def test_paired_reference_cannot_depend_on_candidate() -> None:
    rows = {day: {} for day in DAYS}
    parent = _payload(PARENT_MULTIPLIERS)
    candidate = _payload(CANDIDATE_MULTIPLIERS)
    candidate["planned_child_shares"] = 101.0
    pair = PairedEvaluation(rows, rows, {}, {}, parent, candidate)
    with pytest.raises(ValueError, match="same replica and reference"):
        paired_readouts(pair, _request(), ("impact_curve_mae",))


def _attach_available_panel(
    payload: dict[str, object], missing: tuple[int, int],
) -> None:
    strict = np.asarray(payload["visible_mid"], dtype=float)
    visible = (strict[:, 0] - strict[:, 1]) / 2
    joint = visible * 0.75
    visible_counts = np.full_like(visible, 7.0)
    joint_counts = np.full_like(joint, 6.0)
    execution_peak = visible[:, 8].copy()
    visible[missing] = np.nan
    joint[missing] = np.nan
    visible_counts[missing] = 0
    joint_counts[missing] = 0
    # A strict day is invalid at one child mark even when other replicas
    # supply the available continuous quote estimate.
    strict[:, :, 4] = np.nan
    payload["visible_mid"] = strict.tolist()
    payload["child_visible"] = strict[:, :, SCHEDULE].tolist()
    payload["continuous_panel"] = {
        "visible": visible.tolist(), "joint_difference": joint.tolist(),
        "execution_peak": execution_peak.tolist(), "visible_counts": visible_counts.tolist(),
        "joint_counts": joint_counts.tolist(), "execution_peak_counts": [7] * len(visible),
    }


def test_p2_available_quote_pair_uses_shared_marks_not_strict_curve_support() -> None:
    days = DAYS + (20170315, 20170316)
    before = _payload([1.0] * len(days), replicas=4)
    after = _payload([1.0] * len(days), replicas=4)
    _attach_available_panel(before, (0, 1))
    _attach_available_panel(after, (1, 2))
    rows = {day: {} for day in days}
    request = EvaluationRequest(
        "p2", "evidence_gpt", ModelRef("parent", "parent source"), ModelRef("candidate", "candidate source"),
        (20170103,), days, (103, 137, 211), 12300, "dev", "m5",
    )
    pair = PairedEvaluation(rows, rows, {}, {}, before, after)
    reports = paired_readouts(pair, request, (*CONTINUOUS, IMPACT))
    for metric in CONTINUOUS:
        assert reports[metric]["estimate"] == 0.0
        assert reports[metric]["interval"] == [0.0, 0.0]
        assert reports[metric]["registered_days"] == len(days)
    curve = reports["impact_curve_mae"]
    assert curve["paired_days"] == len(days) - 1
    assert curve["complete_common_days"] == len(days) - 2
    c2 = reports["delta_c2_pass_prob"]
    assert c2["parent_complete_execution_days"] == 0
    assert c2["candidate_complete_execution_days"] == 0
    assert c2["interval"] == [0.0, 0.0]
    assert c2["parent_bootstrap_mean"] == 0.0
    assert c2["candidate_bootstrap_mean"] == 0.0
    assert reports["joint_c2_pass_share"] == 0.0
    assert reports["delta_lob_impact_score"]["interval"] is None
