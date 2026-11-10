from __future__ import annotations

import pytest

from behavior_index import BehaviorIndex
from behavior_profiler import BehaviorProfiler
from island_scheduler import IslandScheduler
from matched_control import ControlPair, MatchedControlBuilder


BOUNDARIES = {
    "performance": [-0.01, 0.0, 0.01, 0.03],
    "tail_risk": [0.005, 0.01, 0.02],
}


def _profile(improvement: float):
    profiler = BehaviorProfiler(
        {"i0": 1.0, "i1": 1.0}, "a" * 64
    )
    return profiler.build_profile(
        "b" * 64,
        {"i0": 1.0 + improvement, "i1": 1.0 + improvement},
        {"i0": 0.2, "i1": 0.2},
        1.0,
        100,
        2.0,
        10,
    )


def test_atlas_archive_keeps_mechanism_and_regime_and_direction():
    index = BehaviorIndex(BOUNDARIES)
    profile = _profile(0.02)
    assert index.assign_atlas_cell(
        profile, mechanism_id="residual", regime="high_vol"
    ) == ("residual", (3, 0), "high_vol")

    entries = [
        {"record_id": "high", "score": 0.2, "status": "ok", "profile": profile,
         "mechanism_id": "residual", "regime": "high_vol"},
        {"record_id": "low", "score": 0.1, "status": "ok", "profile": profile,
         "mechanism_id": "residual", "regime": "high_vol"},
    ]
    archive = index.quality_diversity_archive(entries, direction="max")
    assert archive[next(iter(archive))]["record_id"] == "high"
    archive = index.quality_diversity_archive(entries, direction="min")
    assert archive[next(iter(archive))]["record_id"] == "low"


def test_island_seed_records_are_members_and_min_review(tmp_path):
    scheduler = IslandScheduler(
        tmp_path / "islands.json", num_islands=2, direction="min"
    )
    epochs = scheduler.initialize(["seed"], context_round=0, base_proposal_seed=1)
    assert all(scheduler.get_island_records(
        f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
    ) == ["seed"] for epoch in epochs)
    for index, epoch in enumerate(epochs):
        scheduler.assign_candidate(
            f"{epoch.island_id}_epoch_{epoch.epoch:02d}", f"r{index}"
        )
    replacements = scheduler.run_review(
        10, {"r0": 0.4, "r1": 0.1}, direction="min"
    )
    assert set(replacements) == {"island_00_epoch_00"}
    new_id = replacements["island_00_epoch_00"]
    assert scheduler.get_island_records(new_id) == ["r1"]


def test_matched_control_verdict_is_the_mc_se_paired_effect():
    from bermudan_research import paired_slice_effects

    def metrics(values):
        return {"summaries": [
            {"instance_id": instance, "repeat": repeat,
             "paired_normalized_improvement": value,
             "paired_normalized_standard_error": 0.001}
            for (instance, repeat), value in zip([("a", 0), ("a", 1), ("b", 0)], values)
        ]}

    pair = ControlPair(
        hypothesis_id="h",
        guided_prompt_suffix="g",
        control_prompt_suffix="c",
        shared_parent_id="p",
        shared_seed=7,
        guided_score=0.2,
        control_score=0.1,
        baseline_score=0.0,
    )
    # Heterogeneous cells: a cross-cell t-test (df=2) would be inconclusive,
    # while the per-cell Monte Carlo SEs make the paired effect decisive.
    effect = paired_slice_effects(metrics([0.10, 0.20, 0.30]), metrics([0.0, 0.15, 0.10]))
    result = MatchedControlBuilder.evaluate_pair(pair, paired_effect=effect)
    assert effect["prediction_verdict"] == "supported"
    assert result.verdict == "transfer_supported"
    assert result.paired_cell_count == 3
    assert result.transfer_gain == pytest.approx(effect["effect"])
    assert result.transfer_gain == pytest.approx((0.10 + 0.05 + 0.20) / 3)
    assert result.transfer_gain_standard_error == pytest.approx(effect["standard_error"])
    assert [result.transfer_gain_ci_low, result.transfer_gain_ci_high] == pytest.approx(effect["ci95"])
    assert result.relative_transfer_gain == pytest.approx(result.transfer_gain / 0.1)
    assert result.control_valid is True


def test_matched_control_marks_identity_mismatch():
    pair = ControlPair(
        hypothesis_id="h",
        guided_prompt_suffix="g",
        control_prompt_suffix="c",
        shared_parent_id="p",
        shared_seed=7,
        guided_score=0.2,
        control_score=0.1,
        baseline_score=0.0,
        guided_parent_id="p",
        control_parent_id="other",
    )
    result = MatchedControlBuilder.evaluate_pair(pair)
    assert result.verdict == "invalid_control"
    assert "parent_mismatch" in (result.invalid_control_reason or "")
    assert result.control_valid is False
