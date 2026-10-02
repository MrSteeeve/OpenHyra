"""Tests for harness_v5.V5Bridge with real dependencies (no mocks for core types).

See also test_v5_vertical_loop.py for the full vertical-loop integration tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from harness_v5 import V5Bridge
from schemas_v5 import AnalogyHypothesis, AnalogyResult, ExperimentEvent, IslandEpoch


def _bridge(tmp_path: Path, num_islands: int = 4) -> V5Bridge:
    return V5Bridge(tmp_path / "run", num_islands=num_islands)


def _initialize(bridge: V5Bridge) -> list[IslandEpoch]:
    return bridge.initialize(["seed-0", "seed-1", "seed-2", "seed-3"])


def _evaluate(bridge: V5Bridge, record_id: str, island_epoch_id: str, metrics=None):
    bridge.on_candidate_evaluated(
        record_id=record_id,
        island_epoch_id=island_epoch_id,
        score=0.25,
        status="ok",
        description="minimal candidate",
        parent_ids=["seed-0"],
        metrics=metrics or {},
    )


def test_initialize_creates_islands(tmp_path):
    bridge = _bridge(tmp_path)
    epochs = _initialize(bridge)
    assert len(epochs) == 4
    for epoch in epochs:
        assert isinstance(epoch, IslandEpoch)
        assert epoch.status == "active"


def test_initialize_idempotent(tmp_path):
    bridge = _bridge(tmp_path)
    first = _initialize(bridge)
    second = _initialize(bridge)
    assert len(second) == len(first)


def test_pick_island_deterministic(tmp_path):
    bridge = _bridge(tmp_path)
    _initialize(bridge)
    assert bridge.pick_island(7) == bridge.pick_island(7)


def test_on_candidate_evaluated_writes_event(tmp_path):
    bridge = _bridge(tmp_path)
    _initialize(bridge)
    island_epoch_id = bridge.pick_island(0)
    _evaluate(bridge, "candidate-event", island_epoch_id)
    events = bridge.event_store.read_experiment_events()
    assert len(events) == 1
    assert events[0].record_id == "candidate-event"
    assert isinstance(events[0], ExperimentEvent)


def test_on_candidate_evaluated_assigns_to_island(tmp_path):
    bridge = _bridge(tmp_path)
    _initialize(bridge)
    island_epoch_id = bridge.pick_island(1)
    _evaluate(bridge, "candidate-island", island_epoch_id)
    assert "candidate-island" in bridge.island_scheduler.get_island_records(island_epoch_id)


def test_on_candidate_evaluated_builds_card(tmp_path):
    bridge = _bridge(tmp_path)
    _initialize(bridge)
    island_epoch_id = bridge.pick_island(2)
    _evaluate(bridge, "candidate-card", island_epoch_id)
    assert "candidate-card" in bridge._cards
    card = bridge._cards["candidate-card"]
    assert card.record_id == "candidate-card"


def test_on_candidate_evaluated_builds_profile(tmp_path):
    baseline = {"inst_0": 0.5, "inst_1": 0.4}
    bridge = _bridge(tmp_path)
    bridge.initialize(
        ["seed-0"],
        baseline_scores=baseline,
        probe_suite_sha256="a" * 64,
    )
    island_epoch_id = bridge.pick_island(3)
    metrics = {
        "per_instance_results": {"inst_0": 0.6, "inst_1": 0.5},
        "per_instance_scores": {"inst_0": 0.6, "inst_1": 0.5},
        "per_instance_exercise_rates": {"inst_0": 0.3, "inst_1": 0.4},
        "artifact_sha256": "b" * 64,
        "training_seconds": 120.0,
        "peak_memory_bytes": 1024,
        "inference_us": 50.0,
        "parameter_count": 100,
    }
    _evaluate(bridge, "candidate-profile", island_epoch_id, metrics)
    assert "candidate-profile" in bridge._profiles


def test_on_context_complete_no_review(tmp_path):
    bridge = _bridge(tmp_path)
    _initialize(bridge)
    assert bridge.on_context_complete(context_round=5) == {}


def test_on_context_complete_triggers_review(tmp_path):
    bridge = _bridge(tmp_path)
    epochs = _initialize(bridge)
    for index, epoch in enumerate(epochs):
        island_epoch_id = f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
        _evaluate(bridge, f"candidate-{index}", island_epoch_id)

    replacements = bridge.on_context_complete(context_round=10)
    assert isinstance(replacements, dict)


def test_build_context_returns_portfolio(tmp_path):
    bridge = _bridge(tmp_path)
    _initialize(bridge)
    context = bridge.build_context()
    assert "portfolio_text" in context
    assert isinstance(context["portfolio_text"], str)
    assert len(context["portfolio_text"]) > 0


def _hypothesis(hypothesis_id: str = "hyp-1") -> AnalogyHypothesis:
    return AnalogyHypothesis(
        id=hypothesis_id,
        source_record_ids=["seed-0"],
        target_parent_id="seed-1",
        relation_mapping=[{"source_role": "features", "target_role": "policy", "shared_relation": "smooth_boundary"}],
        non_correspondence=["different_instance_regime"],
        transferable_intervention="add a normalized continuation feature",
        predicted_effect={"metric": "aggregate_score", "direction": "positive", "minimum_effect": 0.01},
        falsifier="score does not improve on the held-out slice",
        matched_control={"strategy": "same_parent_different_seed"},
        status="preregistered",
    )


def test_hypotheses_are_registered_and_retrieved(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.record_hypothesis(_hypothesis())
    # Re-registration is intentionally idempotent.
    bridge.record_hypothesis(_hypothesis())
    context = bridge.build_context()
    assert len(bridge.hypotheses) == 1
    assert "hyp-1" in context["portfolio_text"]
    resumed = _bridge(tmp_path)
    assert [item.id for item in resumed.hypotheses] == ["hyp-1"]


def test_analogy_result_updates_graph_and_is_idempotent(tmp_path):
    bridge = _bridge(tmp_path)
    hypothesis = _hypothesis()
    bridge.record_hypothesis(hypothesis)
    result = AnalogyResult(
        analogy_hypothesis_id=hypothesis.id,
        guided_record_id="guided-1",
        control_record_id="control-1",
        guided_delta=0.04,
        control_delta=0.01,
        transfer_gain=0.03,
        transfer_gain_standard_error=0.01,
        predicted_slice_effect=0.02,
        prediction_direction_correct=True,
        verdict="transfer_supported",
    )
    bridge.record_analogy_result(result)
    bridge.record_analogy_result(result)
    assert len(bridge.event_store.read_analogy_results()) == 1
    assert bridge.analogy_graph.get_edges_by_type("transfer_supported")
    # The preregistered hypothesis remains immutable, but it is no longer
    # offered as pending once an outcome has been recorded.
    context = bridge.build_context()
    assert context["portfolio"].pending_analogy_pairs == []
    completed = context["portfolio"].completed_analogy_results
    assert len(completed) == 1
    assert completed[0]["analogy_hypothesis_id"] == hypothesis.id
    assert completed[0]["guided_record_id"] == "guided-1"
    assert completed[0]["control_record_id"] == "control-1"
    assert completed[0]["verdict"] == "transfer_supported"
    assert completed[0]["transfer_gain"] == 0.03
    assert completed[0]["matched_arm"] == "guided+control"
    assert completed[0]["mechanism_id"] == hypothesis.id
    assert completed[0]["family"] == "features"
    assert completed[0]["generation_operator"] == "local_mutation"
    assert "guided-1" in context["portfolio_text"]
