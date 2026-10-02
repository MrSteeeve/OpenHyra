from __future__ import annotations

import json

import numpy as np

from feedback import (
    DIRECTIONAL_FEEDBACK_SCHEMA,
    FEEDBACK_PACKET_SCHEMA,
    NOT_OBSERVED,
    BeliefReducer,
    DirectionalFeedback,
    FeedbackPacket,
    ProblemState,
    ProblemStateLog,
    is_not_observed,
    not_observed,
    render_feedback_context,
)
from tasks.bermudan_optimal_stopping import evaluator
from harness_v5 import V5Bridge


def _packet(packet_id: str, values: list[float], *, recommendation: dict | None = None) -> FeedbackPacket:
    return FeedbackPacket(
        packet_id=packet_id,
        candidate_id="candidate-a",
        mechanism_id="normalization",
        observed={"status": "observed"},
        recommendation=recommendation or {"action": "switch"},
        evidence={"source": "trusted_evaluator", "record_ids": [packet_id]},
        probe={"probe_version": "probe.v1"},
        data={"split": "public", "suite_id": "suite-a"},
        directional=[
            DirectionalFeedback(
                id=f"{packet_id}-item",
                candidate_id="candidate-a",
                mechanism_id="normalization",
                slice_key="high-vol",
                direction="positive",
                observed={"samples": values},
                # This field must never be consumed as a numeric observation.
                recommendation={"action": "tune", "scope": "parameter"},
                evidence={"record_ids": [packet_id]},
                probe={"probe_version": "probe.v1"},
                data={"split": "public"},
            )
        ],
    )


def test_feedback_packet_roundtrip_keeps_observed_and_recommendation_separate() -> None:
    packet = _packet("p-1", [0.1, 0.2])
    restored = FeedbackPacket.from_dict(packet.to_dict())
    assert restored.schema == FEEDBACK_PACKET_SCHEMA
    assert restored.directional[0].schema == DIRECTIONAL_FEEDBACK_SCHEMA
    assert restored.observed == {"status": "observed"}
    assert restored.recommendation["action"] == "switch"
    assert restored.directional[0].recommendation["scope"] == "parameter"
    assert restored.to_dict() == packet.to_dict()


def test_not_observed_marker_is_explicit_and_not_reduced() -> None:
    packet = FeedbackPacket(
        packet_id="p-missing",
        mechanism_id="m",
        observed={"delta": not_observed("probe unavailable")},
        recommendation={"action": "probe"},
        data={"split": "public"},
    )
    assert is_not_observed(packet.observed["delta"])
    state = BeliefReducer().reduce([packet])
    assert state.cells == {}


def test_belief_reducer_is_order_independent_and_ignores_recommendations() -> None:
    first = _packet("p-1", [1.0, 2.0], recommendation={"action": "bad"})
    second = _packet("p-2", [3.0], recommendation={"action": "worse"})
    reducer = BeliefReducer(confidence_level=0.95, min_observations=2)
    left = reducer.rebuild([first, second], state_id="state-a")
    right = reducer.rebuild([second, first], state_id="state-a")
    assert left.to_dict() == right.to_dict()
    cell = left.get_cell("normalization", "high-vol")
    assert cell is not None
    assert cell.n == 3
    assert np.isclose(cell.mean, 2.0)
    assert np.isclose(cell.variance, 1.0)
    assert np.isclose(cell.se, 1.0 / np.sqrt(3.0))
    assert 0.0 <= cell.p_positive <= 1.0
    assert cell.status == "promising"


def test_problem_state_append_is_immutable_and_idempotent() -> None:
    reducer = BeliefReducer()
    initial = reducer.rebuild([], state_id="state-a")
    once = reducer.append(initial, _packet("p-1", [1.0]))
    twice = reducer.append(once, _packet("p-1", [1.0]))
    assert initial.state_version == 0
    assert once.state_version == 1
    assert twice.to_dict() == once.to_dict()
    assert once.applied_packet_ids == ("p-1",)
    assert once.state_hash == once.hash
    assert once.state_hash == BeliefReducer().rebuild(
        [_packet("p-1", [1.0])], state_id="state-a"
    ).state_hash


def test_problem_state_log_rebuilds_append_only_packets(tmp_path) -> None:
    log = ProblemStateLog(tmp_path / "feedback.jsonl")
    log.append(_packet("p-2", [2.0]))
    log.append(_packet("p-1", [1.0]))
    restored = log.rebuild(BeliefReducer(min_observations=1), state_id="logged")
    assert restored.state_id == "logged"
    assert restored.applied_packet_ids == ("p-1", "p-2")
    assert restored.get_cell("normalization", "high-vol").mean == 1.5
    assert len(log.read()) == 2
    first_line = (tmp_path / "feedback.jsonl").read_text().splitlines()[0]
    assert json.loads(first_line)["schema"] == FEEDBACK_PACKET_SCHEMA


def test_feedback_context_projection_excludes_private_packets() -> None:
    public = _packet("public", [0.2])
    private = FeedbackPacket(
        packet_id="private",
        mechanism_id="hidden",
        observed={"effect": 9.0},
        data={"split": "private"},
    )
    text = render_feedback_context([public, private])
    assert "public" in text
    assert "private" not in text
    assert "normalization" in text


def _request(stage: str) -> dict:
    config = {
        "instance_count": 1,
        "repeats": 1,
        "training_paths": 64,
        "pricing_paths": 128,
        "ridge_alpha": 1e-6,
    }
    if stage == "audit":
        config.update({"outer_paths": 64, "inner_paths": 2})
    return {
        "schema": evaluator.REQUEST_SCHEMA,
        "stage": stage,
        "task": evaluator.TASK_NAME,
        "protocol": evaluator.TASK_PROTOCOL,
        "seed": 17,
        "suite_id": f"feedback-{stage}",
        "config": config,
    }


def test_bermudan_search_emits_domain_feedback_without_score_change() -> None:
    request = _request("search")
    score, metrics, _, evidence = evaluator.evaluate_submission(
        evaluator.BASELINE_PROGRAM, request
    )
    assert np.isclose(score, metrics["search_score"])
    packet = FeedbackPacket.from_dict(metrics["feedback_packet"])
    assert packet.data["split"] == "public"
    assert packet.candidate_id == metrics["candidate_hash"]
    assert packet.observed["runtime_seconds"]["status"] == NOT_OBSERVED
    assert metrics["runtime_seconds"] > 0.0
    assert packet.observed["tail_risk"]["loss_definition"] == (
        "negative_paired_normalized_improvement"
    )
    assert packet.observed["tail_risk"]["var95_by_cell"]
    assert evidence["search"]["feedback_packet_id"] == packet.packet_id
    assert packet.directional
    assert packet.directional[0].observed["effect"] != NOT_OBSERVED


def test_bermudan_audit_marks_unrun_search_geometry_unobserved() -> None:
    request = _request("audit")
    _score, metrics, _, _evidence = evaluator.evaluate_submission(
        evaluator.BASELINE_PROGRAM, request
    )
    packet = FeedbackPacket.from_dict(metrics["feedback_packet"])
    assert packet.data["split"] == "private"
    assert packet.observed["independent_reproduction"]["status"] == NOT_OBSERVED
    assert packet.directional[0].observed["candidate_exercise_rate"]["status"] == NOT_OBSERVED


def test_v5_bridge_persists_public_state_but_hides_private_packets(tmp_path) -> None:
    bridge = V5Bridge(tmp_path / "run")
    bridge.initialize(["seed-0"])
    public = _packet("public-1", [0.4])
    private = FeedbackPacket(
        packet_id="private-1",
        mechanism_id="normalization",
        observed={"delta": 9.0},
        data={"split": "private"},
    )
    bridge.append_feedback_packet(public)
    bridge.append_feedback_packet(private)
    state = bridge.get_problem_state()
    assert state.get_cell("normalization", "high-vol") is not None
    assert bridge.get_problem_state(include_private=True).get_cell(
        "normalization", "global"
    ) is not None
    assert [item.packet_id for item in bridge.read_feedback_packets()] == [
        "public-1"
    ]
    resumed = V5Bridge(tmp_path / "run")
    assert resumed.get_problem_state().to_dict() == state.to_dict()
    assert resumed.get_problem_state(include_private=True).get_cell(
        "normalization", "global"
    ) is not None


def test_v5_bridge_candidate_hook_records_feedback_sidecar(tmp_path) -> None:
    bridge = V5Bridge(tmp_path / "run")
    epoch = bridge.initialize(["seed-0"])[0]
    packet = _packet("hook-packet", [0.2])
    bridge.on_candidate_evaluated(
        record_id="candidate-1",
        island_epoch_id=f"{epoch.island_id}_epoch_{epoch.epoch:02d}",
        score=0.2,
        status="ok",
        parent_ids=["seed-0"],
        metrics={
            "score_metric": "paired_lower_bound_lcb",
            "artifact_sha256": "a" * 64,
            "feedback_packet": packet.to_dict(),
        },
    )
    assert [item.packet_id for item in bridge.read_feedback_packets()] == [
        "hook-packet"
    ]
    event = bridge.event_store.read_experiment_events()[0]
    assert event.feedback_packet_ref
    assert event.feedback_packet_schema == FEEDBACK_PACKET_SCHEMA
