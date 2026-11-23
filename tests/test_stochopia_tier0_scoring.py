"""Native OpenHyra projections from the published Tier 0 development records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_stochopia_tier0_llm_selector import manifest, run_offline_rehearsal  # noqa: F401

from algorithm_discovery import AlgorithmSpec, EvaluationResult, FeedbackOracle
from feedback import FeedbackPacket, is_not_observed, render_feedback_context
from stochopia.harness.contract import ExperimentManifest
from tasks.stochopia_tier0.entry import Entry
from tasks.stochopia_tier0.scoring import Tier0FeedbackOracle
from tasks.stochopia_tier0.store import DevelopmentStore


def test_native_feedback_is_group_scoped_and_preserves_raw_publication(
    manifest: ExperimentManifest, tmp_path: Path,
) -> None:
    run_offline_rehearsal(tmp_path, manifest)
    store = DevelopmentStore(tmp_path / "runs", manifest.exp_id)
    entry = Entry(store)
    raw = store.feedback(1)
    assert raw is not None
    assert all(set(item) == {"trial", "paired_diffs", "results"} for items in raw.values() for item in items)

    evidence, = entry.native_feedback(1, "evidence")
    metrics, = entry.native_feedback(1, "metrics")
    fixed, = entry.native_feedback(1, "fixed")

    assert isinstance(evidence.candidate, AlgorithmSpec)
    assert isinstance(evidence.result, EvaluationResult)
    assert isinstance(evidence.result.feedback, FeedbackPacket)
    assert isinstance(Tier0FeedbackOracle("evidence"), FeedbackOracle)
    assert evidence.candidate.family == "gru"
    assert evidence.candidate.operator == "separate_flow"
    assert evidence.candidate.mechanism_id
    assert evidence.candidate.parent_ids == ("private-base",)
    assert evidence.candidate.prediction and evidence.candidate.falsifier
    assert evidence.result.split == "development"
    assert evidence.result.score == 1.0
    assert evidence.result.status == "supported"
    assert evidence.result.cost == raw["evidence"][0]["trial"]["cost"]
    assert evidence.result.metrics["prediction_verdicts"][0]["status"] == "supported"
    assert evidence.result.feedback.data["group_id"] == "evidence"
    assert evidence.result.feedback.directional[0].observed["paired_delta"] == 1.0
    context, = json.loads(render_feedback_context([evidence.result.feedback]))
    assert context["primary_metric"] == "end_response"
    assert context["directions"][0]["effect"] == 1.0
    assert context["directions"][0]["falsifier"]
    assert metrics.result.metrics["prediction_verdicts"][0]["status"] == "inconclusive"
    assert fixed.result.metrics["prediction_verdicts"] == []
    native_json = json.dumps(evidence.result.to_dict())
    assert metrics.result.metrics["trial_id"] not in native_json
    assert fixed.result.metrics["trial_id"] not in native_json
    assert "heldout-oracle" not in native_json and "frozen-" not in native_json
    assert store.feedback(1) == raw


def test_native_projection_respects_boundary_and_ledger_export_is_idempotent(
    manifest: ExperimentManifest, tmp_path: Path,
) -> None:
    store = DevelopmentStore(tmp_path / "runs", manifest.exp_id)
    entry = Entry(store, manifest)
    with pytest.raises(ValueError, match="published"):
        entry.native_feedback(1, "evidence")
    run_offline_rehearsal(tmp_path, manifest)

    ledger = entry.export_discovery_ledger("evidence")
    replayed = Entry(store).export_discovery_ledger("evidence")
    other = entry.export_discovery_ledger("metrics")
    event, = ledger.read()
    assert replayed.read() == [event]
    assert event["result"]["feedback"]["data"]["group_id"] == "evidence"
    assert event["state_version"] == 1 and event["state_hash"]
    assert other.path != ledger.path and other.read()[0]["result"]["feedback"]["data"]["group_id"] == "metrics"
    entry_result = entry.native_feedback(1, "evidence")[0]
    with pytest.raises(ValueError, match="group"):
        Tier0FeedbackOracle("metrics").project(entry_result.candidate, entry_result.result)


def test_missing_interval_does_not_become_a_native_belief_observation(
    manifest: ExperimentManifest, tmp_path: Path,
) -> None:
    run_offline_rehearsal(tmp_path, manifest)
    entry = Entry(DevelopmentStore(tmp_path / "runs", manifest.exp_id))
    projected = entry.native_feedback(1, "evidence")[0]
    native = projected.result.to_dict()
    native["feedback"] = None
    readout = next(iter(native["metrics"]["paired_diffs"].values()))
    readout["interval"] = None
    readout["reason"] = "too few days"
    result = EvaluationResult.from_dict(native)

    packet = Tier0FeedbackOracle("evidence").project(projected.candidate, result)

    assert packet is not None
    observed = packet.directional[0].observed
    assert observed["point_estimate"] == 1.0
    assert is_not_observed(observed["paired_delta"])
