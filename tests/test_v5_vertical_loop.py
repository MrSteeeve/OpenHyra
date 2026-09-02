"""Real integration test for V5Bridge: init → pick_island → on_candidate_evaluated → on_context_complete.

Uses actual schema objects — no mocks for core types. This is the single vertical
closed-loop validation that the Codex review identified as missing.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from harness_v5 import V5Bridge, _map_status, adapt_bermudan_metrics
from schemas_v5 import ExperimentEvent, IslandEpoch, BehaviorProfile


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "test_run"


class TestV5VerticalLoop:
    """Full init → context → candidate → eval path with real dependencies."""

    def test_init_creates_islands(self, run_dir):
        bridge = V5Bridge(run_dir, num_islands=4)
        epochs = bridge.initialize(
            seed_record_ids=["seed_001"],
            frozen_baseline_score=0.5,
        )
        assert len(epochs) == 4
        for epoch in epochs:
            assert isinstance(epoch, IslandEpoch)
            assert epoch.status == "active"
            assert epoch.seed_record_ids == ["seed_001"]

    def test_init_single_island(self, run_dir):
        bridge = V5Bridge(run_dir, num_islands=1)
        epochs = bridge.initialize(seed_record_ids=["seed_001"])
        assert len(epochs) == 1
        assert epochs[0].island_id == "island_00"

    def test_pick_island_returns_valid_epoch_id(self, run_dir):
        bridge = V5Bridge(run_dir, num_islands=4)
        bridge.initialize(seed_record_ids=["seed_001"])
        epoch_id = bridge.pick_island(context_round=1)
        assert epoch_id.startswith("island_")
        assert "_epoch_" in epoch_id

    def test_on_candidate_evaluated_writes_experiment_event(self, run_dir):
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(seed_record_ids=["seed_001"])
        epoch_id = bridge.pick_island(context_round=1)

        bridge.on_candidate_evaluated(
            record_id="rec_001",
            island_epoch_id=epoch_id,
            score=0.75,
            status="ok",
            parent_ids=["seed_001"],
            metrics={
                "elapsed_s": 10.5,
                "peak_memory_mb": 256.0,
                "protocol": "feature_ir",
            },
        )

        events = bridge.event_store.read_experiment_events()
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, ExperimentEvent)
        assert event.record_id == "rec_001"
        assert event.island_epoch_id == epoch_id
        assert event.score == 0.75
        assert event.status == "ok"
        assert event.parent_ids == ["seed_001"]
        assert event.created_at != ""

    def test_on_candidate_evaluated_assigns_to_island(self, run_dir):
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(seed_record_ids=["seed_001"])
        epoch_id = bridge.pick_island(context_round=1)

        bridge.on_candidate_evaluated(
            record_id="rec_001",
            island_epoch_id=epoch_id,
            score=0.75,
            status="ok",
            parent_ids=[],
            metrics={},
        )

        records = bridge.island_scheduler.get_island_records(epoch_id)
        assert "rec_001" in records

    def test_on_candidate_evaluated_builds_mechanism_card(self, run_dir):
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(seed_record_ids=["seed_001"])
        epoch_id = bridge.pick_island(context_round=1)

        bridge.on_candidate_evaluated(
            record_id="rec_001",
            island_epoch_id=epoch_id,
            score=0.75,
            status="ok",
            parent_ids=[],
            metrics={"protocol": "feature_ir", "entrypoint": "evaluate_features"},
        )

        assert "rec_001" in bridge._cards
        card = bridge._cards["rec_001"]
        assert card.deterministic_facts["protocol"] == "feature_ir"

    def test_on_candidate_evaluated_with_profiler(self, run_dir):
        """When baseline_scores are provided, profiler should build real BehaviorProfiles."""
        baseline = {"inst_0": 0.5, "inst_1": 0.4, "inst_2": 0.6}
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(
            seed_record_ids=["seed_001"],
            baseline_scores=baseline,
            probe_suite_sha256="a" * 64,
        )
        epoch_id = bridge.pick_island(context_round=1)

        bridge.on_candidate_evaluated(
            record_id="rec_001",
            island_epoch_id=epoch_id,
            score=0.55,
            status="ok",
            parent_ids=[],
            metrics={
                "per_instance_results": {"inst_0": 0.6, "inst_1": 0.5, "inst_2": 0.55},
                "per_instance_scores": {"inst_0": 0.6, "inst_1": 0.5, "inst_2": 0.55},
                "per_instance_exercise_rates": {"inst_0": 0.3, "inst_1": 0.4, "inst_2": 0.35},
                "artifact_sha256": "b" * 64,
                "training_seconds": 120.0,
                "peak_memory_bytes": 1024 * 1024 * 512,
                "inference_us": 50.0,
                "parameter_count": 10000,
            },
        )

        assert "rec_001" in bridge._profiles
        profile = bridge._profiles["rec_001"]
        assert isinstance(profile, BehaviorProfile)
        assert profile.policy_artifact_sha256 == "b" * 64

    def test_on_candidate_evaluated_without_profiler_skips_profile(self, run_dir):
        """Without baseline_scores in initialize(), no profile should be built."""
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(seed_record_ids=["seed_001"])
        epoch_id = bridge.pick_island(context_round=1)

        bridge.on_candidate_evaluated(
            record_id="rec_001",
            island_epoch_id=epoch_id,
            score=0.55,
            status="ok",
            parent_ids=[],
            metrics={
                "per_instance_results": {"inst_0": 0.6},
                "per_instance_scores": {"inst_0": 0.6},
                "per_instance_exercise_rates": {"inst_0": 0.3},
            },
        )

        assert "rec_001" not in bridge._profiles

    def test_on_context_complete_no_review_at_round_1(self, run_dir):
        bridge = V5Bridge(run_dir, num_islands=4)
        bridge.initialize(seed_record_ids=["seed_001"])
        result = bridge.on_context_complete(context_round=1)
        assert result == {}

    def test_full_vertical_loop(self, run_dir):
        """Complete loop: init → pick → evaluate × N → context_complete."""
        bridge = V5Bridge(run_dir, num_islands=2)
        epochs = bridge.initialize(
            seed_record_ids=["seed_001"],
            frozen_baseline_score=0.5,
        )
        assert len(epochs) == 2

        for i in range(3):
            epoch_id = bridge.pick_island(context_round=i)
            bridge.on_candidate_evaluated(
                record_id=f"rec_{i:03d}",
                island_epoch_id=epoch_id,
                score=0.5 + i * 0.01,
                status="ok",
                parent_ids=["seed_001"],
                metrics={
                    "elapsed_s": 5.0 + i,
                    "protocol": "feature_ir",
                },
            )

        replacements = bridge.on_context_complete(context_round=3)
        assert isinstance(replacements, dict)

        events = bridge.event_store.read_experiment_events()
        assert len(events) == 3

        diag = bridge.get_island_diagnostics()
        assert diag["active_islands"] == 2
        assert diag["cards_cached"] == 3

    def test_build_context_after_candidates(self, run_dir):
        """build_context() should work after candidates have been evaluated."""
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(
            seed_record_ids=["seed_001"],
            frozen_baseline_score=0.5,
        )
        epoch_id = bridge.pick_island(context_round=1)
        bridge.on_candidate_evaluated(
            record_id="rec_001",
            island_epoch_id=epoch_id,
            score=0.6,
            status="ok",
            parent_ids=["seed_001"],
            metrics={},
        )

        ctx = bridge.build_context(target_island_epoch_id=epoch_id)
        assert "portfolio" in ctx
        assert "portfolio_text" in ctx
        assert isinstance(ctx["portfolio_text"], str)
        assert len(ctx["portfolio_text"]) > 0

    def test_save_and_reload_state(self, run_dir):
        """State should survive save + reload, including cards and profiles."""
        baseline = {"inst_0": 0.5, "inst_1": 0.4}
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(
            seed_record_ids=["seed_001"],
            baseline_scores=baseline,
            probe_suite_sha256="a" * 64,
        )
        epoch_id = bridge.pick_island(context_round=1)
        bridge.on_candidate_evaluated(
            record_id="rec_001",
            island_epoch_id=epoch_id,
            score=0.7,
            status="ok",
            parent_ids=[],
            metrics={
                "per_instance_results": {"inst_0": 0.6, "inst_1": 0.5},
                "per_instance_scores": {"inst_0": 0.6, "inst_1": 0.5},
                "per_instance_exercise_rates": {"inst_0": 0.3, "inst_1": 0.4},
                "artifact_sha256": "c" * 64,
                "training_seconds": 10.0,
                "peak_memory_bytes": 1024,
                "inference_us": 5.0,
                "parameter_count": 50,
                "protocol": "feature_ir",
            },
        )
        bridge.save_state()

        bridge2 = V5Bridge(run_dir, num_islands=2)
        events = bridge2.event_store.read_experiment_events()
        assert len(events) == 1
        assert events[0].record_id == "rec_001"

        island_records = bridge2.island_scheduler.get_island_records(epoch_id)
        assert "rec_001" in island_records

        assert "rec_001" in bridge2._cards, "cards must survive reload"
        assert bridge2._cards["rec_001"].record_id == "rec_001"

        assert "rec_001" in bridge2._profiles, "profiles must survive reload"
        assert isinstance(bridge2._profiles["rec_001"], BehaviorProfile)

    def test_description_param_backward_compat(self, run_dir):
        """harness.py passes description= kwarg; bridge should accept it without error."""
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(seed_record_ids=["seed_001"])
        epoch_id = bridge.pick_island(context_round=1)
        bridge.on_candidate_evaluated(
            record_id="rec_001",
            island_epoch_id=epoch_id,
            score=0.6,
            status="ok",
            description="test candidate",
            parent_ids=[],
            metrics={},
        )
        events = bridge.event_store.read_experiment_events()
        assert len(events) == 1


class TestMatchedControlSchemaAlignment:
    """Verify matched_control.py uses real AnalogyHypothesis/AnalogyResult schemas."""

    def _make_hypothesis(self) -> "AnalogyHypothesis":
        from schemas_v5 import AnalogyHypothesis
        return AnalogyHypothesis(
            id="hyp_001",
            source_record_ids=["src_001", "src_002"],
            target_parent_id="tgt_001",
            relation_mapping=[{
                "source_role": "feature_extractor",
                "target_role": "policy_head",
                "shared_relation": "improves_exercise_boundary",
            }],
            non_correspondence=["different_volatility_regime"],
            transferable_intervention="Apply ReLU activation after feature extraction layer",
            predicted_effect={
                "metric": "aggregate_score",
                "direction": "positive",
                "minimum_effect": 0.01,
            },
            falsifier="If score decreases on high-vol instances",
            matched_control={"strategy": "same_parent_different_seed"},
            status="preregistered",
        )

    def test_build_pair_uses_transferable_intervention(self):
        from matched_control import MatchedControlBuilder
        hypothesis = self._make_hypothesis()
        pair = MatchedControlBuilder.build_pair(hypothesis, parent_id="p1", seed=42)
        assert "transferable_intervention" not in pair.guided_prompt_suffix or \
               "Apply ReLU" in pair.guided_prompt_suffix
        assert hypothesis.transferable_intervention in pair.guided_prompt_suffix

    def test_evaluate_pair_returns_valid_analogy_result(self):
        from matched_control import MatchedControlBuilder, ControlPair
        from schemas_v5 import AnalogyResult
        hypothesis = self._make_hypothesis()
        pair = MatchedControlBuilder.build_pair(hypothesis, parent_id="p1", seed=42)
        pair.guided_record_id = "g1"
        pair.control_record_id = "c1"
        pair.guided_score = 0.6
        pair.control_score = 0.5
        pair.baseline_score = 0.4

        result = MatchedControlBuilder.evaluate_pair(pair, direction="max")
        assert isinstance(result, AnalogyResult)
        result.validate()
        assert result.analogy_hypothesis_id == "hyp_001"
        assert result.guided_record_id == "g1"
        assert result.control_record_id == "c1"
        assert result.guided_delta == pytest.approx(0.2)
        assert result.control_delta == pytest.approx(0.1)
        assert result.transfer_gain == pytest.approx(0.1)
        assert result.verdict == "transfer_supported"

    def test_evaluate_pair_execution_failed(self):
        from matched_control import MatchedControlBuilder, ControlPair
        hypothesis = self._make_hypothesis()
        pair = MatchedControlBuilder.build_pair(hypothesis, parent_id="p1", seed=42)
        pair.guided_score = None
        pair.control_score = 0.5

        result = MatchedControlBuilder.evaluate_pair(pair)
        result.validate()
        assert result.verdict == "execution_failed"

    def test_control_pair_store_roundtrip(self, tmp_path):
        from matched_control import ControlPairStore, ControlPair
        store = ControlPairStore(tmp_path / "pairs.jsonl")
        pair = ControlPair(
            hypothesis_id="h1",
            guided_prompt_suffix="guided",
            control_prompt_suffix="control",
            shared_parent_id="p1",
            shared_seed=42,
            guided_score=0.6,
            control_score=0.5,
        )
        store.save(pair)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].hypothesis_id == "h1"
        assert loaded[0].guided_score == 0.6


class TestStatusMapping:
    """Verify legacy harness statuses map correctly to V5 schema statuses."""

    @pytest.mark.parametrize(
        "legacy,expected",
        [
            ("ok", "ok"),
            ("crash", "runtime_error"),
            ("rejected", "static_rejected"),
            ("cancelled", "cancelled"),
            ("violation", "violation"),
            ("timeout", "timeout"),
        ],
    )
    def test_known_legacy_statuses(self, legacy, expected):
        assert _map_status(legacy) == expected

    @pytest.mark.parametrize(
        "v5_native",
        ["runtime_error", "static_rejected", "artifact_rejected", "oom", "early_stopped"],
    )
    def test_v5_native_statuses_pass_through(self, v5_native):
        assert _map_status(v5_native) == v5_native

    def test_unknown_status_falls_back_to_runtime_error(self):
        assert _map_status("some_new_status") == "runtime_error"


class TestHalfCommitProtection:
    """V5 event write failures must not propagate to the harness worker."""

    def test_crash_status_does_not_raise(self, tmp_path):
        run_dir = tmp_path / "run"
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(seed_record_ids=["seed_001"])
        epoch_id = bridge.pick_island(context_round=1)

        bridge.on_candidate_evaluated(
            record_id="rec_crash",
            island_epoch_id=epoch_id,
            score=None,
            status="crash",
            parent_ids=[],
            metrics={},
        )

        events = bridge.event_store.read_experiment_events()
        assert len(events) == 1
        assert events[0].status == "runtime_error"

    def test_rejected_status_maps_correctly(self, tmp_path):
        run_dir = tmp_path / "run"
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(seed_record_ids=["seed_001"])
        epoch_id = bridge.pick_island(context_round=1)

        bridge.on_candidate_evaluated(
            record_id="rec_reject",
            island_epoch_id=epoch_id,
            score=None,
            status="rejected",
            parent_ids=[],
            metrics={},
        )

        events = bridge.event_store.read_experiment_events()
        assert events[0].status == "static_rejected"

    def test_inner_exception_is_caught(self, tmp_path):
        """If V5 event write somehow fails, it must not propagate."""
        run_dir = tmp_path / "run"
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(seed_record_ids=["seed_001"])

        bridge.on_candidate_evaluated(
            record_id="rec_bad",
            island_epoch_id="nonexistent_epoch_id",
            score=0.5,
            status="ok",
            parent_ids=[],
            metrics={},
        )
        # Should not raise — error is caught and logged to stderr


class TestBermudanMetricsAdapter:
    """Verify adapt_bermudan_metrics extracts V5-compatible fields."""

    def test_basic_field_mapping(self):
        raw = {
            "metric": "paired_lower_bound_lcb",
            "runtime_seconds": 42.5,
            "candidate_hash": "abc123",
        }
        adapted = adapt_bermudan_metrics(raw)
        assert adapted["score_metric"] == "paired_lower_bound_lcb"
        assert adapted["elapsed_s"] == 42.5
        assert adapted["artifact_sha256"] == "abc123"

    def test_summaries_extraction(self):
        raw = {
            "metric": "paired_lower_bound_lcb",
            "summaries": [
                {
                    "instance_id": "inst_0",
                    "repeat": 0,
                    "candidate_lower_bound": 0.6,
                    "baseline_lower_bound": 0.5,
                },
                {
                    "instance_id": "inst_0",
                    "repeat": 1,
                    "candidate_lower_bound": 0.7,
                    "baseline_lower_bound": 0.55,
                },
                {
                    "instance_id": "inst_1",
                    "repeat": 0,
                    "candidate_lower_bound": 0.8,
                    "baseline_lower_bound": 0.7,
                },
            ],
        }
        adapted = adapt_bermudan_metrics(raw)

        scores = adapted["per_instance_scores"]
        assert scores["inst_0"] == pytest.approx(0.65)
        assert scores["inst_1"] == pytest.approx(0.8)

        baselines = adapted["baseline_scores"]
        assert baselines["inst_0"] == pytest.approx(0.525)
        assert baselines["inst_1"] == pytest.approx(0.7)

    def test_empty_summaries_no_crash(self):
        adapted = adapt_bermudan_metrics({"summaries": []})
        assert "per_instance_scores" not in adapted

    def test_missing_summaries_no_crash(self):
        adapted = adapt_bermudan_metrics({})
        assert adapted["score_metric"] == "aggregate_score"

    def test_passthrough_keys(self):
        raw = {
            "generation_operator": "local_mutation",
            "experiment_plan_id": "plan_001",
            "inspiration_ids": ["rec_a"],
        }
        adapted = adapt_bermudan_metrics(raw)
        assert adapted["generation_operator"] == "local_mutation"
        assert adapted["experiment_plan_id"] == "plan_001"
        assert adapted["inspiration_ids"] == ["rec_a"]

    def test_adapted_metrics_survive_bridge_evaluate(self, tmp_path):
        """End-to-end: adapt real evaluator output → feed to V5Bridge → no crash."""
        raw_evaluator_output = {
            "stage": "search",
            "metric": "paired_lower_bound_lcb",
            "search_score": 0.0123,
            "mean_paired_normalized_improvement": 0.0123,
            "paired_aggregate_standard_error": 0.002,
            "instance_count": 2,
            "repeat_count": 1,
            "evaluation_cell_count": 2,
            "runtime_seconds": 35.7,
            "candidate_hash": "d" * 64,
            "feature_program_sha256": "d" * 64,
            "summaries": [
                {
                    "instance_id": "inst_0",
                    "repeat": 0,
                    "candidate_lower_bound": 0.55,
                    "baseline_lower_bound": 0.50,
                    "paired_normalized_improvement": 0.01,
                    "paired_normalized_standard_error": 0.002,
                },
                {
                    "instance_id": "inst_1",
                    "repeat": 0,
                    "candidate_lower_bound": 0.60,
                    "baseline_lower_bound": 0.58,
                    "paired_normalized_improvement": 0.005,
                    "paired_normalized_standard_error": 0.001,
                },
            ],
        }
        adapted = adapt_bermudan_metrics(raw_evaluator_output)

        run_dir = tmp_path / "run"
        bridge = V5Bridge(run_dir, num_islands=2)
        bridge.initialize(seed_record_ids=["seed_001"])
        epoch_id = bridge.pick_island(context_round=1)

        bridge.on_candidate_evaluated(
            record_id="rec_bermudan",
            island_epoch_id=epoch_id,
            score=0.0123,
            status="ok",
            parent_ids=["seed_001"],
            metrics=adapted,
        )

        events = bridge.event_store.read_experiment_events()
        assert len(events) == 1
        e = events[0]
        assert e.score_metric == "paired_lower_bound_lcb"
        assert e.algorithm_bundle_sha256 == "d" * 64
        assert e.per_instance_metrics_ref != ""
        assert e.runtime_metrics_ref != ""
