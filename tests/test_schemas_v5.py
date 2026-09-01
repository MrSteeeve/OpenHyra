from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from schemas_v5 import (
    AlgorithmBundle,
    AnalogyHypothesis,
    AnalogyResult,
    AnnotationEvent,
    BehaviorProfile,
    ExperimentEvent,
    ExperimentPlan,
    FrozenPolicyArtifact,
    IslandEpoch,
    MechanismCard,
)


SHA_A = "a" * 64
SHA_B = "B" * 64


def make_algorithm_bundle() -> AlgorithmBundle:
    return AlgorithmBundle(
        entrypoint="train.py",
        artifact_protocol="openhyra-policy-spec.v1",
        source_files=["train.py", "manifest.json"],
        parent_ids=["sol_0031"],
        inspiration_ids=["sol_0019"],
        generation_operator="analogy_transfer",
        experiment_plan_id="plan_0048",
        candidate_seed=48_000_007,
    )


def make_frozen_policy_artifact() -> FrozenPolicyArtifact:
    return FrozenPolicyArtifact(
        protocol="openhyra-policy-spec.v1",
        instance_id="put_1d_atm_vol20",
        repeat=0,
        artifact_sha256=SHA_A,
        files=[
            {"path": "normalization.json", "sha256": SHA_B},
            {"path": "step_000.npy", "sha256": SHA_A},
        ],
    )


def make_experiment_event() -> ExperimentEvent:
    return ExperimentEvent(
        record_id="sol_0048",
        algorithm_bundle_sha256=SHA_A,
        experiment_plan_id="plan_0048",
        island_epoch_id="island_02_epoch_03",
        status="ok",
        score=0.0182,
        score_metric="paired_lower_bound_lcb",
        per_instance_metrics_ref=f"sha256:{SHA_A}",
        behavior_profile_ref=f"sha256:{SHA_A}",
        runtime_metrics_ref=f"sha256:{SHA_A}",
        parent_ids=["sol_0031"],
        inspiration_ids=["sol_0019"],
        created_at="2026-09-01T00:00:00Z",
    )


def make_behavior_profile() -> BehaviorProfile:
    return BehaviorProfile(
        probe_suite="bermudan-behavior-probe.v1",
        probe_suite_sha256=SHA_A,
        policy_artifact_sha256=SHA_B,
        performance={
            "per_instance_improvement": [0.01, -0.01],
            "paired_mean": 0.0,
            "paired_standard_error": 0.002,
        },
        outcome_distribution={
            "loss_definition": "negative_paired_discounted_payoff_improvement",
            "mean_loss": 0.0,
            "var_95": 0.012,
            "cvar_95": 0.019,
        },
        policy_geometry={
            "exercise_rate_by_instance": [0.21, 0.34],
            "boundary_monotonicity_violations": 0,
            "reference_boundary_agreement": 0.82,
        },
        sensitivity={"moneyness": 0.41},
        robustness={"seed_instability": 0.011},
        compute={"training_seconds": 5.4, "parameter_count": 2433},
    )


def make_mechanism_card() -> MechanismCard:
    return MechanismCard(
        record_id="sol_0048",
        deterministic_facts={"optimizer": "adam"},
        trusted_observations={"strong_slices": ["high_volatility"]},
        llm_inferences=[
            {
                "claim": "normalization may reduce residual burden",
                "confidence": 0.63,
                "evidence_record_ids": ["sol_0019", "sol_0048"],
                "annotation_event_id": "annotation_0082",
            }
        ],
    )


def make_analogy_hypothesis() -> AnalogyHypothesis:
    return AnalogyHypothesis(
        id="analogy_0017",
        source_record_ids=["sol_0019"],
        target_parent_id="sol_0031",
        relation_mapping=[
            {
                "source_role": "log_moneyness_feature",
                "target_role": "mlp_input_normalization",
                "shared_relation": "stabilize_state_scale",
            }
        ],
        non_correspondence=["ridge_coefficients_do_not_map_to_hidden_weights"],
        transferable_intervention="add_log_moneyness_and_hold_budget_fixed",
        predicted_effect={
            "metric": "high_volatility_slice_improvement",
            "direction": "positive",
            "minimum_effect": 0.003,
        },
        falsifier="paired_effect_lcb_le_0",
        matched_control={"same_parent": True, "same_training_budget": True},
        status="preregistered",
    )


def make_analogy_result() -> AnalogyResult:
    return AnalogyResult(
        analogy_hypothesis_id="analogy_0017",
        guided_record_id="sol_0048",
        control_record_id="sol_0049",
        guided_delta=0.006,
        control_delta=0.001,
        transfer_gain=0.005,
        transfer_gain_standard_error=0.002,
        predicted_slice_effect=0.007,
        prediction_direction_correct=True,
        verdict="transfer_supported",
    )


def make_experiment_plan() -> ExperimentPlan:
    return ExperimentPlan(
        id="plan_0048",
        action="continue",
        target_island_epoch_id="island_02_epoch_03",
        generation_operator="analogy_transfer",
        parent_ids=["sol_0031"],
        inspiration_ids=["sol_0019"],
        analogy_hypothesis_id="analogy_0017",
        implementation_intent="add one normalized MLP input",
        negative_constraints=["do_not_change_hidden_width"],
        success_criterion="paired_slice_lcb_gt_0",
        budget={
            "candidate_count": 2,
            "sandbox_seconds_per_cell": 60,
            "max_artifact_bytes": 8_388_608,
        },
    )


def make_island_epoch() -> IslandEpoch:
    return IslandEpoch(
        island_id="island_02",
        epoch=3,
        seed_record_ids=["sol_0031"],
        started_after_context_round=40,
        proposal_seed=230_041,
        status="active",
    )


def make_annotation_event() -> AnnotationEvent:
    return AnnotationEvent(
        id="annotation_0082",
        annotation_type="mechanism_inference",
        target_record_ids=["sol_0048"],
        evidence_record_ids=["sol_0019", "sol_0048"],
        model="model-id",
        backend="backend-id",
        prompt_sha256=SHA_A,
        response_sha256=SHA_B,
        parser_schema="openhyra-mechanism-card.v1",
        created_at="2026-09-01T00:00:00Z",
    )


VALID_FACTORIES: list[tuple[Callable[[], Any], str]] = [
    (make_algorithm_bundle, "openhyra-algorithm-bundle.v1"),
    (make_frozen_policy_artifact, "openhyra-frozen-policy-artifact.v1"),
    (make_experiment_event, "openhyra-experiment-event.v1"),
    (make_behavior_profile, "openhyra-behavior-profile.v1"),
    (make_mechanism_card, "openhyra-mechanism-card.v1"),
    (make_analogy_hypothesis, "openhyra-analogy-hypothesis.v1"),
    (make_analogy_result, "openhyra-analogy-result.v1"),
    (make_experiment_plan, "openhyra-experiment-plan.v1"),
    (make_island_epoch, "openhyra-island-epoch.v1"),
    (make_annotation_event, "openhyra-annotation-event.v1"),
]


@pytest.mark.parametrize(
    ("factory", "expected_schema"),
    VALID_FACTORIES,
    ids=[factory.__name__ for factory, _ in VALID_FACTORIES],
)
def test_valid_construction_and_round_trip(
    factory: Callable[[], Any], expected_schema: str
) -> None:
    instance = factory()
    instance.validate()

    payload = instance.to_dict()
    restored = type(instance).from_dict(payload)

    assert instance.schema == expected_schema
    assert payload["schema"] == expected_schema
    assert restored == instance
    assert restored.to_dict() == payload


def test_to_dict_returns_independent_nested_values() -> None:
    bundle = make_algorithm_bundle()
    payload = bundle.to_dict()
    payload["source_files"].append("unexpected.py")

    assert bundle.source_files == ["train.py", "manifest.json"]


@pytest.mark.parametrize("operator", [
    "local_mutation",
    "ablation",
    "repair",
    "analogy_transfer",
    "composition",
    "restart_from_skeleton",
])
def test_all_generation_operators_are_valid(operator: str) -> None:
    bundle = make_algorithm_bundle()
    bundle.generation_operator = operator
    bundle.validate()

    plan = make_experiment_plan()
    plan.generation_operator = operator
    plan.validate()


def test_algorithm_bundle_rejects_empty_source_files() -> None:
    bundle = make_algorithm_bundle()
    bundle.source_files = []
    with pytest.raises(ValueError):
        bundle.validate()


def test_algorithm_bundle_rejects_unknown_generation_operator() -> None:
    bundle = make_algorithm_bundle()
    bundle.generation_operator = "unknown"
    with pytest.raises(ValueError):
        bundle.validate()


def test_algorithm_bundle_allows_empty_lineage_lists() -> None:
    bundle = make_algorithm_bundle()
    bundle.parent_ids = []
    bundle.inspiration_ids = []
    bundle.validate()


def test_frozen_policy_artifact_rejects_empty_files() -> None:
    artifact = make_frozen_policy_artifact()
    artifact.files = []
    with pytest.raises(ValueError):
        artifact.validate()


@pytest.mark.parametrize("bad_sha", ["a" * 63, "g" * 64, "", 42])
def test_frozen_policy_artifact_rejects_bad_artifact_sha256(
    bad_sha: object,
) -> None:
    artifact = make_frozen_policy_artifact()
    artifact.artifact_sha256 = bad_sha  # type: ignore[assignment]
    with pytest.raises(ValueError):
        artifact.validate()


@pytest.mark.parametrize("bad_sha", ["b" * 65, "not-hex", None])
def test_frozen_policy_artifact_rejects_bad_file_sha256(
    bad_sha: object,
) -> None:
    artifact = make_frozen_policy_artifact()
    artifact.files[0]["sha256"] = bad_sha
    with pytest.raises(ValueError):
        artifact.validate()


def test_frozen_policy_artifact_accepts_uppercase_sha256() -> None:
    artifact = make_frozen_policy_artifact()
    artifact.validate()


def test_frozen_policy_artifact_requires_file_keys() -> None:
    artifact = make_frozen_policy_artifact()
    artifact.files = [{"path": "normalization.json"}]
    with pytest.raises(ValueError):
        artifact.validate()


@pytest.mark.parametrize("status", [
    "ok",
    "early_stopped",
    "static_rejected",
    "artifact_rejected",
    "timeout",
    "oom",
    "violation",
    "runtime_error",
    "cancelled",
])
def test_all_experiment_event_statuses_are_valid(status: str) -> None:
    event = make_experiment_event()
    event.status = status
    event.validate()


def test_experiment_event_rejects_unknown_status() -> None:
    event = make_experiment_event()
    event.status = "pending"
    with pytest.raises(ValueError):
        event.validate()


def test_experiment_event_allows_none_score() -> None:
    event = make_experiment_event()
    event.score = None
    event.validate()


def test_behavior_profile_rejects_mismatched_instance_lengths() -> None:
    profile = make_behavior_profile()
    profile.policy_geometry["exercise_rate_by_instance"] = [0.21]
    with pytest.raises(ValueError):
        profile.validate()


def test_behavior_profile_allows_two_empty_instance_lists() -> None:
    profile = make_behavior_profile()
    profile.performance["per_instance_improvement"] = []
    profile.policy_geometry["exercise_rate_by_instance"] = []
    profile.validate()


@pytest.mark.parametrize("confidence", [-0.001, 1.001])
def test_mechanism_card_rejects_out_of_range_confidence(
    confidence: float,
) -> None:
    card = make_mechanism_card()
    card.llm_inferences[0]["confidence"] = confidence
    with pytest.raises(ValueError):
        card.validate()


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_mechanism_card_accepts_confidence_boundaries(confidence: float) -> None:
    card = make_mechanism_card()
    card.llm_inferences[0]["confidence"] = confidence
    card.validate()


def test_mechanism_card_allows_empty_inferences() -> None:
    card = make_mechanism_card()
    card.llm_inferences = []
    card.validate()


def test_analogy_hypothesis_rejects_empty_relation_mapping() -> None:
    hypothesis = make_analogy_hypothesis()
    hypothesis.relation_mapping = []
    with pytest.raises(ValueError):
        hypothesis.validate()


@pytest.mark.parametrize("status", ["draft", "failed"])
def test_analogy_hypothesis_rejects_unknown_status(status: str) -> None:
    hypothesis = make_analogy_hypothesis()
    hypothesis.status = status
    with pytest.raises(ValueError):
        hypothesis.validate()


def test_analogy_hypothesis_rejects_unknown_direction() -> None:
    hypothesis = make_analogy_hypothesis()
    hypothesis.predicted_effect["direction"] = "flat"
    with pytest.raises(ValueError):
        hypothesis.validate()


@pytest.mark.parametrize("verdict", [
    "transfer_supported",
    "transfer_refuted",
    "inconclusive",
    "invalid_control",
    "execution_failed",
])
def test_all_analogy_result_verdicts_are_valid(verdict: str) -> None:
    result = make_analogy_result()
    result.verdict = verdict
    result.validate()


def test_analogy_result_rejects_unknown_verdict() -> None:
    result = make_analogy_result()
    result.verdict = "pending"
    with pytest.raises(ValueError):
        result.validate()


@pytest.mark.parametrize("action", ["continue", "stop", "reset"])
def test_all_experiment_plan_actions_are_valid(action: str) -> None:
    plan = make_experiment_plan()
    plan.action = action
    plan.validate()


def test_experiment_plan_rejects_unknown_action() -> None:
    plan = make_experiment_plan()
    plan.action = "pause"
    with pytest.raises(ValueError):
        plan.validate()


def test_experiment_plan_rejects_unknown_generation_operator() -> None:
    plan = make_experiment_plan()
    plan.generation_operator = "unknown"
    with pytest.raises(ValueError):
        plan.validate()


def test_experiment_plan_allows_none_analogy_hypothesis_id() -> None:
    plan = make_experiment_plan()
    plan.analogy_hypothesis_id = None
    plan.validate()


@pytest.mark.parametrize("status", ["active", "culled", "completed"])
def test_all_island_epoch_statuses_are_valid(status: str) -> None:
    epoch = make_island_epoch()
    epoch.status = status
    epoch.validate()


def test_island_epoch_rejects_unknown_status() -> None:
    epoch = make_island_epoch()
    epoch.status = "reset"
    with pytest.raises(ValueError):
        epoch.validate()


def test_unconstrained_record_id_lists_may_be_empty() -> None:
    epoch = make_island_epoch()
    epoch.seed_record_ids = []
    epoch.validate()

    annotation = make_annotation_event()
    annotation.target_record_ids = []
    annotation.evidence_record_ids = []
    annotation.validate()


@pytest.mark.parametrize(
    ("factory", "field_name", "bad_value"),
    [
        (make_algorithm_bundle, "candidate_seed", True),
        (make_frozen_policy_artifact, "repeat", "zero"),
        (make_experiment_event, "score", "high"),
        (make_behavior_profile, "compute", {"training_seconds": "fast"}),
        (make_mechanism_card, "llm_inferences", "none"),
        (make_analogy_hypothesis, "source_record_ids", [1]),
        (make_analogy_result, "prediction_direction_correct", 1),
        (make_experiment_plan, "budget", {"candidate_count": 2}),
        (make_island_epoch, "epoch", True),
        (make_annotation_event, "prompt_sha256", 7),
        (make_annotation_event, "target_record_ids", "sol_0048"),
    ],
    ids=[
        "algorithm-int",
        "artifact-int",
        "event-float",
        "profile-float-dict",
        "mechanism-list",
        "hypothesis-string-list",
        "result-bool",
        "plan-budget-shape",
        "epoch-int",
        "annotation-string",
        "annotation-string-list",
    ],
)
def test_validate_rejects_bad_field_types(
    factory: Callable[[], Any], field_name: str, bad_value: object
) -> None:
    instance = factory()
    setattr(instance, field_name, bad_value)
    with pytest.raises(ValueError):
        instance.validate()


def test_from_dict_runs_validation() -> None:
    payload = make_algorithm_bundle().to_dict()
    payload["source_files"] = []
    with pytest.raises(ValueError):
        AlgorithmBundle.from_dict(payload)


@pytest.mark.parametrize("bad_payload", [None, [], "not-a-dict"])
def test_from_dict_requires_dict(bad_payload: object) -> None:
    with pytest.raises(ValueError):
        AnnotationEvent.from_dict(bad_payload)  # type: ignore[arg-type]
