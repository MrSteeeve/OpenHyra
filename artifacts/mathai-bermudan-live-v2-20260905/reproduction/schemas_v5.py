from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


_GENERATION_OPERATORS = {
    "local_mutation",
    "ablation",
    "repair",
    "analogy_transfer",
    "composition",
    "restart_from_skeleton",
}
_EXPERIMENT_STATUSES = {
    "ok",
    "early_stopped",
    "static_rejected",
    "artifact_rejected",
    "timeout",
    "oom",
    "violation",
    "runtime_error",
    "cancelled",
}
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


def _require_str(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str")


def _require_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int")


def _require_float(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a float")


def _require_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def _require_dict(value: object, name: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict")
    return value


def _require_keys(value: object, keys: set[str], name: str) -> dict[Any, Any]:
    mapping = _require_dict(value, name)
    missing = keys.difference(mapping)
    if missing:
        raise ValueError(f"{name} is missing keys: {', '.join(sorted(missing))}")
    return mapping


def _require_str_list(value: object, name: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list[str]")


def _require_float_list(value: object, name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list[float]")
    for index, item in enumerate(value):
        _require_float(item, f"{name}[{index}]")


def _require_str_float_dict(value: object, name: str) -> None:
    mapping = _require_dict(value, name)
    for key, item in mapping.items():
        _require_str(key, f"{name} key")
        _require_float(item, f"{name}[{key!r}]")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")


def _validated_from_dict(cls: type[Any], d: dict[str, Any]) -> Any:
    if not isinstance(d, dict):
        raise ValueError("d must be a dict")
    try:
        instance = cls(**deepcopy(d))
    except TypeError as exc:
        raise ValueError(f"invalid {cls.__name__} fields") from exc
    instance.validate()
    return instance


@dataclass
class AlgorithmBundle:
    schema: str = field(
        default="openhyra-algorithm-bundle.v1", kw_only=True
    )
    entrypoint: str
    artifact_protocol: str
    source_files: list[str]
    parent_ids: list[str]
    inspiration_ids: list[str]
    generation_operator: str
    experiment_plan_id: str
    candidate_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "entrypoint": self.entrypoint,
            "artifact_protocol": self.artifact_protocol,
            "source_files": deepcopy(self.source_files),
            "parent_ids": deepcopy(self.parent_ids),
            "inspiration_ids": deepcopy(self.inspiration_ids),
            "generation_operator": self.generation_operator,
            "experiment_plan_id": self.experiment_plan_id,
            "candidate_seed": self.candidate_seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AlgorithmBundle:
        return _validated_from_dict(cls, d)

    def validate(self) -> None:
        _require_str(self.schema, "schema")
        _require_str(self.entrypoint, "entrypoint")
        _require_str(self.artifact_protocol, "artifact_protocol")
        _require_str_list(self.source_files, "source_files")
        _require_str_list(self.parent_ids, "parent_ids")
        _require_str_list(self.inspiration_ids, "inspiration_ids")
        _require_str(self.generation_operator, "generation_operator")
        _require_str(self.experiment_plan_id, "experiment_plan_id")
        _require_int(self.candidate_seed, "candidate_seed")
        if not self.source_files:
            raise ValueError("source_files must be non-empty")
        if self.generation_operator not in _GENERATION_OPERATORS:
            raise ValueError("generation_operator is not allowed")


@dataclass
class FrozenPolicyArtifact:
    schema: str = field(
        default="openhyra-frozen-policy-artifact.v1", kw_only=True
    )
    protocol: str
    instance_id: str
    repeat: int
    artifact_sha256: str
    files: list[dict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol": self.protocol,
            "instance_id": self.instance_id,
            "repeat": self.repeat,
            "artifact_sha256": self.artifact_sha256,
            "files": deepcopy(self.files),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrozenPolicyArtifact:
        return _validated_from_dict(cls, d)

    def validate(self) -> None:
        _require_str(self.schema, "schema")
        _require_str(self.protocol, "protocol")
        _require_str(self.instance_id, "instance_id")
        _require_int(self.repeat, "repeat")
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        if not isinstance(self.files, list):
            raise ValueError("files must be a list[dict]")
        if not self.files:
            raise ValueError("files must be non-empty")
        for index, item in enumerate(self.files):
            file_record = _require_keys(item, {"path", "sha256"}, f"files[{index}]")
            _require_str(file_record["path"], f"files[{index}].path")
            _require_sha256(file_record["sha256"], f"files[{index}].sha256")


@dataclass
class ExperimentEvent:
    schema: str = field(
        default="openhyra-experiment-event.v1", kw_only=True
    )
    record_id: str
    algorithm_bundle_sha256: str
    experiment_plan_id: str
    island_epoch_id: str
    status: str
    score: float | None
    score_metric: str
    per_instance_metrics_ref: str
    behavior_profile_ref: str
    runtime_metrics_ref: str
    parent_ids: list[str]
    inspiration_ids: list[str]
    created_at: str
    # Optional sidecar references added by the feedback-aware bridge.  Keeping
    # them keyword-only preserves the v1 constructor and lets old event lines
    # load with empty references.
    feedback_packet_ref: str = field(default="", kw_only=True)
    feedback_packet_schema: str = field(default="", kw_only=True)
    problem_state_ref: str = field(default="", kw_only=True)
    problem_state_version: int | None = field(default=None, kw_only=True)
    problem_state_hash: str = field(default="", kw_only=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "record_id": self.record_id,
            "algorithm_bundle_sha256": self.algorithm_bundle_sha256,
            "experiment_plan_id": self.experiment_plan_id,
            "island_epoch_id": self.island_epoch_id,
            "status": self.status,
            "score": self.score,
            "score_metric": self.score_metric,
            "per_instance_metrics_ref": self.per_instance_metrics_ref,
            "behavior_profile_ref": self.behavior_profile_ref,
            "runtime_metrics_ref": self.runtime_metrics_ref,
            "parent_ids": deepcopy(self.parent_ids),
            "inspiration_ids": deepcopy(self.inspiration_ids),
            "created_at": self.created_at,
            "feedback_packet_ref": self.feedback_packet_ref,
            "feedback_packet_schema": self.feedback_packet_schema,
            "problem_state_ref": self.problem_state_ref,
            "problem_state_version": self.problem_state_version,
            "problem_state_hash": self.problem_state_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentEvent:
        return _validated_from_dict(cls, d)

    def validate(self) -> None:
        _require_str(self.schema, "schema")
        _require_str(self.record_id, "record_id")
        _require_str(self.algorithm_bundle_sha256, "algorithm_bundle_sha256")
        _require_str(self.experiment_plan_id, "experiment_plan_id")
        _require_str(self.island_epoch_id, "island_epoch_id")
        _require_str(self.status, "status")
        if self.score is not None:
            _require_float(self.score, "score")
        _require_str(self.score_metric, "score_metric")
        _require_str(self.per_instance_metrics_ref, "per_instance_metrics_ref")
        _require_str(self.behavior_profile_ref, "behavior_profile_ref")
        _require_str(self.runtime_metrics_ref, "runtime_metrics_ref")
        _require_str_list(self.parent_ids, "parent_ids")
        _require_str_list(self.inspiration_ids, "inspiration_ids")
        _require_str(self.created_at, "created_at")
        _require_str(self.feedback_packet_ref, "feedback_packet_ref")
        _require_str(self.feedback_packet_schema, "feedback_packet_schema")
        _require_str(self.problem_state_ref, "problem_state_ref")
        _require_str(self.problem_state_hash, "problem_state_hash")
        if self.problem_state_version is not None:
            _require_int(self.problem_state_version, "problem_state_version")
            if self.problem_state_version < 0:
                raise ValueError("problem_state_version must be nonnegative")
        if self.status not in _EXPERIMENT_STATUSES:
            raise ValueError("status is not allowed")


@dataclass
class BehaviorProfile:
    schema: str = field(
        default="openhyra-behavior-profile.v1", kw_only=True
    )
    probe_suite: str
    probe_suite_sha256: str
    policy_artifact_sha256: str
    performance: dict
    outcome_distribution: dict
    policy_geometry: dict
    sensitivity: dict[str, float]
    robustness: dict[str, float]
    compute: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "probe_suite": self.probe_suite,
            "probe_suite_sha256": self.probe_suite_sha256,
            "policy_artifact_sha256": self.policy_artifact_sha256,
            "performance": deepcopy(self.performance),
            "outcome_distribution": deepcopy(self.outcome_distribution),
            "policy_geometry": deepcopy(self.policy_geometry),
            "sensitivity": deepcopy(self.sensitivity),
            "robustness": deepcopy(self.robustness),
            "compute": deepcopy(self.compute),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BehaviorProfile:
        return _validated_from_dict(cls, d)

    def validate(self) -> None:
        _require_str(self.schema, "schema")
        _require_str(self.probe_suite, "probe_suite")
        _require_str(self.probe_suite_sha256, "probe_suite_sha256")
        _require_str(self.policy_artifact_sha256, "policy_artifact_sha256")

        performance = _require_keys(
            self.performance,
            {"per_instance_improvement", "paired_mean", "paired_standard_error"},
            "performance",
        )
        _require_float_list(
            performance["per_instance_improvement"],
            "performance.per_instance_improvement",
        )
        _require_float(performance["paired_mean"], "performance.paired_mean")
        _require_float(
            performance["paired_standard_error"],
            "performance.paired_standard_error",
        )

        outcome = _require_keys(
            self.outcome_distribution,
            {"loss_definition", "mean_loss", "var_95", "cvar_95"},
            "outcome_distribution",
        )
        _require_str(outcome["loss_definition"], "outcome_distribution.loss_definition")
        _require_float(outcome["mean_loss"], "outcome_distribution.mean_loss")
        _require_float(outcome["var_95"], "outcome_distribution.var_95")
        _require_float(outcome["cvar_95"], "outcome_distribution.cvar_95")

        geometry = _require_keys(
            self.policy_geometry,
            {
                "exercise_rate_by_instance",
                "boundary_monotonicity_violations",
                "reference_boundary_agreement",
            },
            "policy_geometry",
        )
        _require_float_list(
            geometry["exercise_rate_by_instance"],
            "policy_geometry.exercise_rate_by_instance",
        )
        _require_int(
            geometry["boundary_monotonicity_violations"],
            "policy_geometry.boundary_monotonicity_violations",
        )
        _require_float(
            geometry["reference_boundary_agreement"],
            "policy_geometry.reference_boundary_agreement",
        )
        _require_str_float_dict(self.sensitivity, "sensitivity")
        _require_str_float_dict(self.robustness, "robustness")
        _require_str_float_dict(self.compute, "compute")

        if len(performance["per_instance_improvement"]) != len(
            geometry["exercise_rate_by_instance"]
        ):
            raise ValueError(
                "performance.per_instance_improvement and "
                "policy_geometry.exercise_rate_by_instance must have equal lengths"
            )


@dataclass
class MechanismCard:
    schema: str = field(
        default="openhyra-mechanism-card.v1", kw_only=True
    )
    record_id: str
    deterministic_facts: dict
    trusted_observations: dict
    llm_inferences: list[dict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "record_id": self.record_id,
            "deterministic_facts": deepcopy(self.deterministic_facts),
            "trusted_observations": deepcopy(self.trusted_observations),
            "llm_inferences": deepcopy(self.llm_inferences),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MechanismCard:
        return _validated_from_dict(cls, d)

    def validate(self) -> None:
        _require_str(self.schema, "schema")
        _require_str(self.record_id, "record_id")
        _require_dict(self.deterministic_facts, "deterministic_facts")
        _require_dict(self.trusted_observations, "trusted_observations")
        if not isinstance(self.llm_inferences, list):
            raise ValueError("llm_inferences must be a list[dict]")
        for index, item in enumerate(self.llm_inferences):
            inference = _require_keys(
                item,
                {"claim", "confidence", "evidence_record_ids", "annotation_event_id"},
                f"llm_inferences[{index}]",
            )
            _require_str(inference["claim"], f"llm_inferences[{index}].claim")
            _require_float(
                inference["confidence"], f"llm_inferences[{index}].confidence"
            )
            _require_str_list(
                inference["evidence_record_ids"],
                f"llm_inferences[{index}].evidence_record_ids",
            )
            _require_str(
                inference["annotation_event_id"],
                f"llm_inferences[{index}].annotation_event_id",
            )
            if not 0.0 <= inference["confidence"] <= 1.0:
                raise ValueError("llm inference confidence must be in [0.0, 1.0]")


@dataclass
class AnalogyHypothesis:
    schema: str = field(
        default="openhyra-analogy-hypothesis.v1", kw_only=True
    )
    id: str
    source_record_ids: list[str]
    target_parent_id: str
    relation_mapping: list[dict]
    non_correspondence: list[str]
    transferable_intervention: str
    predicted_effect: dict
    falsifier: str
    matched_control: dict
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "id": self.id,
            "source_record_ids": deepcopy(self.source_record_ids),
            "target_parent_id": self.target_parent_id,
            "relation_mapping": deepcopy(self.relation_mapping),
            "non_correspondence": deepcopy(self.non_correspondence),
            "transferable_intervention": self.transferable_intervention,
            "predicted_effect": deepcopy(self.predicted_effect),
            "falsifier": self.falsifier,
            "matched_control": deepcopy(self.matched_control),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnalogyHypothesis:
        return _validated_from_dict(cls, d)

    def validate(self) -> None:
        _require_str(self.schema, "schema")
        _require_str(self.id, "id")
        _require_str_list(self.source_record_ids, "source_record_ids")
        _require_str(self.target_parent_id, "target_parent_id")
        if not isinstance(self.relation_mapping, list):
            raise ValueError("relation_mapping must be a list[dict]")
        if not self.relation_mapping:
            raise ValueError("relation_mapping must be non-empty")
        for index, item in enumerate(self.relation_mapping):
            mapping = _require_keys(
                item,
                {"source_role", "target_role", "shared_relation"},
                f"relation_mapping[{index}]",
            )
            _require_str(mapping["source_role"], f"relation_mapping[{index}].source_role")
            _require_str(mapping["target_role"], f"relation_mapping[{index}].target_role")
            _require_str(
                mapping["shared_relation"],
                f"relation_mapping[{index}].shared_relation",
            )
        _require_str_list(self.non_correspondence, "non_correspondence")
        _require_str(self.transferable_intervention, "transferable_intervention")
        predicted = _require_keys(
            self.predicted_effect,
            {"metric", "direction", "minimum_effect"},
            "predicted_effect",
        )
        _require_str(predicted["metric"], "predicted_effect.metric")
        _require_str(predicted["direction"], "predicted_effect.direction")
        _require_float(predicted["minimum_effect"], "predicted_effect.minimum_effect")
        _require_str(self.falsifier, "falsifier")
        _require_dict(self.matched_control, "matched_control")
        _require_str(self.status, "status")
        if self.status not in {"preregistered", "executing", "completed"}:
            raise ValueError("status is not allowed")
        if predicted["direction"] not in {"positive", "negative"}:
            raise ValueError("predicted_effect.direction is not allowed")


@dataclass
class AnalogyResult:
    schema: str = field(
        default="openhyra-analogy-result.v1", kw_only=True
    )
    analogy_hypothesis_id: str
    guided_record_id: str
    control_record_id: str
    guided_delta: float
    control_delta: float
    transfer_gain: float
    transfer_gain_standard_error: float
    predicted_slice_effect: float
    prediction_direction_correct: bool
    verdict: str
    # Optional evaluator-owned paired-cell statistics.  They are keyword-only
    # so archived v1 constructors and JSON lines remain readable.  The legacy
    # ``transfer_gain_standard_error`` field now means the actual standard
    # error of per-cell transfer gains; it is never a relative-effect ratio.
    paired_cell_count: int = field(default=0, kw_only=True)
    transfer_gain_ci_low: float | None = field(default=None, kw_only=True)
    transfer_gain_ci_high: float | None = field(default=None, kw_only=True)
    relative_transfer_gain: float | None = field(default=None, kw_only=True)
    invalid_control_reason: str | None = field(default=None, kw_only=True)
    control_valid: bool | None = field(default=None, kw_only=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "analogy_hypothesis_id": self.analogy_hypothesis_id,
            "guided_record_id": self.guided_record_id,
            "control_record_id": self.control_record_id,
            "guided_delta": self.guided_delta,
            "control_delta": self.control_delta,
            "transfer_gain": self.transfer_gain,
            "transfer_gain_standard_error": self.transfer_gain_standard_error,
            "predicted_slice_effect": self.predicted_slice_effect,
            "prediction_direction_correct": self.prediction_direction_correct,
            "verdict": self.verdict,
            "paired_cell_count": self.paired_cell_count,
            "transfer_gain_ci_low": self.transfer_gain_ci_low,
            "transfer_gain_ci_high": self.transfer_gain_ci_high,
            "relative_transfer_gain": self.relative_transfer_gain,
            "invalid_control_reason": self.invalid_control_reason,
            "control_valid": self.control_valid,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnalogyResult:
        return _validated_from_dict(cls, d)

    def validate(self) -> None:
        _require_str(self.schema, "schema")
        _require_str(self.analogy_hypothesis_id, "analogy_hypothesis_id")
        _require_str(self.guided_record_id, "guided_record_id")
        _require_str(self.control_record_id, "control_record_id")
        _require_float(self.guided_delta, "guided_delta")
        _require_float(self.control_delta, "control_delta")
        _require_float(self.transfer_gain, "transfer_gain")
        _require_float(
            self.transfer_gain_standard_error, "transfer_gain_standard_error"
        )
        _require_float(self.predicted_slice_effect, "predicted_slice_effect")
        _require_bool(
            self.prediction_direction_correct, "prediction_direction_correct"
        )
        _require_int(self.paired_cell_count, "paired_cell_count")
        if self.paired_cell_count < 0:
            raise ValueError("paired_cell_count must be non-negative")
        if self.transfer_gain_ci_low is not None:
            _require_float(self.transfer_gain_ci_low, "transfer_gain_ci_low")
        if self.transfer_gain_ci_high is not None:
            _require_float(self.transfer_gain_ci_high, "transfer_gain_ci_high")
        if (self.transfer_gain_ci_low is None) != (
            self.transfer_gain_ci_high is None
        ):
            raise ValueError(
                "transfer_gain_ci_low and transfer_gain_ci_high must be set together"
            )
        if self.relative_transfer_gain is not None:
            _require_float(self.relative_transfer_gain, "relative_transfer_gain")
        if self.invalid_control_reason is not None:
            _require_str(self.invalid_control_reason, "invalid_control_reason")
        if self.control_valid is not None:
            _require_bool(self.control_valid, "control_valid")
        if self.control_valid is False and not self.invalid_control_reason:
            raise ValueError(
                "control_valid=False requires invalid_control_reason"
            )
        _require_str(self.verdict, "verdict")
        if self.verdict not in {
            "transfer_supported",
            "transfer_refuted",
            "inconclusive",
            "invalid_control",
            "execution_failed",
        }:
            raise ValueError("verdict is not allowed")

    @property
    def paired_standard_error(self) -> float:
        """Descriptive alias for the evaluator-owned paired-cell SE."""
        return self.transfer_gain_standard_error

    @property
    def paired_ci(self) -> tuple[float, float] | None:
        """Return the confidence interval when per-cell evidence exists."""
        if self.transfer_gain_ci_low is None or self.transfer_gain_ci_high is None:
            return None
        return self.transfer_gain_ci_low, self.transfer_gain_ci_high


@dataclass
class ExperimentPlan:
    schema: str = field(
        default="openhyra-experiment-plan.v1", kw_only=True
    )
    id: str
    action: str
    target_island_epoch_id: str
    generation_operator: str
    parent_ids: list[str]
    inspiration_ids: list[str]
    analogy_hypothesis_id: str | None
    implementation_intent: str
    negative_constraints: list[str]
    success_criterion: str
    budget: dict
    # Feedback-aware typed intervention fields.  They are keyword-only
    # extensions so archived v1 plans and positional constructors remain
    # readable.  The fields describe a probe; they are not evaluator claims.
    phase: str = field(default="numeric", kw_only=True)
    intervention_scope: str | None = field(default=None, kw_only=True)
    intervention_operator: str | None = field(default=None, kw_only=True)
    target_slice: str | None = field(default=None, kw_only=True)
    prediction: str | None = field(default=None, kw_only=True)
    falsifier: str | None = field(default=None, kw_only=True)
    evidence_ids: list[str] = field(default_factory=list, kw_only=True)
    next_probe: str | None = field(default=None, kw_only=True)
    state_version: str | int | None = field(default=None, kw_only=True)
    state_hash: str | None = field(default=None, kw_only=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "id": self.id,
            "action": self.action,
            "target_island_epoch_id": self.target_island_epoch_id,
            "generation_operator": self.generation_operator,
            "parent_ids": deepcopy(self.parent_ids),
            "inspiration_ids": deepcopy(self.inspiration_ids),
            "analogy_hypothesis_id": self.analogy_hypothesis_id,
            "implementation_intent": self.implementation_intent,
            "negative_constraints": deepcopy(self.negative_constraints),
            "success_criterion": self.success_criterion,
            "budget": deepcopy(self.budget),
            "phase": self.phase,
            "intervention_scope": self.intervention_scope,
            "intervention_operator": self.intervention_operator,
            "target_slice": self.target_slice,
            "prediction": self.prediction,
            "falsifier": self.falsifier,
            "evidence_ids": deepcopy(self.evidence_ids),
            "next_probe": self.next_probe,
            "state_version": self.state_version,
            "state_hash": self.state_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentPlan:
        return _validated_from_dict(cls, d)

    def validate(self) -> None:
        _require_str(self.schema, "schema")
        _require_str(self.id, "id")
        _require_str(self.action, "action")
        _require_str(self.target_island_epoch_id, "target_island_epoch_id")
        _require_str(self.generation_operator, "generation_operator")
        _require_str_list(self.parent_ids, "parent_ids")
        _require_str_list(self.inspiration_ids, "inspiration_ids")
        if self.analogy_hypothesis_id is not None:
            _require_str(self.analogy_hypothesis_id, "analogy_hypothesis_id")
        _require_str(self.implementation_intent, "implementation_intent")
        _require_str_list(self.negative_constraints, "negative_constraints")
        _require_str(self.success_criterion, "success_criterion")
        _require_str(self.phase, "phase")
        for name, value in (
            ("intervention_scope", self.intervention_scope),
            ("intervention_operator", self.intervention_operator),
            ("target_slice", self.target_slice),
            ("prediction", self.prediction),
            ("falsifier", self.falsifier),
            ("next_probe", self.next_probe),
            ("state_hash", self.state_hash),
        ):
            if value is not None:
                _require_str(value, name)
        _require_str_list(self.evidence_ids, "evidence_ids")
        if self.state_version is not None and (
            isinstance(self.state_version, bool)
            or not isinstance(self.state_version, (int, str))
        ):
            raise ValueError("state_version must be an int, str, or None")
        budget = _require_keys(
            self.budget,
            {"candidate_count", "sandbox_seconds_per_cell", "max_artifact_bytes"},
            "budget",
        )
        _require_int(budget["candidate_count"], "budget.candidate_count")
        _require_int(
            budget["sandbox_seconds_per_cell"], "budget.sandbox_seconds_per_cell"
        )
        _require_int(budget["max_artifact_bytes"], "budget.max_artifact_bytes")
        if self.action not in {"continue", "stop", "reset"}:
            raise ValueError("action is not allowed")
        if self.generation_operator not in _GENERATION_OPERATORS:
            raise ValueError("generation_operator is not allowed")


@dataclass
class IslandEpoch:
    schema: str = field(
        default="openhyra-island-epoch.v1", kw_only=True
    )
    island_id: str
    epoch: int
    seed_record_ids: list[str]
    started_after_context_round: int
    proposal_seed: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "island_id": self.island_id,
            "epoch": self.epoch,
            "seed_record_ids": deepcopy(self.seed_record_ids),
            "started_after_context_round": self.started_after_context_round,
            "proposal_seed": self.proposal_seed,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IslandEpoch:
        return _validated_from_dict(cls, d)

    def validate(self) -> None:
        _require_str(self.schema, "schema")
        _require_str(self.island_id, "island_id")
        _require_int(self.epoch, "epoch")
        _require_str_list(self.seed_record_ids, "seed_record_ids")
        _require_int(self.started_after_context_round, "started_after_context_round")
        _require_int(self.proposal_seed, "proposal_seed")
        _require_str(self.status, "status")
        if self.status not in {"active", "culled", "completed"}:
            raise ValueError("status is not allowed")


@dataclass
class AnnotationEvent:
    schema: str = field(
        default="openhyra-annotation-event.v1", kw_only=True
    )
    id: str
    annotation_type: str
    target_record_ids: list[str]
    evidence_record_ids: list[str]
    model: str
    backend: str
    prompt_sha256: str
    response_sha256: str
    parser_schema: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "id": self.id,
            "annotation_type": self.annotation_type,
            "target_record_ids": deepcopy(self.target_record_ids),
            "evidence_record_ids": deepcopy(self.evidence_record_ids),
            "model": self.model,
            "backend": self.backend,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "parser_schema": self.parser_schema,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnnotationEvent:
        return _validated_from_dict(cls, d)

    def validate(self) -> None:
        _require_str(self.schema, "schema")
        _require_str(self.id, "id")
        _require_str(self.annotation_type, "annotation_type")
        _require_str_list(self.target_record_ids, "target_record_ids")
        _require_str_list(self.evidence_record_ids, "evidence_record_ids")
        _require_str(self.model, "model")
        _require_str(self.backend, "backend")
        _require_str(self.prompt_sha256, "prompt_sha256")
        _require_str(self.response_sha256, "response_sha256")
        _require_str(self.parser_schema, "parser_schema")
        _require_str(self.created_at, "created_at")


__all__ = [
    "AlgorithmBundle",
    "FrozenPolicyArtifact",
    "ExperimentEvent",
    "BehaviorProfile",
    "MechanismCard",
    "AnalogyHypothesis",
    "AnalogyResult",
    "ExperimentPlan",
    "IslandEpoch",
    "AnnotationEvent",
]

# The feedback/state schemas live in their own module so task plugins can use
# them without importing the full V5 event schema.  Re-exporting them here
# keeps the historical ``schemas_v5`` import surface additive and backwards
# compatible.
from feedback import (  # noqa: E402  (import after legacy declarations)
    BELIEF_CELL_SCHEMA,
    DIRECTIONAL_FEEDBACK_SCHEMA,
    FEEDBACK_PACKET_SCHEMA,
    NOT_OBSERVED,
    PROBLEM_STATE_SCHEMA,
    BeliefCell,
    BeliefReducer,
    DirectionalFeedback,
    FeedbackPacket,
    ProblemState,
    ProblemStateLog,
    is_not_observed,
    not_observed,
)

__all__.extend([
    "FEEDBACK_PACKET_SCHEMA",
    "DIRECTIONAL_FEEDBACK_SCHEMA",
    "PROBLEM_STATE_SCHEMA",
    "BELIEF_CELL_SCHEMA",
    "NOT_OBSERVED",
    "DirectionalFeedback",
    "FeedbackPacket",
    "BeliefCell",
    "ProblemState",
    "BeliefReducer",
    "ProblemStateLog",
    "not_observed",
    "is_not_observed",
])
