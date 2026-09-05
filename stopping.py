"""Structured Context decisions and deterministic review of stop requests."""

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from mechanism_hypotheses import normalize_hypotheses


CONTEXT_PHASES = (
    "numeric",
    "discover",
    "diagnose",
    "transfer",
    "confirm",
    "construct",
    "falsify",
    "formalize",
    "repair_formalization",
)
INTERVENTION_SCOPES = {
    "parameter", "target", "representation", "architecture", "mechanism", "family",
    "objective", "loss", "feature", "protocol", "probe",
    "whole_program", "program_subtree", "fit", "predict", "subsystem",
}
INTERVENTION_OPERATORS = {
    "tune", "replace", "combine", "ablate", "transfer", "abandon", "probe",
    "mutate", "switch", "adjust", "compose", "remove", "restart", "inspect",
    "modify", "change", "add", "delete",
    # Executable whole-program operators.  The legacy spellings above remain
    # accepted for archived Context packets; new research packets should use
    # these four explicit names so the Harness can dispatch them without
    # interpreting free-form prose.
    "whole_program_restart", "ast_mutation", "ast_crossover", "subsystem_rewrite",
}


@dataclass(frozen=True)
class ContextDecision:
    action: str
    analysis: str
    reason: str
    expected_gain: float | None
    confidence: float | None
    next_experiment: str | None
    phase: str = "numeric"
    target_claim_id: str | None = None
    success_criterion: str | None = None
    # Context may propose a small portfolio of mechanisms in addition to its
    # single next-experiment direction.  Keeping this optional preserves the
    # legacy decision shape while letting task-specific proposal flows carry a
    # structured, falsifiable hypothesis list.
    mechanism_candidates: tuple[dict, ...] = ()
    # Typed intervention fields make the Context output executable by a
    # deterministic router.  All are optional for legacy tasks and old JSON
    # analysis files.
    intervention_scope: str | None = None
    intervention_operator: str | None = None
    target_slice: str | None = None
    prediction: str | None = None
    falsifier: str | None = None
    evidence_ids: tuple[str, ...] = ()
    next_probe: str | None = None
    state_version: str | int | None = None
    state_hash: str | None = None

    @property
    def scope(self) -> str | None:
        """Short alias used by compact Context payloads."""
        return self.intervention_scope

    @property
    def operator(self) -> str | None:
        """Short alias used by compact Context payloads."""
        return self.intervention_operator

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("Context decision must be a JSON object")
        action = payload.get("action")
        if action not in {"continue", "stop"}:
            raise ValueError("Context action must be 'continue' or 'stop'")
        analysis = payload.get("analysis")
        reason = payload.get("reason")
        if not isinstance(analysis, str) or not analysis.strip():
            raise ValueError("Context analysis must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Context reason must be a non-empty string")

        def optional_number(name):
            value = payload.get(name)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Context {name} must be numeric or null")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Context {name} must be finite")
            return value

        expected_gain = optional_number("expected_gain")
        confidence = optional_number("confidence")
        if expected_gain is not None and expected_gain < 0:
            raise ValueError("Context expected_gain must be non-negative")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("Context confidence must be within [0, 1]")
        next_experiment = payload.get("next")
        if next_experiment is not None and (
                not isinstance(next_experiment, str) or not next_experiment.strip()):
            raise ValueError("Context next must be a non-empty string or null")
        if action == "continue" and next_experiment is None:
            raise ValueError("A continue decision requires a next experiment")
        if action == "stop" and next_experiment is not None:
            raise ValueError("A stop decision requires next=null")
        phase = payload.get("phase", "numeric")
        if phase not in set(CONTEXT_PHASES):
            raise ValueError("Context phase is not supported")
        target_claim_id = payload.get("target_claim_id")
        if target_claim_id is not None and (
            not isinstance(target_claim_id, str)
            or not target_claim_id.strip()
            or len(target_claim_id) > 64
        ):
            raise ValueError(
                "Context target_claim_id must be bounded text or null"
            )
        success_criterion = payload.get("success_criterion")
        if success_criterion is not None and (
            not isinstance(success_criterion, str)
            or not success_criterion.strip()
            or len(success_criterion) > 500
        ):
            raise ValueError(
                "Context success_criterion must be bounded text or null"
            )
        raw_mechanisms = payload.get("mechanism_candidates", [])
        if raw_mechanisms is None:
            raw_mechanisms = []
        if not isinstance(raw_mechanisms, (list, tuple)):
            # The field is an optional extension.  Keep a valid legacy
            # continue/stop decision usable when an LLM emits a single object
            # or another malformed optional value.
            raw_mechanisms = []
        mechanism_candidates = tuple(
            normalize_hypotheses(
                raw_mechanisms,
                source="context",
                limit=8,
            )
        )
        # Accept both the compact top-level representation and a nested
        # ``intervention`` object.  Keeping aliases in the serialized form
        # lets older Context prompts continue to work while newer routers can
        # consume a typed record directly.
        raw_intervention = payload.get("intervention")
        if not isinstance(raw_intervention, dict):
            raw_intervention = {}

        def optional_text(name, *, limit=500, aliases=()):
            value = payload.get(name)
            if value is None:
                value = raw_intervention.get(name)
            if value is None:
                for alias in aliases:
                    value = payload.get(alias)
                    if value is None:
                        value = raw_intervention.get(alias)
                    if value is not None:
                        break
            if value is None:
                return None
            if name == "target_slice" and isinstance(value, (list, tuple)):
                value = ", ".join(
                    item.strip() for item in value
                    if isinstance(item, str) and item.strip()
                )
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Context {name} must be bounded text or null")
            value = " ".join(value.split()).strip()
            if len(value) > limit:
                raise ValueError(f"Context {name} exceeds character limit")
            return value

        intervention_scope = optional_text(
            "intervention_scope", limit=64, aliases=("scope",)
        )
        intervention_operator = optional_text(
            "intervention_operator", limit=64, aliases=("operator",)
        )
        if intervention_scope is not None and intervention_scope not in INTERVENTION_SCOPES:
            raise ValueError("Context intervention_scope is not supported")
        if intervention_operator is not None and intervention_operator not in INTERVENTION_OPERATORS:
            raise ValueError("Context intervention_operator is not supported")
        target_slice = optional_text("target_slice", limit=240, aliases=("target_slices",))
        prediction = optional_text("prediction", limit=500)
        falsifier = optional_text("falsifier", limit=500, aliases=("failure_condition",))
        next_probe = optional_text("next_probe", limit=500)
        raw_evidence = payload.get("evidence_ids", raw_intervention.get("evidence_ids", ()))
        if isinstance(raw_evidence, str):
            raw_evidence = [raw_evidence]
        if raw_evidence is None:
            raw_evidence = []
        if not isinstance(raw_evidence, (list, tuple)):
            raise ValueError("Context evidence_ids must be a list or null")
        evidence_ids = []
        for evidence_id in raw_evidence:
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise ValueError("Context evidence_ids must contain text")
            evidence_id = evidence_id.strip()
            if len(evidence_id) > 160:
                raise ValueError("Context evidence_id exceeds character limit")
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
            if len(evidence_ids) >= 16:
                break
        state_version = payload.get("state_version", raw_intervention.get("state_version"))
        if isinstance(state_version, bool):
            raise ValueError("Context state_version must be text, integer, or null")
        if isinstance(state_version, float):
            if not state_version.is_integer():
                raise ValueError("Context state_version must be integral")
            state_version = int(state_version)
        elif state_version is not None and not isinstance(state_version, int):
            if not isinstance(state_version, str) or not state_version.strip():
                raise ValueError("Context state_version must be text, integer, or null")
            state_version = state_version.strip()
            if len(state_version) > 128:
                raise ValueError("Context state_version exceeds character limit")
        state_hash = payload.get("state_hash", raw_intervention.get("state_hash"))
        if state_hash is not None:
            if not isinstance(state_hash, str) or not state_hash.strip() or len(state_hash) > 128:
                raise ValueError("Context state_hash must be bounded text or null")
            state_hash = state_hash.strip()
        return cls(
            action=action,
            analysis=analysis.strip(),
            reason=reason.strip(),
            expected_gain=expected_gain,
            confidence=confidence,
            next_experiment=next_experiment.strip() if next_experiment else None,
            phase=phase,
            target_claim_id=(
                target_claim_id.strip() if target_claim_id else None
            ),
            success_criterion=(
                success_criterion.strip() if success_criterion else None
            ),
            mechanism_candidates=mechanism_candidates,
            intervention_scope=intervention_scope,
            intervention_operator=intervention_operator,
            target_slice=target_slice,
            prediction=prediction,
            falsifier=falsifier,
            evidence_ids=tuple(evidence_ids),
            next_probe=next_probe,
            state_version=state_version,
            state_hash=state_hash,
        )

    def to_dict(self):
        payload = asdict(self)
        payload["next"] = payload.pop("next_experiment")
        payload["mechanism_candidates"] = [
            dict(item) for item in self.mechanism_candidates
        ]
        payload["evidence_ids"] = list(self.evidence_ids)
        payload["intervention"] = {
            "scope": self.intervention_scope,
            "operator": self.intervention_operator,
            "target_slice": self.target_slice,
            "prediction": self.prediction,
            "falsifier": self.falsifier,
            "evidence_ids": list(self.evidence_ids),
            "next_probe": self.next_probe,
            "state_version": self.state_version,
            "state_hash": self.state_hash,
        }
        return payload

    def forced_continue(self, next_experiment, reason):
        return replace(
            self,
            action="continue",
            reason=reason,
            next_experiment=next_experiment,
        )


@dataclass(frozen=True)
class StopPolicy:
    enabled: bool = False
    min_contexts_before_stop: int = 6
    stop_patience: int = 4
    meaningful_delta: float = 0.0001
    recent_window: int = 4
    min_successful_candidates: int = 4

    def __post_init__(self):
        if self.min_contexts_before_stop < 0:
            raise ValueError("min_contexts_before_stop must be >= 0")
        if self.stop_patience < 0:
            raise ValueError("stop_patience must be >= 0")
        if self.meaningful_delta < 0:
            raise ValueError("meaningful_delta must be >= 0")
        if self.recent_window < 1:
            raise ValueError("recent_window must be >= 1")
        if self.min_successful_candidates < 0:
            raise ValueError("min_successful_candidates must be >= 0")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class StopReview:
    accepted: bool
    reasons: tuple[str, ...]
    evidence: dict

    def to_dict(self):
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "evidence": self.evidence,
        }


def _pick_score(scores, direction):
    if not scores:
        return None
    return (max if direction == "max" else min)(scores)


def _group_context_records(records):
    """Return complete Context groups, incomplete iterations and seed scores."""
    all_grouped = {}
    baseline_scores = []
    for record in records:
        iteration = record.get("metadata", {}).get("iteration")
        if not isinstance(iteration, int):
            if record.get("status") == "ok" and record.get("score") is not None:
                baseline_scores.append(record["score"])
            continue
        all_grouped.setdefault(iteration, []).append(record)

    grouped = {}
    incomplete = []
    for iteration, iteration_records in all_grouped.items():
        candidate_indexes = {
            record.get("metadata", {}).get("candidate_index")
            for record in iteration_records
            if isinstance(record.get("metadata", {}).get("candidate_index"), int)
        }
        expected_counts = [
            record.get("metadata", {}).get("candidate_count")
            for record in iteration_records
            if isinstance(record.get("metadata", {}).get("candidate_count"), int)
        ]
        if expected_counts and len(candidate_indexes) < max(expected_counts):
            incomplete.append(iteration)
            continue
        grouped[iteration] = iteration_records
    return grouped, sorted(incomplete), baseline_scores


def incomplete_contexts(records):
    """Return Context iterations whose expected candidates are not all in EB."""
    _grouped, incomplete, _baseline_scores = _group_context_records(records)
    return incomplete


def stopping_evidence(
    records,
    *,
    direction,
    policy,
    required_formal_claims=(),
):
    """Summarize completed Contexts using evaluator records, not LLM claims."""
    grouped, incomplete, baseline_scores = _group_context_records(records)

    running_best = _pick_score(baseline_scores, direction)
    last_meaningful_best = running_best
    last_meaningful_position = None
    context_improvements = []
    ordered = sorted(grouped)
    for position, iteration in enumerate(ordered, start=1):
        scores = [
            record["score"] for record in grouped[iteration]
            if record.get("status") == "ok" and record.get("score") is not None
        ]
        context_best = _pick_score(scores, direction)
        incremental_gain = 0.0
        cumulative_gain = 0.0
        meaningful = False
        if context_best is not None:
            if running_best is None:
                meaningful = True
                running_best = context_best
                last_meaningful_best = context_best
            else:
                improvement = (
                    context_best - running_best
                    if direction == "max"
                    else running_best - context_best
                )
                if improvement > 0:
                    incremental_gain = improvement
                    running_best = context_best
                cumulative_gain = (
                    running_best - last_meaningful_best
                    if direction == "max"
                    else last_meaningful_best - running_best
                )
                meaningful = (
                    cumulative_gain > 0 and
                    cumulative_gain + 1e-15 >= policy.meaningful_delta
                )
                if meaningful:
                    last_meaningful_best = running_best
        if meaningful:
            last_meaningful_position = position
        context_improvements.append({
            "iteration": iteration,
            "best_score": context_best,
            "improvement": incremental_gain,
            "cumulative_gain": cumulative_gain,
            "meaningful": meaningful,
        })

    recent_iterations = set(ordered[-policy.recent_window:])
    candidate_outcomes = {}
    directions = set()
    for iteration in recent_iterations:
        for record in grouped[iteration]:
            metadata = record.get("metadata", {})
            candidate_index = metadata.get("candidate_index")
            if not isinstance(candidate_index, int):
                continue
            key = (iteration, candidate_index)
            outcome = candidate_outcomes.setdefault(
                key, {"successful": False, "duplicate": False},
            )
            if record.get("status") == "ok" and record.get("score") is not None:
                outcome["successful"] = True
                outcome["duplicate"] = bool(metadata.get("duplicate_of"))
            direction_label = metadata.get("direction")
            if isinstance(direction_label, str) and direction_label.strip():
                directions.add(direction_label.strip())

    successful = sum(item["successful"] for item in candidate_outcomes.values())
    duplicate = sum(
        item["successful"] and item["duplicate"]
        for item in candidate_outcomes.values()
    )
    completed = len(ordered)
    contexts_since = (
        completed - last_meaningful_position
        if last_meaningful_position is not None else completed
    )
    required_formal_claims = tuple(sorted(set(required_formal_claims)))
    formal_complete_records = []
    formal_targets = {}
    for record in records:
        metrics = record.get("metrics", {})
        if metrics.get("formalization_status") != "verified":
            continue
        refutation_counts = [
            metrics.get(field)
            for field in (
                "refuted_claim_count",
                "refuted_obligation_count",
                "refuted_certificate_count",
            )
        ]
        if not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value == 0
            for value in refutation_counts
        ):
            continue
        grouped_targets = {}
        for item in metrics.get("formal_checked_targets", []):
            target = item.get("target")
            template = item.get("template")
            if not isinstance(target, dict) or not isinstance(template, str):
                continue
            key = json.dumps(target, sort_keys=True, separators=(",", ":"))
            grouped_targets.setdefault(key, set()).add(template)
        for target_key, templates in grouped_targets.items():
            if set(required_formal_claims).issubset(templates):
                formal_complete_records.append(record["id"])
                formal_targets[record["id"]] = json.loads(target_key)
                break

    return {
        "completed_contexts": completed,
        "incomplete_contexts": incomplete,
        "contexts_since_meaningful_improvement": contexts_since,
        "meaningful_delta": policy.meaningful_delta,
        "recent_window": policy.recent_window,
        "recent_candidate_count": len(candidate_outcomes),
        "recent_successful_candidates": successful,
        "recent_duplicate_candidates": duplicate,
        "recent_duplicate_rate": duplicate / successful if successful else None,
        "covered_direction_count": len(directions),
        "best_score": running_best,
        "context_improvements": context_improvements,
        "required_formal_claim_templates": list(required_formal_claims),
        "formal_complete_records": formal_complete_records,
        "formal_complete_targets": formal_targets,
        "proof_complete": (
            bool(formal_complete_records)
            if required_formal_claims else True
        ),
    }


class StopController:
    """Treat an Agent stop as a request gated by deterministic evidence."""

    def __init__(self, policy, direction, required_formal_claims=()):
        self.policy = policy
        self.direction = direction
        self.required_formal_claims = tuple(required_formal_claims)

    def review(self, decision, records):
        evidence = stopping_evidence(
            records,
            direction=self.direction,
            policy=self.policy,
            required_formal_claims=self.required_formal_claims,
        )
        reasons = []
        if decision.action != "stop":
            reasons.append("context_requested_continue")
        if not self.policy.enabled:
            reasons.append("agent_stop_disabled")
        if evidence["incomplete_contexts"]:
            reasons.append("incomplete_contexts_exist")
        if evidence["completed_contexts"] < self.policy.min_contexts_before_stop:
            reasons.append("minimum_contexts_not_met")
        if (evidence["contexts_since_meaningful_improvement"] <
                self.policy.stop_patience):
            reasons.append("patience_not_met")
        if (evidence["recent_successful_candidates"] <
                self.policy.min_successful_candidates):
            reasons.append("insufficient_successful_candidates")
        if not evidence["proof_complete"]:
            reasons.append("required_formal_claims_not_complete")
        accepted = decision.action == "stop" and not reasons
        return StopReview(accepted, tuple(reasons), evidence)


def write_termination(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **payload,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    return item
