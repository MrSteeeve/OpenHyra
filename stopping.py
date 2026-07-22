"""Structured Context decisions and deterministic review of stop requests."""

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ContextDecision:
    action: str
    analysis: str
    reason: str
    expected_gain: float | None
    confidence: float | None
    next_experiment: str | None

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
        return cls(
            action=action,
            analysis=analysis.strip(),
            reason=reason.strip(),
            expected_gain=expected_gain,
            confidence=confidence,
            next_experiment=next_experiment.strip() if next_experiment else None,
        )

    def to_dict(self):
        payload = asdict(self)
        payload["next"] = payload.pop("next_experiment")
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


def stopping_evidence(records, *, direction, policy):
    """Summarize completed Contexts using evaluator records, not LLM claims."""
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

    running_best = _pick_score(baseline_scores, direction)
    last_meaningful_position = None
    context_improvements = []
    ordered = sorted(grouped)
    for position, iteration in enumerate(ordered, start=1):
        scores = [
            record["score"] for record in grouped[iteration]
            if record.get("status") == "ok" and record.get("score") is not None
        ]
        context_best = _pick_score(scores, direction)
        delta = 0.0
        meaningful = False
        if context_best is not None:
            if running_best is None:
                meaningful = True
                delta = math.inf
                running_best = context_best
            else:
                improvement = (
                    context_best - running_best
                    if direction == "max"
                    else running_best - context_best
                )
                if improvement > 0:
                    delta = improvement
                    running_best = context_best
                    meaningful = improvement >= policy.meaningful_delta
        if meaningful:
            last_meaningful_position = position
        context_improvements.append({
            "iteration": iteration,
            "best_score": context_best,
            "improvement": None if math.isinf(delta) else delta,
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
    return {
        "completed_contexts": completed,
        "incomplete_contexts": sorted(incomplete),
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
    }


class StopController:
    """Treat an Agent stop as a request gated by deterministic evidence."""

    def __init__(self, policy, direction):
        self.policy = policy
        self.direction = direction

    def review(self, decision, records):
        evidence = stopping_evidence(
            records, direction=self.direction, policy=self.policy,
        )
        reasons = []
        if decision.action != "stop":
            reasons.append("context_requested_continue")
        if not self.policy.enabled:
            reasons.append("agent_stop_disabled")
        if evidence["completed_contexts"] < self.policy.min_contexts_before_stop:
            reasons.append("minimum_contexts_not_met")
        if (evidence["contexts_since_meaningful_improvement"] <
                self.policy.stop_patience):
            reasons.append("patience_not_met")
        if (evidence["recent_successful_candidates"] <
                self.policy.min_successful_candidates):
            reasons.append("insufficient_successful_candidates")
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
