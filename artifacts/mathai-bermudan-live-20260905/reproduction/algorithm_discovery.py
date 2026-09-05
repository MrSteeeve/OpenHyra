"""Task-independent coordination for open algorithm discovery.

A search space emits complete algorithms, an evaluator returns measured
outcomes, feedback updates the problem state, and the next generation may use
those observations when choosing parents and operators.  The concrete whole-
program implementation lives in :mod:`program_search`; task evaluators remain
responsible for executing and scoring candidate programs.

Providing this loop is an implemented search capability, not evidence that a
scientifically novel algorithm has already been discovered.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from feedback import (
    BeliefReducer,
    FeedbackPacket,
    ProblemState,
    ProblemStateLog,
)


ALGORITHM_DISCOVERY_SCHEMA = "openhyra-algorithm-discovery.v1"
ALGORITHM_SPEC_SCHEMA = "openhyra-algorithm-spec.v1"
EVALUATION_RESULT_SCHEMA = "openhyra-algorithm-evaluation.v1"
DISCOVERY_EVENT_SCHEMA = "openhyra-discovery-event.v1"


def _json(value: Any) -> Any:
    """Return a JSON-compatible copy and reject non-finite numeric values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float is not allowed")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    raise ValueError(f"value is not JSON-compatible: {type(value).__name__}")


def _hash(payload: Any, prefix: str) -> str:
    encoded = json.dumps(
        _json(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a{'n' if not empty else ''} non-empty string")
    return value.strip() if value else value


@dataclass(frozen=True)
class AlgorithmSpec:
    """A complete, serialisable candidate algorithm description.

    ``implementation`` is an opaque task-owned payload (for example a
    manifest or a finite program AST).  It is never evaluated by this module;
    the task evaluator remains the authority for execution and acceptance.
    """

    candidate_id: str
    family: str
    implementation: Any = field(default_factory=dict)
    parent_ids: tuple[str, ...] = ()
    mechanism_id: str = ""
    operator: str = "local_mutation"
    prediction: Any = "not_observed"
    falsifier: Any = "not_observed"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ALGORITHM_SPEC_SCHEMA

    def validate(self) -> None:
        if self.schema != ALGORITHM_SPEC_SCHEMA:
            raise ValueError(f"schema must be {ALGORITHM_SPEC_SCHEMA}")
        _text(self.candidate_id, "candidate_id")
        _text(self.family, "family")
        if not isinstance(self.parent_ids, tuple) or any(
            not isinstance(item, str) or not item for item in self.parent_ids
        ):
            raise ValueError("parent_ids must be tuple[str, ...]")
        _text(self.mechanism_id, "mechanism_id", empty=True)
        _text(self.operator, "operator")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        _json(self.implementation)
        _json(self.prediction)
        _json(self.falsifier)
        _json(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "family": self.family,
            "implementation": _json(self.implementation),
            "parent_ids": list(self.parent_ids),
            "mechanism_id": self.mechanism_id,
            "operator": self.operator,
            "prediction": _json(self.prediction),
            "falsifier": _json(self.falsifier),
            "metadata": _json(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlgorithmSpec":
        if not isinstance(raw, Mapping):
            raise ValueError("algorithm spec must be a mapping")
        allowed = {
            "schema", "candidate_id", "family", "implementation", "parent_ids",
            "mechanism_id", "operator", "prediction", "falsifier", "metadata",
        }
        unknown = set(raw).difference(allowed)
        if unknown:
            raise ValueError("unknown algorithm spec field(s): " + ", ".join(sorted(unknown)))
        item = cls(
            schema=raw.get("schema", ALGORITHM_SPEC_SCHEMA),
            candidate_id=raw.get("candidate_id", ""),
            family=raw.get("family", ""),
            implementation=raw.get("implementation", {}),
            parent_ids=tuple(raw.get("parent_ids", ())),
            mechanism_id=raw.get("mechanism_id", ""),
            operator=raw.get("operator", "local_mutation"),
            prediction=raw.get("prediction", "not_observed"),
            falsifier=raw.get("falsifier", "not_observed"),
            metadata=raw.get("metadata", {}),
        )
        item.validate()
        return item


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluator-owned outcome for one candidate and one data split."""

    candidate_id: str
    status: str
    score: float | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    feedback: FeedbackPacket | None = None
    split: str = "development"
    seed: int = 0
    cost: Mapping[str, Any] = field(default_factory=dict)
    artifact_sha256: str = ""
    schema: str = EVALUATION_RESULT_SCHEMA

    def validate(self) -> None:
        if self.schema != EVALUATION_RESULT_SCHEMA:
            raise ValueError(f"schema must be {EVALUATION_RESULT_SCHEMA}")
        _text(self.candidate_id, "candidate_id")
        _text(self.status, "status")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
        ):
            raise ValueError("score must be finite or None")
        if self.split not in {"public", "development", "held_out", "private", "unknown"}:
            raise ValueError("split is not allowed")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an int")
        if not isinstance(self.metrics, Mapping) or not isinstance(self.cost, Mapping):
            raise ValueError("metrics and cost must be mappings")
        _json(self.metrics)
        _json(self.cost)
        if self.feedback is not None:
            self.feedback.validate()
        if self.artifact_sha256 and (
            not isinstance(self.artifact_sha256, str)
            or len(self.artifact_sha256) != 64
            or any(ch not in "0123456789abcdefABCDEF" for ch in self.artifact_sha256)
        ):
            raise ValueError("artifact_sha256 must be a SHA-256 hex string")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "score": self.score,
            "metrics": _json(self.metrics),
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "split": self.split,
            "seed": self.seed,
            "cost": _json(self.cost),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvaluationResult":
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation result must be a mapping")
        feedback = raw.get("feedback")
        item = cls(
            schema=raw.get("schema", EVALUATION_RESULT_SCHEMA),
            candidate_id=raw.get("candidate_id", ""),
            status=raw.get("status", ""),
            score=raw.get("score"),
            metrics=raw.get("metrics", {}),
            feedback=(feedback if isinstance(feedback, FeedbackPacket)
                      else FeedbackPacket.from_dict(feedback) if feedback else None),
            split=raw.get("split", "development"),
            seed=raw.get("seed", 0),
            cost=raw.get("cost", {}),
            artifact_sha256=raw.get("artifact_sha256", ""),
        )
        item.validate()
        return item


@runtime_checkable
class SearchSpace(Protocol):
    """Task implementation that proposes complete finite algorithms."""

    def propose(self, context: Mapping[str, Any], slot: int) -> AlgorithmSpec:
        ...

    def validate(self, candidate: AlgorithmSpec) -> None:
        ...


def make_python_program_search_space(**kwargs: Any) -> SearchSpace:
    """Construct the repository's concrete whole-Python search space.

    The lazy import keeps the task-independent dataclasses free of a circular
    import while making the implemented search engine available from the same
    public module as the protocol.
    """
    from program_search import PythonProgramSearchSpace

    return PythonProgramSearchSpace(**kwargs)


def make_agent_python_program_search_space(
    workspace_root: str | Path,
    *,
    backend: str | None = None,
    model: str | None = None,
    timeout_s: int = 600,
    initial_files: Mapping[str, str] | None = None,
    **search_space_kwargs: Any,
) -> SearchSpace:
    """Construct a directly runnable LLM-backed Python program search space."""
    from agent_program_generator import AgentWholeProgramGenerator
    from program_search import PythonProgramSearchSpace

    generator = AgentWholeProgramGenerator(
        workspace_root,
        backend=backend,
        model=model,
        timeout_s=timeout_s,
        initial_files=initial_files,
    )
    return PythonProgramSearchSpace(
        generator=generator,
        **search_space_kwargs,
    )


@runtime_checkable
class FeedbackOracle(Protocol):
    """Optional task-specific projection from an evaluator result to feedback."""

    def project(
        self, candidate: AlgorithmSpec, result: EvaluationResult,
    ) -> FeedbackPacket | None:
        ...


@dataclass(frozen=True)
class AcquisitionDecision:
    candidate_id: str
    utility: float
    uncertainty: float
    coverage_bonus: float
    expected_improvement: float
    reason: str
    schema: str = "openhyra-acquisition-decision.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "utility": self.utility,
            "uncertainty": self.uncertainty,
            "coverage_bonus": self.coverage_bonus,
            "expected_improvement": self.expected_improvement,
            "reason": self.reason,
        }


class AcquisitionPolicy:
    """Deterministic uncertainty/EI/coverage policy for open proposals."""

    def __init__(self, *, exploration_weight: float = 0.35, coverage_weight: float = 0.25):
        if exploration_weight < 0 or coverage_weight < 0:
            raise ValueError("acquisition weights must be nonnegative")
        self.exploration_weight = float(exploration_weight)
        self.coverage_weight = float(coverage_weight)

    @staticmethod
    def _cell_key(candidate: AlgorithmSpec) -> str:
        return f"{candidate.mechanism_id or candidate.family}::global"

    def score(self, candidate: AlgorithmSpec, state: ProblemState) -> AcquisitionDecision:
        cell = state.cells.get(self._cell_key(candidate))
        if cell is None:
            uncertainty, ei, coverage = 1.0, 1.0, 1.0
            reason = "untried mechanism family"
        else:
            uncertainty = 1.0 / math.sqrt(cell.n + 1.0)
            # Positive posterior mass times remaining uncertainty is a small,
            # transparent expected-improvement proxy, not a learned scorer.
            ei = max(0.0, cell.p_positive) * uncertainty
            coverage = 0.0 if cell.n else 1.0
            reason = f"{cell.status}; n={cell.n}; p_positive={cell.p_positive:.3f}"
        utility = ei + self.exploration_weight * uncertainty + self.coverage_weight * coverage
        return AcquisitionDecision(
            candidate_id=candidate.candidate_id,
            utility=utility,
            uncertainty=uncertainty,
            coverage_bonus=coverage,
            expected_improvement=ei,
            reason=reason,
        )

    def rank(self, candidates: Sequence[AlgorithmSpec], state: ProblemState) -> list[AcquisitionDecision]:
        decisions = [self.score(candidate, state) for candidate in candidates]
        return sorted(decisions, key=lambda item: (-item.utility, item.candidate_id))


@dataclass(frozen=True)
class DiscoveryEvent:
    """One append-only event tying proposal, outcome, feedback and state hash."""

    round_index: int
    candidate: AlgorithmSpec
    result: EvaluationResult
    state_version: int
    state_hash: str
    acquisition: AcquisitionDecision | None = None
    schema: str = DISCOVERY_EVENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        self.candidate.validate()
        self.result.validate()
        return {
            "schema": self.schema,
            "round_index": self.round_index,
            "candidate": self.candidate.to_dict(),
            "result": self.result.to_dict(),
            "state_version": self.state_version,
            "state_hash": self.state_hash,
            "acquisition": self.acquisition.to_dict() if self.acquisition else None,
        }


class DiscoveryLedger:
    """Append-only JSONL persistence for a discovery run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, event: DiscoveryEvent) -> None:
        payload = event.to_dict()
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def read(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
        return rows


class AlgorithmDiscoveryLoop:
    """Small round-barrier loop reusable by task plugins and experiments.

    The loop is intentionally callback based.  ``evaluate`` is the trusted
    task evaluator; this class only coordinates proposals, feedback and
    append-only state.  All candidates in a round are evaluated before the
    next state snapshot is published, avoiding order-dependent Context reads.
    """

    def __init__(
        self,
        *,
        state: ProblemState | None = None,
        reducer: BeliefReducer | None = None,
        ledger: DiscoveryLedger | None = None,
        state_log: ProblemStateLog | None = None,
        acquisition: AcquisitionPolicy | None = None,
        oracle: FeedbackOracle | None = None,
    ):
        self.reducer = reducer or BeliefReducer()
        self.state = state or ProblemState(state_id="algorithm-discovery")
        self.state.validate()
        self.ledger = ledger
        self.state_log = state_log
        self.acquisition = acquisition or AcquisitionPolicy()
        self.oracle = oracle

    @property
    def state_hash(self) -> str:
        return _hash(self.state.to_dict(), "state")

    def rank(self, candidates: Sequence[AlgorithmSpec]) -> list[AcquisitionDecision]:
        for candidate in candidates:
            candidate.validate()
        return self.acquisition.rank(candidates, self.state)

    def run_round(
        self,
        candidates: Sequence[AlgorithmSpec],
        evaluate: Callable[[AlgorithmSpec], EvaluationResult],
        *,
        round_index: int | None = None,
        select: int | None = None,
    ) -> list[DiscoveryEvent]:
        if not candidates:
            return []
        ranked = self.rank(candidates)
        limit = len(ranked) if select is None else max(0, min(int(select), len(ranked)))
        chosen = ranked[:limit]
        round_index = self.state.state_version if round_index is None else int(round_index)

        # Evaluate the whole selected cohort before mutating state.  Feedback
        # is then applied as one round barrier: callers never observe a
        # partially updated Context state while sibling candidates from the
        # same epoch are still being scored.
        evaluated: list[tuple[AlgorithmSpec, AcquisitionDecision, EvaluationResult]] = []
        for decision in chosen:
            candidate = next(item for item in candidates if item.candidate_id == decision.candidate_id)
            result = evaluate(candidate)
            if not isinstance(result, EvaluationResult):
                result = EvaluationResult.from_dict(result)
            result.validate()
            if result.candidate_id != candidate.candidate_id:
                raise ValueError("evaluator returned a different candidate_id")
            if result.feedback is None and self.oracle is not None:
                feedback = self.oracle.project(candidate, result)
                if feedback is not None:
                    result = EvaluationResult(
                        candidate_id=result.candidate_id,
                        status=result.status,
                        score=result.score,
                        metrics=result.metrics,
                        feedback=feedback,
                        split=result.split,
                        seed=result.seed,
                        cost=result.cost,
                        artifact_sha256=result.artifact_sha256,
                    )
            evaluated.append((candidate, decision, result))

        # Persist packets in evaluator order, but do not emit events until the
        # final state snapshot is available. ``ProblemState.append`` is
        # functional, so intermediate versions stay local to this method and
        # all events from one round carry an identical state version/hash.
        for candidate, decision, result in evaluated:
            if result.feedback is not None:
                if self.state_log is not None:
                    self.state_log.append(result.feedback)
                self.state = self.state.append(result.feedback, self.reducer)

        events: list[DiscoveryEvent] = []
        for candidate, decision, result in evaluated:
            event = DiscoveryEvent(
                round_index=round_index,
                candidate=candidate,
                result=result,
                state_version=self.state.state_version,
                state_hash=self.state_hash,
                acquisition=decision,
            )
            if self.ledger is not None:
                self.ledger.append(event)
            events.append(event)
        return events

    def run_search(
        self,
        search_space: SearchSpace,
        evaluate: Callable[[AlgorithmSpec], EvaluationResult],
        *,
        rounds: int,
        candidates_per_round: int,
        context: Mapping[str, Any] | Callable[[int, ProblemState], Mapping[str, Any]] | None = None,
        select: int | None = None,
    ) -> list[DiscoveryEvent]:
        """Run propose -> evaluate -> observe as one recursive search.

        ``SearchSpace.propose`` is called only from the state published by the
        previous round.  After the whole cohort is evaluated, results are fed
        back to concrete spaces exposing ``observe`` before the next proposal
        is generated.  This makes parent selection depend on measured results
        rather than lineage labels alone.
        """
        if not isinstance(rounds, int) or rounds < 0:
            raise ValueError("rounds must be a nonnegative int")
        if not isinstance(candidates_per_round, int) or candidates_per_round < 1:
            raise ValueError("candidates_per_round must be a positive int")

        all_events: list[DiscoveryEvent] = []
        for round_index in range(rounds):
            if callable(context):
                round_context = context(round_index, self.state)
            else:
                round_context = context or {}
            if not isinstance(round_context, Mapping):
                raise ValueError("search context must be a mapping")
            begin_round = getattr(search_space, "begin_round", None)
            end_round = getattr(search_space, "end_round", None)
            if callable(begin_round):
                begin_round()
            try:
                proposals = [
                    search_space.propose(round_context, slot)
                    for slot in range(candidates_per_round)
                ]
                for candidate in proposals:
                    search_space.validate(candidate)
                events = self.run_round(
                    proposals,
                    evaluate,
                    round_index=round_index,
                    select=select,
                )
                observe = getattr(search_space, "observe", None)
                if callable(observe):
                    for event in events:
                        observe(event.result)
                all_events.extend(events)
            finally:
                if callable(end_round):
                    end_round()
        return all_events


# Friendly aliases for task authors and notebooks.
AlgorithmCandidate = AlgorithmSpec
AlgorithmDiscovery = AlgorithmDiscoveryLoop
DiscoveryState = ProblemState
BeliefStore = ProblemStateLog


__all__ = [
    "ALGORITHM_DISCOVERY_SCHEMA",
    "ALGORITHM_SPEC_SCHEMA",
    "EVALUATION_RESULT_SCHEMA",
    "DISCOVERY_EVENT_SCHEMA",
    "AlgorithmSpec",
    "AlgorithmCandidate",
    "EvaluationResult",
    "SearchSpace",
    "make_python_program_search_space",
    "make_agent_python_program_search_space",
    "FeedbackOracle",
    "AcquisitionDecision",
    "AcquisitionPolicy",
    "DiscoveryEvent",
    "DiscoveryLedger",
    "AlgorithmDiscoveryLoop",
    "AlgorithmDiscovery",
    "DiscoveryState",
    "BeliefStore",
]
