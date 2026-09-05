"""Reproducible experiment scaffolding for the open Python-program track.

This module is deliberately separate from the historical Bermudan artifacts.
It records the *whole program* submitted to ``bermudan_python_search``, its
operator/parent lineage, evaluator-owned cost and failures, and the frozen
run identity.  The runner is usable with a cheap injected evaluator in tests;
the default evaluator calls the task's real public evaluator when requested.

The records are evidence of an experiment run, not evidence of algorithmic
novelty.  In particular, this module never imports a legacy policy runner to
score a Python program.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from algorithm_discovery import (
    AlgorithmDiscoveryLoop,
    AlgorithmSpec,
    DiscoveryEvent,
    EvaluationResult,
)


EXPERIMENT_SCHEMA = "openhyra-python-program-experiment.v1"
RECORD_SCHEMA = "openhyra-python-program-record.v1"


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def source_tree_sha256(source: Mapping[str, str]) -> str:
    """Content digest of every source file, including non-Python notes."""
    if not isinstance(source, Mapping) or not source:
        raise ValueError("source must be a non-empty mapping")
    if any(not isinstance(path, str) or not isinstance(text, str) for path, text in source.items()):
        raise ValueError("source must map paths to text")
    return _sha256({path: source[path] for path in sorted(source)})


def ast_lineage_sha256(source: Mapping[str, str]) -> str:
    """Digest normalized Python ASTs; syntax failures are explicit errors."""
    trees: dict[str, str] = {}
    for path, text in sorted(source.items()):
        if path.endswith(".py"):
            trees[path] = ast.dump(ast.parse(text, filename=path), annotate_fields=True, include_attributes=False)
    if not trees:
        raise ValueError("source must contain at least one Python file")
    return _sha256(trees)


def _git_metadata(root: Path) -> dict[str, Any]:
    def command(args: list[str]) -> str | None:
        try:
            result = subprocess.run(args, cwd=root, check=False, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = command(["git", "status", "--porcelain"])
    return {
        "sha": command(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "dirty_state_sha256": hashlib.sha256((status or "").encode()).hexdigest(),
    }


@dataclass(frozen=True)
class MatchedControlPlan:
    """Pre-registered control policy carried into the run manifest."""

    arms: tuple[str, ...] = ("guided", "no_feedback", "random_restart", "ast_only")
    same_parent: bool = True
    same_seed: bool = True
    same_compute_budget: bool = True

    def to_dict(self) -> dict[str, Any]:
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("matched control arms must be non-empty and unique")
        return {
            "enabled": True,
            "arms": list(self.arms),
            "same_parent": self.same_parent,
            "same_seed": self.same_seed,
            "same_compute_budget": self.same_compute_budget,
        }


@dataclass(frozen=True)
class PythonProgramExperimentConfig:
    """Frozen identity and budget for one fresh search run."""

    root: Path
    output_dir: Path
    trial_seed: int
    search_request: Mapping[str, Any]
    rounds: int = 1
    candidates_per_round: int = 4
    backend: str = "injected"
    model: str = "none"
    matched_controls: MatchedControlPlan = field(default_factory=MatchedControlPlan)
    # These flags make the research boundary explicit in the frozen manifest.
    # They add diagnostics only; the evaluator request remains authoritative.
    research_mode: bool = True
    independent_validation: bool = True
    git_sha: str | None = None

    def manifest(self) -> dict[str, Any]:
        if isinstance(self.trial_seed, bool) or self.trial_seed < 0:
            raise ValueError("trial_seed must be a non-negative integer")
        if self.rounds < 0 or self.candidates_per_round < 1:
            raise ValueError("rounds/candidates_per_round out of range")
        if not isinstance(self.research_mode, bool) or not isinstance(
            self.independent_validation, bool
        ):
            raise ValueError("research_mode and independent_validation must be bool")
        git = _git_metadata(self.root)
        if self.git_sha is not None:
            if self.git_sha != git.get("sha"):
                raise ValueError("configured git_sha does not match repository HEAD")
        return {
            "schema": EXPERIMENT_SCHEMA,
            "task": "bermudan_python_search",
            "protocol": "bermudan-python-program-search.v1",
            "artifact_protocol": "openhyra-python-program.v1",
            "trial_seed": self.trial_seed,
            "rounds": self.rounds,
            "candidates_per_round": self.candidates_per_round,
            "backend": self.backend,
            "model": self.model,
            "git": git,
            "search_request": dict(self.search_request),
            "matched_controls": self.matched_controls.to_dict(),
            "research_mode": self.research_mode,
            "independent_validation": self.independent_validation,
            "evidence_boundary": "executable whole-program run; novelty and generalization unobserved",
        }


def build_python_program_experiment_manifest(**kwargs: Any) -> dict[str, Any]:
    """Convenience constructor used by launchers and notebooks."""
    return PythonProgramExperimentConfig(**kwargs).manifest()


def _failure_result(candidate: AlgorithmSpec, seed: int, exc: BaseException, elapsed: float) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate.candidate_id,
        status="error",
        score=None,
        split="development",
        seed=seed,
        metrics={"failure_type": type(exc).__name__, "failure": str(exc)},
        cost={"wall_seconds": elapsed},
    )


def candidate_record(event: DiscoveryEvent) -> dict[str, Any]:
    """Expand a discovery event with source, AST and explicit failure fields."""
    candidate = event.candidate
    implementation = candidate.implementation
    source = implementation.get("source") if isinstance(implementation, Mapping) else None
    if not isinstance(source, Mapping):
        raise ValueError("Python program candidate has no source mapping")
    result = event.result
    metadata = dict(candidate.metadata)
    matched = {
        key: metadata[key]
        for key in ("matched_arm", "matched_pair_id", "matched_seed", "budget_id")
        if key in metadata
    }
    failure = None if result.status == "ok" else {
        "status": result.status,
        "reason": result.metrics.get("failure", result.metrics.get("error", "unknown")),
    }
    return {
        "schema": RECORD_SCHEMA,
        "round_index": event.round_index,
        "candidate_id": candidate.candidate_id,
        "parent_ids": list(candidate.parent_ids),
        "operator": candidate.operator,
        "mechanism_id": candidate.mechanism_id,
        "source": {path: source[path] for path in sorted(source)},
        "source_sha256": source_tree_sha256(source),
        # Stable aliases make the experiment bundle self-describing for
        # downstream Context tooling while retaining the historical field
        # names above.
        "source_digest": source_tree_sha256(source),
        "parent_lineage": list(candidate.parent_ids),
        "ast_sha256": ast_lineage_sha256(source),
        "research_hypothesis": {
            "mechanism_id": candidate.mechanism_id,
            "family": candidate.family,
            "prediction": candidate.prediction,
            "falsifier": candidate.falsifier,
            "target_slice": metadata.get("target_slice", metadata.get("target_slices")),
        },
        "evaluator_observation": {
            "effect": result.metrics.get("mean_paired_normalized_improvement", result.metrics.get("mean_normalized_confidence_gap")),
            "standard_error": result.metrics.get("paired_aggregate_standard_error", result.metrics.get("aggregate_standard_error")),
            "failure_reason": failure["reason"] if failure else None,
        },
        "metadata": metadata,
        "matched_control": matched or None,
        "result": result.to_dict(),
        "cost": dict(result.cost),
        "failure": failure,
        "state_version": event.state_version,
        "state_hash": event.state_hash,
    }


class PythonProgramExperimentRunner:
    """Run and persist a bounded whole-program search.

    ``evaluate`` receives an :class:`AlgorithmSpec` and may be a tiny fake in
    tests.  Omitting it uses the task evaluator through a temporary materialized
    candidate, with the exact request stored in the manifest.
    """

    def __init__(self, config: PythonProgramExperimentConfig, search_space: Any, *, evaluate: Callable[[AlgorithmSpec], EvaluationResult] | None = None):
        self.config = config
        self.search_space = search_space
        self.evaluate = evaluate or self._evaluate_real

    def _safe_evaluate(self, candidate: AlgorithmSpec) -> EvaluationResult:
        """Turn evaluator exceptions into durable failure records."""
        started = time.perf_counter()
        try:
            result = self.evaluate(candidate)
            if not isinstance(result, EvaluationResult):
                result = EvaluationResult.from_dict(result)
            result.validate()
            if result.candidate_id != candidate.candidate_id:
                raise ValueError("evaluator returned a different candidate_id")
            return result
        except Exception as exc:
            return _failure_result(
                candidate,
                int(self.config.search_request["seed"]),
                exc,
                time.perf_counter() - started,
            )

    def _evaluate_real(self, candidate: AlgorithmSpec) -> EvaluationResult:
        started = time.perf_counter()
        try:
            source = candidate.implementation["source"]
            with tempfile.TemporaryDirectory(prefix="openhyra-python-program-") as directory:
                root = Path(directory)
                for relative, text in source.items():
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(text, encoding="utf-8")
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                from tasks.bermudan_python_search import evaluator
                score, metrics, _normalized, _evidence = evaluator.evaluate_submission(
                    manifest, dict(self.config.search_request), candidate_source_dir=root,
                )
            elapsed = time.perf_counter() - started
            return EvaluationResult(
                candidate_id=candidate.candidate_id, status="ok", score=float(score),
                metrics=metrics, split="development",
                seed=int(self.config.search_request["seed"]),
                cost={"wall_seconds": elapsed},
            )
        except Exception as exc:
            return _failure_result(candidate, int(self.config.search_request["seed"]), exc, time.perf_counter() - started)

    def run(self) -> list[dict[str, Any]]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.config.manifest()
        (self.config.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        events_path = self.config.output_dir / "records.jsonl"
        events_path.write_text("", encoding="utf-8")
        ledger = self.config.output_dir / "discovery_events.jsonl"
        loop = AlgorithmDiscoveryLoop()
        events = loop.run_search(
            self.search_space, self._safe_evaluate,
            rounds=self.config.rounds,
            candidates_per_round=self.config.candidates_per_round,
            context={"experiment_schema": EXPERIMENT_SCHEMA, "trial_seed": self.config.trial_seed},
        )
        records = [candidate_record(event) for event in events]
        with events_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        with ledger.open("w", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return records


__all__ = [
    "EXPERIMENT_SCHEMA", "RECORD_SCHEMA", "MatchedControlPlan",
    "PythonProgramExperimentConfig", "PythonProgramExperimentRunner",
    "source_tree_sha256", "ast_lineage_sha256", "candidate_record",
    "build_python_program_experiment_manifest",
]
