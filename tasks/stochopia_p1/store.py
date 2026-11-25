"""Typed P1 evaluation records and durable development and verification files."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from threading import get_ident
from typing import Any, Literal

from stochopia.common import atomic_write_json


GROUPS = ("evidence", "metrics", "no_llm")
FORMAL_ARMS = ("evidence_gpt", "metrics_gpt", "evidence_ds", "metrics_ds", "no_llm")
NLL = "val_nll_total"
IMPACT = "lob_impact_score"
Json = dict[str, Any]
DailyRows = Mapping[int, Mapping[str, float | None]]


def group_mode(group: str) -> Literal["evidence", "metrics", "no_llm"]:
    """Resolve a registered experiment arm to its permitted information view."""
    if group in ("evidence", "evidence_gpt", "evidence_ds"):
        return "evidence"
    if group in ("metrics", "metrics_gpt", "metrics_ds"):
        return "metrics"
    if group == "no_llm":
        return "no_llm"
    raise ValueError(f"unknown P1 experiment arm {group!r}")


@dataclass(frozen=True)
class ModelRef:
    model_id: str
    source: str


@dataclass(frozen=True)
class MetricRule:
    """Registered improvement direction, threshold and no-LLM forecast."""

    metric: str
    direction: Literal["increase", "decrease"]
    min_effect: float
    no_llm_interval: tuple[float, float]

    def __post_init__(self) -> None:
        low, high = self.no_llm_interval
        if not self.metric or self.direction not in ("increase", "decrease"):
            raise ValueError("a metric rule requires a readout and improvement direction")
        if any(not isfinite(x) for x in (low, high, self.min_effect)) or self.min_effect < 0 or low > high:
            raise ValueError("metric thresholds must be finite and ordered")
        if (self.direction == "increase" and low <= 0) or (self.direction == "decrease" and high >= 0):
            raise ValueError("the no-LLM forecast must respect the improvement direction")

    @classmethod
    def from_dict(cls, value: Json) -> MetricRule:
        return cls(value["metric"], value["direction"], value["min_effect"], tuple(value["no_llm_interval"]))


def validate_dates(dates: tuple[int, ...], name: str) -> None:
    if not dates or tuple(sorted(set(dates))) != tuple(dates):
        raise ValueError(f"{name} must have distinct, strictly chronological dates")


@dataclass(frozen=True, kw_only=True)
class StudyPlan:
    seed_model: ModelRef
    seed_evidence: Mapping[str, object]
    train_dates: tuple[int, ...]
    dev_dates: tuple[int, ...]
    metric_rules: tuple[MetricRule, ...]
    rounds: int = 2
    train_seeds: tuple[int, ...] = (103, 137, 211)
    rollout_seed: int = 12_300
    mutation_seed: int = 29
    nll_guardrail: float = 0.05
    groups: tuple[str, ...] = GROUPS
    trial_prefix: str = ""
    analysis_version: Literal["m4", "m5"] = "m4"

    def __post_init__(self) -> None:
        validate_dates(self.train_dates, "train_dates")
        validate_dates(self.dev_dates, "dev_dates")
        if self.train_dates[-1] >= self.dev_dates[0] or len(self.dev_dates) < 2:
            raise ValueError("development dates must follow training dates and support paired inference")
        if self.rounds < 1 or not self.train_seeds or len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("positive rounds and unique training seeds are required")
        if self.nll_guardrail != 0.05:
            raise ValueError("P1-M4 fixes the NLL guardrail at +0.05 nats/event")
        if self.analysis_version not in ("m4", "m5"):
            raise ValueError("unknown paired analysis protocol")
        if not self.groups or len(set(self.groups)) != len(self.groups):
            raise ValueError("register distinct experiment arms")
        for group in self.groups:
            group_mode(group)
        if self.trial_prefix and not all(char.isalnum() or char in ".-_" for char in self.trial_prefix):
            raise ValueError("trial prefix must be a single path component")
        rules = {rule.metric for rule in self.metric_rules}
        if not rules or len(rules) != len(self.metric_rules):
            raise ValueError("register a unique improvement rule for each permitted core prediction metric")

    @classmethod
    def from_dict(cls, value: Json) -> StudyPlan:
        payload = dict(value)
        payload["seed_model"] = ModelRef(**value["seed_model"])
        payload["metric_rules"] = tuple(MetricRule.from_dict(rule) for rule in value["metric_rules"])
        for field in ("train_dates", "dev_dates", "train_seeds"):
            payload[field] = tuple(value[field])
        if "groups" in value:
            payload["groups"] = tuple(value["groups"])
        return cls(**payload)


@dataclass(frozen=True)
class EvaluationRequest:
    """Immutable common replicate plan for one pair of model sources.

    Per-day outcomes combine all registered training seeds and same-clock
    windows within each day. Training uses ``train_dates`` in both modes.
    Only a frozen verifier stage issues requests with ``split='test'``.
    """

    trial_id: str
    group: str
    parent: ModelRef
    candidate: ModelRef
    train_dates: tuple[int, ...]
    eval_dates: tuple[int, ...]
    train_seeds: tuple[int, ...]
    rollout_seed: int
    split: Literal["dev", "test"]
    analysis_version: Literal["m4", "m5"] = "m4"

    @classmethod
    def from_dict(cls, value: Json) -> EvaluationRequest:
        return cls(
            value["trial_id"], value["group"], ModelRef(**value["parent"]), ModelRef(**value["candidate"]),
            tuple(value["train_dates"]), tuple(value["eval_dates"]), tuple(value["train_seeds"]),
            value["rollout_seed"], value["split"], value.get("analysis_version", "m4"),
        )


@dataclass(frozen=True)
class PairedEvaluation:
    """One parent/candidate day panel and compact experiment-level evidence."""

    parent_rows: DailyRows
    candidate_rows: DailyRows
    parent_evidence: Mapping[str, object]
    candidate_evidence: Mapping[str, object]
    parent_market: Mapping[str, object] | None = None
    candidate_market: Mapping[str, object] | None = None

    def validate(self, request: EvaluationRequest) -> None:
        dates = set(request.eval_dates)
        if set(self.parent_rows) != dates or set(self.candidate_rows) != dates:
            raise ValueError("paired evaluation must return exactly the registered evaluation dates")
        if (self.parent_market is None) != (self.candidate_market is None):
            raise ValueError("both models must use the same market readout protocol")

    def to_dict(self) -> Json:
        return {
            "parent_rows": {str(day): dict(row) for day, row in self.parent_rows.items()},
            "candidate_rows": {str(day): dict(row) for day, row in self.candidate_rows.items()},
            "parent_evidence": dict(self.parent_evidence),
            "candidate_evidence": dict(self.candidate_evidence),
            "parent_market": self.parent_market,
            "candidate_market": self.candidate_market,
        }

    @classmethod
    def from_dict(cls, value: Json) -> PairedEvaluation:
        return cls(
            {int(day): row for day, row in value["parent_rows"].items()},
            {int(day): row for day, row in value["candidate_rows"].items()},
            value["parent_evidence"], value["candidate_evidence"],
            value.get("parent_market"), value.get("candidate_market"),
        )


class CandidateEvaluationFailed(RuntimeError):
    """A candidate failed within evaluation after its paired budget was allocated."""


class NonfiniteCandidateLoss(CandidateEvaluationFailed):
    """One registered fit produced a nonfinite training or validation loss."""


Evaluator = Callable[[EvaluationRequest], PairedEvaluation]


# --- File ownership and publication ---


class RunBusyError(RuntimeError):
    """Another process or thread owns the experiment root's lock."""


class Ledger:
    def __init__(self, root: Path) -> None:
        self.path = root.expanduser().resolve()
        self._owner: tuple[int, int] | None = None

    def file(self, *parts: str) -> Path:
        if any(part in ("", ".", "..") or "/" in part or "\\" in part for part in parts):
            raise ValueError("ledger paths must be single components")
        path = self.path.joinpath(*parts)
        if path.resolve() != path:
            raise ValueError("ledger paths cannot redirect through symlinks")
        return path

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold one nonblocking process/thread lock spanning reads and seals."""
        self.path.mkdir(parents=True, exist_ok=True)
        with self.file(".lock").open("a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunBusyError(f"another worker owns {self.path}") from exc
            pid = os.getpid()
            self._owner = (pid, get_ident())
            try:
                yield
            finally:
                self._owner = None
                if os.getpid() == pid:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def read(self, *parts: str) -> Json | None:
        path = self.file(*parts)
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except FileNotFoundError:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"expected one JSON object at {path}")
        return value

    def seal(self, payload: Json, *parts: str) -> Json:
        """Publish once under the writer lock, accepting equal retry contents."""
        if self._owner != (os.getpid(), get_ident()):
            raise RuntimeError("a ledger seal requires the current worker to own its lock")
        value = json.loads(json.dumps(payload, allow_nan=False))
        previous = self.read(*parts)
        if previous is not None:
            if previous != value:
                raise ValueError(f"sealed ledger record cannot change: {self.file(*parts)}")
            return previous
        atomic_write_json(self.file(*parts), value)
        return value


@dataclass(frozen=True)
class DevelopmentExport:
    """Detached training/dev registration and terminal group rows for freezing."""

    development_root: Path
    plan: StudyPlan
    groups: Mapping[str, tuple[Json, ...]]

    def to_dict(self) -> Json:
        return {"plan": asdict(self.plan), "groups": self.groups, "development_root": str(self.development_root)}

    @classmethod
    def from_dict(cls, value: Json) -> DevelopmentExport:
        return cls(Path(value["development_root"]), StudyPlan.from_dict(value["plan"]),
                   {group: tuple(rows) for group, rows in value["groups"].items()})
