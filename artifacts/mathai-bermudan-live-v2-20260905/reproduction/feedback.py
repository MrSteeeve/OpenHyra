"""Evaluator-owned feedback packets and deterministic belief reduction.

This module is deliberately independent of the proposal/context loop.  A task
evaluator can emit a :class:`FeedbackPacket` containing two clearly separated
planes:

``observed``
    Values measured by the trusted evaluator (or explicit
    ``not_observed`` markers).
``recommendation``
    A provisional next-action suggestion.  It is never treated as evidence by
    :class:`BeliefReducer`.

The packet also keeps evidence, probe, and data provenance in separate maps.
The maps are JSON-only and versioned, so a packet can be persisted as an
append-only object and replayed without invoking an LLM.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Iterator, Mapping, Sequence


FEEDBACK_PACKET_SCHEMA = "openhyra-feedback-packet.v1"
DIRECTIONAL_FEEDBACK_SCHEMA = "openhyra-directional-feedback.v1"
PROBLEM_STATE_SCHEMA = "openhyra-problem-state.v1"
BELIEF_CELL_SCHEMA = "openhyra-belief-cell.v1"
NOT_OBSERVED = "not_observed"

_DIRECTIONS = {
    "positive",
    "negative",
    "neutral",
    "uncertain",
    "increase",
    "decrease",
    "improve",
    "regress",
    "hold",
    "abandon",
}
_DATA_SPLITS = {"public", "development", "held_out", "private", "unknown"}
_NUMERIC_OBSERVATION_KEYS = (
    "delta",
    "paired_delta",
    "paired_normalized_improvement",
    "effect",
    "value",
    "metric_value",
)


def not_observed(reason: str = "") -> dict[str, str]:
    """Return a serialisable marker for an unavailable measurement.

    The short string :data:`NOT_OBSERVED` is accepted as well.  The structured
    form is preferred for evaluator output because it records why a probe was
    unavailable without fabricating a numeric zero.
    """

    marker = {"status": NOT_OBSERVED}
    if reason:
        marker["reason"] = str(reason)
    return marker


def is_not_observed(value: Any) -> bool:
    """Whether ``value`` is an explicit unavailable-observation marker."""

    return value == NOT_OBSERVED or (
        isinstance(value, Mapping) and value.get("status") == NOT_OBSERVED
    )


def _require_string(value: Any, name: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        suffix = "" if allow_empty else " non-empty"
        raise ValueError(f"{name} must be a{suffix} str")


def _require_finite(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_json(value: Any, name: str) -> None:
    """Validate a JSON-compatible value and reject NaN/Infinity."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} has a non-string key")
            _validate_json(item, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{name}[{index}]")
        return
    raise ValueError(f"{name} is not JSON-compatible")


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _canonical_id(payload: Any, prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:16]}"


@dataclass
class DirectionalFeedback:
    """One typed, evaluator-grounded direction for a mechanism/slice.

    ``observed`` and ``recommendation`` intentionally have separate fields.
    The reducer only reads the former, so a speculative recommendation cannot
    silently become a belief update.
    """

    mechanism_id: str
    slice_key: str
    direction: str = "uncertain"
    id: str = ""
    candidate_id: str = ""
    prediction: Any = NOT_OBSERVED
    confidence: Any = NOT_OBSERVED
    observed: dict[str, Any] = field(default_factory=dict)
    recommendation: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    probe: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    version: str = DIRECTIONAL_FEEDBACK_SCHEMA
    falsifier: Any = NOT_OBSERVED
    created_at: str = ""
    schema: str = DIRECTIONAL_FEEDBACK_SCHEMA

    # Common aliases used by task adapters and notebooks.
    @property
    def mechanism(self) -> str:
        return self.mechanism_id

    @property
    def feedback_id(self) -> str:
        return self.id

    @property
    def slice(self) -> str:
        return self.slice_key

    def validate(self) -> None:
        _require_string(self.schema, "schema")
        if self.schema != DIRECTIONAL_FEEDBACK_SCHEMA:
            raise ValueError(f"schema must be {DIRECTIONAL_FEEDBACK_SCHEMA}")
        _require_string(self.version, "version")
        _require_string(self.mechanism_id, "mechanism_id")
        _require_string(self.slice_key, "slice_key")
        _require_string(self.direction, "direction")
        if self.direction not in _DIRECTIONS:
            raise ValueError("direction is not allowed")
        _require_string(self.id, "id", allow_empty=True)
        _require_string(self.candidate_id, "candidate_id", allow_empty=True)
        _require_string(self.created_at, "created_at", allow_empty=True)
        if self.confidence != NOT_OBSERVED and not is_not_observed(self.confidence):
            _require_finite(self.confidence, "confidence")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise ValueError("confidence must be in [0, 1]")
        for name, value in (
            ("prediction", self.prediction),
            ("falsifier", self.falsifier),
            ("observed", self.observed),
            ("recommendation", self.recommendation),
            ("evidence", self.evidence),
            ("probe", self.probe),
            ("data", self.data),
        ):
            _validate_json(value, name)
        for name, value in (
            ("observed", self.observed),
            ("recommendation", self.recommendation),
            ("evidence", self.evidence),
            ("probe", self.probe),
            ("data", self.data),
        ):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a dict")
        split = self.data.get("split")
        if split is not None and split not in _DATA_SPLITS:
            raise ValueError("data.split is not allowed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "version": self.version,
            "id": self.id,
            "candidate_id": self.candidate_id,
            "mechanism_id": self.mechanism_id,
            "slice_key": self.slice_key,
            "direction": self.direction,
            "prediction": _copy(self.prediction),
            "confidence": _copy(self.confidence),
            "observed": _copy(self.observed),
            "recommendation": _copy(self.recommendation),
            "evidence": _copy(self.evidence),
            "probe": _copy(self.probe),
            "data": _copy(self.data),
            "falsifier": _copy(self.falsifier),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DirectionalFeedback":
        if not isinstance(raw, Mapping):
            raise ValueError("directional feedback must be a dict")
        payload = dict(raw)
        # Accept the concise aliases used in task-level diagnostic adapters.
        if "mechanism_id" not in payload and "mechanism" in payload:
            payload["mechanism_id"] = payload.pop("mechanism")
        if "slice_key" not in payload and "slice" in payload:
            payload["slice_key"] = payload.pop("slice")
        allowed = {
            "schema", "version", "id", "candidate_id", "mechanism_id",
            "slice_key", "direction", "prediction", "confidence", "observed",
            "recommendation", "evidence", "probe", "data", "falsifier",
            "created_at",
        }
        unknown = set(payload).difference(allowed)
        if unknown:
            raise ValueError("unknown directional feedback field(s): " + ", ".join(sorted(unknown)))
        item = cls(**payload)
        item.validate()
        return item


@dataclass
class FeedbackPacket:
    """Versioned evaluator output consumed by Context/Proposal layers."""

    packet_id: str
    candidate_id: str = ""
    mechanism_id: str = ""
    directional: list[DirectionalFeedback] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)
    recommendation: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    probe: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    version: str = FEEDBACK_PACKET_SCHEMA
    created_at: str = ""
    schema: str = FEEDBACK_PACKET_SCHEMA

    @property
    def directions(self) -> list[DirectionalFeedback]:
        return self.directional

    @property
    def id(self) -> str:
        return self.packet_id

    @property
    def directional_feedback(self) -> list[DirectionalFeedback]:
        """Long-form alias used by packet consumers."""

        return self.directional

    def validate(self) -> None:
        _require_string(self.schema, "schema")
        if self.schema != FEEDBACK_PACKET_SCHEMA:
            raise ValueError(f"schema must be {FEEDBACK_PACKET_SCHEMA}")
        _require_string(self.version, "version")
        _require_string(self.packet_id, "packet_id")
        _require_string(self.candidate_id, "candidate_id", allow_empty=True)
        _require_string(self.mechanism_id, "mechanism_id", allow_empty=True)
        _require_string(self.created_at, "created_at", allow_empty=True)
        if not isinstance(self.directional, list):
            raise ValueError("directional must be a list")
        for index, item in enumerate(self.directional):
            if not isinstance(item, DirectionalFeedback):
                raise ValueError(f"directional[{index}] must be DirectionalFeedback")
            item.validate()
        for name, value in (
            ("observed", self.observed),
            ("recommendation", self.recommendation),
            ("evidence", self.evidence),
            ("probe", self.probe),
            ("data", self.data),
        ):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a dict")
            _validate_json(value, name)
        split = self.data.get("split")
        if split is not None and split not in _DATA_SPLITS:
            raise ValueError("data.split is not allowed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "version": self.version,
            "packet_id": self.packet_id,
            "candidate_id": self.candidate_id,
            "mechanism_id": self.mechanism_id,
            "observed": _copy(self.observed),
            "recommendation": _copy(self.recommendation),
            "evidence": _copy(self.evidence),
            "probe": _copy(self.probe),
            "data": _copy(self.data),
            "created_at": self.created_at,
            "directional": [item.to_dict() for item in self.directional],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FeedbackPacket":
        if not isinstance(raw, Mapping):
            raise ValueError("feedback packet must be a dict")
        payload = dict(raw)
        if "packet_id" not in payload and "id" in payload:
            payload["packet_id"] = payload.pop("id")
        if "directional" not in payload and "directions" in payload:
            payload["directional"] = payload.pop("directions")
        allowed = {
            "schema", "version", "packet_id", "candidate_id", "mechanism_id",
            "observed", "recommendation", "evidence", "probe", "data",
            "directional", "created_at",
        }
        unknown = set(payload).difference(allowed)
        if unknown:
            raise ValueError("unknown feedback packet field(s): " + ", ".join(sorted(unknown)))
        payload["directional"] = [
            item if isinstance(item, DirectionalFeedback)
            else DirectionalFeedback.from_dict(item)
            for item in payload.get("directional", [])
        ]
        item = cls(**payload)
        item.validate()
        return item


@dataclass(frozen=True)
class BeliefCell:
    """Sufficient statistics for one mechanism × behavior slice."""

    mechanism_id: str
    slice_key: str
    n: int
    mean: float
    variance: float
    se: float
    ci_lower: float
    ci_upper: float
    p_positive: float
    status: str
    confidence_level: float = 0.95
    observation_ids: tuple[str, ...] = ()
    schema: str = BELIEF_CELL_SCHEMA

    @property
    def standard_error(self) -> float:
        return self.se

    @property
    def ci(self) -> tuple[float, float]:
        return (self.ci_lower, self.ci_upper)

    @property
    def mechanism(self) -> str:
        return self.mechanism_id

    @property
    def slice(self) -> str:
        return self.slice_key

    def validate(self) -> None:
        _require_string(self.schema, "schema")
        if self.schema != BELIEF_CELL_SCHEMA:
            raise ValueError(f"schema must be {BELIEF_CELL_SCHEMA}")
        _require_string(self.mechanism_id, "mechanism_id")
        _require_string(self.slice_key, "slice_key")
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 0:
            raise ValueError("n must be a nonnegative int")
        for name, value in (
            ("mean", self.mean), ("variance", self.variance), ("se", self.se),
            ("ci_lower", self.ci_lower), ("ci_upper", self.ci_upper),
            ("p_positive", self.p_positive), ("confidence_level", self.confidence_level),
        ):
            _require_finite(value, name)
        if self.variance < 0.0 or self.se < 0.0:
            raise ValueError("variance and se must be nonnegative")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if not 0.0 <= self.p_positive <= 1.0:
            raise ValueError("p_positive must be in [0, 1]")
        if self.status not in {"untried", "uncertain", "promising", "falsified"}:
            raise ValueError("status is not allowed")
        if not isinstance(self.observation_ids, tuple) or any(
            not isinstance(item, str) for item in self.observation_ids
        ):
            raise ValueError("observation_ids must be tuple[str, ...]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "mechanism_id": self.mechanism_id,
            "slice_key": self.slice_key,
            "n": self.n,
            "mean": self.mean,
            "variance": self.variance,
            "se": self.se,
            "standard_error": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci": [self.ci_lower, self.ci_upper],
            "p_positive": self.p_positive,
            "status": self.status,
            "confidence_level": self.confidence_level,
            "observation_ids": list(self.observation_ids),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BeliefCell":
        if not isinstance(raw, Mapping):
            raise ValueError("belief cell must be a dict")
        payload = dict(raw)
        if "se" not in payload and "standard_error" in payload:
            payload["se"] = payload["standard_error"]
        if ("ci_lower" not in payload or "ci_upper" not in payload) and "ci" in payload:
            interval = payload["ci"]
            if isinstance(interval, (list, tuple)) and len(interval) == 2:
                payload.setdefault("ci_lower", interval[0])
                payload.setdefault("ci_upper", interval[1])
        payload.pop("standard_error", None)
        payload.pop("ci", None)
        payload["observation_ids"] = tuple(payload.get("observation_ids", ()))
        item = cls(**payload)
        item.validate()
        return item


@dataclass(frozen=True)
class ProblemState:
    """Immutable snapshot of an append-only belief log.

    ``observations`` is retained in the snapshot so deleting derived views and
    replaying the same packets yields the same cells.  Callers append by
    obtaining a new snapshot through :meth:`append`; existing snapshots are
    never modified.
    """

    state_id: str
    state_version: int = 0
    confidence_level: float = 0.95
    min_observations: int = 2
    cells: dict[str, BeliefCell] = field(default_factory=dict)
    applied_packet_ids: tuple[str, ...] = ()
    observations: tuple[dict[str, Any], ...] = ()
    created_at: str = ""
    schema: str = PROBLEM_STATE_SCHEMA

    @property
    def beliefs(self) -> dict[str, BeliefCell]:
        return self.cells

    @property
    def state_hash(self) -> str:
        """Content digest used to bind a Context decision to its snapshot."""

        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def hash(self) -> str:
        """Short compatibility alias for callers using ``state.hash``."""

        return self.state_hash

    def get_cell(self, mechanism_id: str, slice_key: str) -> BeliefCell | None:
        """Return a cell without allowing callers to mutate the snapshot."""

        return self.cells.get(f"{mechanism_id}::{slice_key}")

    def get_cells(self) -> dict[str, BeliefCell]:
        """Return a defensive copy of the derived cell map."""

        return dict(self.cells)

    def validate(self) -> None:
        _require_string(self.schema, "schema")
        if self.schema != PROBLEM_STATE_SCHEMA:
            raise ValueError(f"schema must be {PROBLEM_STATE_SCHEMA}")
        _require_string(self.state_id, "state_id")
        _require_string(self.created_at, "created_at", allow_empty=True)
        if not isinstance(self.state_version, int) or isinstance(self.state_version, bool) or self.state_version < 0:
            raise ValueError("state_version must be a nonnegative int")
        _require_finite(self.confidence_level, "confidence_level")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if not isinstance(self.min_observations, int) or isinstance(self.min_observations, bool) or self.min_observations < 1:
            raise ValueError("min_observations must be a positive int")
        if not isinstance(self.cells, dict):
            raise ValueError("cells must be a dict")
        for key, cell in self.cells.items():
            _require_string(key, "cells key")
            if not isinstance(cell, BeliefCell):
                raise ValueError("cells must contain BeliefCell values")
            cell.validate()
        if not isinstance(self.applied_packet_ids, tuple) or any(
            not isinstance(item, str) for item in self.applied_packet_ids
        ):
            raise ValueError("applied_packet_ids must be tuple[str, ...]")
        if len(set(self.applied_packet_ids)) != len(self.applied_packet_ids):
            raise ValueError("applied_packet_ids must be unique")
        if not isinstance(self.observations, tuple):
            raise ValueError("observations must be tuple[dict, ...]")
        for index, item in enumerate(self.observations):
            if not isinstance(item, dict):
                raise ValueError(f"observations[{index}] must be a dict")
            _validate_json(item, f"observations[{index}]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "state_id": self.state_id,
            "state_version": self.state_version,
            "confidence_level": self.confidence_level,
            "min_observations": self.min_observations,
            "cells": {
                key: self.cells[key].to_dict() for key in sorted(self.cells)
            },
            "applied_packet_ids": list(self.applied_packet_ids),
            "observations": [_copy(item) for item in self.observations],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProblemState":
        if not isinstance(raw, Mapping):
            raise ValueError("problem state must be a dict")
        payload = dict(raw)
        cells_raw = payload.get("cells", {})
        if not isinstance(cells_raw, Mapping):
            raise ValueError("cells must be a dict")
        payload["cells"] = {
            str(key): value if isinstance(value, BeliefCell) else BeliefCell.from_dict(value)
            for key, value in cells_raw.items()
        }
        payload["applied_packet_ids"] = tuple(payload.get("applied_packet_ids", ()))
        payload["observations"] = tuple(payload.get("observations", ()))
        item = cls(**payload)
        item.validate()
        return item

    def append(self, packet: FeedbackPacket, reducer: "BeliefReducer | None" = None) -> "ProblemState":
        """Return a new state with ``packet`` appended (idempotently)."""

        return (reducer or BeliefReducer(
            confidence_level=self.confidence_level,
            min_observations=self.min_observations,
        )).append(self, packet)


def _packet_id(packet: FeedbackPacket) -> str:
    return packet.packet_id or _canonical_id(packet.to_dict(), "packet")


def _numeric(value: Any) -> float | None:
    if is_not_observed(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _observation_values(mapping: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    # ``delta``/``effect`` is a scalar observation.  A caller may additionally
    # supply ``samples`` (for example a paired batch); both are retained as
    # explicit observations.  This avoids silently discarding a measured
    # scalar merely because a richer sample list is present.
    for key in _NUMERIC_OBSERVATION_KEYS:
        if key in mapping:
            value = _numeric(mapping[key])
            if value is not None:
                values.append(value)
    samples = mapping.get("samples")
    if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes, bytearray)):
        values.extend(
            item for item in (_numeric(sample) for sample in samples)
            if item is not None
        )
    return values


def _iter_packet_observations(packet: FeedbackPacket) -> Iterator[dict[str, Any]]:
    """Yield explicit numeric observations; never infer from recommendations."""

    pid = _packet_id(packet)
    ordinal = 0
    for item in packet.directional:
        values = _observation_values(item.observed)
        if not values:
            continue
        mechanism = item.mechanism_id or packet.mechanism_id or "unknown"
        slice_key = item.slice_key or "global"
        for value in values:
            yield {
                "observation_id": (
                    f"{pid}:{item.id}:{ordinal}" if item.id
                    else f"{pid}:directional:{ordinal}"
                ),
                "packet_id": pid,
                "mechanism_id": mechanism,
                "slice_key": slice_key,
                "value": value,
                "data_split": item.data.get("split", packet.data.get("split", "unknown")),
            }
            ordinal += 1

    observed_items = packet.observed.get("observations")
    if isinstance(observed_items, list):
        for item_index, item in enumerate(observed_items):
            if not isinstance(item, Mapping):
                continue
            values = _observation_values(item)
            mechanism = str(item.get("mechanism_id", item.get("mechanism", packet.mechanism_id or "unknown")))
            slice_key = str(item.get("slice_key", item.get("slice", "global")))
            for value_index, value in enumerate(values):
                yield {
                    "observation_id": (
                        f"{pid}:{item.get('observation_id')}:{value_index}"
                        if item.get("observation_id") is not None
                        else f"{pid}:observed:{item_index}:{value_index}"
                    ),
                    "packet_id": pid,
                    "mechanism_id": mechanism,
                    "slice_key": slice_key,
                    "value": value,
                    "data_split": item.get("data_split", packet.data.get("split", "unknown")),
                }
    elif packet.mechanism_id:
        values = _observation_values(packet.observed)
        for value_index, value in enumerate(values):
            yield {
                "observation_id": f"{pid}:observed:{value_index}",
                "packet_id": pid,
                "mechanism_id": packet.mechanism_id,
                "slice_key": str(packet.data.get("slice_key", "global")),
                "value": value,
                "data_split": packet.data.get("split", "unknown"),
            }


def _cell_from_observations(
    mechanism_id: str,
    slice_key: str,
    observations: Sequence[Mapping[str, Any]],
    *,
    confidence_level: float,
    min_observations: int,
) -> BeliefCell:
    values = [float(item["value"]) for item in observations]
    values.sort()
    n = len(values)
    if n:
        mean = math.fsum(values) / n
        if n > 1:
            centered = [value - mean for value in values]
            variance = math.fsum(item * item for item in centered) / (n - 1)
        else:
            variance = 0.0
    else:
        mean = variance = 0.0
    se = math.sqrt(variance / n) if n else 0.0
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    ci_lower = mean - z * se
    ci_upper = mean + z * se
    if se > 0.0:
        p_positive = NormalDist().cdf(mean / se)
    elif mean > 0.0:
        p_positive = 1.0
    elif mean < 0.0:
        p_positive = 0.0
    else:
        p_positive = 0.5
    if n == 0:
        status = "untried"
    elif n < min_observations:
        status = "uncertain"
    elif ci_lower > 0.0:
        status = "promising"
    elif ci_upper < 0.0:
        status = "falsified"
    else:
        status = "uncertain"
    cell = BeliefCell(
        mechanism_id=mechanism_id,
        slice_key=slice_key,
        n=n,
        mean=mean,
        variance=variance,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_positive=p_positive,
        status=status,
        confidence_level=confidence_level,
        observation_ids=tuple(str(item["observation_id"]) for item in observations),
    )
    cell.validate()
    return cell


@dataclass(frozen=True)
class BeliefReducer:
    """Deterministically rebuild or append a :class:`ProblemState`."""

    confidence_level: float = 0.95
    min_observations: int = 2

    def __post_init__(self) -> None:
        _require_finite(self.confidence_level, "confidence_level")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if not isinstance(self.min_observations, int) or isinstance(self.min_observations, bool) or self.min_observations < 1:
            raise ValueError("min_observations must be a positive int")

    def reduce(
        self,
        packets: Iterable[FeedbackPacket | Mapping[str, Any]],
        *,
        state_id: str = "problem-state",
        state_version: int | None = None,
    ) -> ProblemState:
        normalized: list[FeedbackPacket] = []
        for raw in packets:
            normalized.append(raw if isinstance(raw, FeedbackPacket) else FeedbackPacket.from_dict(raw))
        # IDs make replay order-independent and make duplicate append retries
        # harmless.  The packet payload remains the source of truth.
        by_id = {_packet_id(item): item for item in normalized}
        observations: list[dict[str, Any]] = []
        for pid in sorted(by_id):
            observations.extend(_iter_packet_observations(by_id[pid]))
        return self._state_from_observations(
            observations,
            state_id=state_id,
            state_version=(len(by_id) if state_version is None else state_version),
            packet_ids=tuple(sorted(by_id)),
        )

    # Explicit alias for callers rebuilding after deleting derived views.
    rebuild = reduce

    def append(self, state: ProblemState, packet: FeedbackPacket | Mapping[str, Any]) -> ProblemState:
        state.validate()
        normalized = packet if isinstance(packet, FeedbackPacket) else FeedbackPacket.from_dict(packet)
        pid = _packet_id(normalized)
        if pid in state.applied_packet_ids:
            return state
        observations = list(state.observations)
        observations.extend(_iter_packet_observations(normalized))
        packet_ids = tuple(sorted((*state.applied_packet_ids, pid)))
        return self._state_from_observations(
            observations,
            state_id=state.state_id,
            state_version=state.state_version + 1,
            packet_ids=packet_ids,
        )

    # ``apply`` is a convenient functional spelling used by reducers in other
    # task plugins.
    apply = append

    # ``update`` is retained as a compatibility alias; it still returns a new
    # immutable snapshot and never mutates the supplied state.
    update = append

    def _state_from_observations(
        self,
        observations: Sequence[Mapping[str, Any]],
        *,
        state_id: str,
        state_version: int,
        packet_ids: tuple[str, ...],
    ) -> ProblemState:
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        unique: dict[str, Mapping[str, Any]] = {}
        for item in observations:
            oid = str(item["observation_id"])
            unique[oid] = dict(item)
        ordered = [unique[key] for key in sorted(unique)]
        for item in ordered:
            mechanism = str(item["mechanism_id"])
            slice_key = str(item["slice_key"])
            grouped.setdefault((mechanism, slice_key), []).append(item)
        cells = {
            f"{mechanism}::{slice_key}": _cell_from_observations(
                mechanism,
                slice_key,
                grouped[(mechanism, slice_key)],
                confidence_level=self.confidence_level,
                min_observations=self.min_observations,
            )
            for mechanism, slice_key in sorted(grouped)
        }
        state = ProblemState(
            state_id=state_id,
            state_version=state_version,
            confidence_level=self.confidence_level,
            min_observations=self.min_observations,
            cells=cells,
            applied_packet_ids=tuple(sorted(set(packet_ids))),
            observations=tuple(_copy(item) for item in ordered),
        )
        state.validate()
        return state


class ProblemStateLog:
    """Tiny append-only JSONL log for packets and derived state snapshots.

    This is optional infrastructure for task runners; it does not alter the
    evaluator's candidate boundary.  Each call writes one complete packet and
    fsyncs it before returning.  Rebuilding uses :class:`BeliefReducer` only.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, packet: FeedbackPacket | Mapping[str, Any]) -> None:
        item = packet if isinstance(packet, FeedbackPacket) else FeedbackPacket.from_dict(packet)
        payload = item.to_dict()
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())

    def read(self) -> list[FeedbackPacket]:
        packets: list[FeedbackPacket] = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    packets.append(FeedbackPacket.from_dict(json.loads(line)))
        return packets

    def rebuild(self, reducer: BeliefReducer | None = None, *, state_id: str = "problem-state") -> ProblemState:
        return (reducer or BeliefReducer()).rebuild(self.read(), state_id=state_id)


def render_feedback_context(
    packets: Iterable[FeedbackPacket | Mapping[str, Any]],
    *,
    max_packets: int = 8,
    max_directions: int = 4,
    max_chars: int = 12_000,
) -> str:
    """Render a compact, public feedback view for Context/Proposal prompts.

    This projection is intentionally shared by the V5 bridge and the
    non-V5 adaptive loop.  It contains measured fields plus the evaluator's
    recommendation as a clearly nested annotation; the belief reducer never
    consumes the recommendation as an observation.
    """
    normalized: list[FeedbackPacket] = []
    for raw in packets:
        try:
            packet = raw if isinstance(raw, FeedbackPacket) else FeedbackPacket.from_dict(raw)
            packet.validate()
        except (TypeError, ValueError, KeyError):
            continue
        if packet.data.get("split") == "private":
            continue
        normalized.append(packet)
    limit = max(0, int(max_packets))
    values = normalized[-limit:] if limit else []
    direction_limit = max(0, int(max_directions))
    rows: list[dict[str, Any]] = []
    for packet in values:
        observed = packet.observed if isinstance(packet.observed, dict) else {}
        row: dict[str, Any] = {
            "packet_id": packet.packet_id,
            "candidate_id": packet.candidate_id,
            "mechanism_id": packet.mechanism_id,
            "split": packet.data.get("split", "unknown"),
            "primary_metric": observed.get("primary_metric"),
            "aggregate_effect": observed.get("aggregate_effect"),
            "aggregate_standard_error": observed.get("aggregate_standard_error"),
            "directions": [],
        }
        for item in packet.directional[:direction_limit]:
            item_observed = item.observed if isinstance(item.observed, dict) else {}
            effect = item_observed.get("effect")
            if not isinstance(effect, (int, float)):
                effect = item_observed.get("paired_delta")
            row["directions"].append({
                "mechanism_id": item.mechanism_id or packet.mechanism_id,
                "slice": item.slice_key,
                "direction": item.direction,
                "effect": effect,
                "recommendation": item.recommendation,
                "falsifier": item.falsifier,
            })
        rows.append(row)
    if not rows:
        return ""
    text = json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2)
    return text[:max(0, int(max_chars))]


__all__ = [
    "FEEDBACK_PACKET_SCHEMA",
    "DIRECTIONAL_FEEDBACK_SCHEMA",
    "PROBLEM_STATE_SCHEMA",
    "BELIEF_CELL_SCHEMA",
    "NOT_OBSERVED",
    "not_observed",
    "is_not_observed",
    "DirectionalFeedback",
    "FeedbackPacket",
    "BeliefCell",
    "ProblemState",
    "BeliefReducer",
    "ProblemStateLog",
    "render_feedback_context",
]
