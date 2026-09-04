"""Deterministic hypothesis queue and intervention acquisition helpers.

The Context Agent is allowed to suggest many mechanisms, but a round should
execute only a bounded subset.  This module keeps that decision explicit and
replayable: hypotheses are persisted as a small JSON document, while ranking
uses only structured evidence supplied by the trusted harness.  It is
deliberately independent of the evaluator and of the V5 schema so legacy
tasks can ignore it completely.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


INTERVENTION_SCOPES = (
    "parameter",
    "target",
    "representation",
    "architecture",
    "mechanism",
    "family",
)
INTERVENTION_OPERATORS = (
    "tune",
    "replace",
    "combine",
    "ablate",
    "transfer",
    "abandon",
    "probe",
    "mutate",
    "switch",
    "adjust",
    "compose",
    "remove",
    "restart",
    "inspect",
)
HYPOTHESIS_STATUSES = (
    "untried",
    "scheduled",
    "tested",
    "supported",
    "refuted",
    "inconclusive",
)


def _text(value: Any, limit: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()[:limit].rstrip()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _ids(value: Any, limit: int = 16) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return ()
    out: list[str] = []
    for item in value:
        item = _text(item, 160)
        if item and item not in out:
            out.append(item)
        if len(out) >= limit:
            break
    return tuple(out)


def _stable_id(payload: Mapping[str, Any]) -> str:
    supplied = _text(
        payload.get("id") or payload.get("hypothesis_id") or payload.get("mechanism_id"),
        96,
    )
    if supplied:
        return supplied
    basis = "|".join(
        _text(payload.get(key), 240)
        for key in ("family", "mechanism", "prediction", "target_slice")
    )
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]
    return f"hypothesis-{digest}"


def normalize_intervention(
    value: Mapping[str, Any] | None,
    *,
    source: str = "context",
    state_version: str | int | None = None,
) -> dict[str, Any]:
    """Normalize one hypothesis/intervention without trusting free text.

    Unknown fields are intentionally ignored.  The returned record is plain
    JSON and can be embedded in ContextDecision or ExperimentPlan metadata.
    Vocabulary values outside the small built-in set are retained as
    ``custom`` text rather than causing a proposal round to fail; task-owned
    extensions can therefore be introduced without changing this module.
    """
    payload = dict(value or {})
    nested = payload.get("intervention")
    if isinstance(nested, Mapping):
        merged = dict(nested)
        merged.update({key: val for key, val in payload.items() if key != "intervention"})
        payload = merged
    scope = _text(payload.get("intervention_scope") or payload.get("scope"), 64)
    operator = _text(
        payload.get("intervention_operator") or payload.get("operator"), 64
    )
    target_slice = payload.get("target_slice", payload.get("target_slices"))
    if isinstance(target_slice, (list, tuple)):
        target_slice = ", ".join(_text(item, 120) for item in target_slice if _text(item, 120))
    target_slice = _text(target_slice, 240)
    version = payload.get("state_version", state_version)
    if isinstance(version, bool):
        version = None
    elif isinstance(version, (int, float)):
        version = int(version) if float(version).is_integer() else str(version)
    elif version is not None:
        version = _text(version, 128) or None
    result = {
        "id": _stable_id(payload),
        "family": _text(payload.get("family"), 80) or "general",
        "mechanism": _text(
            payload.get("mechanism")
            or payload.get("hypothesis")
            or payload.get("intervention")
            or payload.get("idea"),
            500,
        ),
        "prediction": _text(payload.get("prediction"), 500),
        "failure_condition": _text(
            payload.get("failure_condition") or payload.get("falsifier"), 500
        ),
        "matched_control": _text(payload.get("matched_control"), 500),
        "intervention_scope": scope,
        "intervention_operator": operator,
        "target_slice": target_slice,
        "evidence_ids": list(_ids(payload.get("evidence_ids"))),
        "next_probe": _text(payload.get("next_probe"), 500),
        "state_version": version,
        "expected_gain": _finite(payload.get("expected_gain")),
        "confidence": (
            max(0.0, min(1.0, _finite(payload.get("confidence"))))
            if payload.get("confidence") is not None else None
        ),
        "information_gain": max(0.0, _finite(payload.get("information_gain"))),
        "source": _text(payload.get("source"), 80) or source,
    }
    # Keep the old aliases in the normalized packet for human-facing prompts.
    result["scope"] = scope
    result["operator"] = operator
    if result["intervention_scope"] not in INTERVENTION_SCOPES:
        result["intervention_scope"] = result["intervention_scope"] or "mechanism"
        result["scope"] = result["intervention_scope"]
    if result["intervention_operator"] not in INTERVENTION_OPERATORS:
        result["intervention_operator"] = result["intervention_operator"] or "replace"
        result["operator"] = result["intervention_operator"]
    return result


@dataclass
class HypothesisEntry:
    """A queued hypothesis plus deterministic acquisition bookkeeping."""

    hypothesis: dict[str, Any]
    status: str = "untried"
    first_iteration: int | None = None
    last_iteration: int | None = None
    attempts: int = 0
    priority: float = 0.0
    uncertainty: float = 1.0
    expected_gain: float = 0.0
    information_gain: float = 0.0
    result_ids: list[str] = field(default_factory=list)
    last_reason: str = ""

    @property
    def id(self) -> str:
        return str(self.hypothesis.get("id", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": dict(self.hypothesis),
            "status": self.status,
            "first_iteration": self.first_iteration,
            "last_iteration": self.last_iteration,
            "attempts": int(self.attempts),
            "priority": float(self.priority),
            "uncertainty": float(self.uncertainty),
            "expected_gain": float(self.expected_gain),
            "information_gain": float(self.information_gain),
            "result_ids": list(self.result_ids),
            "last_reason": self.last_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HypothesisEntry | None":
        if not isinstance(payload, Mapping):
            return None
        hypothesis = payload.get("hypothesis", payload)
        if not isinstance(hypothesis, Mapping):
            return None
        normalized = normalize_intervention(hypothesis)
        if not normalized.get("mechanism"):
            return None
        status = _text(payload.get("status"), 32) or "untried"
        if status not in HYPOTHESIS_STATUSES:
            status = "untried"
        result_ids = list(_ids(payload.get("result_ids"), 64))
        return cls(
            hypothesis=normalized,
            status=status,
            first_iteration=(
                int(payload["first_iteration"])
                if isinstance(payload.get("first_iteration"), int)
                else None
            ),
            last_iteration=(
                int(payload["last_iteration"])
                if isinstance(payload.get("last_iteration"), int)
                else None
            ),
            attempts=max(0, int(payload.get("attempts", 0) or 0)),
            priority=_finite(payload.get("priority")),
            uncertainty=max(0.0, _finite(payload.get("uncertainty"), 1.0)),
            expected_gain=_finite(payload.get("expected_gain")),
            information_gain=max(0.0, _finite(payload.get("information_gain"))),
            result_ids=result_ids,
            last_reason=_text(payload.get("last_reason"), 500),
        )


class PendingHypothesisQueue:
    """Small crash-tolerant persistent queue for Context hypotheses."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self._entries: dict[str, HypothesisEntry] = {}
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return
        values = payload.get("entries", payload) if isinstance(payload, Mapping) else payload
        if isinstance(values, Mapping):
            values = list(values.values())
        if not isinstance(values, list):
            return
        for value in values:
            entry = HypothesisEntry.from_dict(value)
            if entry is not None and entry.id:
                self._entries[entry.id] = entry

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "openhyra-pending-hypotheses.v1",
            "entries": [self._entries[key].to_dict() for key in sorted(self._entries)],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def enqueue(
        self,
        hypotheses: Iterable[Mapping[str, Any]],
        *,
        iteration: int | None = None,
        state_version: str | int | None = None,
        source: str = "context",
    ) -> list[HypothesisEntry]:
        """Insert/update hypotheses while preserving tested history."""
        added: list[HypothesisEntry] = []
        with self._lock:
            for raw in hypotheses or ():
                if not isinstance(raw, Mapping):
                    continue
                normalized = normalize_intervention(
                    raw, source=source, state_version=state_version
                )
                if not normalized.get("mechanism"):
                    continue
                identifier = normalized["id"]
                existing = self._entries.get(identifier)
                if existing is None:
                    existing = HypothesisEntry(
                        hypothesis=normalized,
                        first_iteration=(int(iteration) if isinstance(iteration, int) else None),
                        expected_gain=_finite(normalized.get("expected_gain")),
                        uncertainty=max(
                            0.0,
                            1.0 - _finite(normalized.get("confidence"), 0.0),
                        ),
                        information_gain=max(
                            0.0, _finite(normalized.get("information_gain"))
                        ),
                    )
                    self._entries[identifier] = existing
                    added.append(existing)
                else:
                    # Context may refine prose or target slices.  Keep prior
                    # evidence/status but update the current hypothesis text.
                    existing.hypothesis.update({
                        key: value for key, value in normalized.items()
                        if value not in ("", [], None)
                    })
                if isinstance(iteration, int):
                    existing.last_iteration = iteration
                if state_version is not None:
                    existing.hypothesis["state_version"] = state_version
            self._save()
        return added

    def entries(self, *, include_terminal: bool = True) -> list[HypothesisEntry]:
        with self._lock:
            values = list(self._entries.values())
            if not include_terminal:
                values = [item for item in values if item.status not in {"supported", "refuted"}]
            return [HypothesisEntry.from_dict(item.to_dict()) for item in values if item is not None]

    def get(self, identifier: str) -> HypothesisEntry | None:
        with self._lock:
            item = self._entries.get(str(identifier))
            return HypothesisEntry.from_dict(item.to_dict()) if item else None

    def mark_scheduled(self, identifiers: Iterable[str], *, iteration: int | None = None, reason: str = "") -> None:
        with self._lock:
            for identifier in identifiers or ():
                item = self._entries.get(str(identifier))
                if item is None:
                    continue
                item.status = "scheduled"
                item.attempts += 1
                item.last_iteration = iteration if isinstance(iteration, int) else item.last_iteration
                item.last_reason = _text(reason, 500)
            self._save()

    def mark_result(
        self,
        identifier: str,
        status: str,
        *,
        result_id: str | None = None,
        iteration: int | None = None,
        reason: str = "",
        expected_gain: float | None = None,
        uncertainty: float | None = None,
        information_gain: float | None = None,
    ) -> None:
        with self._lock:
            item = self._entries.get(str(identifier))
            if item is None:
                return
            status = status if status in HYPOTHESIS_STATUSES else "inconclusive"
            item.status = status
            if result_id and result_id not in item.result_ids:
                item.result_ids.append(str(result_id))
            if isinstance(iteration, int):
                item.last_iteration = iteration
            if expected_gain is not None:
                item.expected_gain = _finite(expected_gain)
            if uncertainty is not None:
                item.uncertainty = max(0.0, _finite(uncertainty))
            if information_gain is not None:
                item.information_gain = max(0.0, _finite(information_gain))
            item.last_reason = _text(reason, 500)
            self._save()

    def mark_tested(
        self,
        identifier: str,
        *,
        result_id: str | None = None,
        iteration: int | None = None,
        reason: str = "",
    ) -> None:
        """Record that a hypothesis reached evaluation but has no verdict yet."""
        self.mark_result(
            identifier,
            "tested",
            result_id=result_id,
            iteration=iteration,
            reason=reason or "candidate evaluated; awaiting paired or held-out verdict",
        )

    def pending(self, limit: int | None = None) -> list[dict[str, Any]]:
        values = [entry.to_dict() for entry in self.entries(include_terminal=False)]
        values.sort(key=lambda item: (-float(item.get("priority", 0.0)), item["hypothesis"]["id"]))
        if limit is not None:
            values = values[: max(0, int(limit))]
        return values

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "openhyra-pending-hypotheses.v1",
            "entries": [entry.to_dict() for entry in self.entries()],
        }

    # Small aliases keep integrations readable and make the queue convenient
    # for notebooks without duplicating state-management logic.
    add = enqueue
    schedule = mark_scheduled
    update = mark_result
    list_pending = pending


class AcquisitionRouter:
    """Rank and select hypotheses using deterministic uncertainty-aware scores."""

    def __init__(self, queue: PendingHypothesisQueue | None = None):
        self.queue = queue or PendingHypothesisQueue()

    @staticmethod
    def _rank_entry(entry: HypothesisEntry, state: Mapping[str, Any] | None = None) -> float:
        hypothesis = entry.hypothesis
        state = state or {}
        # State may expose cells under either a direct id or id::slice key.
        cell = state.get(hypothesis.get("id"), {}) if isinstance(state, Mapping) else {}
        if not cell and isinstance(state, Mapping):
            cells = state.get("cells", state.get("beliefs", {}))
            if isinstance(cells, Mapping):
                candidates = [
                    value for key, value in cells.items()
                    if str(key).split("::", 1)[0] == str(hypothesis.get("id"))
                    and isinstance(value, Mapping)
                ]
                if candidates:
                    # Aggregate slices conservatively: retain the largest
                    # uncertainty and the most optimistic expected gain.
                    cell = {
                        "uncertainty": max(
                            _finite(value.get("se"), 0.0)
                            for value in candidates if isinstance(value, Mapping)
                        ),
                        "expected_gain": max(
                            _finite(value.get("mean"), 0.0)
                            for value in candidates if isinstance(value, Mapping)
                        ),
                    }
        if not isinstance(cell, Mapping):
            cell = {}
        uncertainty = max(
            0.0,
            _finite(cell.get("uncertainty"), entry.uncertainty),
        )
        expected = _finite(
            cell.get("expected_gain", cell.get("mean_delta")), entry.expected_gain
        )
        information = max(
            0.0,
            _finite(cell.get("information_gain"), entry.information_gain),
        )
        novelty = 1.0 if entry.status == "untried" else 0.0
        recency_penalty = min(0.5, max(0, entry.attempts) * 0.05)
        # Stable weighted acquisition rule: exploit expected gain, investigate
        # uncertainty, and reserve a small fraction for unseen mechanisms.
        score = expected + 0.50 * uncertainty + 0.35 * information + 0.20 * novelty - recency_penalty
        if entry.status == "refuted":
            score -= 1.0
        elif entry.status == "supported":
            score += 0.10
        entry.expected_gain = expected
        entry.uncertainty = uncertainty
        entry.information_gain = information
        entry.priority = float(score)
        return float(score)

    def rank(
        self,
        hypotheses: Iterable[Mapping[str, Any]] | None = None,
        *,
        state: Mapping[str, Any] | None = None,
        iteration: int | None = None,
    ) -> list[HypothesisEntry]:
        if hypotheses is not None:
            self.queue.enqueue(hypotheses, iteration=iteration)
        entries = self.queue.entries(include_terminal=False)
        for entry in entries:
            self._rank_entry(entry, state)
        entries.sort(key=lambda item: (-item.priority, item.id))
        # Persist priority updates without exposing mutable entries.
        for entry in entries:
            original = self.queue._entries.get(entry.id)
            if original is not None:
                original.priority = entry.priority
                original.expected_gain = entry.expected_gain
                original.uncertainty = entry.uncertainty
                original.information_gain = entry.information_gain
        self.queue._save()
        return entries

    def select(
        self,
        hypotheses: Iterable[Mapping[str, Any]] | None = None,
        *,
        count: int = 1,
        state: Mapping[str, Any] | None = None,
        iteration: int | None = None,
        mark_scheduled: bool = True,
    ) -> list[dict[str, Any]]:
        ranked = self.rank(hypotheses, state=state, iteration=iteration)
        chosen = ranked[: max(0, int(count))]
        if mark_scheduled:
            self.queue.mark_scheduled(
                [entry.id for entry in chosen],
                iteration=iteration,
                reason="deterministic acquisition selection",
            )
        return [dict(entry.hypothesis) for entry in chosen]

    def observe_result(self, identifier: str, *, improved: bool | None, result_id: str | None = None, iteration: int | None = None, reason: str = "") -> None:
        if improved is True:
            status = "supported"
        elif improved is False:
            status = "refuted"
        else:
            status = "inconclusive"
        self.queue.mark_result(identifier, status, result_id=result_id, iteration=iteration, reason=reason)


def route_hypotheses(
    hypotheses: Iterable[Mapping[str, Any]],
    *,
    count: int,
    queue: PendingHypothesisQueue | None = None,
    state: Mapping[str, Any] | None = None,
    iteration: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convenience API returning ``(selected, pending_snapshot)``."""
    router = AcquisitionRouter(queue)
    selected = router.select(
        hypotheses,
        count=count,
        state=state,
        iteration=iteration,
    )
    pending = router.queue.pending()
    return selected, pending


__all__ = [
    "AcquisitionRouter",
    "HYPOTHESIS_STATUSES",
    "HypothesisEntry",
    "INTERVENTION_OPERATORS",
    "INTERVENTION_SCOPES",
    "PendingHypothesisQueue",
    "normalize_intervention",
    "route_hypotheses",
]
