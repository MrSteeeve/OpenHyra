from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Iterable

from schemas_v5 import (
    AnalogyHypothesis,
    AnalogyResult,
    BehaviorProfile,
    ExperimentEvent,
    ExperimentPlan,
    IslandEpoch,
    MechanismCard,
)
from experience_events import ExperienceEventStore
from behavior_profiler import BehaviorProfiler
from behavior_index import BehaviorIndex
from island_scheduler import IslandScheduler
from mechanism_cards import MechanismCardBuilder, MechanismCardStore
from analogy_graph import AnalogyGraph
from context_retrieval import (
    ContextRetrieval,
    PortfolioPacket,
    AnalysisPacket,
    ProposalPacket,
)
from feedback import (
    BeliefReducer,
    FeedbackPacket,
    ProblemState,
    ProblemStateLog,
    is_not_observed,
    render_feedback_context,
)


_BEHAVIOR_BOUNDARIES = {
    "performance": [-0.01, 0.0, 0.01, 0.03],
    "tail_risk": [0.005, 0.01, 0.02],
}

_LEGACY_STATUS_MAP = {
    "ok": "ok",
    "crash": "runtime_error",
    "rejected": "static_rejected",
    "cancelled": "cancelled",
    "violation": "violation",
    "timeout": "timeout",
    "early_stopped": "early_stopped",
    "runtime_error": "runtime_error",
    "static_rejected": "static_rejected",
    "artifact_rejected": "artifact_rejected",
    "oom": "oom",
}
_SUCCESS_STATUSES = {"ok", "early_stopped"}


def _map_status(legacy_status: str) -> str:
    """Map harness status strings to V5 ExperimentEvent allowed statuses."""
    mapped = _LEGACY_STATUS_MAP.get(legacy_status)
    if mapped is not None:
        return mapped
    return "runtime_error"


def adapt_bermudan_metrics(raw_metrics: dict) -> dict:
    """Extract V5-compatible fields from Bermudan evaluator output.

    The Bermudan evaluator produces: metric, runtime_seconds, summaries,
    candidate_hash, feature_program_sha256, etc. This adapter maps those
    to the field names V5Bridge.on_candidate_evaluated expects.

    Fields that aren't available from the evaluator are omitted rather
    than filled with zeros — the bridge handles missing fields gracefully.
    """
    adapted: dict = {}

    adapted["score_metric"] = raw_metrics.get("metric", "aggregate_score")
    # Prefer sandbox-produced artifact hash; evaluator's candidate_hash is a feature-level hash
    adapted["artifact_sha256"] = raw_metrics.get("artifact_sha256", "")
    if not adapted["artifact_sha256"]:
        adapted["artifact_sha256"] = raw_metrics.get("candidate_hash", "")
    # AlgorithmBundle runs expose a parent-controlled source digest in
    # ``algorithm_bundle_sha256``.  V5's ExperimentEvent field is the code
    # identity, so prefer that digest when present; legacy feature runs do
    # not provide it and retain the historical artifact/candidate fallback.
    if raw_metrics.get("algorithm_bundle_sha256"):
        adapted["artifact_sha256"] = raw_metrics["algorithm_bundle_sha256"]
        adapted["algorithm_bundle_sha256"] = raw_metrics[
            "algorithm_bundle_sha256"
        ]

    runtime_s = raw_metrics.get("runtime_seconds")
    if runtime_s is None:
        runtime_s = raw_metrics.get("total_seconds")
    if runtime_s is not None:
        adapted["elapsed_s"] = float(runtime_s)

    summaries = raw_metrics.get("summaries")
    if isinstance(summaries, list) and summaries:
        per_instance_scores: dict[str, float] = {}
        baseline_scores: dict[str, float] = {}
        instance_counts: dict[str, int] = {}

        for s in summaries:
            iid = s.get("instance_id", "")
            if not iid:
                continue
            candidate_lb = s.get("candidate_lower_bound")
            baseline_lb = s.get("baseline_lower_bound")
            if candidate_lb is not None:
                per_instance_scores[iid] = (
                    per_instance_scores.get(iid, 0.0) + float(candidate_lb)
                )
                instance_counts[iid] = instance_counts.get(iid, 0) + 1
            if baseline_lb is not None:
                baseline_scores[iid] = (
                    baseline_scores.get(iid, 0.0) + float(baseline_lb)
                )

        for iid, count in instance_counts.items():
            if count > 1:
                per_instance_scores[iid] /= count
                if iid in baseline_scores:
                    baseline_scores[iid] /= count

        if per_instance_scores:
            adapted["per_instance_scores"] = per_instance_scores
            adapted["per_instance_results"] = per_instance_scores
        if baseline_scores:
            adapted["baseline_scores"] = baseline_scores

    for key in (
        "generation_operator",
        "experiment_plan_id",
        "inspiration_ids",
        "protocol",
        "artifact_protocol",
        "entrypoint",
        "candidate_mode",
        "mechanism_id",
        "hypothesis_id",
        "matched_pair_id",
        "matched_arm",
        "matched_seed",
        "matched_control_enabled",
        # Behaviour metrics emitted by the Bermudan evaluator.  They are
        # optional, so legacy feature-track records remain unchanged.
        "per_instance_exercise_rates",
        "per_instance_exercise_rate_std",
        "per_instance_stop_time_mean",
        "per_instance_stop_time_std",
        "per_instance_finite_rates",
        "per_instance_by_time_exercise_rates",
        "training_seconds",
        "peak_memory_bytes",
        "inference_us",
        "parameter_count",
        # Evaluator-owned feedback sidecar.  It is persisted as an object
        # reference by V5; recommendations remain separate from observed
        # values inside the packet.
        "feedback_packet",
        "feedback_packet_id",
        "feedback_packet_schema",
        "failure_rate",
        "failure_rate_observed",
        "deterministic_reproduction_passed",
        "deterministic_reproduction_observed",
    ):
        if key in raw_metrics:
            adapted[key] = raw_metrics[key]

    # Direct callers may provide only ``artifact_protocol``.  V5's card
    # builder historically reads ``protocol`` for this field, so project the
    # trusted artifact identity there without changing legacy records.
    if raw_metrics.get("artifact_protocol") and not raw_metrics.get("protocol"):
        adapted["protocol"] = raw_metrics["artifact_protocol"]

    return adapted


def _adapt_common_fields(raw_metrics: dict) -> dict:
    """Extract universally trusted fields from any evaluator output.

    These fields exist regardless of task protocol. Task-specific metrics
    are passed through as-is in the returned dict.
    """
    adapted = dict(raw_metrics)  # preserve all original fields

    # Normalize runtime field name
    runtime_s = raw_metrics.get("runtime_seconds")
    if runtime_s is None:
        runtime_s = raw_metrics.get("total_seconds")
    if runtime_s is not None:
        adapted["elapsed_s"] = float(runtime_s)

    # score_metric comes from the evaluator
    if "metric" in raw_metrics:
        adapted["score_metric"] = raw_metrics["metric"]

    return adapted


def get_metrics_adapter(protocol: str):
    """Return the appropriate metrics adapter for a task protocol.

    Returns a callable (raw_metrics -> adapted_metrics) or None for passthrough.
    """
    _ADAPTERS = {
        "bermudan-lsmc-feature-ir.v1": adapt_bermudan_metrics,
        # The Python AlgorithmBundle track reports the same trusted
        # per-instance lower-bound summaries as the legacy feature track.
        # Reuse the narrow Bermudan projection so V5 islands retain those
        # fields instead of falling back to generic passthrough metrics.
        "bermudan-lsmc-python.v1": adapt_bermudan_metrics,
        "bermudan-lsmc-algorithm-bundle.v1": adapt_bermudan_metrics,
        "bermudan-python-program-search.v1": adapt_bermudan_metrics,
    }
    adapter = _ADAPTERS.get(protocol)
    if adapter is not None:
        return adapter
    return _adapt_common_fields


class V5Bridge:
    """Encapsulates v5 state and provides hooks for the harness pipeline.

    Thread-safe: all public methods acquire self._lock before mutating state.
    """

    def __init__(
        self,
        run_dir: Path,
        num_islands: int = 4,
        hypotheses: Iterable[AnalogyHypothesis] | None = None,
        allowed_operators: Iterable[str] | None = None,
        direction: str = "max",
    ):
        self.run_dir = Path(run_dir)
        self._lock = threading.RLock()

        v5_dir = self.run_dir / "v5"
        self.event_store = ExperienceEventStore(v5_dir / "events")
        # Feedback is an additive sidecar.  It is append-only and rebuilt by
        # BeliefReducer, so the legacy experiment event stream remains
        # readable and the Context path can explicitly exclude private data.
        self._feedback_log = ProblemStateLog(v5_dir / "feedback_packets.jsonl")
        self._problem_states_path = v5_dir / "problem_states.jsonl"
        self._problem_states_path.touch(exist_ok=True)
        self._feedback_reducer = BeliefReducer()
        self._problem_state = self._feedback_reducer.rebuild([], state_id="problem-state")
        self._problem_state_ref = ""
        self._problem_state_hash = ""
        self._feedback_packet_ids: set[str] = set()
        self.island_scheduler = IslandScheduler(
            v5_dir / "islands.json",
            num_islands=num_islands,
            direction=direction,
        )
        self.behavior_index = BehaviorIndex(_BEHAVIOR_BOUNDARIES)
        self.card_store = MechanismCardStore(v5_dir / "mechanism_cards")
        self.analogy_graph = AnalogyGraph.load(v5_dir / "analogy_graph.json")
        self._hypotheses_path = v5_dir / "hypotheses.jsonl"
        self._hypotheses_path.touch(exist_ok=True)
        self._allowed_operators = list(
            allowed_operators
            if allowed_operators is not None
            else ["feature_augment", "residualize"]
        )

        self._profiles: dict[str, BehaviorProfile] = {}
        self._cards: dict[str, MechanismCard] = {}
        self._hypotheses: dict[str, AnalogyHypothesis] = {}
        self._frozen_baseline_score: float | None = None
        self._profiler: BehaviorProfiler | None = None

        self._reload_cached_state()
        for hypothesis in hypotheses or ():
            self.record_hypothesis(hypothesis)

    def _log_sync_error(self, record_id: str, operation: str, error: Exception) -> None:
        import datetime as _dt
        entry = {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "record_id": record_id,
            "operation": operation,
            "error": repr(error),
            "legacy_eb_committed": True,
        }
        ledger = self.run_dir / "v5" / "sync_errors.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def resolve_sync_error(self, record_id: str, resolution: str) -> None:
        entry = {
            "operation": "resolution",
            "record_id": record_id,
            "resolution": resolution,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "legacy_eb_committed": True,
        }
        ledger = self.run_dir / "v5" / "sync_errors.jsonl"
        with self._lock:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(ledger, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def _read_sync_ledger(self) -> tuple[list[dict], bool]:
        """Read sync entries while keeping a torn line visible to diagnostics."""
        ledger = self.run_dir / "v5" / "sync_errors.jsonl"
        if not ledger.is_file():
            return [], False
        entries = []
        malformed = False
        for line_number, line in enumerate(
            ledger.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                malformed = True
                print(
                    f"[v5] warning: ignoring malformed sync ledger line "
                    f"{line_number}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(entry, dict) or not entry.get("record_id"):
                malformed = True
                print(
                    f"[v5] warning: ignoring invalid sync ledger line "
                    f"{line_number}",
                    file=sys.stderr,
                )
                continue
            entries.append(entry)
        return entries, malformed

    def _unresolved_sync_record_ids(
        self, entries: list[dict] | None = None,
    ) -> set[str]:
        """Resolve errors by latest ledger operation, not record-wide sets."""
        if entries is None:
            entries, _malformed = self._read_sync_ledger()
        latest_operation: dict[str, str] = {}
        for entry in entries:
            latest_operation[entry["record_id"]] = entry.get("operation", "")
        return {
            record_id
            for record_id, operation in latest_operation.items()
            if operation != "resolution"
        }

    def _feedback_sidecar_ref(self, metrics: dict | None) -> tuple[str, str]:
        """Validate and persist an evaluator feedback packet as an object."""
        raw = metrics.get("feedback_packet") if isinstance(metrics, dict) else None
        if raw is None:
            return "", ""
        try:
            packet = raw if isinstance(raw, FeedbackPacket) else FeedbackPacket.from_dict(raw)
            packet.validate()
        except (TypeError, ValueError, KeyError):
            return "", ""
        ref = self.event_store.object_store.put_json(
            packet.to_dict(), "feedback_packet.json"
        )
        return ref, packet.schema

    @staticmethod
    def _state_digest(state: ProblemState) -> str:
        """Return the canonical digest for one immutable public state."""

        encoded = json.dumps(
            state.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _persist_problem_state(self, state: ProblemState) -> None:
        """Store the exact state snapshot and retain its content reference."""

        payload = state.to_dict()
        self._problem_state_ref = self.event_store.object_store.put_json(
            payload, "problem_state.json"
        )
        self._problem_state_hash = self._state_digest(state)

    def _reload_cached_state(self) -> None:
        """Rebuild in-memory caches from persisted state (for resume)."""
        # Feedback packets are facts emitted by the evaluator.  Rebuild the
        # public ProblemState from those packets instead of trusting a mutable
        # derived snapshot; private packets remain persisted but are excluded
        # from Context-facing state.
        try:
            packets = self._feedback_log.read()
            self._feedback_packet_ids = {packet.packet_id for packet in packets}
            public_packets = [
                packet for packet in packets
                if packet.data.get("split") not in {"private"}
            ]
            self._problem_state = self._feedback_reducer.rebuild(
                public_packets, state_id="problem-state"
            )
            if self._problem_state.applied_packet_ids:
                self._problem_state_hash = self._state_digest(self._problem_state)
                try:
                    if self.event_store.object_store.exists(self._problem_state_hash):
                        self._problem_state_ref = self._problem_state_hash
                except (OSError, ValueError):
                    self._problem_state_ref = ""
        except (OSError, ValueError, TypeError):
            # Optional sidecar corruption must not prevent legacy V5 resume.
            self._feedback_packet_ids = set()
            self._problem_state = self._feedback_reducer.rebuild(
                [], state_id="problem-state"
            )
            self._problem_state_ref = ""
            self._problem_state_hash = ""

        # Hypotheses are small, append-only records kept next to V5 state so
        # resumed contexts retain the research portfolio.  Malformed lines
        # are ignored just like other optional caches; the legacy event path
        # remains unaffected.
        try:
            for line in self._hypotheses_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    hypothesis = AnalogyHypothesis.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                self._hypotheses[hypothesis.id] = hypothesis
                self.analogy_graph.add_hypothesis(hypothesis)
        except OSError:
            pass

        for card in self.card_store.load_all():
            self._cards[card.record_id] = card

        events = self.event_store.read_experiment_events()
        obj_store = self.event_store.object_store
        for event in events:
            if not event.behavior_profile_ref:
                continue
            profile_path = obj_store.get_path(
                event.behavior_profile_ref, "behavior_profile.json"
            )
            if profile_path is None:
                continue
            try:
                profile_data = json.loads(
                    profile_path.read_text(encoding="utf-8")
                )
                self._profiles[event.record_id] = (
                    BehaviorProfile.from_dict(profile_data)
                )
            except (ValueError, TypeError, json.JSONDecodeError, OSError):
                pass

    @property
    def hypotheses(self) -> list[AnalogyHypothesis]:
        """Return registered mechanism hypotheses in deterministic order."""
        with self._lock:
            return [self._hypotheses[key] for key in sorted(self._hypotheses)]

    def record_hypothesis(
        self, hypothesis: AnalogyHypothesis | dict
    ) -> str:
        """Register a structured hypothesis for retrieval and analogy history.

        Registration is idempotent by hypothesis id.  This is intentionally a
        small positive-development hook: callers can propose many hypotheses
        without introducing another audit or approval subsystem.
        """
        if isinstance(hypothesis, dict):
            hypothesis = AnalogyHypothesis.from_dict(hypothesis)
        if not isinstance(hypothesis, AnalogyHypothesis):
            raise TypeError("hypothesis must be AnalogyHypothesis or dict")
        hypothesis.validate()
        with self._lock:
            existing = self._hypotheses.get(hypothesis.id)
            if existing is not None:
                return existing.id
            with self._hypotheses_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(hypothesis.to_dict(), ensure_ascii=False) + "\n")
                stream.flush()
            self._hypotheses[hypothesis.id] = hypothesis
            self.analogy_graph.add_hypothesis(hypothesis)
            self.analogy_graph.save(self.run_dir / "v5" / "analogy_graph.json")
            return hypothesis.id

    def record_analogy_result(
        self,
        result: AnalogyResult | AnalogyHypothesis | dict,
        hypothesis: AnalogyHypothesis | AnalogyResult | dict | None = None,
    ) -> str:
        """Persist a matched-control/analogy result and update the graph."""
        # Accept both natural call orders used by integrations:
        # ``(result, hypothesis)`` and ``(hypothesis, result)``.
        if isinstance(result, AnalogyHypothesis) and isinstance(
            hypothesis, (AnalogyResult, dict)
        ):
            result, hypothesis = hypothesis, result
        elif isinstance(result, dict) and isinstance(hypothesis, AnalogyResult):
            # The first mapping is the hypothesis payload and the second is
            # the result object (the other mixed representation).
            if "transfer_gain" not in result:
                result, hypothesis = hypothesis, result
        elif (
            isinstance(result, dict)
            and isinstance(hypothesis, dict)
            and "transfer_gain" not in result
            and "analogy_hypothesis_id" in hypothesis
        ):
            result, hypothesis = hypothesis, result
        if isinstance(result, dict):
            result = AnalogyResult.from_dict(result)
        if not isinstance(result, AnalogyResult):
            raise TypeError("result must be AnalogyResult or dict")
        result.validate()
        if hypothesis is not None:
            self.record_hypothesis(hypothesis)
        with self._lock:
            hypothesis_obj = self._hypotheses.get(result.analogy_hypothesis_id)
            # A result has no standalone id in v1; this tuple is its stable
            # identity and prevents duplicate writes on resume/replay.
            existing = self.event_store.read_analogy_results()
            identity = (
                result.analogy_hypothesis_id,
                result.guided_record_id,
                result.control_record_id,
            )
            for item in existing:
                if (
                    item.analogy_hypothesis_id,
                    item.guided_record_id,
                    item.control_record_id,
                ) == identity:
                    if hypothesis_obj is not None:
                        self.analogy_graph.add_result(hypothesis_obj, item)
                        self.analogy_graph.save(
                            self.run_dir / "v5" / "analogy_graph.json"
                        )
                    return item.analogy_hypothesis_id
            self.event_store.append_analogy_result(result)
            if hypothesis_obj is not None:
                self.analogy_graph.add_result(hypothesis_obj, result)
                self.analogy_graph.save(self.run_dir / "v5" / "analogy_graph.json")
            return result.analogy_hypothesis_id

    def _reconcile(self, legacy_record_ids: list[str] | None = None) -> dict:
        v5_events = self.event_store.read_experiment_events()
        v5_ids = {e.record_id for e in v5_events}
        orphan_events = sorted(
            event.record_id
            for event in v5_events
            if event.island_epoch_id
            and event.record_id
            not in self.island_scheduler.get_members(
                event.island_epoch_id
            )
        )
        missing_cards = sorted(
            event.record_id
            for event in v5_events
            if event.record_id not in self._cards
        )
        sync_error_count = 0
        ledger_entries, ledger_malformed = self._read_sync_ledger()
        sync_error_count = sum(
            1
            for entry in ledger_entries
            if entry.get("operation") != "resolution"
        )
        result = {
            "v5_event_count": len(v5_ids),
            "sync_error_count": sync_error_count,
            "orphan_events": orphan_events,
            "missing_cards": missing_cards,
            "malformed_sync_ledger": ledger_malformed,
        }
        if legacy_record_ids is not None:
            legacy_set = set(legacy_record_ids)
            missing_in_v5 = legacy_set - v5_ids
            extra_in_v5 = v5_ids - legacy_set
            result["legacy_record_count"] = len(legacy_set)
            result["missing_in_v5"] = sorted(missing_in_v5)
            result["extra_in_v5"] = sorted(extra_in_v5)
            if missing_in_v5:
                print(f"[v5] reconciliation: {len(missing_in_v5)} legacy records missing from V5 events: {sorted(missing_in_v5)[:5]}...", file=sys.stderr)
            if ledger_entries:
                error_ids = self._unresolved_sync_record_ids(ledger_entries)
                orphan_set = set(orphan_events)
                missing_card_set = set(missing_cards)
                for record_id in sorted(error_ids):
                    if (record_id in v5_ids
                            and record_id not in orphan_set
                            and record_id not in missing_card_set):
                        self.resolve_sync_error(record_id, "auto_reconciled")
        return result

    # ------------------------------------------------------------------
    # Evaluator feedback / ProblemState sidecar
    # ------------------------------------------------------------------
    def append_feedback_packet(
        self, packet: FeedbackPacket | dict, *, rebuild: bool = True
    ) -> str:
        """Persist one evaluator packet and update the public belief view.

        The packet log is append-only and idempotent by ``packet_id``.  A
        packet tagged ``data.split=private`` is still retained for audit
        provenance, but :meth:`get_problem_state` and :meth:`build_context`
        exclude it by default.
        """

        normalized = (
            packet if isinstance(packet, FeedbackPacket)
            else FeedbackPacket.from_dict(packet)
        )
        normalized.validate()
        with self._lock:
            packet_id = normalized.packet_id
            if packet_id in self._feedback_packet_ids:
                return packet_id
            self._feedback_log.append(normalized)
            self._feedback_packet_ids.add(packet_id)
            if rebuild:
                self.update_problem_state()
            return packet_id

    # Short alias for task adapters.
    append_feedback = append_feedback_packet

    def read_feedback_packets(
        self, *, include_private: bool = False
    ) -> list[FeedbackPacket]:
        """Read sidecar packets in deterministic append order."""

        with self._lock:
            packets = self._feedback_log.read()
            if include_private:
                return packets
            return [
                packet for packet in packets
                if packet.data.get("split") != "private"
            ]

    def _append_problem_state_snapshot(self, state: ProblemState) -> None:
        payload = state.to_dict()
        with self._problem_states_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _feedback_context_text(
        self, packets: Iterable[FeedbackPacket], *, max_packets: int = 8,
        max_directions: int = 4, max_chars: int = 12_000,
    ) -> str:
        """Render a compact, public-only feedback view for Context prompts.

        The full packets remain available as structured data.  This short
        projection makes the directional signal visible to proposal prompts
        without asking an LLM to rediscover it from a large JSON blob.
        """
        return render_feedback_context(
            packets,
            max_packets=max_packets,
            max_directions=max_directions,
            max_chars=max_chars,
        )

    def update_problem_state(
        self,
        packet: FeedbackPacket | dict | None = None,
        *,
        include_private: bool = False,
        iteration: int | None = None,
        result_ids: Iterable[str] | None = None,
        matched_results: Iterable[dict] | None = None,
        records: Iterable[dict] | None = None,
        feedback_packets: Iterable[FeedbackPacket | dict] | None = None,
    ) -> ProblemState:
        """Rebuild and persist a deterministic ProblemState snapshot.

        Passing ``packet`` is a convenience for adapters that receive a
        result and state update together; the packet is appended exactly once.
        The state itself is immutable, and every changed public snapshot is
        appended to ``v5/problem_states.jsonl``.
        """

        with self._lock:
            # The main harness calls this hook at the round barrier with the
            # committed EB records.  Extract only evaluator-owned packets;
            # candidate prose and matched-control summaries never become
            # observations.  ``iteration``/``matched_results`` are accepted
            # as provenance-friendly keyword arguments for older bridges and
            # intentionally do not alter the reducer.
            packets_to_append: list[FeedbackPacket] = []
            if packet is not None:
                normalized = (
                    packet if isinstance(packet, FeedbackPacket)
                    else FeedbackPacket.from_dict(packet)
                )
                normalized.validate()
                packets_to_append.append(normalized)
            if feedback_packets is not None:
                for raw_packet in feedback_packets:
                    try:
                        normalized = (
                            raw_packet if isinstance(raw_packet, FeedbackPacket)
                            else FeedbackPacket.from_dict(raw_packet)
                        )
                        normalized.validate()
                    except (TypeError, ValueError, KeyError):
                        continue
                    packets_to_append.append(normalized)
            if records is not None:
                wanted = {str(value) for value in (result_ids or ())}
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    if wanted and str(record.get("id")) not in wanted:
                        continue
                    raw_metrics = record.get("metrics", {})
                    raw_packet = (
                        raw_metrics.get("feedback_packet")
                        if isinstance(raw_metrics, dict) else None
                    )
                    if raw_packet is None:
                        continue
                    try:
                        normalized = (
                            raw_packet if isinstance(raw_packet, FeedbackPacket)
                            else FeedbackPacket.from_dict(raw_packet)
                        )
                        normalized.validate()
                    except (TypeError, ValueError, KeyError):
                        continue
                    packets_to_append.append(normalized)
            for normalized in packets_to_append:
                if normalized.packet_id not in self._feedback_packet_ids:
                    self._feedback_log.append(normalized)
                    self._feedback_packet_ids.add(normalized.packet_id)
            packets = self.read_feedback_packets(include_private=include_private)
            state = self._feedback_reducer.rebuild(
                packets, state_id="problem-state"
            )
            if not include_private:
                previous = self._problem_state.to_dict()
                if state.to_dict() != previous:
                    self._append_problem_state_snapshot(state)
                    self._persist_problem_state(state)
                self._problem_state = state
            return state

    def get_problem_state(
        self, *, include_private: bool = False
    ) -> ProblemState:
        """Return a rebuildable state, hiding private feedback by default."""

        with self._lock:
            if include_private:
                return self._feedback_reducer.rebuild(
                    self.read_feedback_packets(include_private=True),
                    state_id="problem-state",
                )
            # Rebuild on read so deleting derived snapshots cannot make the
            # Context view stale.
            self._problem_state = self._feedback_reducer.rebuild(
                self.read_feedback_packets(include_private=False),
                state_id="problem-state",
            )
            return self._problem_state

    def initialize(
        self,
        seed_record_ids: list[str],
        frozen_baseline_score: float | None = None,
        base_proposal_seed: int = 42,
        baseline_scores: dict[str, float] | None = None,
        probe_suite_sha256: str = "",
        direction: str | None = None,
    ) -> list[IslandEpoch]:
        """Initialize islands from seed records. Call once at pipeline start."""
        with self._lock:
            if direction is not None:
                self.island_scheduler.set_direction(direction)
            self._frozen_baseline_score = frozen_baseline_score
            if baseline_scores:
                self._profiler = BehaviorProfiler(
                    baseline_scores=baseline_scores,
                    probe_suite_sha256=probe_suite_sha256,
                )
            try:
                epochs = self.island_scheduler.initialize(
                    seed_record_ids,
                    context_round=0,
                    base_proposal_seed=base_proposal_seed,
                )
            except RuntimeError:
                epochs = self.island_scheduler.get_active_epochs()
            return epochs

    def record_seed(
        self,
        record_id: str,
        score: float | None,
        metrics: dict,
        island_epoch_id: str | None = None,
    ) -> None:
        """Record a legacy seed as an idempotent V5 experiment event."""
        with self._lock:
            existing_ids = {
                event.record_id
                for event in self.event_store.read_experiment_events()
            }
            if record_id in existing_ids:
                return

            per_instance_ref = ""
            per_instance = metrics.get("per_instance_results")
            if per_instance is not None:
                per_instance_ref = self.event_store.object_store.put_json(
                    per_instance, "per_instance_metrics.json"
                )

            runtime_metrics = {
                key: metrics[key]
                for key in (
                    "elapsed_s",
                    "total_seconds",
                    "runtime_seconds",
                    "solver_seconds",
                    "evaluator_seconds",
                )
                if key in metrics
            }
            runtime_ref = ""
            if runtime_metrics:
                runtime_ref = self.event_store.object_store.put_json(
                    runtime_metrics, "runtime_metrics.json"
                )

            feedback_ref, feedback_schema = self._feedback_sidecar_ref(metrics)

            event = ExperimentEvent(
                record_id=record_id,
                algorithm_bundle_sha256=metrics.get(
                    "artifact_sha256", ""
                ),
                experiment_plan_id="seed",
                island_epoch_id=island_epoch_id or "",
                status="ok",
                score=score,
                score_metric=metrics.get(
                    "score_metric", "aggregate_score"
                ),
                per_instance_metrics_ref=per_instance_ref,
                behavior_profile_ref="",
                runtime_metrics_ref=runtime_ref,
                parent_ids=[],
                inspiration_ids=[],
                created_at=datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                feedback_packet_ref=feedback_ref,
                feedback_packet_schema=feedback_schema,
                problem_state_ref=self._problem_state_ref,
                problem_state_version=self._problem_state.state_version,
                problem_state_hash=self._problem_state_hash,
            )
            event.validate()
            self.event_store.append_experiment_event(event)

            if island_epoch_id:
                try:
                    self.island_scheduler.assign_candidate(
                        island_epoch_id, record_id
                    )
                except (KeyError, RuntimeError):
                    pass

            manifest = {
                "artifact_protocol": metrics.get(
                    "artifact_protocol",
                    metrics.get("protocol", "feature_ir"),
                ),
                "entrypoint": metrics.get(
                    "entrypoint", "evaluate_features"
                ),
                "generation_operator": "seed",
            }
            card = MechanismCardBuilder.from_bundle_manifest(
                record_id, manifest
            )
            self.card_store.save(card)
            self._cards[record_id] = card
            if metrics.get("feedback_packet") is not None:
                self.append_feedback_packet(metrics["feedback_packet"])

    def pick_island(self, context_round: int) -> str:
        """Select an island epoch for the next candidate. Deterministic."""
        with self._lock:
            return self.island_scheduler.sample_island_for_exploration(
                context_round
            )

    def on_candidate_evaluated(
        self,
        record_id: str,
        island_epoch_id: str,
        score: float | None,
        status: str,
        parent_ids: list[str],
        metrics: dict,
        description: str = "",
    ) -> None:
        """Hook called after a candidate is committed to the old EB.

        Wrapped in error handling to prevent half-commit state: if V5
        event writing fails, the error is logged but does not propagate
        to the harness worker (the legacy EB commit already succeeded).
        """
        try:
            self._on_candidate_evaluated_inner(
                record_id, island_epoch_id, score, status,
                parent_ids, metrics,
            )
            feedback_payload = metrics.get("feedback_packet")
            if feedback_payload is not None:
                # Persist the packet immediately for crash recovery, but defer
                # the public belief rebuild until the round barrier in the
                # harness.  This prevents sibling candidates from observing a
                # partially updated Context state.
                self.append_feedback_packet(feedback_payload, rebuild=False)
        except Exception as exc:
            print(
                f"[v5] warning: V5 event recording failed for {record_id}: "
                f"{exc!r}",
                file=sys.stderr,
            )
            self._log_sync_error(record_id, "on_candidate_evaluated", exc)

    def _on_candidate_evaluated_inner(
        self,
        record_id: str,
        island_epoch_id: str,
        score: float | None,
        status: str,
        parent_ids: list[str],
        metrics: dict,
    ) -> None:
        with self._lock:
            # Idempotent: skip if event already recorded (e.g., resume replay)
            existing_ids = {
                e.record_id
                for e in self.event_store.read_experiment_events()
            }
            if record_id in existing_ids:
                self._repair_if_needed(record_id, island_epoch_id, metrics)
                return

            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            v5_status = _map_status(status)

            per_instance_ref = ""
            per_instance = metrics.get("per_instance_results")
            if per_instance:
                per_instance_ref = self.event_store.object_store.put_json(
                    per_instance, "per_instance_metrics.json"
                )

            runtime_metrics = {
                k: metrics[k]
                for k in ("elapsed_s", "peak_memory_mb", "runtime_seconds")
                if k in metrics
            }
            mechanism_lineage = {
                k: metrics[k]
                for k in (
                    "mechanism_id", "hypothesis_id", "matched_pair_id",
                    "matched_arm", "matched_seed", "matched_control_enabled",
                    "generation_operator",
                )
                if k in metrics
            }
            if mechanism_lineage:
                runtime_metrics["mechanism_lineage"] = mechanism_lineage
            runtime_ref = ""
            if runtime_metrics:
                runtime_ref = self.event_store.object_store.put_json(
                    runtime_metrics, "runtime_metrics.json"
                )

            feedback_ref, feedback_schema = self._feedback_sidecar_ref(metrics)

            behavior_profile_ref = ""
            if self._profiler and per_instance and score is not None:
                behavior_profile_ref = self._try_build_profile(
                    record_id, metrics
                )

            event = ExperimentEvent(
                record_id=record_id,
                algorithm_bundle_sha256=metrics.get(
                    "artifact_sha256", ""
                ),
                experiment_plan_id=metrics.get("experiment_plan_id", ""),
                island_epoch_id=island_epoch_id,
                status=v5_status,
                score=score,
                score_metric=metrics.get(
                    "score_metric", "aggregate_score"
                ),
                per_instance_metrics_ref=per_instance_ref,
                behavior_profile_ref=behavior_profile_ref,
                runtime_metrics_ref=runtime_ref,
                parent_ids=list(parent_ids),
                inspiration_ids=list(
                    metrics.get("inspiration_ids", [])
                ),
                created_at=now,
                feedback_packet_ref=feedback_ref,
                feedback_packet_schema=feedback_schema,
                problem_state_ref=self._problem_state_ref,
                problem_state_version=self._problem_state.state_version,
                problem_state_hash=self._problem_state_hash,
            )
            event.validate()
            self.event_store.append_experiment_event(event)

            self.island_scheduler.assign_candidate(
                island_epoch_id, record_id
            )

            manifest = {
                "artifact_protocol": metrics.get(
                    "artifact_protocol",
                    metrics.get("protocol", "feature_ir"),
                ),
                "entrypoint": metrics.get(
                    "entrypoint", "evaluate_features"
                ),
                "generation_operator": metrics.get(
                    "generation_operator", "local_mutation"
                ),
            }
            card = MechanismCardBuilder.from_bundle_manifest(
                record_id, manifest
            )
            self.card_store.save(card)
            self._cards[record_id] = card

    def _try_build_profile(
        self, record_id: str, metrics: dict
    ) -> str:
        """Attempt to build and store a BehaviorProfile. Returns ref or ''."""
        per_instance_scores = metrics.get("per_instance_scores")
        exercise_rates = metrics.get("per_instance_exercise_rates")
        if not (
            isinstance(per_instance_scores, dict)
            and per_instance_scores
        ):
            return ""
        if not (
            isinstance(exercise_rates, dict) and exercise_rates
        ):
            return ""
        try:
            profile = self._profiler.build_profile(
                policy_artifact_sha256=metrics.get(
                    "artifact_sha256", ""
                ),
                per_instance_scores=per_instance_scores,
                per_instance_exercise_rates=exercise_rates,
                training_seconds=float(
                    metrics.get("training_seconds", 0.0)
                ),
                peak_memory_bytes=int(
                    metrics.get("peak_memory_bytes", 0)
                ),
                inference_microseconds_per_state=float(
                    metrics.get("inference_us", 0.0)
                ),
                parameter_count=int(
                    metrics.get("parameter_count", 0)
                ),
            )
            self._profiles[record_id] = profile
            return self.event_store.object_store.put_json(
                profile.to_dict(), "behavior_profile.json"
            )
        except (ValueError, KeyError):
            return ""

    def _repair_if_needed(
        self, record_id: str, island_epoch_id: str, metrics: dict
    ) -> None:
        if record_id not in self.island_scheduler.get_members(
            island_epoch_id
        ):
            try:
                self.island_scheduler.assign_candidate(
                    island_epoch_id, record_id
                )
            except (KeyError, RuntimeError):
                pass

        if record_id not in self._cards:
            manifest = {
                "artifact_protocol": metrics.get(
                    "artifact_protocol",
                    metrics.get("protocol", "feature_ir"),
                ),
                "entrypoint": metrics.get(
                    "entrypoint", "evaluate_features"
                ),
                "generation_operator": metrics.get(
                    "generation_operator", "local_mutation"
                ),
            }
            card = MechanismCardBuilder.from_bundle_manifest(
                record_id, manifest
            )
            self.card_store.save(card)
            self._cards[record_id] = card

    def on_context_complete(self, context_round: int) -> dict[str, str]:
        """Hook called when all candidates for a context round are evaluated.

        Runs island review/cull if due. Returns replacements dict.
        """
        with self._lock:
            if not self.island_scheduler.should_review(context_round):
                return {}

            events = self.event_store.read_experiment_events()
            scores = {}
            for event in events:
                if event.status in _SUCCESS_STATUSES and event.score is not None:
                    scores[event.record_id] = float(event.score)

            if not scores:
                return {}

            active = self.island_scheduler.get_active_epochs()
            scored_epoch_count = 0
            for epoch in active:
                epoch_id = f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
                epoch_records = self.island_scheduler.get_members(epoch_id)
                if any(rid in scores for rid in epoch_records):
                    scored_epoch_count += 1

            if scored_epoch_count <= 1:
                print(
                    f"[v5] review deferred at round {context_round}: "
                    f"only {scored_epoch_count}/{len(active)} epochs have scores",
                    file=sys.stderr,
                )
                return {}
            return self.island_scheduler.run_review(context_round, scores)

    def build_context(
        self, target_island_epoch_id: str | None = None
    ) -> dict:
        """Build v5 retrieval packets for the Context Agent."""
        with self._lock:
            retrieval = self._make_retrieval()
            portfolio, portfolio_prov = retrieval.build_portfolio()

            analysis = None
            analysis_prov = None
            if target_island_epoch_id:
                analysis, analysis_prov = retrieval.build_analysis(
                    target_island_epoch_id
                )

            # Only public/development feedback is exposed to Context.  The
            # packet log retains private entries for audit/replay, but they do
            # not enter the proposal prompt or ProblemState used here.
            problem_state = self.get_problem_state(include_private=False)
            feedback_packets = self.read_feedback_packets(include_private=False)
            feedback_text = self._feedback_context_text(feedback_packets)

            return {
                "portfolio": portfolio,
                "portfolio_text": portfolio.to_text(),
                "portfolio_provenance": portfolio_prov,
                "analysis": analysis,
                "analysis_text": (
                    analysis.to_text() if analysis else ""
                ),
                "analysis_provenance": analysis_prov,
                "problem_state": problem_state,
                "problem_state_dict": problem_state.to_dict(),
                "feedback_packets": feedback_packets,
                "feedback_packet_dicts": [
                    packet.to_dict() for packet in feedback_packets
                ],
                "feedback_text": feedback_text,
            }

    def build_proposal_context(
        self,
        plan: ExperimentPlan,
        parent_source: str,
        candidate_seed: int | None = None,
    ) -> dict:
        """Build a ProposalPacket for the Proposal Agent."""
        with self._lock:
            retrieval = self._make_retrieval()
            packet, prov = retrieval.build_proposal(
                plan, parent_source, candidate_seed=candidate_seed,
            )
            return {
                "proposal": packet,
                "proposal_text": packet.to_text(),
                "provenance": prov,
            }

    def get_island_diagnostics(self) -> dict:
        """Return diagnostic info about island state for logging."""
        with self._lock:
            active = self.island_scheduler.get_active_epochs()
            sync_error_count = 0
            ledger_entries, ledger_malformed = self._read_sync_ledger()
            sync_error_count = sum(
                entry.get("operation") != "resolution"
                for entry in ledger_entries
            )
            unresolved_count = len(
                self._unresolved_sync_record_ids(ledger_entries)
            )
            return {
                "active_islands": len(active),
                "total_epochs": len(
                    self.island_scheduler.get_all_epochs()
                ),
                "profiles_cached": len(self._profiles),
                "cards_cached": len(self._cards),
                "sync_error_count": sync_error_count,
                "unresolved_sync_errors": unresolved_count,
                "malformed_sync_ledger": ledger_malformed,
                "feedback_packets": len(self._feedback_packet_ids),
                "feedback_public_packets": len(
                    self.read_feedback_packets(include_private=False)
                ),
                "problem_state_version": self._problem_state.state_version,
                "problem_state_cells": len(self._problem_state.cells),
                "sync_status": self.sync_status,
                "island_sizes": {
                    f"{e.island_id}_epoch_{e.epoch:02d}": len(
                        self.island_scheduler.get_members(
                            f"{e.island_id}_epoch_{e.epoch:02d}"
                        )
                    )
                    for e in active
                },
            }

    @property
    def sync_status(self) -> str:
        """Return V5 sync health: 'healthy' or 'degraded'."""
        ledger_entries, ledger_malformed = self._read_sync_ledger()
        if ledger_malformed or self._unresolved_sync_record_ids(ledger_entries):
            return "degraded"
        for event in self.event_store.read_experiment_events():
            if not event.island_epoch_id:
                continue
            if not self.island_scheduler.get_members(
                event.island_epoch_id
            ) or event.record_id not in self.island_scheduler.get_members(
                event.island_epoch_id
            ):
                return "degraded"
            if event.record_id not in self._cards:
                return "degraded"
        return "healthy"

    def save_state(self) -> None:
        """Persist analogy graph (other state auto-persists)."""
        with self._lock:
            v5_dir = self.run_dir / "v5"
            self.analogy_graph.save(v5_dir / "analogy_graph.json")

    def _make_retrieval(self) -> ContextRetrieval:
        return ContextRetrieval(
            events=self.event_store,
            islands=self.island_scheduler.get_all_epochs(),
            island_records={
                f"{epoch.island_id}_epoch_{epoch.epoch:02d}": self.island_scheduler.get_members(
                    f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
                )
                for epoch in self.island_scheduler.get_all_epochs()
            },
            profiles=dict(self._profiles),
            cards=dict(self._cards),
            hypotheses=self.hypotheses,
            frozen_baseline_score=self._frozen_baseline_score,
            allowed_operators=self._allowed_operators,
            direction=self.island_scheduler.direction,
        )
