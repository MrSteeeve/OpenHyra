from __future__ import annotations
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
from behavior_index import BehaviorIndex
from experience_events import ExperienceEventStore
from mechanism_hypotheses import mechanism_generation_operator
from schemas_v5 import (AnalogyHypothesis, AnalogyResult, BehaviorProfile,
                        ExperimentEvent, ExperimentPlan, IslandEpoch, MechanismCard)
_BEHAVIOR_BOUNDARIES = {"performance": [-0.01, 0.0, 0.01, 0.03],
                        "tail_risk": [0.005, 0.01, 0.02]}
_SUCCESS_STATUSES = {"ok", "early_stopped"}
def _cap(text: str, char_limit: int) -> str:
    limit = max(0, int(char_limit))
    if len(text) <= limit:
        return text
    marker = "\n\n[truncated at character safety limit]"
    if limit <= len(marker):
        return marker[:limit]
    return text[: limit - len(marker)].rstrip() + marker
def _render(schema: str, sections: list[tuple[str, Any]], char_limit: int) -> str:
    lines = [f"# {schema}"]
    for title, value in sections:
        lines.extend(["", f"## {title}"])
        if title == "Parent Source":
            lines.extend(["```python", str(value), "```"])
        elif isinstance(value, str):
            lines.append(value)
        else:
            lines.append(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return _cap("\n".join(lines), char_limit)
@dataclass
class PortfolioPacket:
    active_islands: list[dict]
    global_best_record_id: str
    global_best_score: float | None
    frozen_baseline_score: float | None
    per_island_representatives: list[dict]
    pending_analogy_pairs: list[dict]
    budget_diagnostics: dict
    stop_diagnostics: dict
    # Completed paired experiments are compact feedback for the next Context
    # round.  Keeping them in the existing portfolio packet makes outcomes
    # visible even when neither arm is an island best/frontier representative.
    completed_analogy_results: list[dict] = field(default_factory=list)
    schema: str = field(default="openhyra-portfolio-packet.v1", kw_only=True)
    def to_text(self, char_limit: int = 16_000) -> str:
        return _render(self.schema, [("Portfolio", asdict(self))], char_limit)
@dataclass
class AnalysisPacket:
    portfolio: PortfolioPacket
    target_island_recent: list[dict]
    representative_profiles: list[dict]
    semantic_rules: list[dict]
    failure_modes: list[dict]
    counterexamples: list[dict]
    schema: str = field(default="openhyra-analysis-packet.v1", kw_only=True)
    def to_text(self, char_limit: int = 48_000) -> str:
        return _render(self.schema, [("Analysis", asdict(self))], char_limit)
@dataclass
class AnalogyPacket:
    target_parent: dict
    source_candidates: list[dict]
    counterexamples: list[dict]
    mechanism_cards: list[dict]
    behavior_slices: list[dict]
    existing_transfer_edges: list[dict]
    allowed_operators: list[str]
    schema: str = field(default="openhyra-analogy-packet.v1", kw_only=True)
    def to_text(self, char_limit: int = 80_000) -> str:
        return _render(self.schema, [("Analogy Evidence", asdict(self))], char_limit)
@dataclass
class ProposalPacket:
    parent_source: str
    inspiration_diffs: list[Any]
    experiment_plan: dict
    protocol_docs: dict
    negative_constraints: list[str]
    candidate_seed: int
    schema: str = field(default="openhyra-proposal-packet.v1", kw_only=True)
    def to_text(self, char_limit: int = 64_000) -> str:
        sections = [("Experiment Plan", self.experiment_plan),
                    ("Protocol Documents", self.protocol_docs),
                    ("Negative Constraints", self.negative_constraints),
                    ("Candidate Seed", self.candidate_seed),
                    ("Inspiration Diffs", self.inspiration_diffs),
                    ("Parent Source", self.parent_source)]
        return _render(self.schema, sections, char_limit)
@dataclass
class PacketProvenance:
    packet_type: str
    selected_record_ids: list[str]
    generation_rules: dict
    char_count: int
    estimated_tokens: float
    schema_version: str = "v1"
class ContextRetrieval:
    def __init__(
        self,
        events: ExperienceEventStore,
        islands: list[IslandEpoch],
        island_records: dict[str, list[str]],
        profiles: dict[str, BehaviorProfile],
        cards: dict[str, MechanismCard] | Iterable[MechanismCard],
        hypotheses: list[AnalogyHypothesis],
        frozen_baseline_score: float | None = None,
        allowed_operators: Iterable[str] | None = None,
        direction: str = "max",
    ) -> None:
        self.events = events
        self.islands = list(islands)
        self.island_records = {key: list(value) for key, value in island_records.items()}
        self.profiles = dict(profiles)
        self.cards = dict(cards) if isinstance(cards, dict) else {
            card.record_id: card for card in cards}
        self.hypotheses = list(hypotheses)
        self.frozen_baseline_score = frozen_baseline_score
        if direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")
        self.direction = direction
        # Keep the historical operator set by default.  Research tasks can
        # opt into a wider portfolio without changing the packet schema or
        # affecting legacy callers/tests.
        self.allowed_operators = list(
            allowed_operators
            if allowed_operators is not None
            else ["feature_augment", "residualize"]
        )
        self.behavior_index = BehaviorIndex(_BEHAVIOR_BOUNDARIES)
    @staticmethod
    def _epoch_id(epoch: IslandEpoch) -> str:
        return f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
    def _island_events(
        self, island_epoch_id: str, events: list[ExperimentEvent]
    ) -> list[ExperimentEvent]:
        mapped_ids = set(self.island_records.get(island_epoch_id, []))
        return [
            event
            for event in events
            if event.island_epoch_id == island_epoch_id or event.record_id in mapped_ids
        ]
    def _evidence_key(self, event: ExperimentEvent) -> tuple[Any, ...]:
        score = (
            float(event.score)
            if event.score is not None and self.direction == "max"
            else (
                -float(event.score)
                if event.score is not None
                else float("-inf")
            )
        )
        return (event.status in _SUCCESS_STATUSES, event.score is not None,
                score,
                event.record_id in self.profiles, event.record_id in self.cards,
                event.created_at, event.record_id)
    def _summary(self, event: ExperimentEvent | None) -> dict:
        if event is None:
            return {}
        summary = {
            "record_id": event.record_id,
            "island_epoch_id": event.island_epoch_id,
            "status": event.status,
            "score": event.score,
            "score_metric": event.score_metric,
            "parent_ids": list(event.parent_ids),
            "inspiration_ids": list(event.inspiration_ids),
            "created_at": event.created_at,
        }
        profile = self.profiles.get(event.record_id)
        if profile is not None:
            summary["behavior_cell"] = list(self.behavior_index.assign_cell(profile))
        # Candidate-level mechanism labels live in the existing runtime object
        # reference rather than requiring a new ExperimentEvent schema field.
        # Expose them to later Context rounds when present.
        if event.runtime_metrics_ref:
            # Older fixtures and records may carry a descriptive ``sha256:...``
            # marker rather than the canonical ObjectStore digest.  Runtime
            # lineage is optional context, so ignore non-canonical/missing
            # references instead of making portfolio construction fail.
            ref = event.runtime_metrics_ref
            runtime_path = None
            if (
                isinstance(ref, str)
                and len(ref) == 64
                and all(char in "0123456789abcdef" for char in ref)
            ):
                try:
                    runtime_path = self.events.object_store.get_path(
                        ref, "runtime_metrics.json"
                    )
                except (OSError, ValueError):
                    runtime_path = None
            if runtime_path is not None:
                try:
                    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                except (OSError, TypeError, ValueError):
                    runtime = None
                if isinstance(runtime, dict) and isinstance(
                    runtime.get("mechanism_lineage"), dict
                ):
                    summary["mechanism_lineage"] = dict(
                        runtime["mechanism_lineage"]
                    )
        return summary
    def _profile_summary(self, record_id: str) -> dict:
        profile = self.profiles.get(record_id)
        if profile is None:
            return {}
        payload = profile.to_dict()
        payload.update(
            {"record_id": record_id, "behavior_cell": list(self.behavior_index.assign_cell(profile))}
        )
        return payload
    def _best(self, events: list[ExperimentEvent]) -> ExperimentEvent | None:
        scored = [
            event
            for event in events
            if event.status in _SUCCESS_STATUSES and event.score is not None
        ]
        if self.direction == "min":
            return min(
                scored,
                key=lambda event: (float(event.score), event.record_id),
                default=None,
            )
        return max(
            scored,
            key=lambda event: (float(event.score), event.record_id),
            default=None,
        )
    def _frontier(self, events: list[ExperimentEvent]) -> ExperimentEvent | None:
        profiled = [event for event in events if event.record_id in self.profiles]
        if len(profiled) < 2:
            return profiled[0] if profiled else max(events, key=self._evidence_key, default=None)
        ranked = []
        for event in profiled:
            others = [self.profiles[item.record_id] for item in profiled if item is not event]
            distance = self.behavior_index.nearest_neighbors(
                self.profiles[event.record_id], others, k=1
            )[0][1]
            ranked.append((distance, self._evidence_key(event), event))
        return max(ranked, key=lambda item: (item[0], item[1]))[2]
    def _coverage(self, events: list[ExperimentEvent]) -> dict:
        profiles = [self.profiles[event.record_id] for event in events if event.record_id in self.profiles]
        return self.behavior_index.cell_diversity(profiles)

    def _atlas_entries(self, events: list[ExperimentEvent]) -> list[dict[str, Any]]:
        """Build compact QD entries from evaluator profiles and lineage."""
        entries: list[dict[str, Any]] = []
        for event in events:
            profile = self.profiles.get(event.record_id)
            if profile is None:
                continue
            summary = self._summary(event)
            lineage = summary.get("mechanism_lineage", {})
            if not isinstance(lineage, dict):
                lineage = {}
            mechanism_id = (
                lineage.get("mechanism_id")
                or lineage.get("hypothesis_id")
                or "unknown"
            )
            # A task may provide a coarse regime label directly.  If it does
            # not, retain ``unknown`` rather than inventing a regime from a
            # free-text direction.
            regime = lineage.get("regime") or lineage.get("target_regime") or "unknown"
            entries.append({
                "record_id": event.record_id,
                "score": event.score,
                "status": event.status,
                "profile": profile,
                "mechanism_id": mechanism_id,
                "regime": regime,
            })
        return entries
    def _recent_improvement(self, events: list[ExperimentEvent]) -> float:
        scored = sorted(
            (
                event
                for event in events
                if event.status in _SUCCESS_STATUSES and event.score is not None
            ),
            key=lambda event: (event.created_at, event.record_id),
        )
        if len(scored) < 2:
            return 0.0
        split = max(1, len(scored) // 2)
        pick = min if self.direction == "min" else max
        earlier = pick(float(event.score) for event in scored[:split])
        recent = pick(float(event.score) for event in scored[split:])
        return (
            earlier - recent
            if self.direction == "min"
            else recent - earlier
        )
    @staticmethod
    def _unique(record_ids: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(record_id for record_id in record_ids if record_id))
    def _provenance(self, kind: str, packet: Any, ids: Iterable[str], rules: dict) -> PacketProvenance:
        char_count = len(packet.to_text())
        return PacketProvenance(kind, self._unique(ids), rules, char_count, char_count / 4)
    def build_portfolio(self) -> tuple[PortfolioPacket, PacketProvenance]:
        events = self.events.read_experiment_events()
        analogy_results = self.events.read_analogy_results()
        hypotheses_by_id = {hypothesis.id: hypothesis for hypothesis in self.hypotheses}
        active = [epoch for epoch in self.islands if epoch.status == "active"]
        active_rows, representatives, selected = [], [], []
        for epoch in active:
            epoch_id = self._epoch_id(epoch)
            items = self._island_events(epoch_id, events)
            failures = [event for event in items if event.status not in _SUCCESS_STATUSES]
            best, frontier = self._best(items), self._frontier(items)
            failure = max(failures, key=self._evidence_key, default=None)
            atlas = self.behavior_index.quality_diversity_archive(
                self._atlas_entries(items),
                direction=self.direction,
            )
            atlas_elites = []
            for elite in atlas.values():
                record_id = elite.get("record_id")
                event = next(
                    (candidate for candidate in items if candidate.record_id == record_id),
                    None,
                )
                if event is not None:
                    row = self._summary(event)
                    row["atlas_cell"] = [
                        elite["atlas_cell"][0],
                        list(elite["atlas_cell"][1]),
                        elite["atlas_cell"][2],
                    ]
                    atlas_elites.append(row)
            active_rows.append(
                {
                    "island_epoch_id": epoch_id,
                    "size": len(items),
                    "best_score": best.score if best else None,
                    "best_record_id": best.record_id if best else "",
                    "recent_improvement": self._recent_improvement(items),
                    "coverage": self._coverage(items),
                    "atlas_coverage": {
                        "occupied_cells": len(atlas),
                        "mechanism_behavior_regime": True,
                    },
                }
            )
            representatives.append(
                {
                    "island_epoch_id": epoch_id,
                    "best": self._summary(best),
                    "frontier": self._summary(frontier),
                    "failure": self._summary(failure),
                    "cell_elites": atlas_elites,
                }
            )
            selected.extend(item.record_id for item in (best, frontier, failure) if item)
            selected.extend(row.get("record_id", "") for row in atlas_elites)
        global_best = self._best(events)
        # Hypotheses are append-only: executing one must not rewrite its
        # preregistered definition.  Treat any recorded AnalogyResult,
        # including an execution failure or refutation, as completion for the
        # pending queue while keeping the original hypothesis available in
        # the graph and history.
        completed_hypothesis_ids = {
            result.analogy_hypothesis_id for result in analogy_results
        }
        pending = [
            item.to_dict()
            for item in self.hypotheses
            if item.status != "completed"
            and item.id not in completed_hypothesis_ids
        ]
        for item in self.hypotheses:
            if (
                item.status != "completed"
                and item.id not in completed_hypothesis_ids
            ):
                selected.extend(item.source_record_ids + [item.target_parent_id])
        # Preserve a bounded, compact result trail for Context.  The result
        # itself is evaluator-produced; the optional hypothesis projection
        # supplies the mechanism/falsifier wording needed to choose the next
        # structure.  This is feedback, not a new verification subsystem.
        completed_results = []
        for result in reversed(analogy_results[-16:]):
            row = {
                "analogy_hypothesis_id": result.analogy_hypothesis_id,
                "guided_record_id": result.guided_record_id,
                "control_record_id": result.control_record_id,
                "guided_delta": result.guided_delta,
                "control_delta": result.control_delta,
                "transfer_gain": result.transfer_gain,
                "transfer_gain_standard_error": result.transfer_gain_standard_error,
                "paired_cell_count": result.paired_cell_count,
                "transfer_gain_ci_low": result.transfer_gain_ci_low,
                "transfer_gain_ci_high": result.transfer_gain_ci_high,
                "relative_transfer_gain": result.relative_transfer_gain,
                "invalid_control_reason": result.invalid_control_reason,
                "control_valid": result.control_valid,
                "predicted_slice_effect": result.predicted_slice_effect,
                "prediction_direction_correct": result.prediction_direction_correct,
                "verdict": result.verdict,
                "matched_arm": "guided+control",
            }
            hypothesis = hypotheses_by_id.get(result.analogy_hypothesis_id)
            if hypothesis is not None:
                control_meta = hypothesis.matched_control
                row["hypothesis"] = {
                    "id": hypothesis.id,
                    "intervention": hypothesis.transferable_intervention,
                    "prediction": hypothesis.predicted_effect,
                    "falsifier": hypothesis.falsifier,
                }
                row["mechanism_id"] = control_meta.get(
                    "mechanism_id", hypothesis.id
                )
                row["family"] = control_meta.get(
                    "family",
                    (
                        hypothesis.relation_mapping[0].get("source_role", "")
                        if hypothesis.relation_mapping else ""
                    ),
                )
                row["generation_operator"] = control_meta.get(
                    "generation_operator",
                    mechanism_generation_operator(
                        {
                            "id": hypothesis.id,
                            "family": row["family"],
                            "mechanism": hypothesis.transferable_intervention,
                        }
                    ),
                )
            else:
                # Results may survive a partial V5 cache reload before their
                # hypothesis line is available.  Keep the joinable identity
                # visible rather than dropping the outcome from Context.
                row["mechanism_id"] = result.analogy_hypothesis_id
                row["family"] = ""
                row["generation_operator"] = ""
            completed_results.append(row)
        packet = PortfolioPacket(
            active_islands=active_rows,
            global_best_record_id=global_best.record_id if global_best else "",
            global_best_score=global_best.score if global_best else None,
            frozen_baseline_score=self.frozen_baseline_score,
            per_island_representatives=representatives,
            pending_analogy_pairs=pending,
            budget_diagnostics={
                "soft_token_budget": 4_000,
                "hard_char_limit": 16_000,
                "event_count": len(events),
                "profile_count": len(self.profiles),
                "completed_analogy_count": len(analogy_results),
            },
            stop_diagnostics={
                "active_island_count": len(active),
                "pending_analogy_count": len(pending),
                "completed_analogy_count": len(analogy_results),
                "successful_event_count": sum(event.status in _SUCCESS_STATUSES for event in events),
                "failed_event_count": sum(event.status not in _SUCCESS_STATUSES for event in events),
                "has_global_best": global_best is not None,
            },
            completed_analogy_results=completed_results,
        )
        if global_best:
            selected.append(global_best.record_id)
        for result in analogy_results[-16:]:
            selected.extend(
                [result.guided_record_id, result.control_record_id]
            )
        rules = {"ranking": "evidence_value_then_deterministic_tiebreak", "per_island": "best_frontier_failure"}
        return packet, self._provenance("PortfolioPacket", packet, selected, rules)
    def build_analysis(self, target_island_epoch_id: str) -> tuple[AnalysisPacket, PacketProvenance]:
        portfolio, portfolio_provenance = self.build_portfolio()
        events = self.events.read_experiment_events()
        target = self._island_events(target_island_epoch_id, events)
        recent_events = sorted(target, key=lambda event: (event.created_at, event.record_id), reverse=True)[:10]
        evidence_ranked = sorted(target + events, key=self._evidence_key, reverse=True)
        profile_ids = self._unique(event.record_id for event in evidence_ranked if event.record_id in self.profiles)[:6]
        relevant_ids = self._unique([event.record_id for event in recent_events] + profile_ids)
        card_ids = [record_id for record_id in relevant_ids if record_id in self.cards]
        card_ids.extend(record_id for record_id in self.cards if record_id not in card_ids)
        semantic_rules = [self.cards[record_id].to_dict() for record_id in card_ids[:4]]
        failures = [event for event in target if event.status not in _SUCCESS_STATUSES]
        counts = Counter(event.status for event in failures)
        failure_modes = [
            {"status": status, "count": count, "record_ids": [event.record_id for event in failures if event.status == status]}
            for status, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        ]
        all_failures = failures + [event for event in events if event.status not in _SUCCESS_STATUSES and event not in failures]
        counterexamples = [self._summary(event) for event in sorted(all_failures, key=self._evidence_key, reverse=True)[:2]]
        packet = AnalysisPacket(
            portfolio=portfolio,
            target_island_recent=[self._summary(event) for event in recent_events],
            representative_profiles=[self._profile_summary(record_id) for record_id in profile_ids],
            semantic_rules=semantic_rules,
            failure_modes=failure_modes,
            counterexamples=counterexamples,
        )
        ids = portfolio_provenance.selected_record_ids + relevant_ids
        ids.extend(event.record_id for event in failures[:2])
        rules = {"target_recent_limit": 10, "profile_limit": 6, "semantic_rule_limit": 4, "failure_mode_limit": 3}
        return packet, self._provenance("AnalysisPacket", packet, ids, rules)
    def build_analogy(self, target_parent_id: str, source_ids: list[str]) -> tuple[AnalogyPacket, PacketProvenance]:
        events = self.events.read_experiment_events()
        event_map = {event.record_id: event for event in events}
        selected_sources = self._unique(source_ids)[:4]
        relevant_ids = self._unique([target_parent_id] + selected_sources)
        results: list[AnalogyResult] = self.events.read_analogy_results()
        related_hypothesis_ids = {
            hypothesis.id
            for hypothesis in self.hypotheses
            if hypothesis.target_parent_id in relevant_ids
            or any(record_id in relevant_ids for record_id in hypothesis.source_record_ids)
        }
        edges = [result.to_dict() for result in results if result.analogy_hypothesis_id in related_hypothesis_ids]
        counterexamples = [edge for edge in edges if edge["verdict"] in {"transfer_refuted", "invalid_control", "execution_failed"}][:2]
        if len(counterexamples) < 2:
            failed = [event for event in events if event.status not in _SUCCESS_STATUSES and event.record_id in relevant_ids]
            counterexamples.extend(self._summary(event) for event in failed[: 2 - len(counterexamples)])
        packet = AnalogyPacket(
            target_parent=self._summary(event_map.get(target_parent_id)),
            source_candidates=[self._summary(event_map[source_id]) for source_id in selected_sources if source_id in event_map],
            counterexamples=counterexamples,
            mechanism_cards=[self.cards[record_id].to_dict() for record_id in relevant_ids if record_id in self.cards],
            behavior_slices=[self._profile_summary(record_id) for record_id in relevant_ids if record_id in self.profiles],
            existing_transfer_edges=edges,
            allowed_operators=list(self.allowed_operators),
        )
        selected = list(relevant_ids)
        for edge in edges:
            selected.extend([edge["guided_record_id"], edge["control_record_id"]])
        rules = {"source_limit": 4, "counterexample_limit": 2, "operators": "prd_v1_enabled_only"}
        return packet, self._provenance("AnalogyPacket", packet, selected, rules)
    def build_proposal(
        self,
        plan: ExperimentPlan,
        parent_source: str,
        inspiration_diffs: list[Any] | None = None,
        candidate_seed: int | None = None,
    ) -> tuple[ProposalPacket, PacketProvenance]:
        target_epoch = next(
            (epoch for epoch in self.islands if self._epoch_id(epoch) == plan.target_island_epoch_id),
            None,
        )
        # If the caller did not provide explicit diffs, derive a compact,
        # evaluator-owned evidence view for the plan's inspiration records.
        # V5 does not retain arbitrary source trees in the event stream, so
        # this is intentionally a behavioral/lineage diff rather than a
        # fabricated code patch.  Proposal agents can use it to understand
        # what changed and why it was retained, while the parent source stays
        # the only editable implementation input.
        if inspiration_diffs is None:
            event_map = {
                event.record_id: event
                for event in self.events.read_experiment_events()
            }
            derived_diffs: list[dict[str, Any]] = []
            for record_id in self._unique(plan.inspiration_ids):
                event = event_map.get(record_id)
                if event is None:
                    continue
                row = self._summary(event)
                row["algorithm_bundle_sha256"] = event.algorithm_bundle_sha256
                row["change_descriptor"] = {
                    "parent_ids": list(event.parent_ids),
                    "inspiration_ids": list(event.inspiration_ids),
                    "mechanism_lineage": row.get("mechanism_lineage", {}),
                }
                derived_diffs.append(row)
                if len(derived_diffs) >= 2:
                    break
            inspiration_diffs = derived_diffs

        packet = ProposalPacket(
            parent_source=parent_source,
            inspiration_diffs=list(inspiration_diffs or [])[:2],
            experiment_plan=plan.to_dict(),
            protocol_docs={
                "context_proposal_separation": "implement the frozen plan without changing its hypothesis or success criterion",
                "artifact": "emit only artifacts allowed by the task protocol",
                "sandbox": "treat candidate execution and logs as untrusted",
                "private_audit": "never available to proposal generation",
            },
            negative_constraints=list(plan.negative_constraints),
            candidate_seed=(
                target_epoch.proposal_seed
                if candidate_seed is None and target_epoch
                else int(candidate_seed or 0)
            ),
        )
        selected = list(plan.parent_ids) + list(plan.inspiration_ids)
        rules = {"full_code_scope": "parent_only", "inspiration_diff_limit": 2, "constraints_source": "experiment_plan"}
        return packet, self._provenance("ProposalPacket", packet, selected, rules)
__all__ = ["AnalysisPacket", "AnalogyPacket", "ContextRetrieval",
           "PacketProvenance", "PortfolioPacket", "ProposalPacket"]
