from __future__ import annotations
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
from behavior_index import BehaviorIndex
from experience_events import ExperienceEventStore
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
    ) -> None:
        self.events = events
        self.islands = list(islands)
        self.island_records = {key: list(value) for key, value in island_records.items()}
        self.profiles = dict(profiles)
        self.cards = dict(cards) if isinstance(cards, dict) else {
            card.record_id: card for card in cards}
        self.hypotheses = list(hypotheses)
        self.frozen_baseline_score = frozen_baseline_score
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
        return (event.status in _SUCCESS_STATUSES, event.score is not None,
                float(event.score) if event.score is not None else float("-inf"),
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
    @staticmethod
    def _best(events: list[ExperimentEvent]) -> ExperimentEvent | None:
        scored = [
            event
            for event in events
            if event.status in _SUCCESS_STATUSES and event.score is not None
        ]
        return max(scored, key=lambda event: (float(event.score), event.record_id), default=None)
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
    @staticmethod
    def _recent_improvement(events: list[ExperimentEvent]) -> float:
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
        earlier = max(float(event.score) for event in scored[:split])
        recent = max(float(event.score) for event in scored[split:])
        return recent - earlier
    @staticmethod
    def _unique(record_ids: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(record_id for record_id in record_ids if record_id))
    def _provenance(self, kind: str, packet: Any, ids: Iterable[str], rules: dict) -> PacketProvenance:
        char_count = len(packet.to_text())
        return PacketProvenance(kind, self._unique(ids), rules, char_count, char_count / 4)
    def build_portfolio(self) -> tuple[PortfolioPacket, PacketProvenance]:
        events = self.events.read_experiment_events()
        active = [epoch for epoch in self.islands if epoch.status == "active"]
        active_rows, representatives, selected = [], [], []
        for epoch in active:
            epoch_id = self._epoch_id(epoch)
            items = self._island_events(epoch_id, events)
            failures = [event for event in items if event.status not in _SUCCESS_STATUSES]
            best, frontier = self._best(items), self._frontier(items)
            failure = max(failures, key=self._evidence_key, default=None)
            active_rows.append(
                {
                    "island_epoch_id": epoch_id,
                    "size": len(items),
                    "best_score": best.score if best else None,
                    "best_record_id": best.record_id if best else "",
                    "recent_improvement": self._recent_improvement(items),
                    "coverage": self._coverage(items),
                }
            )
            representatives.append(
                {
                    "island_epoch_id": epoch_id,
                    "best": self._summary(best),
                    "frontier": self._summary(frontier),
                    "failure": self._summary(failure),
                }
            )
            selected.extend(item.record_id for item in (best, frontier, failure) if item)
        global_best = self._best(events)
        pending = [item.to_dict() for item in self.hypotheses if item.status != "completed"]
        for item in self.hypotheses:
            if item.status != "completed":
                selected.extend(item.source_record_ids + [item.target_parent_id])
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
            },
            stop_diagnostics={
                "active_island_count": len(active),
                "pending_analogy_count": len(pending),
                "successful_event_count": sum(event.status in _SUCCESS_STATUSES for event in events),
                "failed_event_count": sum(event.status not in _SUCCESS_STATUSES for event in events),
                "has_global_best": global_best is not None,
            },
        )
        if global_best:
            selected.append(global_best.record_id)
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
            allowed_operators=["feature_augment", "residualize"],
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
