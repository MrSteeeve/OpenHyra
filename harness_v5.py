from __future__ import annotations

import threading
from pathlib import Path

from schemas_v5 import ExperimentEvent, BehaviorProfile, MechanismCard, ExperimentPlan, IslandEpoch
from object_store import ObjectStore
from experience_events import ExperienceEventStore
from behavior_profiler import BehaviorProfiler
from behavior_index import BehaviorIndex
from island_scheduler import IslandScheduler
from mechanism_cards import MechanismCardBuilder, MechanismCardStore
from analogy_graph import AnalogyGraph
from context_retrieval import ContextRetrieval, PortfolioPacket, AnalysisPacket, ProposalPacket


_BEHAVIOR_BOUNDARIES = {
    "performance": [-0.01, 0.0, 0.01, 0.03],
    "tail_risk": [0.005, 0.01, 0.02],
}


class V5Bridge:
    """Encapsulates v5 state and provides hooks for the harness pipeline.
    
    Thread-safe: all public methods acquire self._lock before mutating state.
    """

    def __init__(self, run_dir: Path, num_islands: int = 4):
        self.run_dir = Path(run_dir)
        self._lock = threading.RLock()
        
        # Initialize v5 subsystems
        v5_dir = self.run_dir / "v5"
        self.object_store = ObjectStore(v5_dir / "objects")
        self.event_store = ExperienceEventStore(v5_dir / "events", self.object_store)
        self.island_scheduler = IslandScheduler(
            v5_dir / "islands.json", num_islands=num_islands,
        )
        self.profiler = BehaviorProfiler()
        self.behavior_index = BehaviorIndex(_BEHAVIOR_BOUNDARIES)
        self.card_store = MechanismCardStore(v5_dir / "mechanism_cards")
        self.analogy_graph = AnalogyGraph.load(v5_dir / "analogy_graph.json")
        
        # In-memory caches
        self._profiles: dict[str, BehaviorProfile] = {}
        self._cards: dict[str, MechanismCard] = {}
        self._frozen_baseline_score: float | None = None

    def initialize(
        self, seed_record_ids: list[str], frozen_baseline_score: float | None = None,
        base_proposal_seed: int = 42,
    ) -> list[IslandEpoch]:
        """Initialize islands from seed records. Call once at pipeline start."""
        with self._lock:
            self._frozen_baseline_score = frozen_baseline_score
            try:
                epochs = self.island_scheduler.initialize(
                    seed_record_ids, context_round=0,
                    base_proposal_seed=base_proposal_seed,
                )
            except RuntimeError:
                # Already initialized (resumed run)
                epochs = self.island_scheduler.get_active_epochs()
            return epochs

    def pick_island(self, context_round: int) -> str:
        """Select an island epoch for the next candidate. Deterministic."""
        with self._lock:
            return self.island_scheduler.sample_island_for_exploration(context_round)

    def on_candidate_evaluated(
        self,
        record_id: str,
        island_epoch_id: str,
        score: float | None,
        status: str,
        description: str,
        parent_ids: list[str],
        metrics: dict,
        source_code: str = "",
        generation_operator: str = "proposal",
    ) -> None:
        """Hook called after a candidate is committed to the old EB.
        
        1. Write ExperimentEvent to v5 event store
        2. Assign to island
        3. Build BehaviorProfile if scored
        4. Build MechanismCard
        """
        with self._lock:
            # 1. Write ExperimentEvent
            event = ExperimentEvent(
                record_id=record_id,
                island_epoch_id=island_epoch_id,
                status=status,
                score=score,
                score_metric=metrics.get("metric", "score"),
                parent_ids=list(parent_ids),
                generation_operator=generation_operator,
                description=description,
            )
            event.validate()
            self.event_store.append_experiment_event(event)

            # 2. Assign to island
            self.island_scheduler.assign_candidate(island_epoch_id, record_id)

            # 3. Build BehaviorProfile if we have per-instance results
            per_instance = metrics.get("per_instance_results")
            baseline_scores = metrics.get("baseline_scores")
            if per_instance and baseline_scores and score is not None:
                profile = self.profiler.build_profile(
                    per_instance_results=per_instance,
                    baseline_scores=baseline_scores,
                    overall_score=score,
                    compute_seconds=metrics.get("elapsed_s", 0.0),
                    memory_mb=metrics.get("peak_memory_mb", 0.0),
                )
                self._profiles[record_id] = profile

            # 4. Build MechanismCard
            manifest = {
                "artifact_protocol": metrics.get("protocol", "feature_ir"),
                "entrypoint": metrics.get("entrypoint", "evaluate_features"),
                "generation_operator": generation_operator,
            }
            card = MechanismCardBuilder.from_bundle_manifest(record_id, manifest)
            self.card_store.save(card)
            self._cards[record_id] = card

    def on_context_complete(self, context_round: int) -> dict[str, str]:
        """Hook called when all candidates for a context round are evaluated.
        
        Runs island review/cull if due. Returns replacements dict (may be empty).
        """
        with self._lock:
            if not self.island_scheduler.should_review(context_round):
                return {}
            
            # Collect scores for all records across active islands
            events = self.event_store.read_experiment_events()
            scores = {}
            for event in events:
                if event.score is not None:
                    scores[event.record_id] = float(event.score)
            
            if not scores:
                return {}
            return self.island_scheduler.run_review(context_round, scores)

    def build_context(self, target_island_epoch_id: str | None = None) -> dict:
        """Build v5 retrieval packets for the Context Agent.
        
        Returns dict with 'portfolio_text', 'analysis_text', and raw packet objects.
        """
        with self._lock:
            retrieval = ContextRetrieval(
                events=self.event_store,
                islands=self.island_scheduler.get_all_epochs(),
                island_records={
                    f"{epoch.island_id}_epoch_{epoch.epoch:02d}": 
                    self.island_scheduler.get_island_records(
                        f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
                    )
                    for epoch in self.island_scheduler.get_all_epochs()
                },
                profiles=dict(self._profiles),
                cards=dict(self._cards),
                hypotheses=[],
                frozen_baseline_score=self._frozen_baseline_score,
            )
            portfolio, portfolio_prov = retrieval.build_portfolio()
            
            analysis = None
            analysis_prov = None
            if target_island_epoch_id:
                analysis, analysis_prov = retrieval.build_analysis(target_island_epoch_id)
            
            return {
                "portfolio": portfolio,
                "portfolio_text": portfolio.to_text(),
                "portfolio_provenance": portfolio_prov,
                "analysis": analysis,
                "analysis_text": analysis.to_text() if analysis else "",
                "analysis_provenance": analysis_prov,
            }

    def build_proposal_context(
        self, plan: ExperimentPlan, parent_source: str,
    ) -> dict:
        """Build a ProposalPacket for the Proposal Agent."""
        with self._lock:
            retrieval = ContextRetrieval(
                events=self.event_store,
                islands=self.island_scheduler.get_all_epochs(),
                island_records={
                    f"{epoch.island_id}_epoch_{epoch.epoch:02d}":
                    self.island_scheduler.get_island_records(
                        f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
                    )
                    for epoch in self.island_scheduler.get_all_epochs()
                },
                profiles=dict(self._profiles),
                cards=dict(self._cards),
                hypotheses=[],
                frozen_baseline_score=self._frozen_baseline_score,
            )
            packet, prov = retrieval.build_proposal(plan, parent_source)
            return {
                "proposal": packet,
                "proposal_text": packet.to_text(),
                "provenance": prov,
            }

    def get_island_diagnostics(self) -> dict:
        """Return diagnostic info about island state for logging."""
        with self._lock:
            active = self.island_scheduler.get_active_epochs()
            return {
                "active_islands": len(active),
                "total_epochs": len(self.island_scheduler.get_all_epochs()),
                "profiles_cached": len(self._profiles),
                "cards_cached": len(self._cards),
                "island_sizes": {
                    f"{e.island_id}_epoch_{e.epoch:02d}": len(
                        self.island_scheduler.get_island_records(
                            f"{e.island_id}_epoch_{e.epoch:02d}"
                        )
                    )
                    for e in active
                },
            }

    def save_state(self) -> None:
        """Persist analogy graph (other state auto-persists)."""
        with self._lock:
            v5_dir = self.run_dir / "v5"
            self.analogy_graph.save(v5_dir / "analogy_graph.json")
