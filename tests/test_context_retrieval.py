from __future__ import annotations

from context_retrieval import (
    AnalysisPacket,
    AnalogyPacket,
    ContextRetrieval,
    PortfolioPacket,
    ProposalPacket,
)
from experience_events import ExperienceEventStore
from schemas_v5 import (
    AnalogyHypothesis,
    BehaviorProfile,
    ExperimentEvent,
    ExperimentPlan,
    IslandEpoch,
    MechanismCard,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def make_event(
    record_id: str,
    island_epoch_id: str,
    score: float | None,
    index: int,
    status: str = "ok",
) -> ExperimentEvent:
    event = ExperimentEvent(
        record_id=record_id,
        algorithm_bundle_sha256=SHA_A,
        experiment_plan_id=f"plan_{index:04d}",
        island_epoch_id=island_epoch_id,
        status=status,
        score=score,
        score_metric="paired_lower_bound_lcb",
        per_instance_metrics_ref=f"sha256:{SHA_A}",
        behavior_profile_ref=f"sha256:{SHA_B}",
        runtime_metrics_ref=f"sha256:{SHA_A}",
        parent_ids=[f"parent_{index:04d}"],
        inspiration_ids=[],
        created_at=f"2026-09-01T00:{index:02d}:00Z",
    )
    event.validate()
    return event


def make_island(index: int) -> IslandEpoch:
    island = IslandEpoch(
        island_id=f"island_{index:02d}",
        epoch=0,
        seed_record_ids=[],
        started_after_context_round=0,
        proposal_seed=10_000 + index,
        status="active",
    )
    island.validate()
    return island


def make_profile(index: int) -> BehaviorProfile:
    improvement = 0.005 * (index + 1)
    profile = BehaviorProfile(
        probe_suite="bermudan-behavior-probe.v1",
        probe_suite_sha256=SHA_A,
        policy_artifact_sha256=SHA_B,
        performance={
            "per_instance_improvement": [improvement, improvement + 0.001],
            "paired_mean": improvement,
            "paired_standard_error": 0.001,
        },
        outcome_distribution={
            "loss_definition": "negative_paired_discounted_payoff_improvement",
            "mean_loss": -improvement,
            "var_95": 0.01 + improvement,
            "cvar_95": 0.02 + improvement,
        },
        policy_geometry={
            "exercise_rate_by_instance": [0.2, 0.3],
            "boundary_monotonicity_violations": 0,
            "reference_boundary_agreement": 0.8,
        },
        sensitivity={
            "moneyness": 0.1,
            "volatility": 0.2,
            "correlation": 0.3,
            "time_to_maturity": 0.4,
        },
        robustness={
            "input_scale_invariance_error": 0.01,
            "state_perturbation_lipschitz_proxy": 0.2,
            "seed_instability": 0.02,
        },
        compute={
            "training_seconds": 5.0 + index,
            "peak_memory_bytes": 1_024.0,
            "inference_microseconds_per_state": 2.0,
            "parameter_count": 128.0,
        },
    )
    profile.validate()
    return profile


def make_card(record_id: str) -> MechanismCard:
    card = MechanismCard(
        record_id=record_id,
        deterministic_facts={"family": "residual"},
        trusted_observations={"high_volatility": "improved"},
        llm_inferences=[
            {
                "claim": "normalization may stabilize the residual",
                "confidence": 0.6,
                "evidence_record_ids": [record_id],
                "annotation_event_id": f"annotation_{record_id}",
            }
        ],
    )
    card.validate()
    return card


def make_hypothesis() -> AnalogyHypothesis:
    hypothesis = AnalogyHypothesis(
        id="analogy_0001",
        source_record_ids=["record_02"],
        target_parent_id="record_00",
        relation_mapping=[
            {
                "source_role": "normalized_feature",
                "target_role": "residual_input",
                "shared_relation": "stable_state_scale",
            }
        ],
        non_correspondence=["source weights do not transfer"],
        transferable_intervention="add normalized input",
        predicted_effect={
            "metric": "paired_lower_bound_lcb",
            "direction": "positive",
            "minimum_effect": 0.001,
        },
        falsifier="paired effect is non-positive",
        matched_control={"same_parent": True, "same_budget": True},
        status="preregistered",
    )
    hypothesis.validate()
    return hypothesis


def make_plan(target: str = "island_00_epoch_00") -> ExperimentPlan:
    plan = ExperimentPlan(
        id="plan_proposal",
        action="continue",
        target_island_epoch_id=target,
        generation_operator="analogy_transfer",
        parent_ids=["record_00"],
        inspiration_ids=["record_02"],
        analogy_hypothesis_id="analogy_0001",
        implementation_intent="add one normalized input",
        negative_constraints=["do_not_change_hidden_width"],
        success_criterion="paired_lcb_gt_0",
        budget={
            "candidate_count": 2,
            "sandbox_seconds_per_cell": 60,
            "max_artifact_bytes": 8_388_608,
        },
    )
    plan.validate()
    return plan


def make_retrieval(tmp_path) -> ContextRetrieval:
    store = ExperienceEventStore(tmp_path / "eb")
    islands = [make_island(index) for index in range(4)]
    island_records = {f"island_{index:02d}_epoch_00": [] for index in range(4)}
    profiles: dict[str, BehaviorProfile] = {}
    for index in range(8):
        record_id = f"record_{index:02d}"
        island_id = f"island_{index // 2:02d}_epoch_00"
        status = "ok" if index % 2 == 0 else "runtime_error"
        event = make_event(
            record_id,
            island_id,
            0.1 + index / 100 if status == "ok" else None,
            index,
            status,
        )
        store.append_experiment_event(event)
        island_records[island_id].append(record_id)
        profiles[record_id] = make_profile(index)
    cards = {record_id: make_card(record_id) for record_id in ["record_00", "record_02"]}
    return ContextRetrieval(
        store,
        islands,
        island_records,
        profiles,
        cards,
        [make_hypothesis()],
        frozen_baseline_score=0.08,
    )


def test_build_portfolio_basic(tmp_path):
    packet, _ = make_retrieval(tmp_path).build_portfolio()

    assert isinstance(packet, PortfolioPacket)
    assert len(packet.active_islands) == 4
    assert {item["island_epoch_id"] for item in packet.active_islands} == {
        f"island_{index:02d}_epoch_00" for index in range(4)
    }
    assert all(item["size"] == 2 for item in packet.active_islands)
    assert packet.global_best_record_id == "record_06"


def test_portfolio_char_limit(tmp_path):
    packet, _ = make_retrieval(tmp_path).build_portfolio()

    assert len(packet.to_text()) <= 16_000
    assert len(packet.to_text(char_limit=500)) <= 500


def test_build_analysis_includes_portfolio(tmp_path):
    packet, _ = make_retrieval(tmp_path).build_analysis("island_00_epoch_00")

    assert isinstance(packet, AnalysisPacket)
    assert isinstance(packet.portfolio, PortfolioPacket)
    assert len(packet.portfolio.active_islands) == 4


def test_build_analysis_target_recent(tmp_path):
    retrieval = make_retrieval(tmp_path)
    for index in range(8, 20):
        event = make_event(
            f"record_{index:02d}",
            "island_00_epoch_00",
            0.1 + index / 100,
            index,
        )
        retrieval.events.append_experiment_event(event)
        retrieval.island_records["island_00_epoch_00"].append(event.record_id)

    packet, _ = retrieval.build_analysis("island_00_epoch_00")

    assert len(packet.target_island_recent) == 10
    assert packet.target_island_recent[0]["record_id"] == "record_19"


def test_build_analogy_packet(tmp_path):
    packet, _ = make_retrieval(tmp_path).build_analogy(
        "record_00", ["record_02", "record_04"]
    )

    assert isinstance(packet, AnalogyPacket)
    assert packet.target_parent["record_id"] == "record_00"
    assert [item["record_id"] for item in packet.source_candidates] == [
        "record_02",
        "record_04",
    ]
    assert packet.allowed_operators == ["feature_augment", "residualize"]


def test_build_proposal_packet(tmp_path):
    parent_source = "def candidate(state):\n    return state[0]\n"
    plan = make_plan()

    packet, _ = make_retrieval(tmp_path).build_proposal(
        plan, parent_source, ["@@ first diff", "@@ second diff", "@@ ignored"]
    )

    assert isinstance(packet, ProposalPacket)
    assert packet.parent_source == parent_source
    assert packet.experiment_plan == plan.to_dict()
    assert len(packet.inspiration_diffs) == 2
    assert packet.candidate_seed == 10_000


def test_provenance_tracking(tmp_path):
    packet, provenance = make_retrieval(tmp_path).build_analogy(
        "record_00", ["record_02"]
    )

    assert provenance.packet_type == "AnalogyPacket"
    assert provenance.selected_record_ids[:2] == ["record_00", "record_02"]
    assert provenance.schema_version == "v1"
    assert provenance.char_count == len(packet.to_text())
    assert provenance.estimated_tokens == provenance.char_count / 4


def test_empty_state(tmp_path):
    retrieval = ContextRetrieval(
        ExperienceEventStore(tmp_path / "empty-eb"), [], {}, {}, {}, []
    )

    portfolio, _ = retrieval.build_portfolio()
    analysis, _ = retrieval.build_analysis("missing_epoch")
    analogy, _ = retrieval.build_analogy("missing_parent", [])
    proposal, _ = retrieval.build_proposal(make_plan("missing_epoch"), "")

    assert portfolio.active_islands == []
    assert portfolio.global_best_score is None
    assert analysis.target_island_recent == []
    assert analysis.representative_profiles == []
    assert analogy.target_parent == {}
    assert analogy.source_candidates == []
    assert proposal.candidate_seed == 0
    assert all(
        item.to_text()
        for item in (portfolio, analysis, analogy, proposal)
    )
