import pytest

from analogy_graph import AnalogyEdge, AnalogyGraph
from mechanism_cards import MechanismCardBuilder, MechanismCardStore
from schemas_v5 import AnalogyHypothesis, AnalogyResult, MechanismCard


def make_card(record_id="sol_1", facts=None, observations=None, inferences=None):
    return MechanismCard(
        record_id=record_id,
        deterministic_facts=facts or {},
        trusted_observations=observations or {},
        llm_inferences=inferences or [],
    )


def make_hypothesis():
    return AnalogyHypothesis(
        id="analogy_1",
        source_record_ids=["source_1", "source_2"],
        target_parent_id="parent_1",
        relation_mapping=[
            {
                "source_role": "normalizer",
                "target_role": "feature_scale",
                "shared_relation": "stabilizes_inputs",
            }
        ],
        non_correspondence=["weights_do_not_transfer"],
        transferable_intervention="add_normalized_feature",
        predicted_effect={
            "metric": "slice_improvement",
            "direction": "positive",
            "minimum_effect": 0.01,
        },
        falsifier="paired_lcb_le_zero",
        matched_control={"same_parent": True},
        status="preregistered",
    )


def make_result(verdict):
    return AnalogyResult(
        analogy_hypothesis_id="analogy_1",
        guided_record_id="guided_1",
        control_record_id="control_1",
        guided_delta=0.05,
        control_delta=0.01,
        transfer_gain=0.04,
        transfer_gain_standard_error=0.01,
        predicted_slice_effect=0.06,
        prediction_direction_correct=verdict == "transfer_supported",
        verdict=verdict,
    )


def test_card_from_manifest():
    card = MechanismCardBuilder.from_bundle_manifest(
        "sol_1",
        {
            "artifact_protocol": "openhyra-policy-spec.v1",
            "entrypoint": "train.py",
            "generation_operator": "analogy_transfer",
        },
    )
    assert card.deterministic_facts == {
        "protocol": "openhyra-policy-spec.v1",
        "entrypoint": "train.py",
        "generation_operator": "analogy_transfer",
    }
    assert card.trusted_observations == {}
    assert card.llm_inferences == []


def test_add_trusted_observations():
    card = make_card(facts={"optimizer": "adam"})
    updated = MechanismCardBuilder.add_trusted_observations(
        card, ["high_volatility"], ["deep_otm"], ["seed_instability"]
    )
    assert updated.trusted_observations["strong_slices"] == ["high_volatility"]
    assert updated.deterministic_facts == card.deterministic_facts
    assert card.trusted_observations == {}


def test_add_llm_inference():
    card = make_card()
    updated = MechanismCardBuilder.add_llm_inference(
        card, "normalization helps", 0.7, ["sol_0", "sol_1"], "annotation_1"
    )
    assert updated.llm_inferences == [
        {
            "claim": "normalization helps",
            "confidence": 0.7,
            "evidence_record_ids": ["sol_0", "sol_1"],
            "annotation_event_id": "annotation_1",
        }
    ]


def test_add_llm_inference_bad_confidence():
    with pytest.raises(ValueError):
        MechanismCardBuilder.add_llm_inference(
            make_card(), "bad", 1.1, [], "annotation_1"
        )


def test_merge_cards():
    inference_a = {
        "claim": "a",
        "confidence": 0.5,
        "evidence_record_ids": ["sol_1"],
        "annotation_event_id": "ann_a",
    }
    inference_b = {
        "claim": "b",
        "confidence": 0.6,
        "evidence_record_ids": ["sol_1"],
        "annotation_event_id": "ann_b",
    }
    base = make_card(
        facts={"protocol": "v1", "optimizer": "adam"},
        observations={"strong_slices": ["a", "b"]},
        inferences=[inference_a],
    )
    update = make_card(
        facts={"optimizer": "sgd"},
        observations={"strong_slices": ["b", "c"], "weak_slices": ["d", "d"]},
        inferences=[inference_b],
    )
    merged = MechanismCardBuilder.merge_cards(base, update)
    assert merged.deterministic_facts == {"protocol": "v1", "optimizer": "sgd"}
    assert merged.trusted_observations == {
        "strong_slices": ["a", "b", "c"],
        "weak_slices": ["d"],
    }
    assert merged.llm_inferences == [inference_a, inference_b]


def test_card_store_save_load(tmp_path):
    store = MechanismCardStore(tmp_path / "cards")
    card = make_card(facts={"protocol": "v1"})
    store.save(card)
    assert store.load("sol_1").to_dict() == card.to_dict()
    assert store.load("missing") is None


def test_card_store_find_by_tag(tmp_path):
    store = MechanismCardStore(tmp_path / "cards")
    store.save(make_card("a", {"tags": ["neural", "fast"]}))
    store.save(make_card("b", {"tags": ["symbolic"]}, {"tags": ["neural"]}))
    store.save(make_card("c", {"tags": ["tree"]}))
    assert [card.record_id for card in store.find_by_tag("tags", "neural")] == [
        "a",
        "b",
    ]


def test_analogy_edge_rejects_unknown_type():
    with pytest.raises(ValueError):
        AnalogyEdge("a", "b", "similar", "evidence_1")


def test_add_hypothesis_creates_edges():
    graph = AnalogyGraph()
    graph.add_hypothesis(make_hypothesis())
    edges = graph.get_edges_by_type("analogy_hypothesized")
    assert {(edge.source_id, edge.target_id) for edge in edges} == {
        ("source_1", "parent_1"),
        ("source_2", "parent_1"),
    }
    assert {edge.evidence_id for edge in edges} == {"analogy_1"}


def test_add_result_supported():
    graph = AnalogyGraph()
    graph.add_result(make_hypothesis(), make_result("transfer_supported"))
    edges = graph.get_edges_by_type("transfer_supported")
    assert {(edge.source_id, edge.target_id) for edge in edges} == {
        ("source_1", "guided_1"),
        ("source_2", "guided_1"),
    }


def test_add_result_refuted():
    graph = AnalogyGraph()
    graph.add_result(make_hypothesis(), make_result("transfer_refuted"))
    assert len(graph.get_edges_by_type("transfer_refuted")) == 2
    assert graph.get_edges_by_type("transfer_supported") == []


def test_get_transfer_history():
    graph = AnalogyGraph()
    graph.add_edge(AnalogyEdge("record", "a", "transfer_supported", "e1"))
    graph.add_edge(AnalogyEdge("b", "record", "transfer_refuted", "e2"))
    graph.add_edge(AnalogyEdge("record", "c", "analogy_hypothesized", "e3"))
    graph.add_edge(AnalogyEdge("record", "d", "behaviorally_near", "e4"))
    history = graph.get_transfer_history("record")
    assert {key: len(value) for key, value in history.items()} == {
        "supported": 1,
        "refuted": 1,
        "hypothesized": 1,
    }


def test_neighbors():
    graph = AnalogyGraph()
    graph.add_edge(AnalogyEdge("record", "a", "behaviorally_near", "e1"))
    graph.add_edge(AnalogyEdge("b", "record", "transfer_supported", "e2"))
    graph.add_edge(AnalogyEdge("record", "c", "composed_from", "e3"))
    assert graph.neighbors("record") == {"a", "b", "c"}
    assert graph.neighbors("record", {"behaviorally_near"}) == {"a"}


def test_graph_persistence(tmp_path):
    graph = AnalogyGraph()
    graph.add_edge(
        AnalogyEdge("a", "b", "structurally_related", "e1", {"score": 0.8})
    )
    graph.add_edge(AnalogyEdge("b", "c", "counterexample_to", "e2"))
    path = tmp_path / "nested" / "graph.json"
    graph.save(path)
    assert AnalogyGraph.load(path).to_dict() == graph.to_dict()
    assert AnalogyGraph.load(tmp_path / "missing.json").edges == []


def test_no_auto_upgrade_behaviorally_near():
    graph = AnalogyGraph()
    edge = AnalogyEdge("a", "b", "behaviorally_near", "e1")
    graph.add_edge(edge)
    graph.add_edge(edge)
    assert len(graph.get_edges_by_type("behaviorally_near")) == 1
    assert graph.get_edges_by_type("transfer_supported") == []
