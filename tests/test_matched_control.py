from types import SimpleNamespace

import pytest

import matched_control
from matched_control import ControlPair, ControlPairStore, MatchedControlBuilder
from schemas_v5 import AnalogyHypothesis, AnalogyResult


@pytest.fixture
def hypothesis() -> AnalogyHypothesis:
    hypothesis = object.__new__(AnalogyHypothesis)
    hypothesis.id = "hypothesis-1"
    hypothesis.source_record_ids = ["source-1", "source-2"]
    hypothesis.target_parent_id = "parent-1"
    hypothesis.transfer_description = "Apply a low-rank residual correction."
    hypothesis.operators = ["residual", "low_rank"]
    hypothesis.status = "preregistered"
    return hypothesis


@pytest.fixture(autouse=True)
def compatible_analogy_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def result_factory(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(matched_control, "AnalogyResult", result_factory)


def test_build_pair_guided_has_transfer(hypothesis: AnalogyHypothesis) -> None:
    pair = MatchedControlBuilder.build_pair(hypothesis, "parent-1", 123)

    assert hypothesis.transfer_description in pair.guided_prompt_suffix


def test_build_pair_control_is_neutral(hypothesis: AnalogyHypothesis) -> None:
    pair = MatchedControlBuilder.build_pair(hypothesis, "parent-1", 123)

    assert hypothesis.transfer_description not in pair.control_prompt_suffix
    assert all(
        source_id not in pair.control_prompt_suffix
        for source_id in hypothesis.source_record_ids
    )


def test_build_pair_shared_seed(hypothesis: AnalogyHypothesis) -> None:
    pair = MatchedControlBuilder.build_pair(hypothesis, "parent-1", 123)

    assert pair.shared_seed == 123
    assert pair.shared_parent_id == "parent-1"


def test_evaluate_supported() -> None:
    pair = ControlPair("hypothesis-1", "guided", "control", "parent-1", 123)
    pair.guided_score = 1.05
    pair.control_score = 1.0

    result = MatchedControlBuilder.evaluate_pair(pair, direction="max")

    assert result.verdict == "transfer_supported"
    assert result.effect_size == pytest.approx(0.05)


def test_evaluate_refuted() -> None:
    pair = ControlPair("hypothesis-1", "guided", "control", "parent-1", 123)
    pair.guided_score = 0.99
    pair.control_score = 1.0

    result = MatchedControlBuilder.evaluate_pair(pair, direction="max")

    assert result.verdict == "transfer_refuted"


def test_evaluate_min_direction() -> None:
    pair = ControlPair("hypothesis-1", "guided", "control", "parent-1", 123)
    pair.guided_score = 0.95
    pair.control_score = 1.0

    result = MatchedControlBuilder.evaluate_pair(pair, direction="min")

    assert result.verdict == "transfer_supported"


def test_evaluate_missing_scores() -> None:
    pair = ControlPair("hypothesis-1", "guided", "control", "parent-1", 123)
    pair.guided_score = None
    pair.control_score = 1.0

    result = MatchedControlBuilder.evaluate_pair(pair)

    assert result.verdict == "execution_failed"


def test_store_round_trip(tmp_path) -> None:
    store = ControlPairStore(tmp_path / "control_pairs.jsonl")
    first = ControlPair("hypothesis-1", "guided-1", "control-1", "parent-1", 1)
    second = ControlPair("hypothesis-2", "guided-2", "control-2", "parent-2", 2)

    store.save(first)
    store.save(second)

    loaded = store.load_all()
    assert [pair.to_dict() for pair in loaded] == [first.to_dict(), second.to_dict()]
    assert [pair.to_dict() for pair in store.find_by_hypothesis("hypothesis-1")] == [
        first.to_dict()
    ]
    assert issubclass(AnalogyResult, object)
