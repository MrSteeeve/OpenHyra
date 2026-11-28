"""Offline regression for prior awakening, clone filtering and dual-track evidence."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import re
import tokenize

import pytest

from tasks.lob_world_model.code_dedup import find_duplicate, normalize_source
from tasks.lob_world_model.folding_ledger import FoldingLedger, promotion_decision
from tasks.lob_world_model.prior_cards import _urgency, advance_cooldown, load_cards, select_active_cards


LABELS = re.compile(r"(?<![A-Za-z0-9])(?:w2|w6|thin|thick|hawkes|long)(?![A-Za-z0-9])", re.I)
PARENT = {"val_nll_total": 4.0, "dt_nll": 1.0, "exhausted_replicate_rate": .5,
          "counterfactual_impact_mae": .30, "sign_acf_abs_error": .4, "memory_gap_64": .05}
SOURCE = "def model(x):\n    return x * 0.125 + 1.0\n"


def test_token_similarity_ignores_comments_layout_and_floats() -> None:
    clone = "# renamed comment\ndef model(x):\n\n    return x*2e-3 + .9 # tuning\n"
    assert normalize_source(clone) == normalize_source(SOURCE)
    duplicate = find_duplicate(clone, {"b": SOURCE, "a": SOURCE})
    assert duplicate is not None
    assert (duplicate.cand_id, duplicate.similarity) == ("a", 1.0)
    assert find_duplicate(clone, {"a": SOURCE}, threshold=1.0) is None
    assert find_duplicate("def create():\n    raise RuntimeError('test')\n", {"a": SOURCE}) is None
    assert normalize_source("x=2") != normalize_source("x=3")
    with pytest.raises(tokenize.TokenError):
        normalize_source("def broken(\n")


def test_competition_cards_are_data_regime_blind_and_bounded() -> None:
    cards = load_cards()
    assert [card.card_id for card in cards] == [f"C{i:02}" for i in range(1, 11)]
    for card in cards:
        assert card.citation and card.symptom_key and card.regime_boundary and card.numerical_guardrails
        assert not LABELS.search(card.render())
    assert list(inspect.signature(select_active_cards).parameters) == ["residual_d", "cooldown_map", "max_cards"]
    symptoms = {"val_nll_total": 4.0, "dt_nll": 2.0}
    initial = [c.card_id for c in select_active_cards(symptoms, {})]
    assert initial == ["C06", "C03", "C08"]
    # Even extraneous regime columns cannot influence the D-only scheduler.
    irrelevant = {**symptoms, "midprice_shift_rate": 1.0, "sign_acf_sum_1_32": 10.0, "interarrival_cv2": 99.0}
    assert [c.card_id for c in select_active_cards(irrelevant, {})] == initial
    queue_cards = [c.card_id for c in select_active_cards({"exhausted_replicate_rate": .8}, {})]
    assert queue_cards[:2] == ["C04", "C05"]
    assert select_active_cards({}, {}) == []
    with pytest.raises(ValueError):
        select_active_cards(symptoms, {}, max_cards=4)


def test_two_future_round_cooling_and_rotation() -> None:
    symptoms = {"val_nll_total": 4.0, "dt_nll": 2.0, "exhausted_replicate_rate": .35}
    selected = [c.card_id for c in select_active_cards(symptoms, {})]
    first = selected[0]
    cooldown = advance_cooldown({}, [first], rejected=True)
    assert cooldown[first] == 2
    for expected in (1, 0):
        assert first not in [c.card_id for c in select_active_cards(symptoms, cooldown)]
        cooldown = advance_cooldown(cooldown, [], rejected=False)
        assert cooldown.get(first, 0) == expected
    assert first in [c.card_id for c in select_active_cards(symptoms, cooldown)]


def test_clock_residual_urgency_is_not_diluted_by_other_heads() -> None:
    symptoms = {'dt_nll': 1.6, 'zero_wait_rate_error': .08, 'interarrival_cv2_rel_error': .5,
                'val_nll_total': 4.0, 'exhausted_replicate_rate': .9, 'free_abs_drift_ticks': 100.0}
    inflated = {**symptoms, 'val_nll_total': 400.0, 'level_nll': 396.0}
    assert _urgency(symptoms)['C06'] == _urgency(inflated)['C06']
    assert _urgency(symptoms)['C06'] > _urgency({'dt_nll': 1.6})['C06']
    ids = [c.card_id for c in select_active_cards(inflated, {})]
    assert ids[0] in {'C04', 'C05'} and ids[1] == 'C06' and len(ids) == 3
    cooled = {card: 2 for card in ('C03', 'C06', 'C08')}
    selected = [c.card_id for c in select_active_cards(symptoms, cooled)]
    assert not set(selected).intersection(cooled) and len(selected) == 3


@pytest.mark.parametrize(("deltas", "expected"), [
    ({"val_nll_total": .005, "counterfactual_impact_mae": -.010, "exhausted_replicate_rate": .05}, "PROMOTED"),
    ({"val_nll_total": .006, "counterfactual_impact_mae": -.020, "exhausted_replicate_rate": .0}, "REJECTED"),
    ({"val_nll_total": -.008, "counterfactual_impact_mae": .009, "exhausted_replicate_rate": .0}, "PROMOTED"),
    ({"val_nll_total": .0, "counterfactual_impact_mae": .0, "exhausted_replicate_rate": -.1}, "PROMOTED"),
    ({"val_nll_total": -.008, "counterfactual_impact_mae": .01, "exhausted_replicate_rate": -.1}, "STEPPING_STONE"),
    ({"val_nll_total": .025, "counterfactual_impact_mae": None, "dt_nll": -.010}, "STEPPING_STONE"),
    ({"val_nll_total": .025, "memory_gap_64": -.05}, "STEPPING_STONE"),
    ({"val_nll_total": .01, "sign_acf_abs_error": -.1}, "STEPPING_STONE"),
    ({"val_nll_total": .0251, "dt_nll": -.2}, "REJECTED"),
    ({"val_nll_total": -.01, "counterfactual_impact_mae": None, "exhausted_replicate_rate": -.2}, "STEPPING_STONE"),
    ({"val_nll_total": -.01, "counterfactual_impact_mae": -.1, "exhausted_replicate_rate": .051}, "REJECTED"),
    ({"val_nll_total": -.01, "counterfactual_impact_mae": {"estimate": -.2, "interval": None,
                                                          "paired_days": 4}, "exhausted_replicate_rate": .0}, "REJECTED"),
    ({"val_nll_total": .01, "dt_nll": {"estimate": -.03, "interval": [-.04, -.02],
                                       "paired_days": 5}}, "REJECTED"),
    ({"val_nll_total": .01, "dt_nll": {"estimate": -.03, "interval": [-.04, -.02],
                                       "paired_days": 6}}, "STEPPING_STONE"),
])
def test_symmetric_paired_promotion_thresholds(deltas: dict, expected: str) -> None:
    candidate = dict(PARENT)
    assert promotion_decision(PARENT, candidate, deltas) == expected
    invalid = {**candidate, "val_nll_total": float("nan")}
    assert promotion_decision(PARENT, invalid, deltas) == "REJECTED"


def test_folded_ledger_preserves_original_but_hides_prose_and_labels(tmp_path: Path) -> None:
    ledger = FoldingLedger(tmp_path)
    ledger.append({"cand_id": "B0", "parent_id": "", "source": SOURCE, "outcome": "SEED"})
    sensitive = "def W6_hawkes(x):\n    return x.long()\n"
    probe = {"cand_id": "c1", "parent_id": "B0", "source": sensitive, "outcome": "STEPPING_STONE",
             "prompt": [{"role": "user", "content": "exact original prompt"}],
             "raw_reply": "W6 thin guessed from latent mechanism",
             "prediction": {"regime_hypothesis": "thought about W2"},
             "parameters": 100, "state_scalars": 20,
             "deltas": {"dt_nll": {"estimate": -.03, "interval": [-.04, -.02], "paired_days": 12,
                                   "reason": "W6 debug text"}}}
    ledger.append(probe)
    ledger.append({"cand_id": "c2", "parent_id": "B0", "source": SOURCE, "outcome": "DUPLICATE_REJECTED"})
    ledger.append({"cand_id": "c3", "parent_id": "B0", "source": "bad syntax!", "outcome": "GATE_REJECTED"})
    ledger.append({"cand_id": "c4", "parent_id": "B0", "source": "not parsed", "outcome": "INVALID_PROPOSAL"})
    expected = {"B0", "c1", "c3"}
    assert set(ledger.evaluated_sources()) == expected
    assert (tmp_path / "candidates/c1/candidate.py").read_text() == sensitive
    assert json.loads((tmp_path / "candidates/c1/raw_reply.json").read_text()) == probe["raw_reply"]
    restored = FoldingLedger(tmp_path)
    assert restored.records == ledger.records
    folded = restored.render("B0", ["c1"], max_tokens=400, token_count=lambda text: len(text.split()))
    assert not LABELS.search(folded)
    assert "thought about" not in folded and "exact original prompt" not in folded
    taboos = folded.split('[Branch taboos]')[1].split('[Stepping stones]')[0]
    assert all(cid in taboos for cid in ('c2', 'c3', 'c4'))
    assert "[Stepping stones]\nc1" in folded
    assert "[Requested nodes]\nc1" in folded
    assert "-0.03" in folded
    with pytest.raises(ValueError):
        ledger.append(probe)
    with pytest.raises(ValueError):
        ledger.append({"cand_id": "../bad", "outcome": "SEED"})


def test_failure_reasons_are_single_line_and_redacted(tmp_path: Path) -> None:
    ledger = FoldingLedger(tmp_path)
    ledger.append({'cand_id': 'B0', 'parent_id': '', 'source': SOURCE, 'outcome': 'SEED'})
    raw = ('Worker crash\n  File "/tmp/private/candidate.py", line 2\n'
           'RuntimeError: /tmp/private/candidate.py in env_alpha W6_hawkes api_key=SECRET123 '
           'Bearer fakecredential ' + 'x' * 150 + '\nignored second line')
    first = ledger.append({'cand_id': 'c01', 'parent_id': 'B0', 'source': SOURCE,
                           'outcome': 'GATE_REJECTED', 'error': raw})
    assert first['error'] == raw
    reason = first['gate_error']
    assert reason.startswith('RuntimeError: candidate.py') and len(reason) <= 120 and '\n' not in reason
    assert all(value not in reason for value in ('/tmp/', 'private', 'env_alpha', 'SECRET123', 'fakecredential'))
    assert not LABELS.search(reason)
    for cid, outcome, text in [('c02', 'INVALID_PROPOSAL', 'invalid fenced block'),
                               ('c03', 'NONFINITE_LOSS', 'NonfiniteModelLoss: nonfinite loss'),
                               ('c04', 'DUPLICATE_REJECTED', '')]:
        row = ledger.append({'cand_id': cid, 'parent_id': 'B0', 'source': SOURCE,
                             'outcome': outcome, 'error': text})
        assert row['gate_error'] and 'reason=' in ledger._line(row)
    folded = ledger.render('B0', max_tokens=700, token_count=lambda text: len(text.split()))
    taboos = folded.split('[Branch taboos]')[1].split('[Stepping stones]')[0]
    assert 'reason=' + reason in taboos
    assert all(cid in taboos for cid in ('c02', 'c03', 'c04'))


def test_tight_history_budget_prioritizes_latest_gate_reason(tmp_path: Path) -> None:
    ledger = FoldingLedger(tmp_path)
    ledger.append({'cand_id': 'B0', 'parent_id': '', 'source': SOURCE, 'outcome': 'SEED'})
    ledger.append({'cand_id': 'c01', 'parent_id': 'B0', 'source': SOURCE,
                   'outcome': 'GATE_REJECTED', 'error': 'RuntimeError: old gate failure'})
    latest = ledger.append({'cand_id': 'c02', 'parent_id': 'B0', 'source': SOURCE,
                            'outcome': 'GATE_REJECTED', 'error': "RuntimeError: shape '[4, 10]' is invalid for input of size 80"})
    ledger.append({'cand_id': 'c03', 'parent_id': 'B0', 'source': SOURCE,
                   'outcome': 'STEPPING_STONE', 'deltas': {'dt_nll': -.03}})
    reason = 'reason=' + latest['gate_error']
    # A full lineage row cannot fit, nor the four conventional headings.
    assert len(ledger._line(latest)) > len(reason)
    assert ledger.render('B0', max_tokens=len(reason), token_count=len) == reason
    compact = '[Branch taboos]\nc02 | GATE_REJECTED | ' + reason
    assert ledger.render('B0', max_tokens=len(compact), token_count=len) == compact
    assert 'old gate failure' not in compact


def test_folded_context_remains_bounded_for_32_rounds(tmp_path: Path) -> None:
    ledger = FoldingLedger(tmp_path)
    ledger.append({"cand_id": "B0", "parent_id": "", "source": SOURCE, "outcome": "SEED"})
    token_count = lambda text: len(re.findall(r"\w+|[^\s\w]", text))
    for round_num in range(1, 33):
        outcome = ("STEPPING_STONE" if round_num % 3 == 1 else "DUPLICATE_REJECTED"
                   if round_num % 3 == 2 else "REJECTED")
        ledger.append({"cand_id": f"c{round_num:02}", "parent_id": "B0", "source": SOURCE,
                       "outcome": outcome, "deltas": {"val_nll_total": .012, "dt_nll": -.02},
                       "raw_reply": "ignore hidden reasoning " * 400})
        folded = ledger.render("B0", ["c01", "c02", "c03"], max_tokens=430, token_count=token_count)
        assert token_count(folded) <= 430
        assert "ignore hidden reasoning" not in folded
    assert len(FoldingLedger(tmp_path).records) == 33
