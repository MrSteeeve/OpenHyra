"""Offline regression for the open catalog, clone filtering and dual-track evidence."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tokenize

import pytest

from tasks.lob_world_model.code_dedup import find_duplicate, normalize_source
from tasks.lob_world_model.folding_ledger import FoldingLedger, promotion_decision
from tasks.lob_world_model.prior_cards import load_cards, shuffled_cards


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


def test_catalog_is_complete_open_reference_without_urls() -> None:
    cards = load_cards()
    assert [card.card_id for card in cards] == [f"C{i:02}" for i in range(1, 11)]
    for card in cards:
        assert card.citation and card.mechanism_formula and card.regime_boundary and card.numerical_guardrails
        assert "http" not in card.render() and card.render().startswith(card.card_id)
    assert "Jusselin & Rosenbaum (2018)" in cards[1].render()
    shuffled = shuffled_cards("env_beta", "rep_2", 3)
    assert sorted(card.card_id for card in shuffled) == [card.card_id for card in cards]


OBJECTIVE, VIABILITY = ("PROMOTED", "objective_improvement"), ("PROMOTED", "viability_recovery")
STONE, REJECT = ("STEPPING_STONE", None), ("REJECTED", None)
SPARSE_CF = {"estimate": -.2, "interval": None, "paired_days": 4}


@pytest.mark.parametrize(("deltas", "expected"), [
    ({"val_nll_total": .005, "counterfactual_impact_mae": -.010, "exhausted_replicate_rate": .05}, OBJECTIVE),
    ({"val_nll_total": .006, "counterfactual_impact_mae": -.020, "exhausted_replicate_rate": .0}, REJECT),
    ({"val_nll_total": -.008, "counterfactual_impact_mae": .009, "exhausted_replicate_rate": .0}, OBJECTIVE),
    ({"val_nll_total": .0, "counterfactual_impact_mae": .0, "exhausted_replicate_rate": -.1}, OBJECTIVE),
    ({"val_nll_total": -.008, "counterfactual_impact_mae": .01, "exhausted_replicate_rate": -.1}, STONE),
    ({"val_nll_total": -.001, "counterfactual_impact_mae": {"estimate": .5, "interval": [.1, .9], "paired_days": 12},
      "exhausted_replicate_rate": -.2}, STONE),
    ({"val_nll_total": .025, "counterfactual_impact_mae": None, "dt_nll": -.010}, STONE),
    ({"val_nll_total": .025, "memory_gap_64": -.05}, STONE),
    ({"val_nll_total": .01, "sign_acf_abs_error": -.1}, STONE),
    ({"val_nll_total": .0251, "dt_nll": -.2}, REJECT),
    ({"val_nll_total": -.01, "counterfactual_impact_mae": None, "exhausted_replicate_rate": -.2}, VIABILITY),
    ({"val_nll_total": -.01, "counterfactual_impact_mae": -.1, "exhausted_replicate_rate": .051}, REJECT),
    ({"val_nll_total": -.01, "counterfactual_impact_mae": SPARSE_CF, "exhausted_replicate_rate": .0}, VIABILITY),
    ({"val_nll_total": .0, "counterfactual_impact_mae": SPARSE_CF, "exhausted_replicate_rate": -.1}, VIABILITY),
    ({"val_nll_total": .0, "counterfactual_impact_mae": SPARSE_CF, "exhausted_replicate_rate": -.09}, REJECT),
    ({"val_nll_total": .006, "counterfactual_impact_mae": SPARSE_CF, "exhausted_replicate_rate": -.5}, STONE),
    ({"val_nll_total": .01, "dt_nll": {"estimate": -.03, "interval": [-.04, -.02],
                                       "paired_days": 5}}, REJECT),
    ({"val_nll_total": .01, "dt_nll": {"estimate": -.03, "interval": [-.04, -.02],
                                       "paired_days": 6}}, STONE),
])
def test_two_stage_paired_promotion_thresholds(deltas: dict, expected: tuple) -> None:
    candidate = dict(PARENT)
    assert promotion_decision(PARENT, candidate, deltas) == expected
    invalid = {**candidate, "val_nll_total": float("nan")}
    assert promotion_decision(PARENT, invalid, deltas) == REJECT


def test_folded_ledger_preserves_original_but_hides_prose(tmp_path: Path) -> None:
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
    assert "W6_hawkes" in folded  # ordinary identifiers are objective edit facts
    assert "thought about" not in folded and "guessed" not in folded and "exact original prompt" not in folded
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
    assert all(value not in reason for value in ('/tmp/', 'private', 'SECRET123', 'fakecredential'))
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
