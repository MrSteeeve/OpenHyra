"""Same-language standard controls, fixed budgets and constraint-gated ranking."""
import json

import pytest
import torch

from stochopia.adapters.structurer_dsl import DSLContract
from structurerdsl.interp import run
from structurerdsl.path import Paths
from algorithm_discovery import EvaluationResult
from tasks.stochopia_structuring.standard_baselines import (
    FAMILIES, baseline_candidates, optimize_standard_baselines,
)


@pytest.mark.parametrize("demand_id", ["upside-protected", "two-way-risk20"])
def test_grid_balances_families_and_all_contracts_execute(demand_id):
    candidates = baseline_candidates(demand_id)
    assert len(candidates) == 24
    assert candidates == baseline_candidates(demand_id)
    assert [c["family"] for c in candidates[:4]] == list(FAMILIES)
    assert all(sum(c["family"] == f for c in candidates) == 6 for f in FAMILIES)
    serialized_programs = {json.dumps(c["artifact"]["contract"]["program"], sort_keys=True) for c in candidates}
    assert len(serialized_programs) == 24
    paths = Paths(spot={"u": torch.tensor([[1., 1.03, 1.06, 1.10], [1., .97, .94, .90]], dtype=torch.float64)},
                  t_years=torch.arange(4, dtype=torch.float64) / 12, rate=.02)
    for candidate in candidates:
        contract = DSLContract.from_payload(candidate["artifact"]["contract"])
        result = run(contract.program, paths, contract.params())
        assert result.cashflows.shape == (2, 4)
        assert bool((result.cashflows[:, -1] >= 1.).all())
    assert json.loads(json.dumps(candidates)) == candidates


def test_callback_uses_identical_constraints_and_rejected_high_scores_never_win():
    calls = []

    def evaluate(artifact):
        calls.append(artifact)
        number = len(calls)
        score = 1000. if number == 3 else number / 100.
        metrics = {"status": "rejected" if number == 3 else "ok",
                   "demand_checks": {"passed": number != 3}, "objective": "test-objective"}
        return score, metrics, artifact, {"split": "development"}

    result = optimize_standard_baselines("two-way-risk20", evaluate, {"max_evaluations": 8})
    assert len(calls) == result["evaluations"] == result["budget"] == 8
    assert result["best"]["score"] == .08
    assert not result["attempts"][2]["eligible"]
    assert result["attempts"][2]["constraints"] == {"passed": False}
    assert all(best is not None for best in result["best_by_family"].values())
    assert not result["feed_to_generator"]
    assert "not global optimum" in result["scope"]


def test_no_eligible_candidates_and_dict_callback_are_explicit():
    result = optimize_standard_baselines("upside-protected", lambda _: {
        "score": None, "metrics": {"status": "rejected", "demand_checks": {"passed": False}}},
        {"max_evaluations": 4})
    assert result["best"] is None
    assert all(best is None for best in result["best_by_family"].values())
    assert result["evaluations"] == 4


def test_actual_harness_evaluation_result_without_feedback():
    def evaluate(artifact):
        return EvaluationResult(candidate_id="standard", status="ok", score=.02,
                                metrics={"demand_checks": {"passed": True}},
                                feedback=None)

    result = optimize_standard_baselines("two-way-protected", evaluate, {"max_evaluations": 4})
    assert result["evaluations"] == 4
    assert result["best"]["score"] == .02
    assert all(row["eligible"] for row in result["attempts"])
    assert all(best is not None for best in result["best_by_family"].values())


def test_harness_rejection_status_takes_precedence_over_a_non_null_score():
    result = optimize_standard_baselines("two-way-protected", lambda artifact:
        EvaluationResult(candidate_id="standard", status="rejected", score=10., metrics={}, feedback=None),
        {"max_evaluations": 4})
    assert result["best"] is None
    assert all(not row["eligible"] for row in result["attempts"])
