"""Search controls: independent proposals, development feedback, and parents."""
import copy
import json
import sys
from types import SimpleNamespace

import pytest

from experiments import run_stochopia_structuring as runner
from tasks.stochopia_structuring.generator import FixtureGenerator, Generation
from tasks.stochopia_structuring.search_feedback import choose_parent, behavior_diagnostics


class RecordingGenerator(FixtureGenerator):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(copy.deepcopy(request))
        return Generation({"proposal_number": len(self.requests)}, usage={"api_calls": 0})


@pytest.fixture
def evaluated_search(monkeypatch):
    evaluated = []

    def evaluate(artifact, request):
        evaluated.append(copy.deepcopy(request))
        number = artifact.get("proposal_number", 0)
        score = None if number == 1 else float(number + 1)
        metrics = {"status": "rejected" if score is None else "ok", "cost": {"evaluation_seconds": 0.1},
                   "accepted_rate": 0 if score is None else 1, "mean_client_return": score,
                   "behavior_vector": [float(number), float(-number)],
                   "episodes": [{"trajectory": "local-private-path"}],
                   "demand_checks": {"passed": score is not None,
                                     "checks": [{"passed": score is not None, "margin": -0.01 if score is None else 0.02}]}}
        return score, metrics, artifact, {}

    monkeypatch.setattr(runner, "evaluate_submission", evaluate)
    monkeypatch.setattr(runner, "public_condition", lambda *args: {"public": "same conditions"})
    return evaluated


def test_independent_has_equal_calls_and_no_previous_experience(tmp_path, evaluated_search):
    feedback, independent = RecordingGenerator(), RecordingGenerator()
    result = runner.run_search(tmp_path / "feedback", feedback, paths=128)
    control = runner.run_search(tmp_path / "independent", independent, paths=128, search_mode="independent")
    assert result["proposal_calls"] == control["proposal_calls"] == 4
    assert control["api_calls"] == 0
    assert control["feedback_scope"] == []
    assert all(r["feedback"] == [] and "parent" not in r and "parent_id" not in r
               for r in independent.requests)
    assert all(r["conditions"] == feedback.requests[0]["conditions"] for r in independent.requests)
    rows = [json.loads(line) for line in (tmp_path / "independent" / "experience.jsonl").read_text().splitlines()]
    assert len(rows) == 5  # Recording history does not put it into the control prompt.
    assert all(row["parent_ids"] == [] for row in rows)
    assert [r["split"] for r in evaluated_search].count("held_out") == 2


def test_feedback_keeps_failed_margins_only_after_cohort(tmp_path, evaluated_search):
    generator = RecordingGenerator()
    summary = runner.run_search(tmp_path / "feedback", generator, paths=128)
    first, sibling, second, second_sibling = generator.requests
    assert first["feedback"] == sibling["feedback"]
    assert second["feedback"] == second_sibling["feedback"]
    assert len(second["feedback"]) == 3
    failed = next(row for row in second["feedback"] if row["candidate_id"] == "r0-s0")
    assert failed["result"]["score"] is None
    assert failed["result"]["metrics"]["demand_checks"]["checks"][0]["margin"] == -0.01
    assert all(row["result"]["split"] == "development" for r in generator.requests for row in r["feedback"])
    assert "local-private-path" not in json.dumps(generator.requests)
    assert [row["development_score"] for row in summary["best_by_round"]] == [1, 3, 5]
    assert summary["behavior_diversity"]["used_in_score"] is False


def test_diverse_parent_is_feasible_and_never_changes_winner(tmp_path, evaluated_search):
    generator = RecordingGenerator()
    summary = runner.run_search(tmp_path / "diverse", generator, paths=128,
                                parent_policy="alternating-diverse")
    assert generator.requests[0]["parent_id"] == "seed"
    # r0-s1 has score 3; seed is the most distant eligible alternative.
    assert generator.requests[2]["parent_id"] == "seed"
    assert summary["parent_selection"][1]["selection"] == "behavior-distant-feasible"
    assert summary["selected"] == "r1-s1" and summary["development_score"] == 5


def test_behavior_diagnostics_excludes_invalid_candidates():
    result = lambda score, vector: SimpleNamespace(score=score, metrics={"behavior_vector": vector})
    results = {"best": result(2, [1, 1]), "other": result(1, [0, 0]), "invalid": result(None, [100, 100])}
    assert choose_parent(results, 1, "alternating-diverse")["candidate_id"] == "other"
    diagnostics = behavior_diagnostics(results)
    assert diagnostics["eligible_vectors"] == 2
    assert diagnostics["max_pairwise_rms_return_distance"] == 1
    assert choose_parent({"best": results["best"]}, 1, "alternating-diverse")["candidate_id"] == "best"


def test_structure_track_allocation_preserves_selection_and_objective(tmp_path, evaluated_search):
    generator = RecordingGenerator()
    summary = runner.run_search(tmp_path / "structural", generator, paths=128,
                                proposal_strategy="balanced-structure")
    assert [r["design_brief"]["track"] for r in generator.requests] == [
        "refine", "structure-exploration", "refine", "structure-exploration"]
    assert "path summaries" in generator.requests[1]["design_brief"]["mechanism_focus"]
    assert "automatic termination" in generator.requests[3]["design_brief"]["mechanism_focus"]
    assert len({r["objective"] for r in generator.requests}) == 1
    assert summary["selected"] == "r1-s1" and summary["development_score"] == 5
    assert summary["behavior_diversity"]["used_in_score"] is False


def test_infeasible_parent_prefers_constraint_margin_to_parse_failure():
    results = {"parse-failure": SimpleNamespace(score=None, metrics={"demand_checks": None,
                                                                    "constraint_violation_sum": None}),
               "near-feasible": SimpleNamespace(score=None, metrics={"constraint_violation_sum": .01}),
               "far-feasible": SimpleNamespace(score=None, metrics={"constraint_violation_sum": .2})}
    assert choose_parent(results, 0, "incumbent", allow_infeasible=True)["candidate_id"] == "near-feasible"


def test_behavior_roundoff_does_not_create_diversity_or_new_parent():
    results = {"best": SimpleNamespace(score=2, metrics={"behavior_vector": [.1, .2]}),
               "equivalent": SimpleNamespace(score=1, metrics={"behavior_vector": [.1 + 1e-15, .2 - 1e-15]})}
    diagnostics = behavior_diagnostics(results)
    assert diagnostics["distinct_vectors"] == 1
    assert diagnostics["scope"] == "numeric probe diversity, not product count"
    assert choose_parent(results, 1, "alternating-diverse")["candidate_id"] == "best"


def test_v2_passes_config_only_to_evaluator(tmp_path, evaluated_search):
    generator = RecordingGenerator()
    summary = runner.run_search(tmp_path / "v2", generator, paths=128,
                                demand_id="upside-protected", demand_protocol="v2",
                                research_goal="return",
                                eval_config={"score_mode": "mean"})
    assert summary["objective"] == "equity-demand-expected-return.v2"
    assert all(r["demand_protocol"] == "v2" and "eval_config" in r for r in evaluated_search)
    assert all("held_out_seed" not in json.dumps(r) for r in generator.requests)
    config = json.loads((tmp_path / "v2" / "run_config.json").read_text())
    assert config["evaluation_config"]["score_mode"] == "mean"
    assert summary["score_evaluation"] == "batch cashflow execution on P scenarios"


def test_v2_failed_search_preserves_diagnostics_without_holdout(tmp_path, evaluated_search, monkeypatch):
    def reject(artifact, request):
        evaluated_search.append(copy.deepcopy(request))
        number = artifact.get("proposal_number", 0)
        metrics = {"status": "rejected", "cost": {}, "accepted_rate": 0,
                   "constraint_violation_sum": (10 - number) / 10,
                   "demand_checks": {"passed": False, "checks": [{"margin": -.01}]},
                   "rejection_stage": "demand_constraints", "behavior_vector": [number],
                   "episodes": [{"trajectory": "secret-path", "P_realization": [4, 5, 6]}],
                   "scenario_probabilities": {"private-scenario": 1},
                   "innovation": {"path_sensitive": False}}
        return None, metrics, artifact, {}

    monkeypatch.setattr(runner, "evaluate_submission", reject)
    generator = RecordingGenerator()
    summary = runner.run_search(tmp_path / "failed", generator, paths=128,
                                demand_id="upside-protected", demand_protocol="v2", research_goal="return")
    assert len(generator.requests) == summary["proposal_calls"] == 4
    assert summary["selected"] is None and summary["status"] == "no_feasible_candidate"
    assert summary["development_eligible_candidates"] == 0
    assert all(row["split"] != "held_out" for row in evaluated_search)
    assert generator.requests[2]["parent_id"] == "r0-s1"
    feedback = generator.requests[2]["feedback"]
    assert all(row["result"]["score"] is None for row in feedback)
    assert "secret-path" not in json.dumps(feedback) and "private-scenario" not in json.dumps(feedback)
    assert feedback[0]["result"]["metrics"]["innovation"] == {"path_sensitive": False}
    held = json.loads((tmp_path / "failed" / "held_out.json").read_text())
    assert held["evidence"]["status"] == "not_performed"


@pytest.mark.parametrize("arguments, expected", [([], "v2"), (["--demand", "pipeline"], "v1"),
                                                (["--demand-protocol", "v1"], "v1")])
def test_cli_uses_new_equity_protocol_and_preserves_explicit_legacy(tmp_path, monkeypatch, arguments, expected):
    calls = []
    monkeypatch.setattr(runner, "run_search", lambda output, generator, **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["run_stochopia_structuring.py", "--output", str(tmp_path), *arguments])
    runner.main()
    assert calls[0]["demand_protocol"] == expected
