"""Mechanism retention, executable witnesses and frozen portfolio evaluation."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from structurerdsl.interp import run
from structurerdsl.path import Paths
from structurerdsl.program import Program
from stochopia.adapters.structurer_dsl import DSLContract, DSL_CONTRACT_VERSION

from experiments import run_stochopia_structuring as runner
from tasks.stochopia_structuring.evaluator import PROTOCOL
from tasks.stochopia_structuring.generator import FixtureGenerator, Generation, system_prompt_for
from tasks.stochopia_structuring.mechanism_archive import (
    MechanismArchive, RESEARCH_GOAL, mechanism_profile,
)
from tasks.stochopia_structuring.equity_protocol_v2 import path_pair_diagnostics


def artifact(body, score=1):
    return {"protocol": PROTOCOL, "fixture_score": score, "contract": {
        "version": DSL_CONTRACT_VERSION, "notional": 1e6,
        "program": Program.from_sexpr("(contract (meta (underlyings u) (maturity 3) "
            "(schedules (obs 1 2 3))) " + body + ")").to_json()}}


def probes(artifact):
    terminals = [.6, .8, .9, .95, 1., 1.05, 1.1, 1.2, 1.4]
    histories = [[x ** (i / 3) for i in range(4)] for x in terminals]
    histories += [p for pair in path_pair_diagnostics() for p in pair["paths"]]
    contract = DSLContract.from_payload(artifact["contract"])
    grid = Paths(spot={"u": torch.tensor(histories, dtype=torch.float64)},
                 t_years=torch.arange(4, dtype=torch.float64) / 12, rate=.02)
    payments = run(contract.program, grid, contract.params()).cashflows[:, 1:].tolist()
    return [dict(name=f"public-{i}", spot_ratios=path, cashflows=pay)
            for i, (path, pay) in enumerate(zip(histories, payments))]


def result(artifact, score=1):
    return SimpleNamespace(score=score, metrics={"public_cashflow_probes": probes(artifact)})


def test_leverage_names_and_equivalent_compositions_do_not_add_mechanisms():
    terminal = artifact("(premium (pay (terminal (spot u)) maturity))")
    amplified = artifact("(premium (pay (mul_const 1000 (terminal (spot u))) maturity))")
    reordered = artifact("(premium (both (pay (mul_const 2 (terminal (spot u))) maturity) "
                         "(pay (terminal (spot u)) maturity)))")
    archive = MechanismArchive([])
    for cid, item in zip(("base", "amplified", "reordered"), (terminal, amplified, reordered)):
        item["contract"]["pitch"] = "completely new product"
        archive.add(cid, item, result(item))
    assert len(archive.groups) == 1
    assert archive.groups[0]["members"] == ["base", "amplified", "reordered"]


def test_dormant_condition_is_not_promoted_by_syntax_alone():
    base = artifact("(premium (pay (terminal (spot u)) maturity))")
    dormant = artifact("(premium (both (pay (terminal (spot u)) maturity) "
        "(when (gt (terminal (spot u)) 100) maturity (pay 10 maturity))))")
    assert mechanism_profile(base) != mechanism_profile(dormant)
    archive = MechanismArchive([])
    archive.add("base", base, result(base))
    archive.add("dormant", dormant, result(dormant))
    assert len(archive.groups) == 1


def test_terminal_mechanism_difference_has_executed_witness():
    linear = artifact("(premium (pay (terminal (spot u)) maturity))")
    capped = artifact("(premium (pay (clip (terminal (spot u)) 0 1.1) maturity))")
    archive = MechanismArchive([])
    archive.add("linear", linear, result(linear, 10))
    archive.add("capped", capped, result(capped, .01))
    assert len(archive.groups) == 2
    assert archive.groups[1]["representative"] == "capped"
    assert archive.groups[1]["contrast_witnesses"][0]["normalized_difference"] > 0
    assert archive.parent({})["candidate_id"] == "linear"
    assert archive.parent({})["candidate_id"] == "capped"


def test_parameters_and_commutative_guards_are_one_profile():
    a = artifact("(premium (when (and (gt (terminal (spot u)) 0.9) "
                 "(lt (terminal (spot u)) 1.1)) maturity (pay 0.01 maturity)))")
    b = artifact("(premium (when (and (lt (terminal (spot u)) 1.2) "
                 "(gt (terminal (spot u)) 0.8)) maturity (pay 100 maturity)))")
    assert mechanism_profile(a) == mechanism_profile(b)


def test_executed_tiers_are_retained_but_recognized_as_known_family():
    one = "(when (or (lt (terminal (spot u)) 0.9) (gt (terminal (spot u)) 1.1)) maturity (pay 0.1 maturity))"
    two = "(when (or (lt (terminal (spot u)) 0.8) (gt (terminal (spot u)) 1.2)) maturity (pay 0.2 maturity))"
    single = artifact(f"(funded 1 $issue_price {one})")
    tiered = artifact(f"(funded 1 $issue_price (both {one} {two}))")
    archive = MechanismArchive([{"name": "double-two-way-digital", "profile": mechanism_profile(tiered)}])
    archive.add("single", single, result(single))
    archive.add("tiered", tiered, result(tiered, .01))
    assert len(archive.groups) == 2
    assert archive.groups[1]["matched_references"] == ["double-two-way-digital"]
    assert archive.groups[1]["novelty_status"] == "reference-profile"


def test_payment_time_difference_needs_execution_even_with_constant_amounts():
    maturity = artifact("(funded 1 $issue_price (pay 0 maturity))")
    early = artifact("(funded 1 $issue_price (terminate_if (gt (spot u) 0) obs 0))")
    archive = MechanismArchive([])
    archive.add("maturity", maturity, result(maturity))
    archive.add("early", early, result(early, .01))
    assert len(archive.groups) == 2
    assert archive.groups[1]["contrast_witnesses"][0]["payment_months_differ"]


def test_constraint_failure_and_missing_execution_are_not_discoveries():
    item = artifact("(premium (pay (terminal (spot u)) maturity))")
    archive = MechanismArchive([])
    archive.add("rejected", item, result(item, None))
    archive.add("no-evidence", item, SimpleNamespace(score=100, metrics={}))
    assert archive.groups == []
    assert archive.unresolved == [{"candidate_id": "no-evidence", "reason": "no executed public cashflow evidence"}]


class SequenceGenerator(FixtureGenerator):
    def __init__(self):
        self.requests = []
        self.artifacts = [
            artifact("(premium (pay (terminal (spot u)) maturity))", 10),
            artifact("(premium (pay (avg_arith (spot u) obs) maturity))", .01),
            artifact("(premium (pay (mul_const 2 (avg_arith (spot u) obs)) maturity))", .02),
            artifact("(premium (pay (mul_const 1000 (terminal (spot u))) maturity))", 100),
        ]

    def generate(self, request):
        self.requests.append(deepcopy(request))
        return Generation(deepcopy(self.artifacts[(len(self.requests)-1) % 4]), usage={"api_calls": 0})


@pytest.fixture
def evaluated_discovery(monkeypatch):
    evaluated = []

    def evaluate(item, request):
        evaluated.append(deepcopy(request))
        score = item.get("fixture_score", 1)
        public = probes(item)
        metrics = {"status": "ok" if score is not None else "rejected", "cost": {},
                   "public_cashflow_probes": public, "accepted_rate": int(score is not None),
                   "mean_client_return": score, "behavior_vector": [sum(p["cashflows"]) for p in public],
                   "demand_checks": {"passed": score is not None}, "constraint_violation_sum": 0.01,
                   "episodes": [{"trajectory": "private-evaluation-path"}],
                   "scored_paths": 12, "representative_replay_count": 3}
        return score, metrics, item, {}

    monkeypatch.setattr(runner, "evaluate_submission", evaluate)
    monkeypatch.setattr(runner, "public_condition", lambda *args: {"public": "same conditions"})
    return evaluated


def run_search(path, generator, **kwargs):
    return runner.run_search(path, generator, paths=128, demand_id="two-way-risk20", demand_protocol="v2", **kwargs)


def test_low_return_mechanism_survives_receives_parent_slot_and_holdout(tmp_path, evaluated_discovery):
    generator = SequenceGenerator()
    summary = run_search(tmp_path / "discovery", generator)
    assert summary["research_goal"] == summary["objective"] == RESEARCH_GOAL
    assert summary["selected"] is None and summary["selection_policy"] == "mechanism-portfolio"
    assert summary["economic_reference"]["candidate_id"] == "r1-s1"
    assert "r1-s0" in summary["selected_mechanisms"]  # Asian mechanism has much lower return.
    assert generator.requests[3]["parent_id"] == "r0-s1"
    assert generator.requests[2]["feedback"] == generator.requests[3]["feedback"]
    assert "private-evaluation-path" not in json.dumps(generator.requests)
    assert all(r["objective"] == RESEARCH_GOAL and r["economic_objective"] == "equity-demand-expected-return.v2"
               for r in generator.requests)
    held = [r for r in evaluated_discovery if r["split"] == "held_out"]
    assert len(held) == summary["primary_measures"]["feasible_mechanism_profiles"] == 3
    assert summary["held_out_portfolio_replay_count"] == 9
    archive = json.loads((tmp_path / "discovery/mechanism_archive.json").read_text())
    assert archive["model_group"] == "offline-control"
    assert "held_out" not in json.dumps(archive)
    assert all("representative_artifact" in row for row in archive["mechanisms"])
    config = json.loads((tmp_path / "discovery/run_config.json").read_text())
    assert config["objective"] == RESEARCH_GOAL
    assert config["economic_objective"] == "equity-demand-expected-return.v2"
    selected = json.loads((tmp_path / "discovery/selected.json").read_text())
    assert len(selected["mechanisms"]) == 3


def test_independent_has_same_calls_and_goal_but_no_archive_or_parent(tmp_path, evaluated_discovery):
    generator = SequenceGenerator()
    summary = run_search(tmp_path / "independent", generator, search_mode="independent")
    assert summary["proposal_calls"] == 4
    assert all(r["research_goal"] == RESEARCH_GOAL for r in generator.requests)
    assert all(r["feedback"] == [] and "parent" not in r and "mechanism_archive" not in r
               for r in generator.requests)
    assert len(summary["selected_mechanisms"]) == 3


def test_restart_rechecks_development_and_never_reads_old_holdout(tmp_path, evaluated_discovery):
    run_search(tmp_path / "previous", SequenceGenerator())
    source = tmp_path / "previous/mechanism_archive.json"
    data = json.loads(source.read_text())
    data["held_out_secret"] = "DO-NOT-FEED-THIS"
    source.write_text(json.dumps(data))
    evaluated_discovery.clear()
    generator = SequenceGenerator()
    summary = run_search(tmp_path / "restart", generator, initial_archive=source, rounds=1, per_round=1)
    assert summary["proposal_calls"] == 1
    assert len([r for r in evaluated_discovery if r["split"] == "development"]) == 5  # seed, 3 imports, proposal
    assert "DO-NOT-FEED-THIS" not in json.dumps(generator.requests)
    assert any(row["candidate_id"].startswith("initial-") for row in generator.requests[0]["feedback"])
    assert generator.requests[0]["parent_id"] == "initial-1"  # Continue opportunity counts, not always seed.
    data["model_group"] = "different-model"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="same model"):
        run_search(tmp_path / "mixed", SequenceGenerator(), initial_archive=source)


def test_all_infeasible_search_has_no_selected_mechanism_or_holdout(tmp_path, evaluated_discovery, monkeypatch):
    original = runner.evaluate_submission

    def reject(item, request):
        _, metrics, normalized, evidence = original(item, request)
        return None, {**metrics, "status": "rejected"}, normalized, evidence

    monkeypatch.setattr(runner, "evaluate_submission", reject)
    summary = run_search(tmp_path / "rejected", SequenceGenerator())
    assert summary["selected_mechanisms"] == []
    assert summary["status"] == "no_feasible_mechanism"
    assert not any(row["split"] == "held_out" for row in evaluated_discovery)


def test_holdout_failure_is_reported_without_changing_frozen_portfolio(tmp_path, evaluated_discovery, monkeypatch):
    original = runner.evaluate_submission

    def fail_asian_holdout(item, request):
        score, metrics, normalized, evidence = original(item, request)
        if request["split"] == "held_out" and "avg_arith" in json.dumps(item):
            score, metrics = None, {**metrics, "status": "rejected"}
        return score, metrics, normalized, evidence

    monkeypatch.setattr(runner, "evaluate_submission", fail_asian_holdout)
    summary = run_search(tmp_path / "held-failure", SequenceGenerator())
    assert "r1-s0" in summary["selected_mechanisms"]
    assert summary["primary_measures"]["feasible_mechanism_profiles"] == 3
    assert summary["primary_measures"]["held_out_feasible_mechanisms"] == 2
    selected = json.loads((tmp_path / "held-failure/selected.json").read_text())
    assert any(row["candidate"]["candidate_id"] == "r1-s0" for row in selected["mechanisms"])


def test_discovery_prompt_does_not_require_beating_other_mechanisms():
    request = {"demand": {"schema": "stochopia-equity-demand.v2"}, "research_goal": RESEARCH_GOAL,
               "design_brief": {"track": "mechanism-explore"}}
    prompt = system_prompt_for(request)
    assert "PRIMARY research goal" in prompt
    assert "Lower-return mechanisms remain" in prompt
    assert "Terminal-only designs" in prompt
    assert "mean ranks" not in prompt
    assert "return ranking" not in prompt
