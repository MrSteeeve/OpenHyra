"""API or offline DSL search using OpenHyra's existing discovery loop and EB."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithm_discovery import AlgorithmDiscoveryLoop, AlgorithmSpec, DiscoveryLedger, EvaluationResult
from eb import ExperienceBank
from feedback import FeedbackPacket
from tasks.stochopia_structuring.evaluator import PROTOCOL, STOCHOPIA, TOPICS, evaluate_submission, public_condition, task_suite
from tasks.stochopia_structuring.generator import APIContractGenerator, FixtureGenerator, EquityFixtureGenerator
from tasks.stochopia_structuring.equity_demands import DEMAND_IDS, DEMAND_OBJECTIVE, demand_spec, equity_fixture
from tasks.stochopia_structuring.api_config import DEFAULT_API_MODELS, api_generator
from tasks.stochopia_structuring.search_feedback import choose_parent, behavior_diagnostics, proposal_brief
from tasks.stochopia_structuring.mechanism_archive import (
    MechanismArchive, RESEARCH_GOAL, discovery_brief, mechanism_profile,
)
from examples.dsl_roundtrip import fixture_payload
import torch


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append_jsonl(path, payload):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def generation_feedback(result, demand_protocol="v1"):
    """Expose evaluation feedback without machine paths or elapsed time."""
    metrics = json.loads(json.dumps(result.metrics))
    if demand_protocol == "v2":
        # Distribution parameters are public; realized batch scenarios are not
        # prompt data. Expose aggregate feedback and fixed behavior probes only.
        keys = {"status", "error", "rejection_stage", "demand_checks", "objective", "score_mode",
                "distribution_means", "distribution_standard_errors", "distribution_statistics",
                "pricing_audit", "behavior_vector", "path_pair_diagnostics",
                "path_dependence_diagnostics", "structural_diagnostics", "constraint_violation_sum",
                "expected_net_return", "worst_distribution_return", "innovation",
                "distribution_return_std", "distribution_replication_gap", "public_cashflow_probes"}
        metrics = {name: value for name, value in metrics.items() if name in keys}
    metrics.get("cost", {}).pop("evaluation_seconds", None)
    for episode in metrics.get("episodes", []):
        episode.pop("trajectory", None)
    return {"status": result.status, "score": result.score,
            "split": result.split, "metrics": metrics}


def run_search(output, generator, *, rounds=2, per_round=2, paths=4096,
               pricing_seed=42, objective=None, demand_id=None, demand_protocol="v1",
               eval_config=None, search_mode="feedback", parent_policy="incumbent", proposal_strategy="default",
               research_goal=None, initial_archive=None):
    if demand_protocol not in {"v1", "v2"}:
        raise ValueError("demand_protocol must be v1 or v2")
    if search_mode not in {"feedback", "independent"}:
        raise ValueError("search_mode must be feedback or independent")
    if parent_policy not in {"incumbent", "alternating-diverse"}:
        raise ValueError("unsupported parent_policy")
    if proposal_strategy not in {"default", "balanced-structure"}:
        raise ValueError("unsupported proposal_strategy")
    if demand_protocol == "v1" and eval_config:
        raise ValueError("eval_config is for demand_protocol v2")
    if demand_protocol == "v2" and demand_id is None:
        raise ValueError("demand_protocol v2 requires an equity demand")
    research_goal = research_goal or ("discovery" if demand_protocol == "v2" else "return")
    if research_goal not in {"discovery", "return"}:
        raise ValueError("research_goal must be discovery or return")
    discovery = research_goal == "discovery"
    if discovery and demand_protocol != "v2":
        raise ValueError("mechanism discovery uses the v2 equity evaluation")
    if initial_archive and not discovery:
        raise ValueError("initial_archive requires mechanism discovery")
    spec = demand_spec(demand_id) if demand_id is not None else None
    if demand_protocol == "v2":
        from tasks.stochopia_structuring.equity_protocol_v2 import protocol_config, demand_spec_v2
        eval_config = protocol_config(eval_config)
        spec = demand_spec_v2(demand_id, eval_config)
        expected_objective = ("equity-demand-robust-return.v2" if eval_config["score_mode"] == "worst_distribution"
                              else "equity-demand-expected-return.v2")
    else:
        expected_objective = DEMAND_OBJECTIVE if spec else "accepted-client-return.v1"
    public_spec = json.loads(json.dumps(spec)) if spec else None
    if demand_protocol == "v2":
        public_spec["protocol_config"]["split_seeds"].pop("held_out", None)
        public_spec["p_distribution"]["split_seeds"].pop("held_out", None)
    objective = objective or expected_objective
    if spec and objective != expected_objective:
        raise ValueError("equity demand search requires its fixed expected-return objective")
    if rounds < 1 or per_round < 1 or paths < 2:
        raise ValueError("rounds/per_round must be positive; paths must be >=2")
    if isinstance(generator, APIContractGenerator) and rounds * per_round > generator.max_calls:
        raise ValueError("proposal count exceeds configured API call budget")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = {"protocol": PROTOCOL, "generator": generator.metadata(), "rounds": rounds,
              "candidates_per_round": per_round, "paths": paths, "pricing_seed": pricing_seed,
              "objective": objective, "python": sys.version, "platform": platform.platform(),
              "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
              "candidate_mode": "fixed-dsl-artifact", "feedback_splits": ["public", "development"],
              "demand": spec,
              "condition_collection_cost": {"episodes": 1 if spec else 2,
                  "client_queries": (1 if spec else 2) * len(TOPICS), "quotes": 0},
              "held_out_policy": "evaluate selected artifact once after final selection"}
    config.update(demand_protocol=demand_protocol, evaluation_config=eval_config,
                  search_mode=search_mode, parent_policy=parent_policy if search_mode == "feedback" else None,
                  proposal_strategy=proposal_strategy,
                  planned_proposal_calls=rounds * per_round,
                  score_owner="task evaluator; behavior diversity is not a reward")
    if search_mode == "independent":
        config["feedback_splits"] = []
    group_id = config["generator"].get("group_id", "offline-control")
    config["model_group"] = group_id
    config.update(research_goal=RESEARCH_GOAL if discovery else "economic-return",
                  objective=RESEARCH_GOAL if discovery else objective,
                  economic_objective=objective,
                  selection_policy="mechanism-portfolio" if discovery else "economic-incumbent",
                  initial_archive=str(initial_archive) if initial_archive else None)
    if discovery:
        config.update(parent_policy="least-explored-mechanism" if search_mode == "feedback" else None,
                      proposal_strategy="discovery-tracks",
                      held_out_policy="freeze and evaluate every mechanism representative; no reranking")
    config["dependencies"] = {name: version(name) for name in ("numpy", "structurerdsl", "httpx", "python-dotenv")}
    config["repositories"] = {}
    for repo in (ROOT, STOCHOPIA, ROOT.parent / "StructurerDSL"):
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True))
        config["repositories"][repo.name] = {"head": head, "dirty": dirty}
    write_json(output / "run_config.json", config)
    development_tasks = task_suite("public" if demand_protocol == "v2" else "development", demand_id)
    conditions = [public_condition(task, paths, pricing_seed)
                  for _, task in (development_tasks[:1] if spec else development_tasks)]
    write_json(output / "public_conditions.json", conditions)
    loop = AlgorithmDiscoveryLoop(ledger=DiscoveryLedger(output / "discovery.jsonl"))
    bank = ExperienceBank(output / "experience_bank", direction="max")
    candidates, records, results, proposals = {}, {}, {}, {}
    best_by_round, parent_history = [], []
    seed_contract = equity_fixture(spec["family"]) if spec else fixture_payload("spread")
    seed = AlgorithmSpec("seed", "dsl", implementation={"protocol": PROTOCOL, "contract": seed_contract},
                         operator="handwritten", mechanism_id="dsl")
    archive = None
    if discovery:
        from tasks.stochopia_structuring.standard_baselines import baseline_candidates
        references = [{"name": "handwritten-seed", "profile": mechanism_profile(seed.implementation)}]
        for row in baseline_candidates(demand_id):
            reference = {"name": row["family"], "profile": mechanism_profile(row["artifact"])}
            if reference not in references:
                references.append(reference)
        archive = MechanismArchive(references)
        write_json(output / "reference_mechanisms.json", references)

    def evaluation_request(split, folder):
        request = {"split": split, "paths": paths, "pricing_seed": pricing_seed,
                   "objective": objective, "demand_id": demand_id, "output": str(folder)}
        if demand_protocol == "v2":
            request.update(demand_protocol="v2", eval_config=eval_config)
        return request

    def evaluate(candidate):
        folder = output / "candidates" / candidate.candidate_id
        folder.mkdir(parents=True)
        write_json(folder / "contract.json", candidate.implementation)
        score, metrics, normalized, evidence = evaluate_submission(candidate.implementation,
            evaluation_request("development", folder / "development"))
        generation = proposals.get(candidate.candidate_id, {})
        cost = {**metrics["cost"], "generation": generation.get("generation", {}).get("usage", {}),
                "generation_seconds": generation.get("generation", {}).get("seconds", 0.)}
        feedback = FeedbackPacket(packet_id=f"feedback-{candidate.candidate_id}",
            candidate_id=candidate.candidate_id, mechanism_id="dsl",
            observed={"value": score} if score is not None else {},
            evidence={"metrics": metrics}, data={"split": "development"})
        result = EvaluationResult(candidate.candidate_id, metrics["status"], score,
            metrics=metrics, feedback=feedback, split="development", seed=pricing_seed, cost=cost)
        write_json(folder / "result.json", result.to_dict())
        parent = records.get(candidate.parent_ids[0], {}).get("id") if candidate.parent_ids else None
        records[candidate.candidate_id] = bank.commit(folder, score, result.status,
            candidate.operator, parent, metrics.get("error", ""), metrics=metrics,
            metadata={"candidate_id": candidate.candidate_id, "parent_ids": list(candidate.parent_ids),
                      "operator": candidate.operator, "cost": cost, "split": "development",
                      "model_group": group_id})
        results[candidate.candidate_id] = result
        if discovery:
            archive.add(candidate.candidate_id, candidate.implementation, result)
        append_jsonl(output / "experience.jsonl", {"schema": "stochopia-generation-experience.v1",
            "model_group": group_id, "candidate_id": candidate.candidate_id, "split": "development", "conditions": conditions,
            "artifact": candidate.implementation, "parent_ids": list(candidate.parent_ids),
            "operator": candidate.operator, "proposal": generation, "result": result.to_dict()})
        return result

    def best_candidate():
        if demand_protocol == "v2" and not any(r.score is not None for r in results.values()):
            return None
        return candidates[choose_parent(results, 0, "incumbent")["candidate_id"]]

    def record_round(round_index):
        best = best_candidate()
        best_by_round.append({"round": round_index, "best": best.candidate_id if best else None,
                              "development_score": results[best.candidate_id].score if best else None,
                              "eligible_candidates": sum(r.score is not None for r in results.values()),
                              "behavior_diversity": behavior_diagnostics(results)})
        history = {"best_by_round": best_by_round, "parent_selection": parent_history}
        if discovery:
            report = archive.report()
            report.update(model_group=group_id, demand_id=demand_id)
            for group in report["mechanisms"]:
                group["representative_artifact"] = candidates[group["representative"]].implementation
            write_json(output / "mechanism_archive.json", report)
            record = best_by_round[-1]
            record.update(mechanism_count=len(archive.groups),
                          economic_reference_candidate=record.pop("best"),
                          economic_reference_score=record.pop("development_score"))
            history["mechanisms_by_round"] = history.pop("best_by_round")
            history["research_goal"] = RESEARCH_GOAL
        write_json(output / "search_history.json", history)

    candidates["seed"] = seed
    loop.run_round([seed], evaluate, round_index=-1)
    record_round(-1)
    # Public example is independent of ranking; stored for debugging / future data.
    public_score, public_metrics, _, _ = evaluate_submission(seed.implementation,
        evaluation_request("public", output / "public_seed"))
    write_json(output / "public_seed.json", {"score": public_score, "metrics": public_metrics})
    if initial_archive:
        initial = json.loads(Path(initial_archive).read_text())
        if initial["model_group"] != group_id or initial["demand_id"] != demand_id:
            raise ValueError("initial archive must belong to the same model group and demand")
        for index, group in enumerate(initial["mechanisms"]):
            cid = f"initial-{index}"
            candidate = AlgorithmSpec(cid, "dsl", implementation=group["representative_artifact"],
                                      operator="archive-restart", mechanism_id="dsl")
            candidates[cid] = candidate
            loop.run_round([candidate], evaluate, round_index=-1)
            mid = archive.assignments.get(cid)
            if mid is not None:
                previous_uses = initial.get("parent_uses", {}).get(group["mechanism_id"], 0)
                archive.parent_uses[mid] = max(archive.parent_uses.get(mid, 0), previous_uses)
        record_round(-1)
    for round_index in range(rounds):
        if search_mode == "feedback" and not discovery:
            selection = choose_parent(results, round_index, parent_policy,
                                      allow_infeasible=demand_protocol == "v2")
            parent = candidates[selection["candidate_id"]]
            parent_history.append({"round": round_index, **selection})
        elif search_mode == "independent":
            parent = None
            parent_history.append({"round": round_index, "candidate_id": None,
                                   "selection": "independent", "distance": None})
        # Freeze this cohort's feedback and parent before generating any sibling.
        feedback = ([{"candidate_id": cid, "artifact": candidates[cid].implementation,
                      "result": generation_feedback(result, demand_protocol)} for cid, result in results.items()]
                    if search_mode == "feedback" else [])
        cohort = []
        for slot in range(per_round):
            if discovery and search_mode == "feedback":
                selection = archive.parent(results)
                parent = candidates[selection["candidate_id"]]
                parent_history.append({"round": round_index, "slot": slot, **selection})
            request = {"schema": "stochopia-contract-generation-request.v1", "round_index": round_index,
                       "slot": slot, "conditions": conditions, "feedback": feedback,
                       "objective": objective, "horizon_months": 3}
            if parent is not None:
                request.update(parent_id=parent.candidate_id, parent=parent.implementation)
            if demand_protocol == "v2" or search_mode != "feedback":
                request.update(search_mode=search_mode, demand_protocol=demand_protocol)
            if spec:
                request["demand"] = public_spec
            if discovery:
                request.update(research_goal=RESEARCH_GOAL,
                               objective=RESEARCH_GOAL, economic_objective=objective,
                               design_brief=discovery_brief(round_index * per_round + slot))
                if search_mode == "feedback":
                    request["mechanism_archive"] = archive.public_context()
            elif proposal_strategy == "balanced-structure":
                request["design_brief"] = proposal_brief(round_index, slot, per_round)
            generation = generator.generate(request)
            cid = f"r{round_index}-s{slot}"
            proposals[cid] = {"request": request, "generation": asdict(generation)}
            append_jsonl(output / "generation.jsonl", {"model_group": group_id, "candidate_id": cid, **proposals[cid]})
            candidate = AlgorithmSpec(cid, "dsl", implementation=generation.artifact,
                parent_ids=(parent.candidate_id,) if parent is not None else (),
                operator="api-design" if isinstance(generator, APIContractGenerator) else "fixture-proposal",
                mechanism_id="dsl")
            candidates[cid] = candidate
            cohort.append(candidate)
        loop.run_round(cohort, evaluate, round_index=round_index)
        record_round(round_index)
        best = best_candidate()
        progress = ({"round": round_index, "mechanism_profiles": len(archive.groups)} if discovery else
                    {"round": round_index, "best": best.candidate_id if best else None})
        print(json.dumps(progress, ensure_ascii=False), flush=True)

    winner = best_candidate()
    # Freeze winner before any held-out evaluation; no further generate calls.
    portfolio_results = []
    if discovery:
        portfolio = [{"mechanism_id": group["mechanism_id"],
                      "candidate": candidates[group["representative"]].to_dict(),
                      "novelty_status": group["novelty_status"]} for group in archive.groups]
        write_json(output / "selected.json", {"selection_kind": "mechanism-portfolio", "mechanisms": portfolio})
        for group in archive.groups:
            candidate = candidates[group["representative"]]
            score, metrics, _, evidence = evaluate_submission(candidate.implementation,
                evaluation_request("held_out", output / "held_out" / candidate.candidate_id))
            portfolio_results.append({"mechanism_id": group["mechanism_id"], "candidate_id": candidate.candidate_id,
                                      "score": score, "metrics": metrics, "evidence": evidence})
        write_json(output / "held_out.json", {"selection_kind": "frozen-mechanism-portfolio",
                                             "mechanisms": portfolio_results})
        reference = next((row for row in portfolio_results if winner and row["candidate_id"] == winner.candidate_id), None)
        score, metrics = (reference["score"], reference["metrics"]) if reference else (None, {})
    elif winner:
        write_json(output / "selected.json", winner.to_dict())
        score, metrics, _, evidence = evaluate_submission(winner.implementation,
            evaluation_request("held_out", output / "held_out"))
    else:
        write_json(output / "selected.json", {"status": "no_feasible_candidate"})
        score, metrics, evidence = None, {}, {"status": "not_performed", "reason": "no feasible development winner"}
    if not discovery:
        write_json(output / "held_out.json", {"candidate_id": winner.candidate_id if winner else None, "score": score,
                                             "metrics": metrics, "evidence": evidence})
    summary = {"protocol": PROTOCOL, "model_group": group_id, "demand_id": demand_id, "objective": objective, "selected": winner.candidate_id if winner else None,
               "development_score": results[winner.candidate_id].score if winner else None, "held_out_score": score,
               "held_out_mean_client_return": metrics.get("mean_client_return"),
               "held_out_accepted_rate": metrics.get("accepted_rate"), "candidate_count": len(candidates),
               "api_calls": getattr(generator, "calls", 0), "output": str(output)}
    summary.update(demand_protocol=demand_protocol, search_mode=search_mode,
                   proposal_strategy=proposal_strategy,
                   status="selected" if winner else "no_feasible_candidate",
                   parent_policy=config["parent_policy"], proposal_calls=len(proposals),
                   planned_proposal_calls=rounds * per_round,
                   api_call_budget=getattr(generator, "max_calls", None),
                   feedback_scope=config["feedback_splits"], best_by_round=best_by_round,
                   parent_selection=parent_history, behavior_diversity=behavior_diagnostics(results),
                   interpretation="feedback-guided artifact search; not a test of recursive self improvement")
    if demand_protocol == "v2":
        development_metrics = results[winner.candidate_id].metrics if winner else {}
        development_score = results[winner.candidate_id].score if winner else None
        seed_score = results["seed"].score
        summary.update(score_evaluation="batch cashflow execution on P scenarios",
                       replay_evaluation="representative quote-submit-hold episodes; inspect held_out evidence",
                       score_mode=eval_config["score_mode"],
                       development_distribution_means=development_metrics.get("distribution_means"),
                       held_out_distribution_means=metrics.get("distribution_means"),
                       held_out_expected_net_return=metrics.get("expected_net_return"),
                       held_out_worst_distribution_return=metrics.get("worst_distribution_return"),
                       held_out_scored_paths=metrics.get("scored_paths", 0),
                       held_out_representative_replay_count=metrics.get("representative_replay_count", 0),
                       development_holdout_gap=(development_score - score
                                                if development_score is not None and score is not None else None),
                       handwritten_seed_development_score=seed_score,
                       development_improvement_over_handwritten_seed=(development_score - seed_score
                           if development_score is not None and seed_score is not None else None),
                       development_pricing_audit=development_metrics.get("pricing_audit"),
                       held_out_pricing_audit=metrics.get("pricing_audit"))
    usage = [p["generation"]["usage"] for p in proposals.values()]
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")
    summary["token_reporting_calls"] = {name: sum(isinstance(u.get(name), (int, float)) for u in usage)
                                        for name in token_fields}
    summary["reported_tokens"] = {name: (sum(u[name] for u in usage if isinstance(u.get(name), (int, float)))
                                        if summary["token_reporting_calls"][name] else None)
                                  for name in token_fields}
    summary["api_usage_reported_calls"] = sum("total_tokens" in u for u in usage)
    summary["failed_generations"] = sum(bool(p["generation"]["error"]) for p in proposals.values())
    summary["development_eligible_candidates"] = sum(r.score is not None for r in results.values())
    summary.update(research_goal=config["research_goal"], selection_policy=config["selection_policy"])
    if discovery:
        summary["economic_reference"] = {"candidate_id": summary["selected"],
            "development_score": summary["development_score"], "held_out_score": summary["held_out_score"],
            "role": "secondary economic comparison, not the discovery winner"}
        summary.update(selected=None, development_score=None, held_out_score=None,
            objective=RESEARCH_GOAL, economic_objective=objective,
            interpretation="mechanism hypotheses with executed contrasts; not proof of market novelty or recursive self improvement",
            selected_mechanisms=[row["candidate_id"] for row in portfolio_results],
            status="portfolio_selected" if portfolio_results else "no_feasible_mechanism",
            parent_policy=config["parent_policy"], proposal_strategy=config["proposal_strategy"],
            primary_measures={"feasible_mechanism_profiles": len(archive.groups),
                "unmatched_profiles_with_public_witness": sum(not group["matched_references"] and bool(group["contrast_witnesses"]) for group in archive.groups),
                "held_out_feasible_mechanisms": sum(row["score"] is not None for row in portfolio_results)},
            discovery_scope=archive.report()["scope"],
            held_out_portfolio_scored_paths=sum(row["metrics"].get("scored_paths", 0) for row in portfolio_results),
            held_out_portfolio_replay_count=sum(row["metrics"].get("representative_replay_count", 0) for row in portfolio_results),
            economic_score_scope="return fields are nested under economic_reference; no novelty bonus")
        economic_fields = ("held_out_mean_client_return", "held_out_accepted_rate", "score_mode",
            "development_distribution_means", "held_out_distribution_means", "held_out_expected_net_return",
            "held_out_worst_distribution_return", "held_out_scored_paths", "held_out_representative_replay_count",
            "development_holdout_gap", "handwritten_seed_development_score",
            "development_improvement_over_handwritten_seed", "development_pricing_audit", "held_out_pricing_audit")
        for name in economic_fields:
            if name in summary:
                summary["economic_reference"][name] = summary.pop(name)
        summary["mechanisms_by_round"] = summary.pop("best_by_round")
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", choices=("offline", "api"), default="offline")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--models", type=Path, default=DEFAULT_API_MODELS)
    parser.add_argument("--env-file", type=Path, default=STOCHOPIA / ".env")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--per-round", type=int, default=2)
    parser.add_argument("--paths", type=int, default=4096)
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, help="optional experiment output cap; default uses provider settings")
    parser.add_argument("--demand-protocol", choices=("v1", "v2"),
                        help="default v2 for equity demands; v1 for pipeline; explicit v1 reproduces historical demands")
    parser.add_argument("--eval-config", type=Path, help="v2 evaluator configuration JSON")
    parser.add_argument("--search-mode", choices=("feedback", "independent"), default="feedback")
    parser.add_argument("--parent-policy", choices=("incumbent", "alternating-diverse"), default="incumbent")
    parser.add_argument("--proposal-strategy", choices=("default", "balanced-structure"), default="default")
    parser.add_argument("--research-goal", choices=("discovery", "return"), help="default: discovery for v2 equity, return for legacy")
    parser.add_argument("--initial-archive", type=Path, help="resume same-model, same-demand representatives; re-evaluate on development only")
    parser.add_argument("--demand", choices=("pipeline", *DEMAND_IDS), default="upside-protected")
    parser.add_argument("--objective", choices=("accepted-client-return.v1", "acceptance-rate.v1", DEMAND_OBJECTIVE,
                                               "equity-demand-expected-return.v2", "equity-demand-robust-return.v2"))
    args = parser.parse_args()
    demand_id = None if args.demand == "pipeline" else args.demand
    demand_protocol = args.demand_protocol or ("v2" if demand_id else "v1")
    generator = EquityFixtureGenerator() if demand_id else FixtureGenerator()
    if args.generator == "api":
        from dotenv import load_dotenv
        from stochopia.llm import load_model_registry
        load_dotenv(args.env_file)
        if args.models.suffix == ".json":
            generator = api_generator(args.model, config_path=args.models,
                max_calls=args.max_calls, max_tokens=args.max_tokens)
        else:
            # Historical YAML presets remain explicitly usable for old runs.
            cfg = load_model_registry(args.models)[args.model]
            if cfg.provider != "openai-compatible":
                parser.error("use the native JSON preset for Anthropic Messages")
            generator = APIContractGenerator(base_url=cfg.base_url, model=cfg.model,
                api_key_env=cfg.api_key_env, max_calls=args.max_calls, max_tokens=args.max_tokens)
        args.output = args.output / generator.group_id / (demand_id or "pipeline")
    print(json.dumps(run_search(args.output, generator, rounds=args.rounds, per_round=args.per_round,
        paths=args.paths, objective=args.objective, demand_id=demand_id,
        demand_protocol=demand_protocol, eval_config=json.loads(args.eval_config.read_text()) if args.eval_config else None,
        search_mode=args.search_mode, parent_policy=args.parent_policy,
        proposal_strategy=args.proposal_strategy, research_goal=args.research_goal,
        initial_archive=args.initial_archive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
