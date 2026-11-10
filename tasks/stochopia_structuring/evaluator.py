"""Trusted Stochopia evaluation of a data-only DSL artifact.

No LLM call occurs in the environment or evaluator. The fixed public policy
asks three questions, requests one quote, submits it, then holds to closure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch

# Local sibling checkouts; alternatively install their packages in this runtime.
STOCHOPIA = Path(__file__).resolve().parents[3] / "Stochopia"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(STOCHOPIA))

from examples.structuring_roundtrip import synthetic_task
from stochopia.adapters.structurer_dsl import DSLContract
from stochopia.environment import replay_and_evaluate, run_agent_episode
from stochopia.environment.agent_runner import AgentDecision
from stochopia.environment.dsl import DSL_AGENT_PROMPT, StochopiaDSLEnv, parse_dsl_action
from stochopia.environment.trajectory import to_jsonable
from stochopia.environment.types import AskClient
from tasks.stochopia_structuring.equity_demands import (
    DEMAND_OBJECTIVE, demand_spec, demand_tasks, normalize_and_check,
)


PROTOCOL = "stochopia-structuring-search.v1"
TOPICS = ("capital", "return_hurdle", "loss_tolerance")
OBJECTIVES = {"accepted-client-return.v1", "acceptance-rate.v1", DEMAND_OBJECTIVE}


def task_suite(split, demand_id=None):
    """Small fixed synthetic split; this is a pipeline check, not a benchmark."""
    if demand_id is not None:
        return demand_tasks(synthetic_task(), split, demand_id)
    cases = {
        "public": [("example", (6000, 6100, 6200, 6300), .20)],
        "development": [("dev-up", (6000, 6080, 6200, 6260), .20),
                        ("dev-flat", (6000, 5950, 6060, 6020), .25)],
        "held_out": [("hold-down", (6000, 5820, 5680, 5700), .20),
                     ("hold-rebound", (6000, 5730, 6150, 6240), .25)],
    }[split]
    base = synthetic_task()
    return [(name, replace(base,
        snapshots=tuple(replace(s, episode_id=name, spot=float(spot), realized_vol_20d=vol)
                        for s, spot in zip(base.snapshots, spots)),
        version="synthetic-dsl-search.v1", task_seed=100 + i))
        for i, (name, spots, vol) in enumerate(cases)]


def public_condition(task, paths=4096, pricing_seed=42):
    """Use actual public observations and query responses, never task.manifest."""
    env = StochopiaDSLEnv(task, dsl_paths=paths, dsl_seed=pricing_seed)
    observation, _ = env.reset(seed=0)
    history = []
    for topic in TOPICS:
        observation, reward, terminated, truncated, info = env.step(AskClient(topic))
        history.append({"action": {"action": "ask_client", "topic": topic},
                        "observation": to_jsonable(observation), "reward": to_jsonable(reward)})
    return {"observation": to_jsonable(observation), "history": history,
            "action_schema": env.action_schema}


class ContractPolicy:
    kind = "fixed-dsl-artifact"
    name = "fixed-disclosure-quote-submit-hold.v1"
    total_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    system_prompt = DSL_AGENT_PROMPT

    def __init__(self, contract):
        self.contract = contract
        self.quoted = self.submitted = False

    def reproducibility_metadata(self):
        return {"policy": self.name, "contract": self.contract, "topics": list(TOPICS)}

    async def act(self, request):
        observation = request.observation
        if observation["round_num"] != 1 or self.submitted:
            action = {"action": "skip", "reason": "hold contract"}
        else:
            topic = next((t for t in TOPICS if t not in observation["disclosed_client"]), None)
            if topic:
                action = {"action": "ask_client", "topic": topic}
            elif observation["quotes"]:
                action = {"action": "submit_design", "quote_id": observation["quotes"][-1]["quote_id"]}
                self.submitted = True
            elif self.quoted:
                action = {"action": "skip", "reason": "quote failed"}
            else:
                action = {"action": "request_dsl_quote", "contract": self.contract}
                self.quoted = True
        return AgentDecision(parse_dsl_action(action), json.dumps(action))


def evaluate_submission(submission, request=None):
    """Harness-compatible (score, metrics, normalized artifact, evidence)."""
    request = request or {}
    if "config" in request:  # Existing OpenHyra search/audit request envelope.
        audit = request["stage"] == "audit"
        config = {**request["config"], "split": "held_out" if audit else "development"}
        if audit:
            # Held-out pricing paths come from the harness-drawn audit seed, so
            # the frozen design is re-issued at audit prices; its identity
            # remains the frozen artifact (re-issued notional is in metrics).
            config["pricing_seed"] = request["seed"]
            score, metrics, _reissued, evidence = evaluate_submission(submission, config)
            return score, metrics, submission, evidence
        request = config
    if "output" not in request:
        # Per-evaluation scratch, never a cwd-relative directory shared by
        # candidates; it is removed, so no trajectory paths are reported.
        with tempfile.TemporaryDirectory(prefix="stochopia-evaluation-") as scratch:
            score, metrics, normalized, evidence = evaluate_submission(
                submission, {**request, "output": scratch})
        for row in metrics.get("episodes", []):
            row.pop("trajectory", None)
        return score, metrics, normalized, evidence
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    if request.get("demand_protocol", "v1") == "v2":
        from tasks.stochopia_structuring.evaluator_v2 import evaluate_v2
        return evaluate_v2(submission, request, output)
    split = request.get("split", "development")
    demand_id = request.get("demand_id")
    spec = demand_spec(demand_id) if demand_id is not None else None
    objective = request.get("objective", DEMAND_OBJECTIVE if spec else "accepted-client-return.v1")
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown evaluator objective: {objective}")
    if (objective == DEMAND_OBJECTIVE) != (spec is not None):
        raise ValueError("equity demands require the equity-demand objective")
    paths, seed = int(request.get("paths", 4096)), int(request.get("pricing_seed", 42))
    started = time.monotonic()
    demand_checks = None
    # Generation / syntax failures remain measured failed candidates, not repairs.
    try:
        if submission.get("protocol") != PROTOCOL:
            raise ValueError("unsupported structuring artifact protocol")
        if submission.get("generation_error"):
            raise ValueError(submission["generation_error"])
        parsed = DSLContract.from_payload(submission["contract"])
        if parsed.program.maturity > 3:
            raise ValueError("this synthetic task supports maturity <=3 months")
        if spec:
            parsed, demand_checks = normalize_and_check(parsed, task_suite(split, demand_id)[0][1], spec,
                                                        paths=paths, pricing_seed=seed)
            if not demand_checks["passed"]:
                failed = [c["name"] for c in demand_checks["checks"] if not c["passed"]]
                raise ValueError("equity demand not met: " + ", ".join(failed))
        contract = parsed.to_payload()
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        metrics = {"status": "rejected", "error": str(exc), "split": split,
                   "objective": objective, "accepted_rate": 0., "episodes": [],
                   "cost": {"evaluation_seconds": time.monotonic() - started}}
        if spec:
            metrics.update(demand=spec, demand_checks=demand_checks)
            metrics["cost"]["normalization_valuations"] = 2 if demand_checks else 0
        return None, metrics, submission, {"split": split}

    async def evaluate_all():
        rows = []
        for name, task in task_suite(split, demand_id):
            folder = output / name
            folder.mkdir(parents=True, exist_ok=True)
            env = StochopiaDSLEnv(task, dsl_paths=paths, dsl_seed=seed)
            run = await run_agent_episode(env, ContractPolicy(contract), trajectory_path=folder / "trajectory.json")
            result = replay_and_evaluate(folder / "trajectory.json", output_path=folder / "evaluation.json", replay=True)
            actions = [entry["transition"]["action"]["action"] for entry in run.trajectory["entries"]]
            rows.append({"case": name, "status": result["status"],
                         "constraints": result["constraint_summary"],
                         "metrics": result["metrics"], "rewards": result["reward_summary"],
                         "replay_verified": result["replay_verified"],
                         "interaction_cost": {"client_queries": actions.count("ask_client"),
                                              "quotes": actions.count("request_dsl_quote"),
                                              "submissions": actions.count("submit_design")},
                         "trajectory": str(folder / "trajectory.json"),
                         "infrastructure_error": run.infrastructure_error})
        return rows

    rows = asyncio.run(evaluate_all())
    accepted = [r["constraints"]["accepted_submissions"] == 1 and
                r["constraints"]["invalid_actions"] == 0 and r["status"] == "complete" for r in rows]
    acceptance = sum(accepted) / len(rows)
    returns = [r["metrics"]["client_flow_adjusted_return"] for r in rows]
    # Lexicographic eligibility: all dev cases must trade before ranking returns.
    score = acceptance if objective == "acceptance-rate.v1" else (sum(returns) / len(returns) if all(accepted) else None)
    metrics = {"status": "ok" if all(accepted) else "rejected", "split": split,
               "objective": objective, "accepted_rate": acceptance,
               "client_flow_adjusted_return": sum(returns) / len(returns),
               "mean_client_return": sum(returns) / len(returns), "episodes": rows,
               "cost": {"evaluation_seconds": time.monotonic() - started,
                        "episodes": len(rows), "pricing_paths_per_quote": paths,
                        "client_queries": sum(r["interaction_cost"]["client_queries"] for r in rows),
                        "quotes": sum(r["interaction_cost"]["quotes"] for r in rows),
                        "steps": sum(r["constraints"]["steps"] for r in rows)}}
    if spec:
        expected_return = sum(returns) / len(returns)
        metrics.update(demand=spec, demand_checks=demand_checks,
                       expected_net_return=expected_return,
                       expected_net_profit_cny=expected_return * spec["investment_budget_cny"],
                       scenario_probabilities={r["case"]: 1. / len(rows) for r in rows})
        metrics["cost"]["normalization_valuations"] = 2
    return score, metrics, {"protocol": PROTOCOL, "contract": contract}, {"split": split, "pricing_seed": seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("request", type=Path, nargs="?")
    args = parser.parse_args()
    torch.set_num_threads(1)
    request = json.loads(args.request.read_text()) if args.request else {}
    try:
        score, metrics, normalized, evidence = evaluate_submission(json.loads(args.artifact.read_text()), request)
        print(json.dumps({"score": score, "metrics": metrics, "normalized_solution": normalized, "evidence": evidence}))
    except (ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))


if __name__ == "__main__":
    main()
