"""Four separate API searches; default prints a plan without model calls."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from experiments.run_stochopia_structuring import run_search, write_json
from tasks.stochopia_structuring.api_config import DEFAULT_API_MODELS, api_generator, load_api_models
from tasks.stochopia_structuring.equity_demands import DEMAND_IDS
from tasks.stochopia_structuring.evaluator import STOCHOPIA
from tasks.stochopia_structuring.mechanism_archive import RESEARCH_GOAL


def comparison_plan(*, models=None, demands=DEMAND_IDS, config_path=DEFAULT_API_MODELS,
                    rounds=2, per_round=2, max_calls=4, max_tokens=None, paths=4096,
                    demand_protocol="v2", eval_config=None, search_mode="feedback", parent_policy="incumbent",
                    research_goal=None):
    config = load_api_models(config_path)
    models = list(config["models"]) if models is None else list(models)
    if rounds < 1 or per_round < 1 or rounds * per_round > max_calls:
        raise ValueError("positive proposal count must fit the per-model, per-demand call budget")
    if (max_tokens is not None and max_tokens < 1) or paths < 2:
        raise ValueError("max_tokens must be positive and pricing paths >=2")
    if len(set(models)) != len(models) or len(set(demands)) != len(demands):
        raise ValueError("select each model and demand once")
    if not models or not demands or set(demands) - set(DEMAND_IDS):
        raise ValueError("select models and supported equity demands")
    groups = [{"group_id": name, **config["models"][name],
               "credential_present": bool(os.environ.get(config["models"][name]["api_key_env"]))}
              for name in models]
    calls = len(models) * len(demands) * rounds * per_round
    if demand_protocol not in ("v1", "v2") or search_mode not in ("feedback", "independent"):
        raise ValueError("select a supported demand protocol and search mode")
    if demand_protocol == "v2":
        from tasks.stochopia_structuring.equity_protocol_v2 import protocol_config
        eval_config = protocol_config(eval_config)
    research_goal = research_goal or ("discovery" if demand_protocol == "v2" else "return")
    if research_goal not in {"discovery", "return"} or research_goal == "discovery" and demand_protocol != "v2":
        raise ValueError("discovery requires v2; research_goal must be discovery or return")
    return {"schema": "stochopia-model-comparison-plan.v2", "reasoning_policy": config["reasoning_policy"],
            "verified_on": config["verified_on"], "groups": groups, "demands": list(demands),
            "rounds": rounds, "per_round": per_round, "max_calls_per_model_demand": max_calls,
            "max_tokens_per_call": max_tokens, "paths": paths, "pricing_seed": 42,
            "demand_protocol": demand_protocol, "eval_config": eval_config,
            "search_mode": search_mode,
            "parent_policy": ("least-explored-mechanism" if research_goal == "discovery" else parent_policy)
                             if search_mode == "feedback" else None,
            "research_goal": RESEARCH_GOAL if research_goal == "discovery" else "economic-return",
            "selection_policy": "mechanism-portfolio" if research_goal == "discovery" else "economic-incumbent",
            "output_budget_policy": "explicit" if max_tokens is not None else "provider-default-or-required-maximum",
            "planned_max_api_calls": calls,
            "planned_max_output_tokens": calls * max_tokens if max_tokens is not None else None}


def run_comparison(output, *, models=None, demands=DEMAND_IDS, config_path=DEFAULT_API_MODELS,
                   rounds=2, per_round=2, max_calls=4, max_tokens=None, paths=4096,
                   demand_protocol="v2", eval_config=None, search_mode="feedback", parent_policy="incumbent",
                   research_goal=None):
    plan = comparison_plan(models=models, demands=demands, config_path=config_path,
        rounds=rounds, per_round=per_round, max_calls=max_calls, max_tokens=max_tokens, paths=paths,
        demand_protocol=demand_protocol, eval_config=eval_config, search_mode=search_mode, parent_policy=parent_policy,
        research_goal=research_goal)
    missing = [g["api_key_env"] for g in plan["groups"] if not g["credential_present"]]
    if missing:
        raise ValueError("missing API credentials: " + ", ".join(missing))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    index = {"schema": "stochopia-model-comparison-index.v1", "groups": {}}
    write_json(output / "index.json", index)
    for group in plan["groups"]:
        name = group["group_id"]
        folder = output / name
        folder.mkdir()
        group_summary = {"model_group": name, "requested_model": group["model"], "demands": {}}
        for demand in plan["demands"]:
            # Fresh search, budget and experience bank for every model/demand pair.
            generator = api_generator(name, config_path=config_path,
                max_calls=max_calls, max_tokens=max_tokens)
            group_summary["demands"][demand] = run_search(folder / demand, generator,
                rounds=rounds, per_round=per_round, paths=paths, demand_id=demand,
                demand_protocol=demand_protocol, eval_config=eval_config,
                search_mode=search_mode, parent_policy=parent_policy, research_goal=research_goal)
            write_json(folder / "model_summary.json", group_summary)
            index["groups"][name] = str(folder / "model_summary.json")
            write_json(output / "index.json", index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="preset group IDs; defaults to all four")
    parser.add_argument("--demands", nargs="+", choices=DEMAND_IDS, default=list(DEMAND_IDS))
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_MODELS)
    parser.add_argument("--env-file", type=Path, default=STOCHOPIA / ".env")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--per-round", type=int, default=2)
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, help="optional experiment output cap; default uses provider settings")
    parser.add_argument("--paths", type=int, default=4096)
    parser.add_argument("--demand-protocol", choices=("v1", "v2"), default="v2")
    parser.add_argument("--eval-config", type=Path, help="v2 evaluation JSON; never an API credentials file")
    parser.add_argument("--search-mode", choices=("feedback", "independent"), default="feedback")
    parser.add_argument("--parent-policy", choices=("incumbent", "alternating-diverse"), default="incumbent")
    parser.add_argument("--research-goal", choices=("discovery", "return"), help="default: discovery for v2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="perform paid API generation")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    options = dict(models=args.models, demands=args.demands, config_path=args.api_config,
        rounds=args.rounds, per_round=args.per_round, max_calls=args.max_calls,
        max_tokens=args.max_tokens, paths=args.paths, demand_protocol=args.demand_protocol,
        eval_config=json.loads(args.eval_config.read_text()) if args.eval_config else None,
        search_mode=args.search_mode, parent_policy=args.parent_policy, research_goal=args.research_goal)
    plan = comparison_plan(**options)
    if args.execute:
        if args.output is None:
            parser.error("--execute requires a fresh --output directory")
        result = run_comparison(args.output, **options)
    else:
        result = plan
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
