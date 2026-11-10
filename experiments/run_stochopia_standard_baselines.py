"""Offline standard-family controls, using the same v2 evaluator as model designs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from experiments.run_stochopia_structuring import write_json
from tasks.stochopia_structuring.evaluator import evaluate_submission
from tasks.stochopia_structuring.equity_demands import DEMAND_IDS
from tasks.stochopia_structuring.equity_protocol_v2 import protocol_config
from tasks.stochopia_structuring.standard_baselines import baseline_config, optimize_standard_baselines


def run_baselines(output, *, demand_id, paths=4096, eval_config=None, max_evaluations=24):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = protocol_config(eval_config)
    counter = 0

    def evaluate(artifact, split="development"):
        nonlocal counter
        counter += 1
        return evaluate_submission(artifact, {"demand_protocol": "v2", "demand_id": demand_id,
            "eval_config": config, "paths": paths, "split": split,
            "output": str(output / split / f"evaluation-{counter:03d}")})

    result = optimize_standard_baselines(demand_id, evaluate, baseline_config({"max_evaluations": max_evaluations}))
    # Select only on development; audit the selected standard family exactly once.
    best = result["best"]
    if best:
        score, metrics, _, evidence = evaluate(best["artifact"], "held_out")
        result["held_out"] = {"score": score, "metrics": metrics, "evidence": evidence,
                              "candidate_id": best["candidate_id"]}
    else:
        result["held_out"] = {"status": "not_performed", "reason": "no feasible standard candidate"}
    result.update(eval_config=config, paths=paths, api_calls=0,
                  evaluation_budget_scope="standard family optimization is a separate comparator budget, not model proposal calls")
    write_json(output / "standard_baselines.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demand", choices=DEMAND_IDS, required=True)
    parser.add_argument("--paths", type=int, default=4096)
    parser.add_argument("--eval-config", type=Path)
    parser.add_argument("--max-evaluations", type=int, default=24)
    args = parser.parse_args()
    result = run_baselines(args.output, demand_id=args.demand, paths=args.paths,
        eval_config=json.loads(args.eval_config.read_text()) if args.eval_config else None,
        max_evaluations=args.max_evaluations)
    print(json.dumps({"output": str(args.output), "evaluations": result["evaluations"],
        "best": result["best"]["candidate_id"] if result["best"] else None,
        "score": result["best"]["score"] if result["best"] else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
