#!/usr/bin/env python3
"""Run the Stochopia world-model search loop.

    propose (generator: offline fixtures or an API model)
      -> sandboxed walk-forward forecasts on development origins (Stochopia panel)
      -> trusted CRPS scoring against the realised market
      -> experience (mechanism, prediction, measured outcome) -> next proposal
    and, once at the end, a held-out test audit of the selected model.

Run from the OpenHyra directory with Stochopia's interpreter, e.g.

    ../Stochopia/.venv/bin/python experiments/run_stochopia_world_model.py --generator offline \
        --rounds 1 --per-round 2 --output artifacts/stochopia-world-model/offline

API generators print the call plan and exit unless ``--execute`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
STOCHOPIA = ROOT.parent / "Stochopia"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm_discovery import (  # noqa: E402
    AlgorithmDiscoveryLoop,
    AlgorithmSpec,
    DiscoveryLedger,
    EvaluationResult,
)
from tasks.stochopia_world_model.evaluator import (  # noqa: E402
    SEED_PROGRAM,
    TASK_PROTOCOL,
    evaluate_program,
    public_feedback,
)
from tasks.stochopia_world_model.generator import (  # noqa: E402
    DIRECTIONS,
    APIProgramGenerator,
    OfflineProgramGenerator,
)

EXPERIENCE_WINDOW = 12
SEED_ID = "seed"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def prediction_check(predicted: list[str], report: Mapping[str, Any] | None,
                     parent: Mapping[str, Any] | None) -> dict[str, Any]:
    """Did each target/family the proposer named improve over its parent?"""

    if not report or not parent:
        return {"items": {}, "hit_rate": None}
    items: dict[str, Any] = {}
    for name in predicted:
        for table in ("family_skill", "target_skill"):
            if name in report.get(table, {}) and name in parent.get(table, {}):
                before, after = parent[table][name], report[table][name]
                items[name] = {"parent": round(before, 4), "candidate": round(after, 4), "improved": after > before}
                break
        else:
            items[name] = {"improved": None, "note": "not a scored target or family"}
    judged = [item["improved"] for item in items.values() if item["improved"] is not None]
    return {"items": items, "hit_rate": (sum(judged) / len(judged)) if judged else None}


class WorldModelSearchSpace:
    """Keeps the current best world model (the loop's "cognition") and its experience."""

    def __init__(self, generator, *, seed_program: str, seed_report: Mapping[str, Any],
                 per_round: int, output: Path):
        self.generator = generator
        self.per_round = per_round
        self.output = output
        self.population: dict[str, dict[str, Any]] = {
            SEED_ID: {"program": seed_program, "report": seed_report, "score": seed_report["score"],
                      "mechanism": "EWMA Gaussian GBM; Q random walk (reference)"},
        }
        self.best_id = SEED_ID
        self.experience: list[dict[str, Any]] = []
        self.round_index = -1
        self.parent_id = SEED_ID

    def begin_round(self) -> None:
        self.round_index += 1
        self.parent_id = self.best_id  # every sibling in a round shares one frozen parent

    def end_round(self) -> None:
        pass

    def request(self, slot: int) -> dict[str, Any]:
        parent = self.population[self.parent_id]
        direction = DIRECTIONS[(self.round_index * self.per_round + slot) % len(DIRECTIONS)]
        return {
            "protocol": TASK_PROTOCOL,
            "round": self.round_index,
            "slot": slot,
            "direction": direction,
            "parent": {
                "candidate_id": self.parent_id,
                "mechanism": parent["mechanism"],
                "development_feedback": public_feedback(parent["report"]),
                "program": parent["program"],
            },
            "experience": self.experience[-EXPERIENCE_WINDOW:],
        }

    def propose(self, context: Mapping[str, Any], slot: int) -> AlgorithmSpec:
        request = self.request(slot)
        generation = self.generator.generate(request)
        candidate_id = f"r{self.round_index:02d}s{slot:02d}"
        append_jsonl(self.output / "generations.jsonl", {
            "candidate_id": candidate_id, "request": {k: v for k, v in request.items() if k != "parent"},
            "parent_id": self.parent_id, "raw_output": generation.raw_output, "usage": generation.usage,
            "seconds": round(generation.seconds, 3), "error": generation.error,
        })
        proposal = generation.proposal or {}
        implementation = (
            {"program": proposal["program"]} if generation.proposal
            else {"generation_error": generation.error or "no proposal"}
        )
        return AlgorithmSpec(
            candidate_id=candidate_id,
            family="world_model",
            implementation=implementation,
            parent_ids=(self.parent_id,),
            mechanism_id=request["direction"]["id"],
            operator="program_revision",
            prediction=proposal.get("predicted_improvements", []),
            falsifier=proposal.get("falsifier", ""),
            metadata={"mechanism": proposal.get("mechanism", ""), "usage": generation.usage},
        )

    def validate(self, candidate: AlgorithmSpec) -> None:
        candidate.validate()

    def record(self, candidate: AlgorithmSpec, result: EvaluationResult, report: Mapping[str, Any] | None) -> None:
        parent_score = self.population[self.parent_id]["score"]
        entry = {
            "candidate_id": candidate.candidate_id,
            "parent_id": self.parent_id,
            "direction": candidate.mechanism_id,
            "mechanism": candidate.metadata.get("mechanism", ""),
            "predicted_improvements": candidate.prediction,
            "falsifier": candidate.falsifier,
            "status": result.status,
            "dev_score": result.score,
            "delta_vs_parent": None if result.score is None else round(result.score - parent_score, 4),
            "prediction_check": result.metrics.get("prediction_check"),
            "family_skill": (report or {}).get("family_skill"),
            "error": (result.metrics.get("error") or "")[-400:] or None,
        }
        self.experience.append(entry)
        append_jsonl(self.output / "experience.jsonl", entry)
        if report is not None and result.score is not None:
            program = candidate.implementation["program"]
            self.population[candidate.candidate_id] = {
                "program": program, "report": report, "score": result.score, "mechanism": entry["mechanism"],
            }
            (self.output / "programs").mkdir(parents=True, exist_ok=True)
            (self.output / "programs" / f"{candidate.candidate_id}.py").write_text(program, encoding="utf-8")
            if result.score > self.population[self.best_id]["score"]:
                self.best_id = candidate.candidate_id


def run(output: Path, generator, *, panel, rounds: int, per_round: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    workdir = output / "sandboxes"
    seed_program = SEED_PROGRAM.read_text(encoding="utf-8")
    seed_eval = evaluate_program(seed_program, panel=panel, split="dev", workdir=workdir)
    if seed_eval["status"] != "ok" or seed_eval["score"] is None:
        raise RuntimeError(f"seed program produced no development score: {seed_eval.get('error')}")
    space = WorldModelSearchSpace(generator, seed_program=seed_program, seed_report=seed_eval["report"],
                                  per_round=per_round, output=output)
    (output / "programs").mkdir(parents=True, exist_ok=True)
    (output / "programs" / f"{SEED_ID}.py").write_text(seed_program, encoding="utf-8")

    def evaluate(candidate: AlgorithmSpec) -> EvaluationResult:
        implementation = candidate.implementation
        parent_report = space.population[space.parent_id]["report"]
        if "generation_error" in implementation:
            result = EvaluationResult(candidate.candidate_id, "generation_error", None,
                                      {"error": implementation["generation_error"]}, split="development")
            space.record(candidate, result, None)
            return result
        try:
            outcome = evaluate_program(implementation["program"], panel=panel, split="dev", workdir=workdir)
        except Exception as exc:  # one bad proposal must not end the run
            outcome = {"status": "evaluator_error", "score": None,
                       "error": f"{type(exc).__name__}: {exc}"[-1500:], "cost": {}}
        report = outcome.get("report")
        metrics: dict[str, Any] = {"error": outcome.get("error")}
        if report is not None:
            metrics["feedback"] = public_feedback(report)
            metrics["prediction_check"] = prediction_check(list(candidate.prediction or []), report, parent_report)
        result = EvaluationResult(candidate.candidate_id, outcome["status"], outcome["score"], metrics,
                                  split="development", cost=outcome["cost"])
        space.record(candidate, result, report)
        return result

    loop = AlgorithmDiscoveryLoop(ledger=DiscoveryLedger(output / "ledger.jsonl"))
    loop.run_search(space, evaluate, rounds=rounds, candidates_per_round=per_round)

    # One-shot held-out audit of the selected model (and the reference seed).
    best = space.population[space.best_id]
    audit = {}
    for label, program in (("selected", best["program"]), ("seed", seed_program)):
        outcome = evaluate_program(program, panel=panel, split="test", workdir=workdir)
        audit[label] = {"status": outcome["status"], "error": outcome.get("error"),
                        **({k: outcome["report"][k] for k in ("origins", "score", "score_lcb_95", "family_skill",
                                                              "target_skill")} if outcome.get("report") else {})}
    final = {
        "selected_candidate": space.best_id,
        "selected_mechanism": best["mechanism"],
        "selected_dev_score": best["score"],
        "selected_dev_lcb_95": best["report"]["score_lcb_95"],
        "test_audit": audit,
        "selection_rule": "highest development score; test scored once after search",
    }
    write_json(output / "final_audit.json", final)
    summary = {
        "protocol": TASK_PROTOCOL,
        "generator": generator.metadata(),
        "rounds": rounds,
        "per_round": per_round,
        "candidates": len(space.experience),
        "evaluated_ok": sum(1 for item in space.experience if item["status"] == "ok"),
        "prediction_hit_rates": [item["prediction_check"]["hit_rate"] for item in space.experience
                                 if item.get("prediction_check")],
        **final,
    }
    write_json(output / "summary.json", summary)
    return summary


def build_generator(name: str, max_calls: int):
    if name == "offline":
        return OfflineProgramGenerator()
    return APIProgramGenerator.from_preset(name, max_calls=max_calls)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", default="offline",
                        help="'offline' or a preset name from tasks/stochopia_structuring/api_models.json")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--per-round", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, default=STOCHOPIA / "data" / "derived" / "world_model")
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".config" / "stochopia" / "model-api.env")
    parser.add_argument("--execute", action="store_true", help="required before any API call is made")
    args = parser.parse_args()

    calls = args.rounds * args.per_round
    if args.generator != "offline" and not args.execute:
        print(json.dumps({"plan_only": True, "generator": args.generator, "api_calls": calls,
                          "rounds": args.rounds, "per_round": args.per_round,
                          "note": "add --execute to spend the API budget"}, ensure_ascii=False))
        return 0
    if args.generator != "offline" and args.env_file.is_file():
        from dotenv import load_dotenv

        load_dotenv(args.env_file)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output}")

    from stochopia.research.market_panel import load_panel

    generator = build_generator(args.generator, calls)
    if hasattr(generator, "credential_present") and not generator.credential_present():
        raise SystemExit(f"missing API credential in {generator.api_key_env}")
    panel = load_panel(args.panel_dir)
    summary = run(args.output, generator, panel=panel,
                  rounds=args.rounds, per_round=args.per_round)
    print(json.dumps({k: summary[k] for k in ("candidates", "evaluated_ok", "selected_candidate",
                                               "selected_dev_score", "prediction_hit_rates")}, ensure_ascii=False))
    print(json.dumps(summary["test_audit"]["selected"].get("family_skill"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
