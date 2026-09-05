#!/usr/bin/env python3
"""Fresh numerical replay of generated programs, audits and all references.

The output directory must be empty. Recorded failures of model generation are
retained, not converted to numerical successes or sent to the model again.
The evaluator and candidate sources are loaded from the original bundle.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle, output = args.bundle.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Replay requires a new empty output directory; recorded evidence is never reused.")
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("frozen_workshop", bundle / "reproduction/experiments/run_bermudan_live_workshop.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows = [json.loads(line) for line in (bundle / "candidate_ledger.jsonl").read_text().splitlines()]
    audits = read(bundle / "summary.json")["private_audits"]
    jobs = [("public", f"{r['mode']}-s{r['seed']}-{r['candidate_id']}",
             bundle / r["program_path"], r["result"]) for r in rows]
    jobs += [("audit", f"{r['mode']}-s{r['seed']}-{r['candidate_id']}",
              bundle / r["program_path"], r["result"]) for r in audits]
    for path in sorted((bundle / "references").glob("s*/*.json")):
        if path.name.endswith(".request.json"):
            continue
        jobs.append(("reference", f"{path.parent.name}-{path.stem}",
                     bundle / "reproduction/tasks/bermudan_python_search/research_candidates" / path.stem,
                     read(path)))
    reports, recorded_failures = [], []
    for kind, identifier, source, old in jobs:
        if old["status"] != "ok":
            recorded_failures.append({"kind": kind, "id": identifier,
                                      "reason": old["failure_reason"],
                                      "scope": "recorded_generation_or_execution_failure_not_reexecuted"})
            continue
        path = output / kind / (identifier + ".json")
        new = runner.evaluate(source, old["request"], path)
        model_hashes = lambda result: [c["model_file_sha256"] for c in result["metrics"].get("training_cells", [])]
        same_models = model_hashes(new) == model_hashes(old) and bool(model_hashes(old))
        old_paths = [s.get("paired_pathwise_improvements") for s in old["metrics"].get("summaries", [])]
        new_paths = [s.get("paired_pathwise_improvements") for s in new["metrics"].get("summaries", [])]
        same_paths = new_paths == old_paths
        probes = new["metrics"].get("independent_validation", {})
        probes_pass = kind != "audit" or all(probes.get(k, {}).get("status") == "passed"
                                             for k in ("lookahead_probe", "deterministic_replay"))
        passed = (new["status"] == "ok" and new["score"] == old["score"] and
                  new["source_digest"] == old["source_digest"] and new["source_unchanged"] and
                  same_models and same_paths and probes_pass)
        reports.append({"kind": kind, "id": identifier, "all_equal": passed,
                        "score_equal": new["score"] == old["score"], "models_equal": same_models,
                        "pathwise_outcomes_equal": same_paths, "independent_probes_passed": probes_pass,
                        "evaluator_wall_seconds": new["wall_seconds"],
                        "result_path": str(path.relative_to(output))})
        print(kind, identifier, passed, flush=True)
    runner.save(output / "replay_verification.json", {
        "schema": "openhyra-bermudan-fresh-bundle-replay.v1",
        "rows": reports, "all_equal": bool(reports) and all(r["all_equal"] for r in reports),
        "fresh_output_required": True, "recorded_failures": recorded_failures,
        "total_evaluator_wall_seconds": sum(r["evaluator_wall_seconds"] for r in reports)})
    if not all(r["all_equal"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
