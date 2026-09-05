#!/usr/bin/env python3
"""Rebuild paired statistics and check an exported real-model Bermudan bundle.

This checks the recorded experiment, not statistical generalization or the
truth of candidate-authored mechanisms. Numerical execution uses --replay in
the frozen live runner and is checked here as a separate evidence item.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_bytes(rows, pairs):
    rows = json.loads(json.dumps(rows))
    rows.sort(key=lambda row: (int(row.get("iteration", 0)), str(row.get("record_id", ""))))
    paired = {p["pair"]["guided_record_id"]: p for p in pairs}
    for row in rows:
        row["prediction_basis"] = "candidate_vs_evaluator_baseline"
        if row.get("record_id") in paired:
            observation = paired[row["record_id"]].get("prediction_test", {})
            row.update(matched_observation=observation,
                       prediction_basis="guided_vs_matched_control_on_target_slice",
                       prediction_verdict=observation.get("prediction_verdict", "not_observed"),
                       next_action=observation.get("next_action", "revise"))
    return (json.dumps({"schema": "openhyra-prediction-table.v1", "rows": rows,
                        "row_count": len(rows)}, ensure_ascii=False,
                       sort_keys=True, indent=2) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(bundle / "reproduction"))
    from sandbox import source_tree_hash
    from bermudan_research import paired_slice_effects
    from proposal_agent import prepare_draft, _apply_python_program_operator

    checks = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    manifest = read(bundle / "manifest.json")
    bad = [p for p, h in manifest["code_sha256"].items()
           if not (bundle / "reproduction" / p).is_file()
           or sha(bundle / "reproduction" / p) != h]
    check("frozen_source_hashes", not bad, bad)
    supplemental = read(bundle / "supplemental_verification.json")
    check("supplemental_verification_source_hashes", all(
          sha(bundle / "reproduction" / p) == h for p, h in supplemental["code_sha256"].items()))
    rows = jsonl(bundle / "candidate_ledger.jsonl")
    summary = read(bundle / "summary.json")
    seeds, modes = manifest["seeds"], manifest["modes"]
    matrix = Counter((r["mode"], r["seed"], r["round"], r["matched_arm"]) for r in rows)
    expected = {(mode, seed, rnd, arm): 2 for mode in modes for seed in seeds
                for rnd in range(2) for arm in ("guided", "control")}
    check("complete_public_matrix", matrix == expected, {"rows": len(rows)})
    for mode in modes:
        for seed in seeds:
            for rnd, operators in enumerate(manifest["operator_schedule"]):
                observed = [r["proposal"]["operator"] for r in rows if r["mode"] == mode
                            and r["seed"] == seed and r["round"] == rnd and r["matched_arm"] == "guided"]
                check(f"{mode}/s{seed}/r{rnd}:two_actual_operator_classes", sorted(observed) == sorted(operators))
    for r in rows:
        label = f"{r['mode']}/s{r['seed']}/{r['candidate_id']}"
        result = r["result"]
        path = Path(r["program_path"])
        check(label + ":relocatable_source", not path.is_absolute())
        digest = source_tree_hash(bundle / path, 1024 * 1024)[0]
        check(label + ":source_digest", digest == result["source_digest"] == r["proposal"]["source_digest"])
        check(label + ":source_unchanged", result["source_unchanged"])
        if r["matched_arm"] == "guided":
            execution = r["proposal"]["execution"]
            # The receipt preserves the concrete sub-operation, e.g. binary_operator
            # or fit_predict_program_composition, not a repeated canonical label.
            # Rebuild the pre-LLM product from both original parents and the slot.
            parents = bundle / "runs" / f"{r['mode']}-s{r['seed']}" / "eb/solutions"
            intervention = {**r["hypothesis"], "matched_arm": "guided", "slot": r["candidate_seed"]}
            if len(r["proposal"]["parent_ids"]) == 2:
                intervention["secondary_parent_path"] = str(parents / r["proposal"]["parent_ids"][1])
            with tempfile.TemporaryDirectory(prefix="openhyra-operator-rebuild-") as tmp:
                draft = Path(tmp) / "draft"
                prepare_draft(parents / r["proposal"]["parent_ids"][0], draft)
                applied = _apply_python_program_operator(draft, entrypoint="algorithm.py",
                          source_files=["algorithm.py", "manifest.json"], intervention=intervention)
                materialized = sha(draft / "algorithm.py")
            check(label + ":operator_materialized", execution.get("executed") and
                  execution.get("declared_operator") == r["proposal"]["operator"] and
                  applied == execution.get("applied_operator") and
                  materialized == execution.get("materialized_source_sha256"),
                  {"reconstructed_sub_operation": applied, "reconstructed_source_sha256": materialized})
        if result["status"] != "ok":
            check(label + ":failure_retained", bool(result["failure_reason"]))
            continue
        cells = result["metrics"]["training_cells"]
        check(label + ":four_fresh_training_cells", len(cells) == 4 and
              len({(c["instance_id"], c["repeat"]) for c in cells}) == 4 and
              len({c["train_seed"] for c in cells}) == 4)
        check(label + ":native_isolation", all(c["isolation"] == "seatbelt" and
              c["research_fallback"] is False for c in cells))
        check(label + ":training_provenance", all(c["fit_wall_seconds"] > 0 and
              c["model_file_sha256"] and c["policy_file_sha256"] and c["training_paths_sha256"]
              and c["payoffs_sha256"] and c["target_sha256"] for c in cells))

    rebuilt_pairs = []
    for old in summary["pairs"]:
        pair = [r for r in rows if r["mode"] == old["mode"] and r["seed"] == old["seed"]
                and r["pair_id"] == old["pair_id"]]
        guided = next(r for r in pair if r["matched_arm"] == "guided")
        control = next(r for r in pair if r["matched_arm"] == "control")
        label = f"{old['mode']}/s{old['seed']}/{old['pair_id']}"
        check(label + ":matched_design", guided["proposal"]["parent_digest"] == control["proposal"]["parent_digest"]
              and guided["candidate_seed"] == control["candidate_seed"]
              and guided["result"]["request"] == control["result"]["request"]
              and control["proposal"]["source_digest"] == control["proposal"]["parent_digest"])
        gm, cm = guided["result"]["metrics"], control["result"]["metrics"]
        stats = paired_slice_effects(gm, cm, guided["hypothesis"].get("target_slice", ""))
        if guided["result"]["status"] != "ok" or control["result"]["status"] != "ok":
            stats.update(prediction_verdict="execution_failed", next_action="falsify")
        check(label + ":reconstructed_statistics", all(old.get(k) == v for k, v in stats.items()))
        if stats["status"] == "observed":
            # Independent calculation from exported path samples, including covariance.
            matched = {(c["instance_id"], c["repeat"]): c for c in cm["summaries"]}
            independent = []
            for cell in stats["cells"]:
                a = next(c for c in gm["summaries"] if (c["instance_id"], c["repeat"]) ==
                         (cell["instance_id"], cell["repeat"]))
                b = matched[cell["instance_id"], cell["repeat"]]
                differences = [x-y for x, y in zip(a["paired_pathwise_improvements"], b["paired_pathwise_improvements"])]
                mean = math.fsum(differences) / len(differences)
                variance = math.fsum((x-mean)**2 for x in differences) / (len(differences)-1) / len(differences)
                independent.append((mean, variance))
            effect = math.fsum(x[0] for x in independent) / len(independent)
            se = math.sqrt(math.fsum(x[1] for x in independent)) / len(independent)
            check(label + ":independent_paired_moments", math.isclose(effect, stats["effect"], abs_tol=1e-13)
                  and math.isclose(se, stats["standard_error"], abs_tol=1e-13))
            gc = {(c["instance_id"], c["repeat"]): c for c in gm["training_cells"]}
            cc = {(c["instance_id"], c["repeat"]): c for c in cm["training_cells"]}
            fields = ("train_seed", "path_seed", "input_bundle_sha256", "training_paths_sha256", "payoffs_sha256", "training_path_count")
            check(label + ":matched_actual_training", gc.keys() == cc.keys() and
                  all(gc[k][field] == cc[k][field] for k in gc for field in fields))
        rebuilt_pairs.append({"mode": old["mode"], "seed": old["seed"], "pair_id": old["pair_id"], **stats})

    chains = []
    for seed in seeds:
        run = bundle / "runs" / f"context_proposal-s{seed}"
        context = read(run / "context-1.json")
        meta = context["metadata"]["prediction_table"]
        previous = [r for r in jsonl(run / "research/prediction_ledger.jsonl") if r["iteration"] == 0]
        pairs = [r for r in jsonl(run / "research/matched_controls.jsonl") if r["iteration"] == 0]
        table = table_bytes(previous, pairs)
        digest = hashlib.sha256(table).hexdigest()
        (args.output / f"context-s{seed}-consumed-round0-table.json").write_bytes(table)
        check(f"context-s{seed}:exact_previous_table", meta["consumed"] and meta["row_count"] == 4
              and meta["rows_in_prompt"] == 4 and digest == meta["sha256"])
        call = read(run / "agent_calls" / (context["context_call_ids"][0] + ".json"))
        check(f"context-s{seed}:actual_model_call", call["role"] == "context" and call["returncode"] == 0
              and all(r["record_id"] in call["prompt"] for r in previous))
        chains.append({"seed": seed, "previous_table_sha256": digest,
                       "context_call_id": call["id"], "decision": context["decision"]})

    frozen = read(bundle / "public_selection_frozen.json")
    audits = summary["private_audits"]
    check("private_audit_matrix", {(a["mode"], a["seed"]) for a in audits} ==
          {(mode, seed) for mode in modes for seed in seeds} and len(audits) == 6)
    for a in audits:
        label = f"audit:{a['mode']}/s{a['seed']}"
        winner = next(r["winner"] for r in frozen if r["mode"] == a["mode"] and r["seed"] == a["seed"])
        check(label + ":frozen_selection", winner == a["candidate_id"])
        check(label + ":successful_heldout_evaluation", a["result"]["status"] == "ok")
        validation = a["result"]["metrics"].get("independent_validation", {})
        check(label + ":observed_probes", all(validation.get(k, {}).get("status") == "passed"
              and validation[k].get("observed") for k in ("deterministic_replay", "lookahead_probe")))
        check(label + ":measured_probe_cost", all(validation.get("cost", {}).get(k, 0) > 0
              for k in ("replay_wall_seconds", "probe_wall_seconds")))

    replay_dir = args.replay_dir or bundle / "numerical_replay"
    replay = read(replay_dir / "replay_verification.json")
    references = [read(p) for p in (bundle / "references").glob("s*/*.json")
                  if not p.name.endswith(".request.json")]
    success_count = sum(r["result"]["status"] == "ok" for r in rows + audits)
    check("all_successful_rows_numerically_replayed", replay["all_equal"] and
          sum(r["kind"] != "reference" for r in replay["rows"]) == success_count,
          {"successful_public_and_audit_rows": success_count})
    check("all_reference_programs_replayed", len(references) == 21 and
          all(r["status"] == "ok" for r in references) and
          sum(r["kind"] == "reference" and r["all_equal"] for r in replay["rows"]) == 21)
    all_results = [r["result"] for r in rows + audits if r["result"]["status"] == "ok"] + references
    all_cells = [c for result in all_results for c in result["metrics"].get("training_cells", [])]
    check("all_primary_fits_used_native_isolation", len(all_cells) == 268 and all(
          c["isolation"] == "seatbelt" and c["research_fallback"] is False for c in all_cells),
          {"primary_fit_cells": len(all_cells)})
    training = read(bundle / "training_validation/training_validation.json")
    check("real_mlp_and_hybrid_training", {r["family"] for r in training["rows"]} == {"mlp", "residual_hybrid"}
          and all(all(r[k] for k in ("gradient_updates_observed", "training_loss_decreased",
                     "terminal_backward_target_matches_mc_payoff", "finite_predictions")) for r in training["rows"]))
    calls = [read(p) for p in (bundle / "runs").glob("*/agent_calls/call-*.json")]
    report = {"schema": "openhyra-bermudan-bundle-audit.v1", "checks": checks,
              "all_passed": all(c["passed"] for c in checks),
              "model_calls": len(calls), "reported_token_calls": sum(c["tokens_reported"] is not None for c in calls),
              "reported_tokens_are_incomplete": any(c["tokens_reported"] is None for c in calls),
              "failure_reasons": [r["result"]["failure_reason"] for r in rows if r["result"]["status"] != "ok"],
              "context_chains": chains, "rebuilt_pairs": rebuilt_pairs}
    (args.output / "bundle_audit.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"checks": len(checks), "all_passed": report["all_passed"],
                      "failures": [c for c in checks if not c["passed"]]}, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
