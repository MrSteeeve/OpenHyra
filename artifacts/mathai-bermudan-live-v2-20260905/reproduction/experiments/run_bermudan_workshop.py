#!/usr/bin/env python3
"""Auditable 3-seed, 2-round Bermudan program-search pilot.

Proposal generation is deterministic and repository-owned in this pilot.
It exercises real program operators and the evaluator, but does not estimate
an LLM search effect. The two paths differ in whether prior measured feedback
selects the next composition parent.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tasks.bermudan_optimal_stopping import evaluator
from program_search import PythonProgramSearchSpace
from proposal_agent import _rebuild_program_subsystem

FAMILIES = ROOT / "tasks" / "bermudan_python_search" / "research_candidates"
SEEDS = (17, 29, 41)
MODES = ("context_proposal", "direct_generation")

def source(family):
    root = FAMILIES / family
    return {name: (root / name).read_text() for name in ("algorithm.py", "manifest.json")}

def digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def materialize(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root

def request(stage, seed):
    cfg = dict(instance_count=1, repeats=1, training_paths=64, pricing_paths=64,
               research_mode=True, independent_validation=True, training_timeout_s=30)
    if stage == "audit":
        cfg.update(outer_paths=64, inner_paths=2)
    return dict(schema=evaluator.REQUEST_SCHEMA, stage=stage,
                task="bermudan_python_search", protocol=evaluator.PYTHON_PROGRAM_TASK_PROTOCOL,
                seed=int(seed), suite_id="bermudan-python-" + ("public" if stage == "search" else "hidden") + "-v2",
                config=cfg)

def evaluate(path, stage, seed):
    started = time.perf_counter()
    try:
        score, metrics, evidence, _ = evaluator.evaluate_submission(
            None, request(stage, seed), candidate_source_dir=path)
        return dict(status="ok", score=float(score), metrics=metrics, evidence=evidence,
                    wall_seconds=time.perf_counter()-started, failure_reason=None)
    except Exception as exc:
        return dict(status="error", score=None, metrics={}, evidence={},
                    wall_seconds=time.perf_counter()-started,
                    failure_reason=type(exc).__name__ + ": " + str(exc)[:2000])

def generated_source(operator, seed, secondary_family, parent_source=None):
    parent_source = dict(parent_source or source("linear_ridge"))
    parent_ref = "source:" + digest(parent_source)[:12]
    if operator == "whole_program_restart":
        # The generator callback has no parent program in its request.
        def generator(_request):
            return {"source": source(secondary_family),
                    "prediction": "a new training rule improves the paired score",
                    "falsifier": "the unchanged ridge control is at least as good"}
        space = PythonProgramSearchSpace(generator=generator, seeds=[parent_source],
                                         required_symbol="predict")
        child = space.propose(context={"operator": "llm_generate"}, slot=seed)
        return dict(child.implementation["source"]), [parent_ref], {"generator": "deterministic_repository_program", "discarded_parent": True}
    if operator in {"ast_mutation", "ast_crossover"}:
        seeds = [parent_source, source(secondary_family)]
        space = PythonProgramSearchSpace(seeds=seeds, required_symbol="predict")
        left, right = space.candidates[:2]
        child = (space.mutate(left, slot=seed) if operator == "ast_mutation" else
                 space.crossover(left, right, slot=seed))
        lineage = [parent_ref]
        if operator == "ast_crossover":
            lineage.append("source:" + digest(source(secondary_family))[:12])
        return dict(child.implementation["source"]), lineage, dict(child.metadata)
    # Replace the complete fit body while the parent predict/CLI stays intact.
    # The two regime fits emit their mean coefficients under the parent's
    # model contract, so the subsystem replacement is executable by that same
    # untouched predict implementation.
    proposed = source("gated_ridge")["algorithm.py"].replace(
        'coefficients_lo=lo_coef, coefficients_hi=hi_coef,',
        'coefficients=(lo_coef + hi_coef) * 0.5, coefficients_lo=lo_coef, coefficients_hi=hi_coef,')
    rebuilt = _rebuild_program_subsystem(parent_source["algorithm.py"], proposed, "fit")
    return {**parent_source, "algorithm.py": rebuilt}, [parent_ref], {"subsystem": "fit", "untouched_subsystem": "predict"}

def summarize_pair(pair):
    guided, control = pair
    g, c = guided["result"], control["result"]
    effect = None if g["score"] is None or c["score"] is None else g["score"]-c["score"]
    se = max(float(g["metrics"].get("paired_aggregate_standard_error") or 0.0),
             float(c["metrics"].get("paired_aggregate_standard_error") or 0.0))
    if effect is None: verdict = "execution_failed"
    elif effect > 1.96*se and effect > 0: verdict = "supported"
    elif effect < -1.96*se: verdict = "refuted"
    else: verdict = "inconclusive"
    return dict(schema="openhyra-bermudan-workshop-pair.v1",
                mode=guided["mode"], seed=guided["seed"], round=guided["round"],
                pair_id=guided["pair_id"], operator=guided["proposal"]["operator"],
                same_baseline_parent=True, same_candidate_seed=True,
                same_evaluator_request=True, same_compute_budget=True,
                effect_guided_minus_control=effect, standard_error=se,
                uncertainty_note="max of arm standard errors; pilot diagnostic, not an independent paired estimator",
                prediction_verdict=verdict,
                next_action=("compose" if verdict=="supported" else "restart" if verdict in {"refuted","execution_failed"} else "revise"))

def run(output):
    output.mkdir(parents=True, exist_ok=True)
    rows, pairs = [], []
    baseline = source("linear_ridge")
    for mode in MODES:
        for seed in SEEDS:
            prior_effect = None
            successful = []
            context_parent = dict(baseline)
            for round_index in range(2):
                if round_index == 0:
                    operators = ("whole_program_restart", "ast_mutation")
                    secondary = "mlp" if mode == "context_proposal" else "residual_hybrid"
                else:
                    operators = ("ast_crossover", "subsystem_rewrite")
                    secondary = ("residual_hybrid" if prior_effect is not None and prior_effect > 0 else "mlp") if mode == "context_proposal" else "pca_ridge"
                round_pairs = []
                round_guided = []
                for pair_index, operator in enumerate(operators):
                    trial_seed = seed*1000 + round_index*10 + pair_index
                    pair_id = f"{mode}-s{seed}-r{round_index}-p{pair_index}"
                    parent_source = (
                        context_parent if mode == "context_proposal" else baseline
                    )
                    child_source, lineage, operator_detail = generated_source(
                        operator, trial_seed, secondary, parent_source
                    )
                    hypothesis = dict(id=pair_id, mechanism=f"{operator} with {secondary}",
                                      prediction="the guided program improves the matched public score",
                                      falsifier="the unchanged parent control is at least as good",
                                      target_slice="public_suite_instance_0",
                                      evidence_ids=[] if round_index==0 else [f"{mode}-s{seed}-r0"],
                                      previous_effect=prior_effect,
                                      next_probe="private hidden audit after both public rounds")
                    pair = []
                    for arm, files in (("guided", child_source), ("control", parent_source)):
                        candidate_id = pair_id+"-"+arm
                        path = materialize(output/"programs"/candidate_id, files)
                        result = evaluate(path, "search", seed+round_index*100+pair_index)
                        row = dict(schema="openhyra-bermudan-workshop-candidate.v1",
                                   candidate_id=candidate_id, mode=mode, seed=seed,
                                   round=round_index, pair_id=pair_id, matched_arm=arm,
                                   candidate_seed=trial_seed, hypothesis=hypothesis,
                                   proposal=dict(operator=operator if arm=="guided" else "identity_control",
                                                 executed=(arm == "guided"), source_digest=digest(files),
                                   baseline_parent_digest=digest(parent_source),
                                                 parent_lineage=lineage if arm=="guided" else ["linear_ridge"],
                                                 operator_detail=operator_detail if arm=="guided" else {}),
                                   result=result)
                        rows.append(row); pair.append(row)
                        if arm=="guided" and result["status"]=="ok":
                            successful.append((row,path))
                            round_guided.append((row, child_source))
                    summary = summarize_pair(pair); pairs.append(summary); round_pairs.append(summary)
                observed = [p["effect_guided_minus_control"] for p in round_pairs if p["effect_guided_minus_control"] is not None]
                prior_effect = max(observed) if observed else None
                if mode == "context_proposal" and round_guided:
                    # Context feedback changes the parent source for the next
                    # round; direct generation intentionally keeps its frozen
                    # baseline parent schedule.
                    chosen, chosen_source = max(
                        round_guided,
                        key=lambda item: item[0]["result"]["score"],
                    )
                    context_parent = dict(chosen_source)
            winner, path = max(successful, key=lambda x: x[0]["result"]["score"])
            audit = evaluate(path, "audit", seed+9000)
            rows.append(dict(schema="openhyra-bermudan-workshop-audit.v1",
                             mode=mode, seed=seed, round="private_audit",
                             candidate_id=winner["candidate_id"], proposal=winner["proposal"],
                             public_selection="best successful guided score after two rounds",
                             result=audit))
    with (output/"candidate_ledger.jsonl").open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    summary = dict(schema="openhyra-bermudan-workshop-summary.v1",
                   seeds=list(SEEDS), rounds=2, modes=list(MODES),
                   candidate_count_per_round=4, private_audits_per_mode_seed=1,
                   proposal_backend="deterministic_repository_program_generator",
                   search_request=request("search",17), audit_request=request("audit",9017),
                   rows=pairs, row_count=len(pairs),
                   successful_rows=sum(r["result"]["status"]=="ok" for r in rows),
                   failed_rows=sum(r["result"]["status"]!="ok" for r in rows),
                   claim_boundary="evaluator-guided open Python program search on Bermudan; LLM search effect, novelty and out-of-suite superiority remain unobserved")
    (output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    lines=["# Bermudan Math AI workshop pilot (2026-09-05)","",
           "This bundle runs 3 seeds x 2 public rounds for both a deterministic",
           "Context-to-Proposal path and a direct-generation path. Every round",
           "executes two operator classes and two guided/control pairs. Each",
           "control uses the same baseline source, candidate seed, data request,",
           "and compute cap. One private hidden audit follows each mode/seed.","",
           "The four executed operators are whole_program_restart, ast_mutation,",
           "ast_crossover, and subsystem_rewrite. The Context path uses observed",
           "round-one effects to choose its round-two composition parent. The",
           "direct path follows a frozen parent schedule.","",
           "This is a deterministic pilot of the protocol, not an estimate of an",
           "LLM search effect. The full metrics preserve source/model digests,",
           "training path and payoff hashes, evaluator target-stream hashes,",
           "fit time, independent replay and causal lookahead probes. Candidate",
           "internal continuation targets are explicitly unobserved.","",
           f"Rows: {len(rows)}; public pairs: {len(pairs)}; failures: {summary['failed_rows']}.",
           "The claim remains evaluator-guided open Python program search on Bermudan.",
           "manifest.json records source digests, request matrix, and rebuild command."]
    (output/"EXPERIMENT_SUMMARY.md").write_text("\n".join(lines)+"\n")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    candidate_digests = {
        family: {
            name: file_digest(FAMILIES / family / name)
            for name in ("algorithm.py", "manifest.json")
        }
        for family in sorted(p.name for p in FAMILIES.iterdir() if p.is_dir())
    }
    manifest = {
        "schema": "openhyra-bermudan-workshop-manifest.v1",
        "repository": "OpenHyra",
        "git_revision": revision,
        "runner": {
            "path": "experiments/run_bermudan_workshop.py",
            "sha256": file_digest(Path(__file__)),
        },
        "evaluator": {
            "path": "tasks/bermudan_optimal_stopping/evaluator.py",
            "sha256": file_digest(ROOT / "tasks/bermudan_optimal_stopping/evaluator.py"),
        },
        "candidate_families": candidate_digests,
        "matrix": {
            "seeds": list(SEEDS), "rounds": 2, "modes": list(MODES),
            "operators_by_round": [
                ["whole_program_restart", "ast_mutation"],
                ["ast_crossover", "subsystem_rewrite"],
            ],
            "pairs_per_round": 2,
            "private_audits_per_mode_seed": 1,
        },
        "requests": {
            "search": request("search", 17),
            "audit": request("audit", 9017),
        },
        "artifacts": {
            "candidate_ledger.jsonl": file_digest(output / "candidate_ledger.jsonl"),
            "summary.json": file_digest(output / "summary.json"),
            "EXPERIMENT_SUMMARY.md": file_digest(output / "EXPERIMENT_SUMMARY.md"),
        },
        "rebuild": {
            "command": "python3 experiments/run_bermudan_workshop.py --output artifacts/mathai-bermudan-workshop-20260905",
            "generator_is_deterministic": True,
            "llm_search_effect_observed": False,
        },
    }
    (output/"manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=ROOT/"artifacts"/"mathai-bermudan-workshop-20260905")
    a=p.parse_args(); run(a.output); print(a.output)
if __name__=="__main__": main()
