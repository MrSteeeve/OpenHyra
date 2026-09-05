#!/usr/bin/env python3
"""Real model Context -> Proposal and direct Proposal experiments on Bermudan.

Resumable artifacts, real operator dispatch, fresh subprocess evaluator, fixed
public requests, and private audits only after all public selections freeze.
Use --replay to reconstruct numerical results from the exported program trees
without calling the model again. The deterministically seeded fit contract does
not imply deterministic LLM generation; every LLM response is frozen instead.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import context_agent
import proposal_agent
from bermudan_research import paired_slice_effects
from eb import ExperienceBank
from harness import Task, _mechanism_slots, _write_prediction_observation, _finalize_matched_controls, _secondary_program_parent
from llm_backend import run_agent
from sandbox import NUMERIC_THREAD_ENV, snapshot_source_tree, source_tree_hash
from stopping import ContextDecision
from tasks.bermudan_optimal_stopping import evaluator

SEEDS = (17, 29, 41)
MODES = ("context_proposal", "direct_generation")
OPERATORS = (("whole_program_restart", "ast_mutation"), ("ast_crossover", "subsystem_rewrite"))
FAMILIES = ROOT / "tasks/bermudan_python_search/research_candidates"
EVAL_PATH = ROOT / "tasks/bermudan_optimal_stopping/evaluator.py"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(stage, seed):
    cfg = dict(instance_count=2, repeats=2 if stage == "search" else 1,
               training_paths=256, pricing_paths=512, training_timeout_s=30,
               independent_validation=stage == "audit", research_mode=True)
    if stage == "search": cfg["public_pathwise_samples"] = True
    else: cfg.update(outer_paths=128, inner_paths=4)
    return dict(schema=evaluator.REQUEST_SCHEMA, stage=stage, task="bermudan_python_search",
                protocol=evaluator.PYTHON_PROGRAM_TASK_PROTOCOL, seed=int(seed),
                suite_id="bermudan-python-" + ("public" if stage == "search" else "hidden") + "-v2", config=cfg)


def evaluate(source, req, destination):
    """A fresh trusted evaluator process; no candidate imports in this driver."""
    if destination.exists(): return read(destination)
    req_path = destination.with_suffix(".request.json"); save(req_path, req)
    source_digest = source_tree_hash(source, 1024 * 1024)[0]
    started = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, str(EVAL_PATH), str(source), str(req_path)],
                           cwd=ROOT, env={**os.environ, **NUMERIC_THREAD_ENV}, text=True,
                           capture_output=True, timeout=300)
        payload = json.loads(r.stdout)
        if r.returncode or "error" in payload:
            raise ValueError(payload.get("error", r.stderr[-2000:]))
        result = dict(status="ok", score=payload["score"], metrics=payload["metrics"],
                      evidence=payload["evidence"], failure_reason=None)
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        result = dict(status="error", score=None, metrics={}, evidence={},
                      failure_reason=f"{type(exc).__name__}: {exc}"[-4000:])
    result.update(wall_seconds=time.perf_counter()-started, source_digest=source_digest,
                  source_unchanged=source_tree_hash(source, 1024 * 1024)[0] == source_digest,
                  request=req)
    save(destination, result)
    return result


class CallLogger:
    def __init__(self, root): self.root, self.calls = root, []
    def __call__(self, prompt, **kwargs):
        index = len(self.calls); started = time.perf_counter()
        try:
            response = run_agent(prompt, **kwargs)
        except subprocess.TimeoutExpired as exc:
            def decode(v): return v.decode(errors="replace") if isinstance(v, bytes) else v or ""
            response = subprocess.CompletedProcess([], 124, decode(exc.stdout), decode(exc.stderr))
        stderr = response.stderr or ""
        tokens = re.findall(r"tokens used\s*\n([\d,]+)", stderr)
        models = re.findall(r"^model:\s*(.+)$", stderr, flags=re.M)
        record = {"id": f"call-{index:03d}", "prompt": prompt,
                  "stdout": response.stdout, "returncode": response.returncode,
                  "model": models[-1] if models else kwargs.get("model"),
                  "tokens_reported": int(tokens[-1].replace(",", "")) if tokens else None,
                  "wall_seconds": time.perf_counter()-started,
                  "role": "proposal" if kwargs.get("writable") else "context"}
        # Preserve the actual backend output for budget/model verification.
        (self.root / f"call-{index:03d}.stderr.txt").write_text(stderr)
        save(self.root / f"call-{index:03d}.json", record)
        self.calls.append(record)
        return response


def make_row(mode, seed, iteration, slot, record, result, item, call_records):
    return {"schema": "openhyra-bermudan-live-candidate.v1", "mode": mode, "seed": seed,
            "round": iteration, "candidate_id": record["id"], "pair_id": item["matched_pair_id"],
            "matched_arm": item["matched_arm"], "candidate_seed": item["candidate_seed"],
            "hypothesis": slot["mechanism"], "proposal": {
                "operator": slot["mechanism"]["intervention_operator"] if item["matched_arm"] == "guided" else "identity_control",
                "execution": item.get("operator_execution", {}), "source_digest": result["source_digest"],
                "parent_ids": [item["parent"]["id"]] + ([item["secondary_parent"]["id"]] if item.get("secondary_parent") else []),
                "parent_digest": source_tree_hash(Path(item["parent"]["path"]), 1024*1024)[0],
                "call_ids": [c["id"] for c in call_records],
                "generation_seconds": sum(c["wall_seconds"] for c in call_records),
                "tokens_reported": sum(c["tokens_reported"] or 0 for c in call_records)},
            "program_path": str(Path(record["path"]).resolve()), "result": result}


def public_run(output, seed, mode, backend):
    run_dir = output / "runs" / f"{mode}-s{seed}"
    if (run_dir / "public_complete.json").exists(): return read(run_dir / "public_complete.json")
    if (run_dir / "eb/records.jsonl").exists():
        raise RuntimeError(f"Incomplete public run retained at {run_dir}; inspect it instead of silently restarting model calls")
    run_dir.mkdir(parents=True, exist_ok=True)
    calls_dir = run_dir / "agent_calls"; calls_dir.mkdir(exist_ok=True)
    logger = CallLogger(calls_dir)
    context_agent.run_agent = logger; proposal_agent.run_agent = logger
    task = Task("bermudan_python_search", "live-workshop")
    task.run_dir = run_dir; task.candidates_per_context = 4
    bank = ExperienceBank(run_dir / "eb", direction="max")
    for family in ("linear_ridge", "mlp", "residual_hybrid"):
        ref = read(output / "references" / f"s{seed}" / f"{family}.json")
        if ref["status"] == "ok":
            bank.commit(FAMILIES/family, ref["score"], "ok", f"shared reference {family}", None, "", metrics=ref["metrics"])
    frozen_parent = bank.best()
    if frozen_parent is None: raise RuntimeError("no successful reference baseline")
    all_rows = []
    for iteration, operators in enumerate(OPERATORS):
        print(f"public {mode} seed={seed} round={iteration} Context/Proposal", flush=True)
        base_description = (task.dir / "TASK.md").read_text().split("## Workshop evidence loop")[0]
        task.description = base_description + "\nFor this preregistered experimental round, return exactly two mechanism_candidates using, respectively: " + ", ".join(operators) + ". Each intervention_operator must be explicit; subsystem_rewrite must use intervention_scope fit or predict. Target an observed instance slice using its exact instance:ID label. Predict positive guided-minus-unchanged-parent mean discounted payoff on that slice; an upper 95% CI below zero refutes the direction. You may invent any complete finite algorithm. Use completed evidence to choose revise, compose, restart or falsify and cite exact record IDs. Do not read files or call tools. Return the requested JSON only."
        before_context = len(logger.calls)
        if mode == "context_proposal":
            decision, parent, prompt, _, meta = context_agent.build_inspiration(task, bank, iteration, backend=backend, trial_seed=seed)
            if len(logger.calls) == before_context or logger.calls[-1]["returncode"] != 0:
                raise RuntimeError("real Context call failed; protocol fallback is not counted as model evidence")
            if not decision.mechanism_candidates:
                raise RuntimeError("Context did not produce mechanisms; inspect recorded response")
            slots = _mechanism_slots(task, meta, iteration, parent, 4)
        else:
            parent = frozen_parent
            mechanisms = [{"id": f"direct_r{iteration}_{i}", "family": "open_program_generation",
                           "mechanism": "Use the declared operation to improve the parent program.",
                           "prediction": "positive mean paired payoff on the declared slice",
                           "failure_condition": "upper 95% paired CI is below zero",
                           "matched_control": "unchanged parent with identical input and budget",
                           "target_slice": "instance:public-put-atm", "intervention_scope": "fit" if op == "subsystem_rewrite" else "whole_program",
                           "intervention_operator": op} for i, op in enumerate(operators)]
            meta = {"trial_seed": seed, "mechanism_candidates": mechanisms, "selected_mechanism_candidates": mechanisms}
            slots = _mechanism_slots(task, meta, iteration, parent, 4)
            decision = None
            prompt = base_description + "\nGenerate the assigned complete finite Python candidate directly. No Context analysis or prior search outcomes are supplied. Improve expected discounted payoff under the unchanged evaluator budget. The registered family names do not restrict the algorithms you may write."
        save(run_dir / f"context-{iteration}.json", {"decision": decision.to_dict() if decision else None,
             "metadata": meta, "slots": slots, "parent_id": parent["id"], "proposal_prompt": prompt,
             "context_call_ids": [c["id"] for c in logger.calls[before_context:]],
             "context_seconds": sum(c["wall_seconds"] for c in logger.calls[before_context:])})
        completed = []
        secondary = _secondary_program_parent(bank, parent, "max")
        for i, slot in enumerate(slots):
            print(f"public {mode} seed={seed} round={iteration} slot={i} {slot['matched_arm']}", flush=True)
            draft = run_dir / "drafts" / f"r{iteration}-c{i}"
            item = {**slot, "iteration": iteration, "parent": parent,
                    "candidate_seed": slot["matched_seed"], "operator_execution": {}}
            intervention = {**slot["mechanism"], "matched_arm": slot["matched_arm"], "slot": slot["matched_seed"]}
            if intervention["intervention_operator"] == "ast_crossover" and secondary:
                item["secondary_parent"] = secondary
                intervention["secondary_parent_path"] = secondary["path"]
            calls_before = len(logger.calls)
            if slot["matched_arm"] == "control":
                proposal_agent.prepare_draft(Path(parent["path"]), draft)
                ok, description = True, "identity control; candidate generation not charged as an LLM call"
            else:
                ok, description = proposal_agent.propose(Path(parent["path"]), draft, prompt,
                    ["algorithm.py", "manifest.json"], timeout_s=180, backend=backend,
                    candidate_mode="python_program", entrypoint="algorithm.py",
                    artifact_protocol="openhyra-python-program.v1", source_files=["algorithm.py", "manifest.json"],
                    intervention=intervention, execution_metadata=item["operator_execution"])
            sealed = run_dir / "programs" / f"r{iteration}-c{i}"
            snapshot_source_tree(draft, sealed, 1024*1024)
            result_path = run_dir / "evaluations" / f"r{iteration}-c{i}.json"
            if ok:
                result = evaluate(sealed, request("search", seed), result_path)
            else:
                result = {"status": "error", "score": None, "metrics": {}, "failure_reason": description,
                          "evidence": {}, "wall_seconds": 0., "source_digest": source_tree_hash(sealed,1024*1024)[0],
                          "source_unchanged": True, "request": request("search",seed)}
                save(result_path, result)
            result["metrics"]["source_snapshot_sha256"] = result["source_digest"]
            record = bank.commit(sealed, result["score"], result["status"], description, parent["id"],
                                 result["failure_reason"] or "", metrics=result["metrics"], metadata=item)
            _write_prediction_observation(task, item, record, {**result, "log_tail": result["failure_reason"] or ""})
            completed.append({"item": item, "records": [record]})
            row = make_row(mode, seed, iteration, slot, record, result, item, logger.calls[calls_before:])
            all_rows.append(row); save(run_dir / "candidate_rows.json", all_rows)
        _finalize_matched_controls(task, iteration, completed, bank)
    successes = [r for r in all_rows if r["matched_arm"] == "guided" and r["result"]["status"] == "ok"]
    winner = max(successes, key=lambda r:r["result"]["score"]) if successes else None
    result = {"mode": mode, "seed": seed, "rows": all_rows, "winner": winner,
              "selection": "maximum fixed-request public score among successful guided candidates",
              "agent_calls": [{k:v for k,v in c.items() if k not in {"prompt", "stdout"}} for c in logger.calls]}
    save(run_dir / "public_complete.json", result)
    return result


def summarize(output, runs, audits):
    pairs = []
    for run in runs:
        for i in range(0, len(run["rows"]), 2):
            g, c = run["rows"][i:i+2]
            paired = paired_slice_effects(g["result"]["metrics"], c["result"]["metrics"], g["hypothesis"].get("target_slice", ""))
            if g["result"]["status"] != "ok" or c["result"]["status"] != "ok":
                paired.update(prediction_verdict="execution_failed", next_action="falsify")
            paired.update(mode=run["mode"], seed=run["seed"], round=g["round"],
                          pair_id=g["pair_id"], hypothesis=g["hypothesis"], guided_id=g["candidate_id"],
                          control_id=c["candidate_id"], operator=g["proposal"]["operator"],
                          same_parent=g["proposal"]["parent_digest"]==c["proposal"]["parent_digest"],
                          same_seed=g["candidate_seed"]==c["candidate_seed"],
                          same_request=g["result"]["request"]==c["result"]["request"])
            pairs.append(paired)
    by_mode = {}
    for mode in MODES:
        rows = [r for run in runs if run["mode"]==mode for r in run["rows"]]
        p = [r for r in pairs if r["mode"]==mode]
        calls = [c for run in runs if run["mode"]==mode for c in run["agent_calls"]]
        by_mode[mode] = {"candidate_count": len(rows), "failure_count": sum(r["result"]["status"]!="ok" for r in rows),
                        "prediction_supported_rate": sum(r.get("prediction_verdict")=="supported" for r in p)/len(p),
                        "prediction_direction_hit_rate": sum(r.get("prediction_direction_correct",False) for r in p)/len(p),
                        "prediction_denominator_includes_failures": len(p),
                        "training_seconds": sum(c["fit_wall_seconds"] for r in rows for c in r["result"]["metrics"].get("training_cells",[])),
                        "evaluator_wall_seconds": sum(r["result"]["wall_seconds"] for r in rows),
                        "generation_wall_seconds": sum(c["wall_seconds"] for c in calls),
                        "model_tokens_reported": sum(c["tokens_reported"] or 0 for c in calls),
                        "private_audit_wall_seconds": sum(a["result"]["wall_seconds"] for a in audits if a["mode"]==mode)}
        by_mode[mode]["total_wall_cost_seconds"] = sum(by_mode[mode][k] for k in ("evaluator_wall_seconds","generation_wall_seconds","private_audit_wall_seconds"))
    result = {"schema":"openhyra-bermudan-live-summary.v1", "seeds":list(SEEDS), "modes":by_mode,
              "pairs":pairs, "private_audits":audits, "generation_backend":"real_model_via_llm_backend",
              "claim":"Bermudan-only evaluator-guided open program search; exploratory finite-budget evidence, no novelty or general superiority claim"}
    save(output/"summary.json",result)
    rows = [r for run in runs for r in run["rows"]]
    with (output/"candidate_ledger.jsonl").open("w") as f:
        for row in rows: f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    return result


def freeze_manifest(output, backend):
    # Export all tracked evaluator dependencies, candidates, and the runner.
    # Reproduction does not depend on a mutable checkout or model service.
    paths = subprocess.check_output(["git","ls-files","*.py","*.json","*.toml","*.md"],cwd=ROOT,text=True).splitlines()
    paths += ["bermudan_research.py", "experiments/run_bermudan_live_workshop.py"]
    code_root = output/"reproduction"; code_hashes = {}
    for rel in sorted(set(paths)):
        if rel.startswith(("artifacts/","runs/","tests/")): continue
        src=ROOT/rel
        if not src.is_file(): continue
        dest=code_root/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dest)
        code_hashes[rel]=file_hash(src)
    manifest = {"schema":"openhyra-bermudan-live-manifest.v1","git_base":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "code_sha256":code_hashes,"backend":backend,"python":sys.version,"numpy":np.__version__,
                "platform":sys.platform,"seeds":list(SEEDS),"rounds":2,"pairs_per_round":2,"modes":list(MODES),
                "operator_schedule":[list(x) for x in OPERATORS],
                "evaluation_budget":{"search":request("search",17)["config"],"audit":request("audit",9017)["config"]},
                "generation_budget":{"proposal_timeout_seconds":180,"context_timeout_seconds":240,
                                      "shared_reference_families":7,"identity_controls_make_no_model_calls":True},
                "private_policy":"all public candidates and winners frozen before any hidden audit",
                "generation_reproducibility":"LLM outputs frozen; generation API has no guaranteed sampling-seed replay",
                "numerical_replay":"python3 reproduction/experiments/run_bermudan_live_workshop.py --replay --output BUNDLE_ABSOLUTE_PATH"}
    save(output/"manifest.json",manifest)


def replay(output):
    reports=[]
    rows=[json.loads(l) for l in (output/"candidate_ledger.jsonl").read_text().splitlines()]
    audits=read(output/"summary.json")["private_audits"]
    for i,row in enumerate(rows+audits):
        old=row["result"]
        if old["status"]!="ok": continue
        original=Path(row["program_path"])
        # Program paths are made relative to the bundle when exported below.
        source=output/original
        result=evaluate(source,old["request"],output/"replay_results"/f"{i:03d}.json")
        same = result["status"]=="ok" and result["score"]==old["score"]
        same = same and [c["model_file_sha256"] for c in result["metrics"].get("training_cells",[])] == [c["model_file_sha256"] for c in old["metrics"].get("training_cells",[])]
        reports.append({"index":i,"score_and_models_equal":same})
        print("replay",i,same,flush=True)
    save(output/"replay_verification.json",{"rows":reports,"all_equal":bool(reports) and all(r["score_and_models_equal"] for r in reports)})


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--backend",default="codex");p.add_argument("--replay",action="store_true")
    a=p.parse_args(); output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if a.replay: replay(output); return
    if not (output/"manifest.json").exists(): freeze_manifest(output,a.backend)
    for seed in SEEDS:
        for family in sorted(x.name for x in FAMILIES.iterdir() if x.is_dir()):
            print("reference",seed,family,flush=True)
            evaluate(FAMILIES/family,request("search",seed),output/"references"/f"s{seed}"/f"{family}.json")
    runs=[public_run(output,seed,mode,a.backend) for mode in MODES for seed in SEEDS]
    save(output/"public_selection_frozen.json",[{"mode":r["mode"],"seed":r["seed"],"winner":r["winner"]["candidate_id"] if r["winner"] else None} for r in runs])
    audits=[]
    for run in runs:
        winner=run["winner"]
        if winner is None: raise RuntimeError("no successful generated candidate for audit")
        source=Path(winner["program_path"])
        print("private audit",run["mode"],run["seed"],flush=True)
        result=evaluate(source,request("audit",run["seed"]+9000),output/"audits"/f"{run['mode']}-s{run['seed']}.json")
        audits.append({"mode":run["mode"],"seed":run["seed"],"candidate_id":winner["candidate_id"],"program_path":str(source.relative_to(output)),"result":result})
    for run in runs:
        for row in run["rows"]: row["program_path"]=str(Path(row["program_path"]).relative_to(output))
    summarize(output,runs,audits)
    save(output/"artifact_hashes.json",{str(f.relative_to(output)):file_hash(f) for f in sorted(output.rglob("*")) if f.is_file() and f.name!="artifact_hashes.json"})
    print("complete",output,flush=True)

if __name__=="__main__": main()
