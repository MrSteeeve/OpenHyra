#!/usr/bin/env python3
"""Frozen-evaluator supplement; source programs execute only through Seatbelt."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, platform, re, subprocess, sys, time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
OLD = ROOT / "artifacts/mathai-bermudan-live-v2-20260905"
FROZEN = OLD / "reproduction"
sys.path.insert(0, str(FROZEN))
import numpy as np
import proposal_agent
from llm_backend import run_agent
from sandbox import NUMERIC_THREAD_ENV, snapshot_source_tree, source_tree_hash
SEEDS = [17, 29, 41]
ARMS = ["structured_memory", "score_only"]
OPS = ["whole_program_restart", "subsystem_rewrite"]
FAMILIES = sorted(p.name for p in (FROZEN / "tasks/bermudan_python_search/research_candidates").iterdir() if p.is_dir())
EVAL = FROZEN / "tasks/bermudan_optimal_stopping/evaluator.py"
PROTOCOL_PATH = OUT / "protocol.json"

def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
def save(p, x, immutable=False):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if immutable and p.exists():
        if canonical(load(p)) != canonical(x): raise RuntimeError(f"Changed immutable artifact: {p}")
        return
    tmp=p.with_suffix(p.suffix+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
    tmp.replace(p)

def request(role,seed):
    if role=="legacy": return load(OLD/f"audits/context_proposal-s{seed}.request.json")
    cfg=dict(instance_count=2,repeats=1,training_paths=256,pricing_paths=1024,
             outer_paths=128,inner_paths=8,training_timeout_s=30,
             independent_validation=False,research_mode=True)
    evaluation_seed=330000+seed
    if role=="holdout":
        cfg.update(instance_count=4,repeats=1,training_paths=512,pricing_paths=2048,
                   outer_paths=256,inner_paths=16,independent_validation=True)
        evaluation_seed=660000+seed
    return dict(schema="openhyra-evaluation-request.v1",stage="audit",
                protocol="bermudan-python-program-search.v1",task="bermudan_python_search",
                suite_id="bermudan-python-hidden-v2",seed=evaluation_seed,config=cfg)

def freeze_protocol():
    files=[p for p in FROZEN.rglob("*.py") if "__pycache__" not in p.parts]
    design={
      "schema":"openhyra-paper-supplement.v1",
      "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
      "frozen_original_manifest_sha256":sha(OLD/"manifest.json"),
      "frozen_source_hashes":{str(p.relative_to(FROZEN)):sha(p) for p in files},
      "driver_sha256":sha(__file__),"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),
      "seeds":SEEDS,"baseline_families":FAMILIES,
      "calibration_requests":{str(s):request("calibration",s) for s in SEEDS},
      "holdout_requests":{str(s):request("holdout",s) for s in SEEDS},
      "legacy_completion":"All seven references on original hidden requests; retrospective, not a new untouched test.",
      "feedback_ablation":{
        "arms":ARMS,"operators":OPS,"unit":"seed x operator; 6 matched pairs, 3 independent seed blocks",
        "model":"gpt-5.6-sol","backend":"codex","context_timeout_s":120,"proposal_timeout_s":180,
        "calls_per_attempt":"one Context call and one Proposal call; no retries",
        "parent":"Same best calibration-gap reference among linear_ridge/mlp/residual_hybrid for both arms and both operators.",
        "history":"Same frozen union of original successful generated programs and references, re-evaluated on calibration panel.",
        "treatment":"Structured memory adds per-instance bounds/uncertainty and labeled historical behavior/mechanism records; score-only keeps IDs, source digests, scalar gaps, status.",
        "context_output":"Same short JSON schema and 500-word instruction in both arms.",
        "budget":"Identical call counts and timeout/evaluator caps, not equal realized tokens or runtime.",
        "scope":"Fixed-history, fixed-parent feedback ablation; not a new multiround recursive search or an unrestricted architecture superiority test.",
        "failures":"Retain all failures; no repairs; failed attempts cannot replace incumbent.",
        "selection":"Minimum calibration confidence-gap score, including parent as deployable incumbent; best generated tracked separately.",
        "holdout":"Evaluate all frozen generated programs and references after all arms finish and selections freeze; no holdout feedback or reselection.",
        "analysis":"Per-seed paired differences, mean across two predefined operators within seed. Three seed blocks are exploratory; no population significance claim."},
      "data_roles":"Evaluator API calls all dual evaluations audit; calibration is public search feedback. Its seeds and instances are disjoint from final holdout.",
      "frozen_upper":"One evaluator-owned Ridge LSMC dual construction shared by all Python policies under each identical request.",
      "trajectory":"Original public-score-selected incumbents re-evaluated on calibration dual panel. Post-hoc diagnostic, not original gap feedback.",
      "report_all":"No score-based exclusion or hidden-based winner selection; include original timeouts and seed incumbents."}
    if PROTOCOL_PATH.exists(): design["created_utc"]=load(PROTOCOL_PATH)["created_utc"]
    save(PROTOCOL_PATH,design,immutable=True)

def verify_frozen():
    p=load(PROTOCOL_PATH)
    if sha(__file__)!=p["driver_sha256"]: raise RuntimeError("Driver changed after preregistration")
    for rel,h in p["frozen_source_hashes"].items():
        if sha(FROZEN/rel)!=h: raise RuntimeError(f"Frozen runtime changed: {rel}")

def snap(source,key):
    dest=OUT/"programs"/key; h=source_tree_hash(source,1024*1024)[0]
    if not dest.exists(): snapshot_source_tree(source,dest,1024*1024)
    if source_tree_hash(dest,1024*1024)[0]!=h: raise RuntimeError(f"Snapshot mismatch: {key}")
    return dest

def evaluate(source,req,destination):
    destination=Path(destination); digest=source_tree_hash(source,1024*1024)[0]
    if destination.exists():
        result=load(destination)
        if result["source_digest"]!=digest or canonical(result["request"])!=canonical(req):
            raise RuntimeError(f"Cached provenance mismatch: {destination}")
        return result
    destination.parent.mkdir(parents=True,exist_ok=True)
    rp=destination.with_suffix(".request.json"); save(rp,req,immutable=True)
    start=time.perf_counter(); stderr=""
    try:
        done=subprocess.run([sys.executable,str(EVAL),str(source),str(rp)],cwd=FROZEN,
            env={**os.environ,**NUMERIC_THREAD_ENV},text=True,capture_output=True,timeout=480)
        stderr=done.stderr; payload=json.loads(done.stdout)
        if done.returncode or "error" in payload: raise RuntimeError(payload.get("error",stderr[-2500:]))
        result=dict(status="ok",score=payload["score"],metrics=payload["metrics"],evidence=payload["evidence"],failure_reason=None)
    except Exception as exc:
        result=dict(status="error",score=None,metrics={},evidence={},failure_reason=f"{type(exc).__name__}: {exc}"[-4000:])
    result.update(request=req,source_digest=digest,source_unchanged=source_tree_hash(source,1024*1024)[0]==digest,
                  wall_seconds=time.perf_counter()-start,evaluator_sha256=sha(EVAL))
    if not result["source_unchanged"]: raise RuntimeError("Evaluator source input mutated")
    save(destination,result,immutable=True)
    destination.with_suffix(".stderr.txt").write_text(stderr)
    print(f"EVAL {destination.relative_to(OUT)} {result['status']} score={result['score']} {result['wall_seconds']:.1f}s",flush=True)
    return result

def old_runs():
    return [load(OLD/f"runs/{mode}-s{s}/public_complete.json") for s in SEEDS for mode in ["context_proposal","direct_generation"]]

def catalog():
    result=[]
    for seed in SEEDS:
        for family in FAMILIES:
            key=f"reference-s{seed}-{family}"
            dest=snap(FROZEN/"tasks/bermudan_python_search/research_candidates"/family,key)
            result.append(dict(key=key,seed=seed,kind="reference",method=family,program_path=str(dest.relative_to(OUT)),
                original_public_result=str(OLD/f"references/s{seed}/{family}.json")))
    for run in old_runs():
        for row in run["rows"]:
            if row["matched_arm"]!="guided" or row["result"]["status"]!="ok": continue
            key=f"legacy-{run['mode']}-s{run['seed']}-{row['candidate_id']}"
            src=Path(row["program_path"])
            if not src.is_absolute(): src=OLD/src
            dest=snap(src,key)
            result.append(dict(key=key,seed=run["seed"],kind="legacy_generated",method=run["mode"],
                candidate_id=row["candidate_id"],round=row["round"],
                is_original_winner=row["candidate_id"]==run["winner"]["candidate_id"],
                program_path=str(dest.relative_to(OUT)),original_row=row))
    save(OUT/"catalog.json",result,immutable=True); return result

def prep(workers=2):
    freeze_protocol(); items=catalog(); jobs=[]
    for item in items:
        src=OUT/item["program_path"]
        if item["kind"]=="reference": jobs.append((src,request("legacy",item["seed"]),OUT/"raw/legacy_completion"/f"{item['key']}.json"))
        jobs.append((src,request("calibration",item["seed"]),OUT/"raw/calibration"/f"{item['key']}.json"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(evaluate,*job) for job in jobs]
        for f in concurrent.futures.as_completed(futures): f.result()
    for seed in SEEDS:
        rows=[]
        for item in items:
            if item["seed"]!=seed: continue
            ev=load(OUT/"raw/calibration"/f"{item['key']}.json")
            base=dict(id=item["key"],source_digest=ev["source_digest"],status=ev["status"],gap_score=ev["score"])
            details=dict(per_instance=ev["metrics"].get("summaries",[]),evaluation_seconds=ev["wall_seconds"],failure=ev["failure_reason"])
            if "original_row" in item:
                original=item["original_row"]; details["historical_mechanism"]=original["hypothesis"]
                details["historical_public_behavior"]=[{k:v for k,v in cell.items() if k in {
                    "instance_id","candidate_exercise_rate","candidate_stop_time_mean",
                    "paired_normalized_improvement","paired_normalized_standard_error"}}
                    for cell in original["result"]["metrics"].get("summaries",[])]
                details["historical_behavior_scope"]="Original public suite; distinct from calibration dual panel."
            rows.append({**base,"details":details})
        choices=[r for r in rows if r["status"]=="ok" and any(r["id"]==f"reference-s{seed}-{name}" for name in ["linear_ridge","mlp","residual_hybrid"])]
        parent=min(choices,key=lambda r:(r["gap_score"],r["id"]))
        item=next(i for i in items if i["key"]==parent["id"])
        save(OUT/f"ablation/s{seed}/frozen_parent.json",item,immutable=True)
        for arm in ARMS:
            packet=rows if arm=="structured_memory" else [{k:v for k,v in r.items() if k!="details"} for r in rows]
            save(OUT/f"ablation/s{seed}/{arm}/memory.json",packet,immutable=True)
    paths=[p for p in (OUT/"raw/calibration").glob("*.json") if not p.name.endswith(".request.json")]
    save(OUT/"preparation_complete.json",{"evaluations":len(jobs),"items":len(items),"all_ok":all(load(p)["status"]=="ok" for p in paths)},immutable=True)
    print("PREPARATION COMPLETE",flush=True)

class Logger:
    def __init__(self,path):
        self.path=path; path.mkdir(parents=True,exist_ok=True); self.calls=[]
    def __call__(self,prompt,**kwargs):
        idx=len(self.calls); start=time.perf_counter(); kwargs["model"]="gpt-5.6-sol"; kwargs["backend"]="codex"
        try: r=run_agent(prompt,**kwargs)
        except subprocess.TimeoutExpired as exc:
            def dec(x): return x.decode(errors="replace") if isinstance(x,bytes) else x or ""
            r=subprocess.CompletedProcess([],124,dec(exc.stdout),dec(exc.stderr))
        except Exception as exc: r=subprocess.CompletedProcess([],125,"",f"{type(exc).__name__}: {exc}")
        usage=re.findall(r"tokens used\s*\n([\d,]+)",r.stderr or "")
        model=re.findall(r"^model:\s*(.+)$",r.stderr or "",re.M)
        rec=dict(id=f"call-{idx:02d}",role="proposal" if kwargs.get("writable") else "context",
            model=model[-1] if model else "gpt-5.6-sol",prompt=prompt,stdout=r.stdout,returncode=r.returncode,
            tokens_reported=int(usage[-1].replace(",","")) if usage else None,
            wall_seconds=time.perf_counter()-start,timeout_s=kwargs["timeout_s"])
        save(self.path/f"call-{idx:02d}.json",rec,immutable=True)
        (self.path/f"call-{idx:02d}.stderr.txt").write_text(r.stderr or ""); self.calls.append(rec)
        print(f"CALL {self.path.relative_to(OUT)} {rec['role']} rc={r.returncode} {rec['wall_seconds']:.1f}s",flush=True)
        return r

def parse_json(text):
    text=text.strip(); fence=chr(96)*3
    if text.startswith(fence):
        text=text[len(fence):].lstrip()
        if text.startswith("json"): text=text[4:].lstrip()
        if text.endswith(fence): text=text[:-len(fence)].rstrip()
    return json.loads(text)

def ablation(seed):
    verify_frozen()
    if not (OUT/"preparation_complete.json").exists(): raise RuntimeError("Complete frozen calibration first")
    parent=load(OUT/f"ablation/s{seed}/frozen_parent.json"); source=OUT/parent["program_path"]
    parent_code=(source/"algorithm.py").read_text()
    contract=(FROZEN/"tasks/bermudan_python_search/TASK.md").read_text().split("## Search semantics")[0]
    for oi,op in enumerate(OPS):
        order=ARMS if (SEEDS.index(seed)+oi)%2==0 else list(reversed(ARMS))
        for arm in order:
            run=OUT/f"ablation/s{seed}/{arm}/{op}"
            if (run/"record.json").exists(): continue
            if (run/"started.json").exists(): raise RuntimeError(f"Interrupted attempt retained; do not re-call silently: {run}")
            run.mkdir(parents=True,exist_ok=True)
            save(run/"started.json",{"seed":seed,"arm":arm,"operator":op},immutable=True)
            logger=Logger(run/"calls"); mp=run.parent/"memory.json"; memory=load(mp)
            prompt=("You are the Context Agent for a controlled Bermudan optimal-stopping program experiment. "
              "Use only the information below. Do not read files, run tools, or use a browser. "
              "Minimize normalized primal-dual confidence-gap score. The evaluator's common upper-bound "
              "construction is fixed across candidates, so improving expected discounted payoff reduces "
              "the true gap. Propose one concrete algorithmic intervention under the assigned operator "
              "and fixed parent. There is no registered model-family restriction. "
              "Budget: 256 training paths, 30 seconds of fitting per instance. Return JSON only, at most "
              "500 words, with keys mechanism, implementation, expected_behavior, failure_condition. "
              "A previous score does not prove a causal mechanism.\n"
              f"Operator: {op}; scope: {'fit' if op=='subsystem_rewrite' else 'whole_program'}.\n"
              "Memory records:\n"+json.dumps(memory,ensure_ascii=False)+"\nParent program:\n"+parent_code)
            cwd=run/"context_workspace"; cwd.mkdir()
            c=logger(prompt,cwd=cwd,writable=False,timeout_s=120)
            ok=c.returncode==0; plan={}; failure=None
            if ok:
                try:
                    plan=parse_json(c.stdout)
                    if not isinstance(plan,dict) or not isinstance(plan.get("implementation"),str): raise ValueError("Missing implementation string")
                except Exception as exc: ok,failure=False,f"invalid Context JSON: {exc}"
            else: failure=f"Context call failed ({c.returncode})"
            save(run/"context.json",{"plan":plan,"memory_sha256":sha(mp)},immutable=True)
            metadata={}; draft=run/"draft"
            if ok:
                proposal_agent.run_agent=logger
                intervention=dict(intervention_operator=op,intervention_scope="fit" if op=="subsystem_rewrite" else "whole_program",
                    mechanism=plan.get("mechanism",""),prediction=plan.get("expected_behavior",""),
                    falsifier=plan.get("failure_condition",""),target_slice="all")
                pp=(contract+"\nImplement the Context plan below as an executable program. Optimize actual "
                    "causal stopping decisions for expected discounted payoff, measured by primal-dual "
                    "confidence gap. Do not alter evaluation or read outside this draft directory. "
                    "Use only Python standard library and NumPy. Fit cap: 30 seconds; training paths: 256. "
                    "Queries contain only causal history and current state. Use terminal/files only to edit "
                    "this draft; do not search the repository or access network. Do not execute the generated "
                    "candidate here: the independent evaluator performs execution.\nContext plan:\n"+json.dumps(plan,ensure_ascii=False))
                ok,description=proposal_agent.propose(source,draft,pp,["algorithm.py","manifest.json"],
                    timeout_s=180,backend="codex",model="gpt-5.6-sol",candidate_mode="python_program",
                    entrypoint="algorithm.py",artifact_protocol="openhyra-python-program.v1",
                    source_files=["algorithm.py","manifest.json"],intervention=intervention,execution_metadata=metadata)
                if not ok: failure=description
            program=run/"program"
            if draft.exists(): snapshot_source_tree(draft,program,1024*1024)
            if ok: ev=evaluate(program,request("calibration",seed),run/"calibration.json")
            else:
                ev=dict(status="error",score=None,metrics={},evidence={},failure_reason=failure,wall_seconds=0,
                    source_digest=source_tree_hash(program,1024*1024)[0] if program.exists() else None,
                    source_unchanged=True,request=request("calibration",seed))
                save(run/"calibration.json",ev,immutable=True)
            rec=dict(seed=seed,arm=arm,operator=op,parent_key=parent["key"],
                parent_source_digest=source_tree_hash(source,1024*1024)[0],memory_sha256=sha(mp),
                context_plan=plan,operator_execution=metadata,program_path=str(program.relative_to(OUT)) if program.exists() else None,
                calls=[{k:v for k,v in x.items() if k not in {"prompt","stdout"}} for x in logger.calls],result=ev)
            save(run/"record.json",rec,immutable=True)
            print(f"ATTEMPT seed={seed} arm={arm} op={op} {ev['status']} {ev['score']}",flush=True)
    print(f"ABLATION SEED {seed} COMPLETE",flush=True)

def freeze_selection():
    selections=[]
    for seed in SEEDS:
        parent=load(OUT/f"ablation/s{seed}/frozen_parent.json")
        pr=load(OUT/"raw/calibration"/f"{parent['key']}.json")
        for arm in ARMS:
            attempts=[load(OUT/f"ablation/s{seed}/{arm}/{op}/record.json") for op in OPS]
            valid=[a for a in attempts if a["result"]["status"]=="ok"]
            best=min(valid,key=lambda a:(a["result"]["score"],a["operator"])) if valid else None
            inc=best if best and best["result"]["score"]<pr["score"] else None
            path=inc["program_path"] if inc else parent["program_path"]
            selections.append(dict(seed=seed,arm=arm,parent_key=parent["key"],selected_program_path=path,
                selected_operator=inc["operator"] if inc else "initial_reference",
                best_generated_program_path=best["program_path"] if best else None,
                best_generated_operator=best["operator"] if best else None,
                calibration_score=inc["result"]["score"] if inc else pr["score"],
                source_digest=source_tree_hash(OUT/path,1024*1024)[0]))
    save(OUT/"selections_frozen.json",{"protocol_sha256":sha(PROTOCOL_PATH),
        "rule":"calibration only; parent included; no holdout reselection","selections":selections},immutable=True)

def holdout(workers=2):
    verify_frozen(); freeze_selection(); index=[]
    for item in load(OUT/"catalog.json"):
        if item["kind"]=="reference" or item.get("is_original_winner"):
            index.append({k:v for k,v in item.items() if k!="original_row"})
    for seed in SEEDS:
        for arm in ARMS:
            for op in OPS:
                rec=load(OUT/f"ablation/s{seed}/{arm}/{op}/record.json")
                if rec["result"]["status"]=="ok":
                    index.append(dict(key=f"ablation-{arm}-s{seed}-{op}",seed=seed,kind="ablation",
                        method=arm,operator=op,program_path=rec["program_path"]))
    save(OUT/"holdout_index.json",index,immutable=True)
    jobs=[(OUT/i["program_path"],request("holdout",i["seed"]),OUT/"raw/holdout"/f"{i['key']}.json") for i in index]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(evaluate,*job) for job in jobs]
        for f in concurrent.futures.as_completed(fs): f.result()
    save(OUT/"experiments_complete.json",{"n_holdout_programs":len(index),
        "selections_sha256":sha(OUT/"selections_frozen.json"),"protocol_sha256":sha(PROTOCOL_PATH)},immutable=True)
    print("FINAL HOLDOUT COMPLETE",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=["prepare","ablation","holdout"])
    p.add_argument("--seed",type=int,choices=SEEDS); p.add_argument("--workers",type=int,default=2); a=p.parse_args()
    if a.phase=="prepare": prep(a.workers)
    elif a.phase=="ablation": ablation(a.seed)
    else: holdout(a.workers)
