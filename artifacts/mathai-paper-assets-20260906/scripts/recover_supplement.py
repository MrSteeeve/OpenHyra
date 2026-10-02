#!/usr/bin/env python3
"""Resume pre-Proposal infrastructure failures without changing plans or scores."""
import argparse, concurrent.futures, os, time
from pathlib import Path
import run_supplement as base

OUT=base.OUT
ACTIVE=OUT/"ablation_recovered"
AMEND=OUT/"recovery_amendment.json"

def stage_parent(source,dest):
    """New writable editing copy, byte-identical to the immutable source input."""
    dest.mkdir(parents=True,exist_ok=True)
    for p in source.rglob("*"):
        if p.is_symlink(): raise RuntimeError("Unexpected source symlink")
        q=dest/p.relative_to(source)
        if p.is_dir(): q.mkdir(exist_ok=True)
        elif p.is_file():
            if q.exists():
                if q.read_bytes()!=p.read_bytes(): raise RuntimeError("Staging source drift")
            else: q.write_bytes(p.read_bytes())
    h=base.source_tree_hash(source,1024*1024)[0]
    if base.source_tree_hash(dest,1024*1024)[0]!=h: raise RuntimeError("Staging hash mismatch")
    if not all(os.access(dest/n,os.W_OK) for n in ["algorithm.py","manifest.json"]):
        raise RuntimeError("Editing input is not writable")
    return h

def initialize():
    base.verify_frozen()
    if AMEND.exists(): verify(); return
    attempts=[]; smoke=[]
    for seed in base.SEEDS:
        parent=base.load(OUT/f"ablation/s{seed}/frozen_parent.json")
        source=OUT/parent["program_path"]
        staging=OUT/f"editing_parents/s{seed}"
        digest=stage_parent(source,staging)
        for op in base.OPS:
            draft=OUT/f"workspace_preflight/s{seed}/{op}"
            base.proposal_agent.prepare_draft(staging,draft)
            applied=base.proposal_agent._apply_python_program_operator(draft,
                entrypoint="algorithm.py",source_files=["algorithm.py","manifest.json"],
                intervention={"intervention_operator":op,"intervention_scope":"fit" if op=="subsystem_rewrite" else "whole_program"})
            smoke.append(dict(seed=seed,operator=op,applied=applied,parent_digest=digest))
            for arm in base.ARMS:
                p=OUT/f"ablation/s{seed}/{arm}/{op}/record.json"
                r=base.load(p); failure=r["result"]["failure_reason"] or ""
                eligible=("Permission denied" in failure and "could not apply Python program operator" in failure)
                if any(c["role"]=="proposal" for c in r["calls"]): raise RuntimeError("Proposal already called")
                if not eligible and not failure.startswith("Context call failed"): raise RuntimeError(f"Unexpected failure: {failure}")
                attempts.append(dict(seed=seed,arm=arm,operator=op,eligible=eligible,
                    record_path=str(p.relative_to(OUT)),record_sha256=base.sha(p),
                    context_sha256=base.sha(p.parent/"context.json"),
                    memory_sha256=r["memory_sha256"],original_failure=failure))
    amendment=dict(schema="openhyra-infrastructure-recovery.v1",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        original_protocol_sha256=base.sha(base.PROTOCOL_PATH),recovery_driver_sha256=base.sha(__file__),
        reason="Frozen evaluator snapshots were copied with read-only modes into Proposal drafts, preventing scaffold materialization before any Proposal model call.",
        change="Create writable, byte-identical editing-parent copies in a separate namespace. Frozen sources and evaluator are unchanged.",
        no_reroll="Reuse exactly the ten successful frozen Context plans. The two Context timeouts remain failures. One originally planned Proposal call for each eligible attempt; no candidate repairs or reruns.",
        supersedes="The original no-repairs rule is amended only for the pre-generation filesystem preparation defect. All twelve original records and actual call costs remain in the package.",
        unchanged="Model, prompts, plans, parents, histories, operators, evaluation seeds/budgets, selection and holdout rules unchanged.",
        active_record_directory=str(ACTIVE.relative_to(OUT)),attempts=attempts,workspace_preflight=smoke)
    base.save(AMEND,amendment,immutable=True)
    print("RECOVERY AMENDMENT FROZEN; 6 workspace checks passed; 10 planned Proposal calls, 2 retained timeouts",flush=True)

def verify():
    base.verify_frozen(); a=base.load(AMEND)
    if a["recovery_driver_sha256"]!=base.sha(__file__): raise RuntimeError("Recovery driver drift")
    if a["original_protocol_sha256"]!=base.sha(base.PROTOCOL_PATH): raise RuntimeError("Original protocol drift")
    for r in a["attempts"]:
        p=OUT/r["record_path"]
        if base.sha(p)!=r["record_sha256"] or base.sha(p.parent/"context.json")!=r["context_sha256"]:
            raise RuntimeError("Frozen attempt changed")
    return a

def recover(seed):
    a=verify()
    parent=base.load(OUT/f"ablation/s{seed}/frozen_parent.json")
    source=OUT/parent["program_path"]
    staging=OUT/f"editing_parents/s{seed}"
    stage_parent(source,staging)
    contract=(base.FROZEN/"tasks/bermudan_python_search/TASK.md").read_text().split("## Search semantics")[0]
    for oi,op in enumerate(base.OPS):
        order=base.ARMS if (base.SEEDS.index(seed)+oi)%2==0 else list(reversed(base.ARMS))
        for arm in order:
            entry=next(x for x in a["attempts"] if (x["seed"],x["arm"],x["operator"])==(seed,arm,op))
            old=base.load(OUT/entry["record_path"])
            run=ACTIVE/f"s{seed}/{arm}/{op}"
            if (run/"record.json").exists(): continue
            if (run/"started.json").exists(): raise RuntimeError("Interrupted recovery attempt; never re-call silently")
            run.mkdir(parents=True,exist_ok=True)
            base.save(run/"started.json",dict(seed=seed,arm=arm,operator=op,amendment_sha256=base.sha(AMEND)),immutable=True)
            rec=dict(old)
            rec["infrastructure_recovery"]={**entry,"amendment_sha256":base.sha(AMEND)}
            if not entry["eligible"]:
                base.save(run/"record.json",rec,immutable=True)
                print(f"RETAIN seed={seed} arm={arm} op={op}: Context timeout",flush=True)
                continue
            plan=old["context_plan"]
            logger=base.Logger(run/"calls")
            logger.calls=list(old["calls"])
            base.proposal_agent.run_agent=logger
            intervention=dict(intervention_operator=op,intervention_scope="fit" if op=="subsystem_rewrite" else "whole_program",
                mechanism=plan.get("mechanism",""),prediction=plan.get("expected_behavior",""),
                falsifier=plan.get("failure_condition",""),target_slice="all")
            pp=(contract+"\nImplement the Context plan below as an executable program. Optimize actual "
                "causal stopping decisions for expected discounted payoff, measured by primal-dual "
                "confidence gap. Do not alter evaluation or read outside this draft directory. "
                "Use only Python standard library and NumPy. Fit cap: 30 seconds; training paths: 256. "
                "Queries contain only causal history and current state. Use terminal/files only to edit "
                "this draft; do not search the repository or access network. Do not execute the generated "
                "candidate here: the independent evaluator performs execution.\nContext plan:\n"+base.json.dumps(plan,ensure_ascii=False))
            metadata={}; draft=run/"draft"
            ok,description=base.proposal_agent.propose(staging,draft,pp,["algorithm.py","manifest.json"],
                timeout_s=180,backend="codex",model="gpt-5.6-sol",candidate_mode="python_program",
                entrypoint="algorithm.py",artifact_protocol="openhyra-python-program.v1",
                source_files=["algorithm.py","manifest.json"],intervention=intervention,execution_metadata=metadata)
            program=run/"program"
            if draft.exists(): base.snapshot_source_tree(draft,program,1024*1024)
            if ok: ev=base.evaluate(program,base.request("calibration",seed),run/"calibration.json")
            else:
                ev=dict(status="error",score=None,metrics={},evidence={},failure_reason=description,wall_seconds=0,
                    source_digest=base.source_tree_hash(program,1024*1024)[0] if program.exists() else None,
                    source_unchanged=True,request=base.request("calibration",seed))
                base.save(run/"calibration.json",ev,immutable=True)
            rec.update(operator_execution=metadata,program_path=str(program.relative_to(OUT)),
                calls=[{k:v for k,v in x.items() if k not in {"prompt","stdout"}} for x in logger.calls],result=ev)
            base.save(run/"record.json",rec,immutable=True)
            if base.source_tree_hash(staging,1024*1024)[0]!=old["parent_source_digest"]:
                raise RuntimeError("Editing parent mutated")
            print(f"RECOVERED seed={seed} arm={arm} op={op} {ev['status']} {ev['score']}",flush=True)
    print(f"RECOVERY SEED {seed} COMPLETE",flush=True)

def holdout(workers):
    verify(); selections=[]; index=[]
    for seed in base.SEEDS:
        parent=base.load(OUT/f"ablation/s{seed}/frozen_parent.json")
        pr=base.load(OUT/"raw/calibration"/f"{parent['key']}.json")
        for arm in base.ARMS:
            attempts=[base.load(ACTIVE/f"s{seed}/{arm}/{op}/record.json") for op in base.OPS]
            valid=[x for x in attempts if x["result"]["status"]=="ok"]
            best=min(valid,key=lambda x:(x["result"]["score"],x["operator"])) if valid else None
            inc=best if best and best["result"]["score"]<pr["score"] else None
            path=inc["program_path"] if inc else parent["program_path"]
            selections.append(dict(seed=seed,arm=arm,parent_key=parent["key"],selected_program_path=path,
                selected_operator=inc["operator"] if inc else "initial_reference",
                best_generated_program_path=best["program_path"] if best else None,
                best_generated_operator=best["operator"] if best else None,
                calibration_score=inc["result"]["score"] if inc else pr["score"],
                source_digest=base.source_tree_hash(OUT/path,1024*1024)[0]))
    freeze=dict(protocol_sha256=base.sha(base.PROTOCOL_PATH),amendment_sha256=base.sha(AMEND),
        rule="calibration only; parent included; no holdout reselection",selections=selections)
    base.save(OUT/"selections_frozen.json",freeze,immutable=True)
    for item in base.load(OUT/"catalog.json"):
        if item["kind"]=="reference" or item.get("is_original_winner"):
            index.append({k:v for k,v in item.items() if k!="original_row"})
    for seed in base.SEEDS:
        for arm in base.ARMS:
            for op in base.OPS:
                rec=base.load(ACTIVE/f"s{seed}/{arm}/{op}/record.json")
                if rec["result"]["status"]=="ok":
                    index.append(dict(key=f"ablation-{arm}-s{seed}-{op}",seed=seed,kind="ablation",
                        method=arm,operator=op,program_path=rec["program_path"]))
    base.save(OUT/"holdout_index.json",index,immutable=True)
    jobs=[(OUT/i["program_path"],base.request("holdout",i["seed"]),OUT/"raw/holdout"/f"{i['key']}.json") for i in index]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(base.evaluate,*job) for job in jobs]
        for f in concurrent.futures.as_completed(fs): f.result()
    base.save(OUT/"experiments_complete.json",dict(n_holdout_programs=len(index),
        selections_sha256=base.sha(OUT/"selections_frozen.json"),protocol_sha256=base.sha(base.PROTOCOL_PATH),
        amendment_sha256=base.sha(AMEND),active_record_directory=str(ACTIVE.relative_to(OUT))),immutable=True)
    print("FINAL HOLDOUT COMPLETE",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=["initialize","recover","holdout"])
    p.add_argument("--seed",type=int,choices=base.SEEDS); p.add_argument("--workers",type=int,default=2); a=p.parse_args()
    if a.phase=="initialize": initialize()
    elif a.phase=="recover": recover(a.seed)
    else: holdout(a.workers)
