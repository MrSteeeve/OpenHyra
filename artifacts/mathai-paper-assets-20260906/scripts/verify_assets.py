#!/usr/bin/env python3
"""Check scientific data identities, provenance, selection, and output integrity."""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
from pypdf import PdfReader
from types import SimpleNamespace

OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/"frozen_runtime"))
from sandbox import source_tree_hash
base=SimpleNamespace(OUT=OUT,source_tree_hash=source_tree_hash,
                     OPS=["whole_program_restart","subsystem_rewrite"])
checks=[]
def load(p): return json.loads(Path(p).read_text())
def check(name,ok,details=None):
    if not bool(ok): raise AssertionError(f"{name}: {details}")
    checks.append(dict(name=name,passed=True,details=details))
def close(a,b): return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-11)
def main():
    protocol=load(OUT/"protocol.json")
    amendment=load(OUT/"recovery_amendment.json")
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    check("Original driver hash",sha(OUT/"scripts/run_supplement.py")==protocol["driver_sha256"])
    check("Recovery driver hash",sha(OUT/"scripts/recover_supplement.py")==amendment["recovery_driver_sha256"])
    check("Recovery binds original protocol",sha(OUT/"protocol.json")==amendment["original_protocol_sha256"])
    for rel,digest in protocol["frozen_source_hashes"].items():
        check("Frozen runtime file hash",sha(OUT/"frozen_runtime"/rel)==digest,rel)
    for r in amendment["attempts"]:
        path=OUT/r["record_path"]
        check("Original interrupted record remains unchanged",sha(path)==r["record_sha256"])
        check("Original Context remains unchanged",sha(path.parent/"context.json")==r["context_sha256"])
    recovery=SimpleNamespace(ACTIVE=OUT/amendment["active_record_directory"])
    check("Frozen original runtime and both protocol drivers",True)
    p=pd.read_csv(OUT/"data/program_evaluations.csv")
    cells=pd.read_csv(OUT/"data/evaluation_cells.csv")
    attempts=pd.read_csv(OUT/"data/original_search_attempts.csv")
    feedback=pd.read_csv(OUT/"data/feedback_attempts.csv")
    calls=pd.read_csv(OUT/"data/model_calls.csv")
    source_count=0; formula_count=0
    for _,row in p.iterrows():
        record=load(OUT/row.source_json)
        ev=record["result"] if "result" in record else record
        check(f"Recorded status: {row.stage}/{row.key}",row.status==ev["status"])
        if ev["status"]!="ok": continue
        ss=ev["metrics"]["summaries"]; z=ev["metrics"]["confidence_component_z"]
        gaps=[]
        for c in ss:
            check("Raw gap identity",close(c["raw_primal_dual_gap"],c["upper_bound"]-c["lower_bound"]))
            check("Confidence margin identity",close(c["confidence_gap"],c["raw_primal_dual_gap"]+z*(c["upper_bound_standard_error"]+c["lower_bound_standard_error"])))
            cc=cells[(cells.stage==row.stage)&(cells.key==row.key)&(cells.instance_id==c["instance_id"])&(cells["repeat"]==c["repeat"])]
            check("Exactly one matching exported cell",len(cc)==1)
            check("Cell value exported correctly",close(cc.iloc[0].lower,c["lower_bound"]) and close(cc.iloc[0].upper,c["upper_bound"]))
            gaps.append(c["normalized_primal_dual_confidence_gap"]); formula_count+=1
        expected=np.mean(gaps)+.25*np.quantile(gaps,.9)
        check("Score mean plus tail identity",close(expected,ev["score"]) and close(expected,row.score))
        if isinstance(row.program_path,str) and isinstance(row.source_digest,str):
            actual=base.source_tree_hash(OUT/row.program_path,1024*1024)[0]
            check("Evaluated source digest",actual==row.source_digest); source_count+=1
    for name,part in cells.groupby(["stage","seed","instance_id","repeat"]):
        check("Common upper estimate within matched request",np.ptp(part.upper.to_numpy())<1e-10,str(name))
        check("Common upper SE within matched request",np.ptp(part.upper_se.to_numpy())<1e-10,str(name))
        check("Shared evaluator-owned dual",set(part.dual_verifier)=={"evaluator_owned_ridge_lsmc"},str(name))
    check("Original attempts and failures retained",len(attempts)==24 and int((attempts.status!="ok").sum())==5)
    check("Twelve planned feedback attempts retained",len(feedback)==12)
    for (seed,op),part in feedback.groupby(["seed","operator"]):
        check("Two arms per matched attempt",set(part.arm)=={"structured_memory","score_only"})
        check("Matched parent source",part.parent_source_digest.nunique()==1)
        orig=OUT/f"ablation/s{seed}"
        a=load(orig/"structured_memory/memory.json"); b=load(orig/"score_only/memory.json")
        check("Same candidates and scalar scores in feedback packets",[{k:v for k,v in r.items() if k!="details"} for r in a]==b)
        for arm in ["structured_memory","score_only"]:
            r=load(recovery.ACTIVE/f"s{seed}/{arm}/{op}/record.json")
            check("Exactly one Context call per attempt",sum(c["role"]=="context" for c in r["calls"])==1)
            check("At most one Proposal call; no model retries",sum(c["role"]=="proposal" for c in r["calls"])<=1)
            if r["result"]["status"]=="ok":
                meta=r["operator_execution"]
                expected=op if op=="whole_program_restart" else "subsystem_rewrite:fit"
                check("Declared operator materialized",meta.get("declared_operator")==op and meta.get("applied_operator")==expected)
    chosen=load(OUT/"selections_frozen.json")
    protocol=load(OUT/"protocol.json")
    for r in chosen["selections"]:
        seed=r["seed"]; arm=r["arm"]
        parent=load(OUT/f"ablation/s{seed}/frozen_parent.json")
        parent_ev=load(OUT/"raw/calibration"/f"{parent['key']}.json")
        candidates=[parent_ev["score"]]
        for op in base.OPS:
            rr=load(recovery.ACTIVE/f"s{seed}/{arm}/{op}/record.json")
            if rr["result"]["status"]=="ok": candidates.append(rr["result"]["score"])
        check("Selection minimized calibration score including parent",close(r["calibration_score"],min(candidates)))
        check("Selected source frozen",base.source_tree_hash(OUT/r["selected_program_path"],1024*1024)[0]==r["source_digest"])
    for item in load(OUT/"holdout_index.json"):
        result_path=OUT/"raw/holdout"/f"{item['key']}.json"
        ev=load(result_path)
        check("Holdout request matches preregistered request",ev["request"]==protocol["holdout_requests"][str(item["seed"])])
        check("Selection file precedes every holdout request",(OUT/"selections_frozen.json").stat().st_mtime_ns<=result_path.with_suffix(".request.json").stat().st_mtime_ns)
    check("Calibration and holdout random seeds disjoint",set(x["seed"] for x in protocol["calibration_requests"].values()).isdisjoint(x["seed"] for x in protocol["holdout_requests"].values()))
    traj=pd.read_csv(OUT/"data/original_search_trajectory.csv")
    for _,g in traj.groupby(["mode","seed"]):
        check("All original trajectory points present",len(g)==5)
        check("Public-score incumbent monotonic",all(np.diff(g.public_incumbent_score)>=-1e-12))
    main=pd.read_csv(OUT/"data/main_results_by_seed.csv")
    summary=pd.read_csv(OUT/"data/main_results_summary.csv")
    for _,r in summary.iterrows():
        good=main[(main.method==r.method)&(main.status=="ok")]
        check("Three requested seeds and truthful success count",len(main[main.method==r.method])==3 and len(good)==r.successful_seeds,r.method)
        for c in ["lower_normalized","upper_normalized","raw_gap_normalized","confidence_gap_normalized","score","wall_seconds","fit_seconds"]:
            check("Table mean agrees with seed data",close(r[c+"_mean"],good[c].mean()))
            check("Table SD agrees with seed data",close(r[c+"_sd"],good[c].std(ddof=1)))
    current=calls[calls.experiment=="feedback_ablation"]
    behavior=pd.read_csv(OUT/"data/stopping_behavior.csv")
    for _,g in behavior.groupby(["seed","mode","instance_id","repeat","actor"]):
        check("Stopping distribution covers every path",close(g.stop_fraction.sum(),1))
        check("Valid stopping frequencies",bool(g.stop_fraction.between(0,1).all()))
    check("Twelve Context calls and ten once-only recovered Proposals",int((current.role=="context").sum())==12 and int((current.role=="proposal").sum())==10)
    check("No duplicate logical model calls",not calls.duplicated(["experiment","method","seed","operator","id"]).any())
    book=openpyxl.load_workbook(OUT/"data/paper_data.xlsx",read_only=True,data_only=True)
    check("Excel summary rows agree",book["Main summary"].max_row==len(summary)+1)
    check("Excel trial rows agree",book["Feedback attempts"].max_row==13)
    check("Excel model call rows agree",book["Model calls"].max_row==len(calls)+1)
    book.close()
    figures=load(OUT/"figures/captions.json")
    check("Seven expected figures",len(figures)==7)
    for name in figures:
        check("PDF has one readable page",len(PdfReader(OUT/"figures"/f"{name}.pdf").pages)==1,name)
        for ext in ["png","svg","pdf"]:
            check("Nonempty figure output",(OUT/"figures"/f"{name}.{ext}").stat().st_size>3000,f"{name}.{ext}")
    result=dict(status="passed",check_count=len(checks),evaluated_source_hash_checks=source_count,
        numeric_cells_recomputed=formula_count,recorded_evaluation_rows=len(p),
        original_attempts=len(attempts),feedback_attempts=len(feedback),
        scope="Data, formulas, pairing, selection/provenance, and file integrity; not a proof of new algorithms or population superiority.",checks=checks)
    (OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(f"PASSED {len(checks)} checks; {formula_count} numerical cells; {source_count} source hashes")

if __name__=="__main__": main()
