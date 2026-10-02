#!/usr/bin/env python3
"""Replay recorded holdout programs through the packaged frozen evaluator."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1]
RUNTIME=OUT/"frozen_runtime"
sys.path.insert(0,str(RUNTIME))
from sandbox import NUMERIC_THREAD_ENV, source_tree_hash

def load(p): return json.loads(Path(p).read_text())
def main(keys):
    frozen=load(OUT/"protocol.json")["frozen_source_hashes"]
    for path,digest in frozen.items():
        if hashlib.sha256((RUNTIME/path).read_bytes()).hexdigest()!=digest: raise RuntimeError("Packaged runtime drift")
    catalog={x["key"]:x for x in load(OUT/"holdout_index.json")}
    results=[]; out=OUT/"verification_replays"; out.mkdir(exist_ok=True)
    for key in keys:
        item=catalog[key]; original=load(OUT/"raw/holdout"/f"{key}.json")
        source=OUT/item["program_path"]
        if source_tree_hash(source,1024*1024)[0]!=original["source_digest"]: raise RuntimeError("Replay source mismatch")
        req=out/f"{key}.request.json"
        if req.exists(): raise RuntimeError("Replay already exists; do not overwrite verification evidence")
        req.write_text(json.dumps(original["request"],sort_keys=True,indent=2)+"\n")
        start=time.perf_counter()
        run=subprocess.run([sys.executable,str(RUNTIME/"tasks/bermudan_optimal_stopping/evaluator.py"),str(source),str(req)],
            cwd=RUNTIME,env={**os.environ,**NUMERIC_THREAD_ENV},text=True,capture_output=True,timeout=480)
        (out/f"{key}.stderr.txt").write_text(run.stderr)
        if run.returncode: raise RuntimeError(run.stderr[-2000:])
        payload=json.loads(run.stdout)
        (out/f"{key}.json").write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
        cols=["lower_bound","upper_bound","lower_bound_standard_error","upper_bound_standard_error",
              "raw_primal_dual_gap","confidence_gap","normalized_primal_dual_confidence_gap"]
        a=np.array([[r[c] for c in cols] for r in original["metrics"]["summaries"]])
        b=np.array([[r[c] for c in cols] for r in payload["metrics"]["summaries"]])
        exact=bool(np.array_equal(a,b) and original["score"]==payload["score"])
        if not exact: raise RuntimeError(f"Numerical replay differs: {key}; {np.max(abs(a-b))}")
        results.append(dict(key=key,source_digest=original["source_digest"],exact_numeric_match=exact,
            score=payload["score"],wall_seconds=time.perf_counter()-start,verified_scalar_fields=int(a.size+1)))
    (out/"summary.json").write_text(json.dumps(dict(status="passed",results=results),indent=2)+"\n")
    print(f"{len(results)} relocated frozen-evaluator replays match exactly")

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("keys",nargs="+"); args=parser.parse_args()
    main(args.keys)
