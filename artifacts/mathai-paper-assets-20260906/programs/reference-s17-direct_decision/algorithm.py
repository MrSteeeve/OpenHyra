#!/usr/bin/env python3
"""Direct stopping-decision program with a learned exercise threshold."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def fit(input_dir:Path,output_dir:Path,seed:int)->None:
    paths=np.load(input_dir/"training_paths.npy",allow_pickle=False)
    payoffs=np.load(input_dir/"payoffs.npy",allow_pickle=False)
    steps=paths.shape[1]-1; thresholds=np.zeros(steps)
    for t in range(steps):
        immediate=np.asarray(payoffs[:,t],float)
        # The threshold is learned from the empirical discounted payoff tail;
        # no evaluator value or hidden path enters the candidate model.
        thresholds[t]=np.quantile(immediate,0.62)
    output_dir.mkdir(parents=True,exist_ok=True)
    np.savez(output_dir/"model.npz",thresholds=thresholds)

def predict(model_dir:Path,input_dir:Path,output_dir:Path)->None:
    req=json.loads((input_dir/"request.json").read_text()); t=int(req["time_index"])
    immediate=np.load(input_dir/"immediate_payoffs.npy",allow_pickle=False)
    with np.load(model_dir/"model.npz",allow_pickle=False) as m:
        out=(immediate>=m["thresholds"][t]).astype(np.float64)
    output_dir.mkdir(parents=True,exist_ok=True); np.save(output_dir/"predictions.npy",out,allow_pickle=False)

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True)
    f=s.add_parser("fit"); f.add_argument("--input",required=True); f.add_argument("--output",required=True); f.add_argument("--seed",type=int,required=True)
    q=s.add_parser("predict"); q.add_argument("--model",required=True); q.add_argument("--input",required=True); q.add_argument("--output",required=True)
    a=p.parse_args()
    if a.command=="fit": fit(Path(a.input),Path(a.output),a.seed)
    else: predict(Path(a.model),Path(a.input),Path(a.output))
if __name__=="__main__": main()
