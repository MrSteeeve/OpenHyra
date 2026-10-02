#!/usr/bin/env python3
"""Two-regime gated ridge continuation program."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def features(states, immediate):
    z = np.log(np.maximum(np.asarray(states, float), 1e-12))
    return np.column_stack([np.ones(len(z)), z, z*z, np.mean(z,1),
                            np.max(z,1), np.asarray(immediate, float)])

def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False)
    steps = paths.shape[1] - 1; x0 = features(paths[:,0,:], payoffs[:,0])
    lo_coef = np.zeros((steps, x0.shape[1])); hi_coef = np.zeros_like(lo_coef)
    means = np.zeros_like(lo_coef); scales = np.ones_like(lo_coef)
    thresholds = np.zeros(steps); cash = np.asarray(payoffs[:,-1], float).copy()
    for t in range(steps-1, -1, -1):
        x = features(paths[:,t,:], payoffs[:,t])
        gate = np.mean(np.log(np.maximum(paths[:,t,:], 1e-12)), 1)
        thresholds[t] = np.median(gate)
        lo, hi = gate <= thresholds[t], gate > thresholds[t]
        mean, scale = x.mean(0), x.std(0); mean[0] = 0.; scale[scale < 1e-10] = 1.
        xn = (x - mean) / scale; reg = np.eye(x.shape[1]) * 1e-6
        def solve(mask):
            if int(mask.sum()) < x.shape[1]: mask = np.ones(len(x), bool)
            return np.linalg.solve(xn[mask].T @ xn[mask] + reg, xn[mask].T @ cash[mask])
        lo_coef[t], hi_coef[t] = solve(lo), solve(hi)
        pred = np.where(lo, xn @ lo_coef[t], xn @ hi_coef[t])
        cash = np.where(payoffs[:,t] >= pred, payoffs[:,t], cash)
        means[t], scales[t] = mean, scale
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir/"model.npz", coefficients_lo=lo_coef, coefficients_hi=hi_coef,
             means=means, scales=scales, thresholds=thresholds)

def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    req=json.loads((input_dir/"request.json").read_text()); t=int(req["time_index"])
    states=np.load(input_dir/"states.npy", allow_pickle=False)
    immediate=np.load(input_dir/"immediate_payoffs.npy", allow_pickle=False)
    with np.load(model_dir/"model.npz", allow_pickle=False) as m:
        x=features(states, immediate); x=(x-m["means"][t])/m["scales"][t]
        gate=np.mean(np.log(np.maximum(states,1e-12)),1)
        out=np.where(gate <= m["thresholds"][t], x@m["coefficients_lo"][t], x@m["coefficients_hi"][t])
    output_dir.mkdir(parents=True, exist_ok=True); np.save(output_dir/"predictions.npy", np.asarray(out,float), allow_pickle=False)

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True)
    f=s.add_parser("fit"); f.add_argument("--input",required=True); f.add_argument("--output",required=True); f.add_argument("--seed",type=int,required=True)
    q=s.add_parser("predict"); q.add_argument("--model",required=True); q.add_argument("--input",required=True); q.add_argument("--output",required=True)
    a=p.parse_args()
    if a.command=="fit": fit(Path(a.input),Path(a.output),a.seed)
    else: predict(Path(a.model),Path(a.input),Path(a.output))
if __name__=="__main__": main()
