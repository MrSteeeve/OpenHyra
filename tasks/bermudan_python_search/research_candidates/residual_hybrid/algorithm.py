#!/usr/bin/env python3
"""Ridge plus a trained MLP residual on Monte Carlo backward targets.

Both the least-squares fit and the network optimizer belong to this candidate.
No evaluator labels or prices are needed: future discounted path cash flows
supply the regression targets, with stopping decisions updated backward.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def features(states, immediate):
    z = np.log(np.maximum(np.asarray(states, float), 1e-12))
    return np.column_stack([np.ones(len(z)), z, z * z, np.mean(z, 1), immediate])


def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False)
    steps, width = paths.shape[1] - 1, 16
    d = features(paths[:, 0], payoffs[:, 0]).shape[1]
    coefs = np.zeros((steps, d)); means = np.zeros_like(coefs); scales = np.ones_like(coefs)
    w1s = np.zeros((steps, d, width)); b1s = np.zeros((steps, width))
    w2s = np.zeros((steps, width)); b2s = np.zeros(steps); target_scales = np.ones(steps)
    targets = np.zeros((steps, len(paths))); losses = np.zeros((steps, 2)); updates = np.zeros(steps)
    cash = payoffs[:, -1].copy()
    for t in range(steps - 1, -1, -1):
        targets[t] = cash
        x = features(paths[:, t], payoffs[:, t])
        mean, scale = x.mean(0), x.std(0); mean[0] = 0.; scale[scale < 1e-10] = 1.
        x = (x - mean) / scale
        coef = np.linalg.solve(x.T @ x + np.eye(d) * 1e-5, x.T @ cash)
        residual = cash - x @ coef
        target_scale = max(float(residual.std()), 1e-5)
        y = residual / target_scale
        w1 = rng.normal(0., .15, (d, width)); initial = w1.copy()
        b1 = np.zeros(width); w2 = rng.normal(0., .1, width); b2 = 0.
        losses[t, 0] = np.mean((np.tanh(x @ w1 + b1) @ w2 + b2 - y) ** 2)
        for _ in range(64):
            h = np.tanh(x @ w1 + b1); err = h @ w2 + b2 - y
            g = 2. * err / len(x); hidden_g = (g[:, None] * w2) * (1. - h * h)
            gw1, gb1, gw2, gb2 = x.T @ hidden_g, hidden_g.sum(0), h.T @ g, g.sum()
            w1 -= .03 * gw1; b1 -= .03 * gb1; w2 -= .03 * gw2; b2 -= .03 * gb2
        residual_pred = np.tanh(x @ w1 + b1) @ w2 + b2
        losses[t, 1] = np.mean((residual_pred - y) ** 2)
        updates[t] = np.linalg.norm(w1 - initial)
        continuation = x @ coef + target_scale * residual_pred
        cash = np.where(payoffs[:, t] >= continuation, payoffs[:, t], cash)
        coefs[t], means[t], scales[t] = coef, mean, scale
        w1s[t], b1s[t], w2s[t], b2s[t], target_scales[t] = w1, b1, w2, b2, target_scale
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "model.npz", coefficients=coefs, means=means, scales=scales,
             w1=w1s, b1=b1s, w2=w2s, b2=b2s, target_scales=target_scales)
    np.savez(output_dir / "training_trace.npz", backward_targets=targets,
             loss_before_after=losses, first_layer_update_norm=updates, seed=np.array(seed))


def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    t = int(json.loads((input_dir / "request.json").read_text())["time_index"])
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    immediate = np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False)
    with np.load(model_dir / "model.npz", allow_pickle=False) as m:
        x = (features(states, immediate) - m["means"][t]) / m["scales"][t]
        residual = np.tanh(x @ m["w1"][t] + m["b1"][t]) @ m["w2"][t] + m["b2"][t]
        out = x @ m["coefficients"][t] + m["target_scales"][t] * residual
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", out, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit"); f.add_argument("--input", required=True); f.add_argument("--output", required=True); f.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("predict"); p.add_argument("--model", required=True); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    a = parser.parse_args()
    if a.command == "fit": fit(Path(a.input), Path(a.output), a.seed)
    else: predict(Path(a.model), Path(a.input), Path(a.output))

if __name__ == "__main__": main()
