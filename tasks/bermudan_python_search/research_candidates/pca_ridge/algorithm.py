#!/usr/bin/env python3
"""Complete PCA plus ridge continuation program."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def raw_features(states, immediate):
    states = np.asarray(states, dtype=float)
    z = np.log(np.maximum(states, 1e-12))
    return np.column_stack([np.ones(len(z)), z, z * z,
                            np.mean(z, axis=1), np.asarray(immediate, float)])

def transform(states, immediate, center, components):
    z = np.log(np.maximum(np.asarray(states, dtype=float), 1e-12))
    score = (z - center) @ components.T
    return np.column_stack([raw_features(states, immediate), score])

def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False)
    z = np.log(np.maximum(paths[:, 1:, :].reshape(-1, paths.shape[-1]), 1e-12))
    center = z.mean(axis=0)
    _, _, vh = np.linalg.svd(z - center, full_matrices=False)
    components = vh[:min(3, vh.shape[0])]
    steps = paths.shape[1] - 1
    design0 = transform(paths[:, 0, :], payoffs[:, 0], center, components)
    coefs = np.zeros((steps, design0.shape[1]))
    means = np.zeros_like(coefs)
    scales = np.ones_like(coefs)
    cash = np.asarray(payoffs[:, -1], float).copy()
    for t in range(steps - 1, -1, -1):
        x = transform(paths[:, t, :], payoffs[:, t], center, components)
        mean, scale = x.mean(0), x.std(0)
        mean[0], scale[scale < 1e-10] = 0.0, 1.0
        xn = (x - mean) / scale
        coef = np.linalg.solve(xn.T @ xn + np.eye(xn.shape[1]) * 1e-6,
                               xn.T @ cash)
        cash = np.where(payoffs[:, t] >= xn @ coef, payoffs[:, t], cash)
        coefs[t], means[t], scales[t] = coef, mean, scale
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "model.npz", center=center, components=components,
             coefficients=coefs, means=means, scales=scales)

def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    request = json.loads((input_dir / "request.json").read_text())
    t = int(request["time_index"])
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    immediate = np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False)
    with np.load(model_dir / "model.npz", allow_pickle=False) as model:
        x = transform(states, immediate, model["center"], model["components"])
        x = (x - model["means"][t]) / model["scales"][t]
        out = x @ model["coefficients"][t]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", np.asarray(out, float), allow_pickle=False)

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit"); f.add_argument("--input", required=True); f.add_argument("--output", required=True); f.add_argument("--seed", type=int, required=True)
    q = sub.add_parser("predict"); q.add_argument("--model", required=True); q.add_argument("--input", required=True); q.add_argument("--output", required=True)
    a = p.parse_args()
    if a.command == "fit": fit(Path(a.input), Path(a.output), a.seed)
    else: predict(Path(a.model), Path(a.input), Path(a.output))

if __name__ == "__main__":
    main()
