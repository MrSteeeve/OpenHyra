#!/usr/bin/env python3
"""Causal local-polynomial Bellman program for Bermudan options.

The model is deliberately self contained (NumPy only).  Backward induction uses
leave-one-out local neighbours, preventing a path's own future cashflow from
determining its stopping decision.  Prediction blends the resulting local
linear estimate with a global polynomial continuation surface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _coordinates(states: np.ndarray) -> np.ndarray:
    """Scale-free Markov coordinates; suitable for all supported products."""
    s = np.maximum(np.asarray(states, dtype=np.float64), 1e-12)
    z = np.log(s)
    # The raw log coordinates preserve all state information.  Mean, extrema,
    # and dispersion make basket/max geometry easier for Euclidean neighbours.
    if z.shape[1] == 1:
        return z
    return np.column_stack((z, z.mean(1), z.min(1), z.max(1), z.std(1)))


def _basis(coords: np.ndarray) -> np.ndarray:
    """Compact cubic polynomial with pairwise interactions."""
    x = np.asarray(coords, dtype=np.float64)
    cols = [np.ones(len(x))]
    cols.extend(x[:, j] for j in range(x.shape[1]))
    cols.extend(x[:, j] ** 2 for j in range(x.shape[1]))
    cols.extend(x[:, j] ** 3 for j in range(x.shape[1]))
    for j in range(x.shape[1]):
        for k in range(j + 1, x.shape[1]):
            cols.append(x[:, j] * x[:, k])
    return np.column_stack(cols)


def _ridge_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = _basis(x)
    center = a.mean(0)
    scale = a.std(0)
    center[0] = 0.0
    scale[scale < 1e-10] = 1.0
    q = (a - center) / scale
    # Scale the penalty by sample size; do not penalize the intercept.
    penalty = np.full(q.shape[1], 2e-4 * len(q))
    penalty[0] = 1e-10
    coef = np.linalg.solve(q.T @ q + np.diag(penalty), q.T @ y)
    return coef, center, scale


def _ridge_predict(x: np.ndarray, coef: np.ndarray, center: np.ndarray,
                   scale: np.ndarray) -> np.ndarray:
    return ((_basis(x) - center) / scale) @ coef


def _loo_local(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Vectorized leave-one-out adaptive-kernel regression."""
    n = len(x)
    k = min(max(12, k), n - 1)
    d2 = np.maximum(((x[:, None, :] - x[None, :, :]) ** 2).sum(2), 0.0)
    np.fill_diagonal(d2, np.inf)
    ix = np.argpartition(d2, k - 1, axis=1)[:, :k]
    ds = np.take_along_axis(d2, ix, axis=1)
    bandwidth = np.maximum(np.max(ds, axis=1), 1e-10)
    w = np.exp(-2.0 * ds / bandwidth[:, None])
    return np.sum(w * y[ix], axis=1) / np.maximum(w.sum(1), 1e-12)


def _local_predict(query: np.ndarray, train: np.ndarray, target: np.ndarray,
                   k: int) -> np.ndarray:
    """Adaptive local-linear prediction in bounded batches."""
    n, dim = len(train), train.shape[1]
    k = min(max(12, k), n)
    ans = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 256):
        q = query[start:start + 256]
        d2 = ((q[:, None, :] - train[None, :, :]) ** 2).sum(2)
        ix = np.argpartition(d2, k - 1, axis=1)[:, :k]
        for row in range(len(q)):
            xx = train[ix[row]] - q[row]
            yy = target[ix[row]]
            dd = d2[row, ix[row]]
            h = max(float(dd.max()), 1e-10)
            w = np.exp(-2.0 * dd / h)
            design = np.column_stack((np.ones(k), xx))
            gram = design.T @ (w[:, None] * design)
            # A small slope penalty controls sparse multidimensional tails.
            reg = np.eye(dim + 1) * (1e-5 + 1e-3 * np.trace(gram) / (dim + 1))
            reg[0, 0] = 1e-10
            rhs = design.T @ (w * yy)
            try:
                ans[start + row] = np.linalg.solve(gram + reg, rhs)[0]
            except np.linalg.LinAlgError:
                ans[start + row] = np.average(yy, weights=w)
    return ans


def fit(input_dir, output_dir, seed):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoff = np.asarray(np.load(input_dir / "payoffs.npy", allow_pickle=False), dtype=np.float64)
    if paths.ndim != 3 or payoff.shape[:2] != paths.shape[:2]:
        raise ValueError("incompatible training arrays")

    n, nt, _ = paths.shape
    # Standardize distances separately at each date.  k grows mildly with
    # dimension, trading variance for robustness in baskets.
    coord_dim = _coordinates(paths[:, 0, :]).shape[1]
    k = min(n - 1, 48 if coord_dim == 1 else 72)
    cash = payoff[:, -1].copy()
    xs, mus, sigmas, coefs, centers, scales, targets = [], [], [], [], [], [], []

    for t in range(nt - 2, -1, -1):
        raw = _coordinates(paths[:, t, :])
        mu, sigma = raw.mean(0), raw.std(0)
        sigma[sigma < 1e-8] = 1.0
        x = (raw - mu) / sigma
        coef, center, scale = _ridge_fit(x, cash)
        global_c = _ridge_predict(x, coef, center, scale)
        local_c = _loo_local(x, cash, k)
        # Local smoothing dominates in 1D (the declared ATM-put target), while
        # the polynomial receives more weight as neighbour geometry sparsifies.
        local_weight = 0.78 if coord_dim == 1 else 0.58
        continuation = local_weight * local_c + (1.0 - local_weight) * global_c
        exercise = (payoff[:, t] > 0.0) & (payoff[:, t] >= continuation)

        xs.append(x.astype(np.float32))
        targets.append(cash.astype(np.float32))
        mus.append(mu); sigmas.append(sigma); coefs.append(coef)
        centers.append(center); scales.append(scale)
        cash = np.where(exercise, payoff[:, t], cash)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Lists were accumulated backwards; object arrays are avoided so loading
    # remains allow_pickle=False. Feature sizes are constant within an instance.
    np.savez_compressed(
        output_dir / "model.npz",
        train_x=np.stack(xs[::-1]), targets=np.stack(targets[::-1]),
        mu=np.stack(mus[::-1]), sigma=np.stack(sigmas[::-1]),
        coef=np.stack(coefs[::-1]), center=np.stack(centers[::-1]),
        scale=np.stack(scales[::-1]), k=np.array(k),
        local_weight=np.array(0.78 if coord_dim == 1 else 0.58),
    )


def predict(model_dir, input_dir, output_dir):
    model_dir, input_dir, output_dir = Path(model_dir), Path(input_dir), Path(output_dir)
    request = json.loads((input_dir / "request.json").read_text())
    t = int(request["time_index"])
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    with np.load(model_dir / "model.npz", allow_pickle=False) as m:
        x = (_coordinates(states) - m["mu"][t]) / m["sigma"][t]
        local = _local_predict(x, m["train_x"][t], m["targets"][t], int(m["k"]))
        global_c = _ridge_predict(x, m["coef"][t], m["center"][t], m["scale"][t])
        w = float(m["local_weight"])
        prediction = w * local + (1.0 - w) * global_c
    prediction = np.nan_to_num(prediction, nan=0.0, posinf=1e6, neginf=0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", prediction.astype(np.float64), allow_pickle=False)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    f = commands.add_parser("fit")
    f.add_argument("--input", required=True); f.add_argument("--output", required=True)
    f.add_argument("--seed", required=True, type=int)
    p = commands.add_parser("predict")
    p.add_argument("--model", required=True); p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit(args.input, args.output, args.seed)
    else:
        predict(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
