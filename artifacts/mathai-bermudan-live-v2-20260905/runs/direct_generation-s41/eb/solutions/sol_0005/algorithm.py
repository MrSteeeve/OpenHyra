#!/usr/bin/env python3
"""Cross-fitted ridge plus a shrinkage-controlled neural residual.

This is an AST-level repair/mutation of the materialized residual-hybrid
parent.  The ridge backbone and trained tanh residual are retained.  Stopping
targets are now updated from out-of-fold predictions, and the neural branch is
allowed to affect inference only to the extent that its out-of-fold residual
predictions improve squared error.  Features are expressed in moneyness, which
is particularly useful for the targeted at-the-money put.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def features(states: np.ndarray, immediate: np.ndarray, strike: float) -> np.ndarray:
    """Finite, scale-free state representation shared by fit and predict."""
    s = np.asarray(states, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    m = s / max(float(strike), 1e-12)
    logm = np.log(np.maximum(m, 1e-12))
    payoff = np.asarray(immediate, dtype=np.float64).reshape(-1) / max(float(strike), 1e-12)
    columns = [
        np.ones(len(s)), m, m * m, m * m * m, logm,
        np.mean(m, axis=1), np.min(m, axis=1), np.max(m, axis=1),
        np.mean(logm * logm, axis=1), payoff, payoff * payoff,
    ]
    return np.column_stack(columns)


def _train(x: np.ndarray, target: np.ndarray, rng: np.random.Generator, width: int = 12):
    """Fit the inherited ridge backbone and MLP residual with stable scaling."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    mean[0] = 0.0                 # preserve the constant feature/intercept
    scale[scale < 1e-10] = 1.0
    xn = (x - mean) / scale
    d = xn.shape[1]
    gram = xn.T @ xn
    gram.flat[:: d + 1] += 2e-5 * len(xn)
    coef = np.linalg.solve(gram, xn.T @ target)
    residual = target - xn @ coef
    residual_scale = max(float(residual.std()), 1e-8)
    y = residual / residual_scale

    w1 = rng.normal(0.0, 0.10, (d, width))
    b1 = np.zeros(width)
    w2 = np.zeros(width)          # initially exactly the ridge parent
    b2 = 0.0
    # Full-batch residual training is deterministic for a supplied seed.  The
    # modest step and gradient clipping prevent the unstable parent updates.
    for _ in range(72):
        h = np.tanh(xn @ w1 + b1)
        err = h @ w2 + b2 - y
        g = 2.0 * err / len(xn)
        gh = g[:, None] * w2 * (1.0 - h * h)
        gw1, gb1 = xn.T @ gh, gh.sum(axis=0)
        gw2, gb2 = h.T @ g, float(g.sum())
        norm = max(1.0, float(np.sqrt(np.sum(gw1 * gw1) + np.sum(gw2 * gw2))))
        rate = 0.025 / norm
        w1 -= rate * gw1
        b1 -= rate * gb1
        w2 -= rate * gw2
        b2 -= rate * gb2
    return mean, scale, coef, w1, b1, w2, b2, residual_scale


def _parts(x: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    mean, scale, coef, w1, b1, w2, b2, residual_scale = model
    xn = (x - mean) / scale
    ridge = xn @ coef
    residual = residual_scale * (np.tanh(xn @ w1 + b1) @ w2 + b2)
    return ridge, residual


def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False)
    instance = json.loads((input_dir / "instance.json").read_text(encoding="utf-8"))
    strike = float(instance.get("strike", 1.0))
    n, n_times, _ = paths.shape
    steps = n_times - 1
    d = features(paths[:, 0], payoffs[:, 0], strike).shape[1]
    width = 12
    means = np.zeros((steps, d)); scales = np.ones((steps, d))
    coefs = np.zeros((steps, d)); w1s = np.zeros((steps, d, width))
    b1s = np.zeros((steps, width)); w2s = np.zeros((steps, width))
    b2s = np.zeros(steps); residual_scales = np.ones(steps); blends = np.zeros(steps)
    cash = np.asarray(payoffs[:, -1], dtype=np.float64).copy()

    # A seeded balanced split makes the backward stopping recursion honest:
    # every path's exercise decision is based on a model not fit to that path.
    rng = np.random.default_rng(int(seed))
    fold = np.empty(n, dtype=np.int8)
    fold[rng.permutation(n)] = np.arange(n) % 2
    for t in range(steps - 1, -1, -1):
        x = features(paths[:, t], payoffs[:, t], strike)
        target = cash.copy()
        ridge_oof = np.empty(n)
        residual_oof = np.empty(n)
        for held_out in (0, 1):
            train_mask = fold != held_out
            model = _train(x[train_mask], target[train_mask], rng)
            r, q = _parts(x[~train_mask], model)
            ridge_oof[~train_mask] = r
            residual_oof[~train_mask] = q
        denom = float(residual_oof @ residual_oof)
        blend = 0.0 if denom < 1e-14 else float(
            np.clip(residual_oof @ (target - ridge_oof) / denom, 0.0, 0.75)
        )
        continuation_oof = np.clip(
            ridge_oof + blend * residual_oof, 1e-10 * strike, strike
        )
        cash = np.where(payoffs[:, t] >= continuation_oof, payoffs[:, t], cash)

        # Store a full-data model for independent evaluator paths.
        model = _train(x, target, rng)
        (means[t], scales[t], coefs[t], w1s[t], b1s[t], w2s[t],
         b2s[t], residual_scales[t]) = model
        blends[t] = blend

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "model.npz", means=means, scales=scales,
        coefficients=coefs, w1=w1s, b1=b1s, w2=w2s, b2=b2s,
        residual_scales=residual_scales, blends=blends,
        strike=np.asarray(strike),
    )


def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    request = json.loads((input_dir / "request.json").read_text(encoding="utf-8"))
    t = int(request["time_index"])
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    immediate = np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False)
    with np.load(model_dir / "model.npz", allow_pickle=False) as m:
        strike = float(m["strike"])
        x = features(states, immediate, strike)
        xn = (x - m["means"][t]) / m["scales"][t]
        ridge = xn @ m["coefficients"][t]
        residual = m["residual_scales"][t] * (
            np.tanh(xn @ m["w1"][t] + m["b1"][t]) @ m["w2"][t] + m["b2"][t]
        )
        predictions = ridge + m["blends"][t] * residual
    predictions = np.nan_to_num(
        predictions, nan=1e-10 * strike,
        posinf=strike, neginf=1e-10 * strike,
    )
    # Discounted option values are nonnegative and bounded by strike for every
    # admitted payoff.  A tiny positive floor avoids exercising a worthless
    # option merely because an evaluator uses a non-strict comparison.
    predictions = np.clip(predictions, 1e-10 * strike, strike)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", predictions.astype(np.float64), allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--seed", required=True, type=int)
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit(Path(args.input), Path(args.output), args.seed)
    else:
        predict(Path(args.model), Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
