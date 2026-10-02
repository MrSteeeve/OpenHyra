#!/usr/bin/env python3
"""Cross-fitted spline Longstaff--Schwartz continuation program.

The important distinction from a plain least-squares Monte Carlo fit is that
the stopping decisions used to construct earlier targets are made by models
which did not see the path being classified.  The final query models are
refitted on all paths.  Everything needed for fitting and inference is stored
in a small, deterministic NumPy archive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


N_KNOTS = 7
N_FOLDS = 4


def _coordinate(states: np.ndarray) -> np.ndarray:
    """A stable scalar ordering, exact for the target one-asset put."""
    s = np.asarray(states, dtype=np.float64)
    if s.ndim == 1:
        return s
    s = s.reshape(len(s), -1)
    # The minimum is useful for puts; the mean makes this degrade gracefully
    # on basket contracts without introducing a dimension-dependent model.
    return 0.75 * np.min(s, axis=1) + 0.25 * np.mean(s, axis=1)


def _basis(states: np.ndarray, immediate: np.ndarray, center: float,
           scale: float, knots: np.ndarray, payoff_scale: float) -> np.ndarray:
    z = np.clip((_coordinate(states) - center) / scale, -6.0, 6.0)
    p = np.asarray(immediate, dtype=np.float64).reshape(-1)
    q = p / max(float(payoff_scale), 1e-8)
    # Linear splines are robust at the exercise boundary; low-order global
    # terms carry extrapolation outside the knot range.
    cols = [np.ones_like(z), z, z * z, q, q * q]
    cols.extend(np.maximum(z - k, 0.0) for k in knots)
    return np.column_stack(cols)


def _ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Scale penalty by sample size.  Do not penalize the intercept.
    penalty = np.eye(x.shape[1]) * (2e-4 * max(len(x), 1))
    penalty[0, 0] = 1e-10
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _geometry(states: np.ndarray, eligible: np.ndarray) -> tuple[float, float, np.ndarray]:
    u = _coordinate(states)
    sample = u[eligible] if np.count_nonzero(eligible) >= 32 else u
    center = float(np.median(sample))
    scale = max(float(np.quantile(sample, .9) - np.quantile(sample, .1)),
                abs(center) * 0.05, 1e-8)
    z = np.clip((sample - center) / scale, -6.0, 6.0)
    knots = np.quantile(z, np.linspace(.12, .88, N_KNOTS))
    return center, scale, np.asarray(knots, dtype=np.float64)


def fit(input_dir, output_dir, seed):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.asarray(np.load(input_dir / "payoffs.npy", allow_pickle=False),
                         dtype=np.float64)
    if paths.ndim != 3 or payoffs.shape != paths.shape[:2]:
        raise ValueError("incompatible training path and payoff shapes")
    n, n_times = payoffs.shape
    steps = n_times - 1
    width = 5 + N_KNOTS
    coefficients = np.zeros((steps, width), dtype=np.float64)
    centers = np.zeros(steps); scales = np.ones(steps)
    payoff_scales = np.ones(steps)
    knots_all = np.zeros((steps, N_KNOTS))
    cash = payoffs[:, -1].copy()
    targets = np.zeros((steps, n), dtype=np.float64)

    # Seeded permutation preserves replay determinism while preventing any
    # simulator ordering from lining up with folds.
    rng = np.random.default_rng(int(seed))
    fold = np.empty(n, dtype=np.int64)
    fold[rng.permutation(n)] = np.arange(n) % N_FOLDS

    for t in range(steps - 1, -1, -1):
        targets[t] = cash
        immediate = payoffs[:, t]
        eligible = immediate > 0.0
        center, scale, knots = _geometry(paths[:, t, :], eligible)
        payoff_scale = max(float(np.max(immediate)), scale * 0.05, 1e-8)
        x = _basis(paths[:, t, :], immediate, center, scale, knots, payoff_scale)
        trainable = eligible if np.count_nonzero(eligible) >= max(48, width * 3) else np.ones(n, bool)

        # Cross-fitted values are used only to improve the recursively formed
        # cash-flow labels.  Each path's stopping rule is out of sample.
        continuation = np.empty(n, dtype=np.float64)
        for k in range(N_FOLDS):
            tr = trainable & (fold != k)
            if np.count_nonzero(tr) < width:
                tr = fold != k
            beta = _ridge(x[tr], targets[t, tr])
            continuation[fold == k] = x[fold == k] @ beta
        exercise = eligible & (immediate >= continuation)
        cash = np.where(exercise, immediate, cash)

        full = trainable
        coefficients[t] = _ridge(x[full], targets[t, full])
        centers[t], scales[t], knots_all[t] = center, scale, knots
        payoff_scales[t] = payoff_scale

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "model.npz", coefficients=coefficients, centers=centers,
             scales=scales, payoff_scales=payoff_scales, knots=knots_all,
             n_times=np.array(n_times),
             seed=np.array(int(seed), dtype=np.int64))


def predict(model_dir, input_dir, output_dir):
    model_dir, input_dir, output_dir = Path(model_dir), Path(input_dir), Path(output_dir)
    request = json.loads((input_dir / "request.json").read_text(encoding="utf-8"))
    t = int(request["time_index"])
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    immediate = np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False)
    with np.load(model_dir / "model.npz", allow_pickle=False) as model:
        x = _basis(states, immediate, float(model["centers"][t]),
                   float(model["scales"][t]), model["knots"][t],
                   float(model["payoff_scales"][t]))
        result = x @ model["coefficients"][t]
    result = np.nan_to_num(result, nan=0.0, posinf=1e100, neginf=-1e100)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", result, allow_pickle=False)


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
