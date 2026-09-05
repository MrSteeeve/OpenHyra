"""Validation-guarded ridge plus nonlinear residual continuation model.

The parent program's backward cash-flow regression, polynomial/log features,
ridge component, and trained tanh residual are retained.  The material AST
mutation is a control-flow guard: a residual is used only when a deterministic
holdout shows lower target error than ridge alone.  The gate also learns a
shrinkage weight, which is useful near an at-the-money exercise boundary where
an overfit residual can change many stopping decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def features(states, immediate):
    """Causal state features; no future portion of a path is consumed."""
    states = np.asarray(states, dtype=float)
    immediate = np.asarray(immediate, dtype=float).reshape(-1)
    z = np.log(np.maximum(states, 1e-12))
    return np.column_stack(
        [np.ones(len(z)), z, z * z, np.mean(z, axis=1), immediate]
    )


def _ridge(x, y, penalty=1e-5):
    gram = x.T @ x
    regularizer = np.eye(x.shape[1]) * penalty
    regularizer[0, 0] = penalty * 0.01
    try:
        return np.linalg.solve(gram + regularizer, x.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + regularizer, x.T @ y, rcond=None)[0]


def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False)
    steps, width = paths.shape[1] - 1, 16
    n_paths = len(paths)
    d = features(paths[:, 0], payoffs[:, 0]).shape[1]

    coefs = np.zeros((steps, d))
    means = np.zeros_like(coefs)
    scales = np.ones_like(coefs)
    w1s = np.zeros((steps, d, width))
    b1s = np.zeros((steps, width))
    w2s = np.zeros((steps, width))
    b2s = np.zeros(steps)
    target_scales = np.ones(steps)
    residual_weights = np.zeros(steps)
    targets = np.zeros((steps, n_paths))
    losses = np.zeros((steps, 3))
    updates = np.zeros(steps)

    # Discounting is already represented in evaluator-owned payoff units.
    cash = payoffs[:, -1].astype(float, copy=True)
    for t in range(steps - 1, -1, -1):
        targets[t] = cash
        raw_x = features(paths[:, t], payoffs[:, t])
        mean, scale = raw_x.mean(axis=0), raw_x.std(axis=0)
        mean[0] = 0.0
        scale[scale < 1e-10] = 1.0
        x = (raw_x - mean) / scale

        # LSM learns the boundary from paths that can actually exercise.  Fall
        # back to all paths if an unusual contract produces too few such rows.
        eligible = np.flatnonzero(payoffs[:, t] > 0.0)
        if len(eligible) < max(24, 3 * d):
            eligible = np.arange(n_paths)
        shuffled = rng.permutation(eligible)
        n_valid = max(1, min(len(shuffled) // 4, 1024))
        valid = shuffled[:n_valid]
        train = shuffled[n_valid:]
        if len(train) < d + 2:
            train, valid = shuffled, shuffled

        coef = _ridge(x[train], cash[train])
        ridge_train = x[train] @ coef
        residual = cash[train] - ridge_train
        target_scale = max(float(np.std(residual)), 1e-8)
        y = residual / target_scale

        w1 = rng.normal(0.0, 0.15, (d, width))
        initial = w1.copy()
        b1 = np.zeros(width)
        w2 = rng.normal(0.0, 0.1, width)
        b2 = 0.0
        losses[t, 0] = np.mean((cash[valid] - x[valid] @ coef) ** 2)
        for _ in range(64):
            h = np.tanh(x[train] @ w1 + b1)
            err = h @ w2 + b2 - y
            g = 2.0 * err / len(train)
            hidden_g = g[:, None] * w2 * (1.0 - h * h)
            gw1 = x[train].T @ hidden_g
            gb1 = hidden_g.sum(axis=0)
            gw2 = h.T @ g
            gb2 = g.sum()
            # The parent's full-batch optimizer, with bounded gradients so a
            # rare path cannot create a non-finite serialized model.
            w1 -= 0.03 * np.clip(gw1, -10.0, 10.0)
            b1 -= 0.03 * np.clip(gb1, -10.0, 10.0)
            w2 -= 0.03 * np.clip(gw2, -10.0, 10.0)
            b2 -= 0.03 * float(np.clip(gb2, -10.0, 10.0))

        valid_residual = target_scale * (
            np.tanh(x[valid] @ w1 + b1) @ w2 + b2
        )
        valid_error = cash[valid] - x[valid] @ coef
        denom = float(valid_residual @ valid_residual)
        weight = 0.0 if denom <= 1e-14 else float(
            np.clip((valid_residual @ valid_error) / denom, 0.0, 1.0)
        )
        guarded_valid = x[valid] @ coef + weight * valid_residual
        guarded_loss = float(np.mean((cash[valid] - guarded_valid) ** 2))

        # Material control-flow guard. A small margin avoids switching on a
        # residual due only to floating-point or negligible holdout gains.
        if (not np.isfinite(guarded_loss)) or guarded_loss >= losses[t, 0] * (1.0 - 1e-4):
            weight = 0.0
            guarded_loss = losses[t, 0]
        losses[t, 1] = guarded_loss
        losses[t, 2] = weight
        updates[t] = np.linalg.norm(w1 - initial)

        residual_all = target_scale * (np.tanh(x @ w1 + b1) @ w2 + b2)
        continuation = np.maximum(x @ coef + weight * residual_all, 0.0)
        cash = np.where(payoffs[:, t] >= continuation, payoffs[:, t], cash)

        coefs[t], means[t], scales[t] = coef, mean, scale
        w1s[t], b1s[t], w2s[t], b2s[t] = w1, b1, w2, b2
        target_scales[t], residual_weights[t] = target_scale, weight

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "model.npz",
        coefficients=coefs,
        means=means,
        scales=scales,
        w1=w1s,
        b1=b1s,
        w2=w2s,
        b2=b2s,
        target_scales=target_scales,
        residual_weights=residual_weights,
    )
    np.savez(
        output_dir / "training_trace.npz",
        backward_targets=targets,
        validation_ridge_guarded_weight=losses,
        first_layer_update_norm=updates,
        seed=np.array(seed),
    )


def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    request = json.loads((input_dir / "request.json").read_text())
    t = int(request["time_index"])
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    immediate = np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False)
    with np.load(model_dir / "model.npz", allow_pickle=False) as model:
        x = (features(states, immediate) - model["means"][t]) / model["scales"][t]
        residual = np.tanh(x @ model["w1"][t] + model["b1"][t]) @ model["w2"][t] + model["b2"][t]
        out = x @ model["coefficients"][t]
        out += model["residual_weights"][t] * model["target_scales"][t] * residual
    # A discounted nonnegative payoff has nonnegative continuation value; this
    # finite fallback keeps the interface valid on extreme external queries.
    out = np.maximum(np.nan_to_num(out, nan=0.0, posinf=1e100, neginf=0.0), 0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", out, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--seed", type=int, required=True)
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
