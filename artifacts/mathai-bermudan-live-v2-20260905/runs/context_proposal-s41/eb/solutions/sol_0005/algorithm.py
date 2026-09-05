"""Cross-fitted, robust backward induction for Bermudan continuation values.

The inherited ridge-plus-tanh-residual representation is retained.  The fit
subtree uses out-of-fold continuation estimates for stopping updates, Huber
IRLS for noisy backward targets, and state-dependent residual winsorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def features(states, immediate):
    z = np.log(np.maximum(np.asarray(states, float), 1e-12))
    return np.column_stack([np.ones(len(z)), z, z * z, np.mean(z, axis=1), immediate])


def _standardize(x, mean=None, scale=None):
    if mean is None:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        mean[0] = 0.0
        scale[scale < 1e-10] = 1.0
    return (x - mean) / scale, mean, scale


def _robust_model(x_raw, target, rng, iterations, width=16):
    """Fit the parent's ridge/residual model with robust target updates."""
    x, mean, scale = _standardize(x_raw)
    d = x.shape[1]
    ridge = 1e-5 * np.eye(d)
    weights = np.ones(len(x))
    coef = np.zeros(d)
    # Huber IRLS stabilizes the linear continuation backbone.
    for _ in range(4):
        sw = np.sqrt(weights)
        coef = np.linalg.solve((x * sw[:, None]).T @ (x * sw[:, None]) + ridge,
                               (x * sw[:, None]).T @ (target * sw))
        err = target - x @ coef
        robust_scale = max(1.4826 * float(np.median(np.abs(err - np.median(err)))), 1e-6)
        ratio = np.abs(err) / (1.5 * robust_scale)
        weights = np.where(ratio <= 1.0, 1.0, 1.0 / np.maximum(ratio, 1e-12))

    residual = target - x @ coef
    residual_scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1e-5)
    # Permit more residual range in unusual states, but bound pathwise target noise.
    state_radius = np.sqrt(np.mean(x[:, 1:] ** 2, axis=1))
    clip = residual_scale * (2.25 + 0.5 * np.minimum(state_radius, 2.0))
    y = np.clip(residual, -clip, clip) / residual_scale

    w1 = rng.normal(0.0, 0.15, (d, width))
    b1 = np.zeros(width)
    w2 = rng.normal(0.0, 0.1, width)
    b2 = 0.0
    loss_before = float(np.mean((np.tanh(x @ w1) @ w2 - y) ** 2))
    initial = w1.copy()
    for _ in range(iterations):
        h = np.tanh(x @ w1 + b1)
        err = h @ w2 + b2 - y
        # Huber gradients also protect the nonlinear residual learner.
        g = np.clip(err, -1.5, 1.5) / len(x)
        hidden_g = g[:, None] * w2 * (1.0 - h * h)
        w1 -= 0.03 * (x.T @ hidden_g)
        b1 -= 0.03 * hidden_g.sum(axis=0)
        w2 -= 0.03 * (h.T @ g)
        b2 -= 0.03 * g.sum()
    pred = x @ coef + residual_scale * (np.tanh(x @ w1 + b1) @ w2 + b2)
    loss_after = float(np.mean((pred - target) ** 2))
    model = (coef, mean, scale, w1, b1, w2, float(b2), residual_scale)
    return model, pred, (loss_before, loss_after, float(np.linalg.norm(w1 - initial)))


def _apply(model, x_raw):
    coef, mean, scale, w1, b1, w2, b2, residual_scale = model
    x, _, _ = _standardize(x_raw, mean, scale)
    return x @ coef + residual_scale * (np.tanh(x @ w1 + b1) @ w2 + b2)


def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False)
    steps = paths.shape[1] - 1
    d, width = features(paths[:, 0], payoffs[:, 0]).shape[1], 16
    n = len(paths)

    coefs = np.zeros((steps, d)); means = np.zeros((steps, d)); scales = np.ones((steps, d))
    w1s = np.zeros((steps, d, width)); b1s = np.zeros((steps, width))
    w2s = np.zeros((steps, width)); b2s = np.zeros(steps); target_scales = np.ones(steps)
    targets = np.zeros((steps, n)); losses = np.zeros((steps, 2)); updates = np.zeros(steps)

    # One fixed split prevents time-by-time fold drift from leaking fitted decisions.
    order = rng.permutation(n)
    folds = np.empty(n, dtype=np.int8)
    folds[order] = np.arange(n) % 2
    cash = payoffs[:, -1].astype(float, copy=True)
    for t in range(steps - 1, -1, -1):
        targets[t] = cash
        x_raw = features(paths[:, t], payoffs[:, t])
        oof = np.empty(n)
        # Two 12-step fold fits plus one 32-step final fit stay below the parent's
        # 64 optimizer steps while making stopping decisions genuinely out-of-fold.
        for fold in (0, 1):
            train, valid = folds != fold, folds == fold
            fold_model, _, _ = _robust_model(x_raw[train], cash[train], rng, 12, width)
            oof[valid] = _apply(fold_model, x_raw[valid])
        cash = np.where(payoffs[:, t] >= oof, payoffs[:, t], cash)

        model, _, trace = _robust_model(x_raw, targets[t], rng, 32, width)
        coef, mean, scale, w1, b1, w2, b2, target_scale = model
        coefs[t], means[t], scales[t] = coef, mean, scale
        w1s[t], b1s[t], w2s[t], b2s[t], target_scales[t] = w1, b1, w2, b2, target_scale
        losses[t] = trace[:2]; updates[t] = trace[2]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "model.npz", coefficients=coefs, means=means, scales=scales,
             w1=w1s, b1=b1s, w2=w2s, b2=b2s, target_scales=target_scales)
    np.savez(output_dir / "training_trace.npz", backward_targets=targets,
             loss_before_after=losses, first_layer_update_norm=updates,
             folds=folds, seed=np.array(seed))


def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    t = int(json.loads((input_dir / "request.json").read_text())["time_index"])
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    immediate = np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False)
    with np.load(model_dir / "model.npz", allow_pickle=False) as m:
        x = (features(states, immediate) - m["means"][t]) / m["scales"][t]
        residual = np.tanh(x @ m["w1"][t] + m["b1"][t]) @ m["w2"][t] + m["b2"][t]
        out = x @ m["coefficients"][t] + m["target_scales"][t] * residual
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", np.asarray(out, dtype=float), allow_pickle=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit"); f.add_argument("--input", required=True); f.add_argument("--output", required=True); f.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("predict"); p.add_argument("--model", required=True); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit(Path(args.input), Path(args.output), args.seed)
    else:
        predict(Path(args.model), Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
