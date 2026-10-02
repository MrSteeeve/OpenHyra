"""Cross-fitted, exercise-gated ridge continuation algorithm.

The evaluator owns paths, payoffs, and the stopping rule.  This program only
learns discounted continuation values from the supplied causal state paths.
"""

import argparse
import json
import os

import numpy as np


LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0)
N_FOLDS = 8


def _features(states, payoff):
    """Compact symmetric state description plus the current payoff."""
    x = np.asarray(states, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    p = np.asarray(payoff, dtype=np.float64).reshape(-1)
    # Prices are positive in the benchmark.  Clipping also makes malformed or
    # numerically underflowed inputs harmless at prediction time.
    z = np.log(np.maximum(x, 1.0e-12))
    ordered = np.sort(z, axis=1)
    gap = ordered[:, 1] - ordered[:, 0] if z.shape[1] > 1 else np.zeros(z.shape[0])
    return np.column_stack((
        z,
        z * z,
        p,
        p * p,
        np.min(z, axis=1),
        np.mean(z, axis=1),
        np.max(z, axis=1),
        gap,
    ))


def _constant_target(y):
    if len(y) == 0:
        return 0.0
    return float(max(0.0, np.mean(y)))


def _ridge_fit(x, y, lam):
    """Fit a standardized ridge with an unpenalized, centered intercept."""
    n, d = x.shape
    mu = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1.0e-10, scale, 1.0)
    z = (x - mu) / scale
    intercept = float(np.mean(y))
    yc = y - intercept
    system = z.T @ z
    system.flat[::d + 1] += lam
    rhs = z.T @ yc
    try:
        coef = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return mu, scale, coef, intercept


def _ridge_predict(model, x):
    mu, scale, coef, intercept = model
    return intercept + ((x - mu) / scale) @ coef


def _fit_or_constant(x, y, lam, min_samples):
    if len(y) < min_samples:
        return None, _constant_target(y)
    try:
        model = _ridge_fit(x, y, lam)
        if all(np.all(np.isfinite(a)) for a in model):
            return model, _constant_target(y)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        pass
    return None, _constant_target(y)


def _predict_or_constant(model, constant, x):
    if model is None:
        return np.full(x.shape[0], constant, dtype=np.float64)
    result = _ridge_predict(model, x)
    return np.where(np.isfinite(result), result, constant)


def fit(input_dir, output_dir, seed):
    paths = np.load(os.path.join(input_dir, "training_paths.npy"), allow_pickle=False)
    payoffs = np.load(os.path.join(input_dir, "payoffs.npy"), allow_pickle=False)
    paths = np.asarray(paths, dtype=np.float64)
    payoffs = np.asarray(payoffs, dtype=np.float64)
    if paths.ndim != 3 or payoffs.ndim != 2:
        raise ValueError("expected paths (n,t,d) and payoffs (n,t)")
    if paths.shape[:2] != payoffs.shape:
        raise ValueError("path and payoff grids do not match")

    n_paths, n_times, _ = paths.shape
    folds = np.empty(n_paths, dtype=np.int64)
    order = np.random.default_rng(int(seed)).permutation(n_paths)
    folds[order] = np.arange(n_paths) % min(N_FOLDS, n_paths)
    fold_count = min(N_FOLDS, n_paths)

    # cashflow is always a discounted payoff, so no additional discounting is
    # needed while walking backward through evaluator-computed payoffs.
    cashflow = np.maximum(payoffs[:, -1], 0.0).copy()
    targets = np.zeros((max(0, n_times - 1), n_paths), dtype=np.float64)
    selected = np.zeros(max(0, n_times - 1), dtype=np.float64)
    positive_counts = np.zeros(max(0, n_times - 1), dtype=np.int64)
    oof_mse = np.full(max(0, n_times - 1), np.nan, dtype=np.float64)

    for t in range(n_times - 2, -1, -1):
        target = cashflow.copy()
        targets[t] = target
        p = payoffs[:, t]
        x = _features(paths[:, t, :], p)
        positive = p > 0.0
        positive_counts[t] = int(np.sum(positive))
        min_samples = max(6, x.shape[1] + 2)

        losses = []
        for lam in LAMBDAS:
            squared_error = 0.0
            count = 0
            for fold in range(fold_count):
                train = positive & (folds != fold)
                valid = positive & (folds == fold)
                if not np.any(valid):
                    continue
                model, const = _fit_or_constant(
                    x[train], target[train], lam, min_samples
                )
                pred = _predict_or_constant(model, const, x[valid])
                squared_error += float(np.sum((pred - target[valid]) ** 2))
                count += int(np.sum(valid))
            losses.append(squared_error / count if count else float("inf"))

        best_index = int(np.argmin(losses))
        lam = LAMBDAS[best_index]
        selected[t] = lam
        oof_mse[t] = losses[best_index]
        continuation = np.zeros(n_paths, dtype=np.float64)
        for fold in range(fold_count):
            train = positive & (folds != fold)
            held_out = folds == fold
            model, const = _fit_or_constant(
                x[train], target[train], lam, min_samples
            )
            continuation[held_out] = _predict_or_constant(
                model, const, x[held_out]
            )
        continuation = np.maximum(continuation, 0.0)
        exercise = positive & (p > continuation)
        cashflow[exercise] = p[exercise]

    # Refit deployment models on all gated samples, using targets produced by
    # the cross-fitted recursion rather than self-fitted stopping labels.
    feature_dim = _features(paths[:, 0, :], payoffs[:, 0]).shape[1]
    means = np.zeros((n_times - 1, feature_dim), dtype=np.float64)
    scales = np.ones_like(means)
    coefs = np.zeros_like(means)
    intercepts = np.zeros(n_times - 1, dtype=np.float64)
    constants = np.zeros(n_times - 1, dtype=np.float64)
    is_constant = np.ones(n_times - 1, dtype=np.uint8)

    for t in range(n_times - 1):
        p = payoffs[:, t]
        x = _features(paths[:, t, :], p)
        positive = p > 0.0
        min_samples = max(6, feature_dim + 2)
        model, const = _fit_or_constant(
            x[positive], targets[t, positive], selected[t], min_samples
        )
        constants[t] = const
        if model is not None:
            means[t], scales[t], coefs[t], intercepts[t] = model
            is_constant[t] = 0
        else:
            intercepts[t] = const

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "model.npz"),
        means=means,
        scales=scales,
        coefs=coefs,
        intercepts=intercepts,
        constants=constants,
        is_constant=is_constant,
        n_times=np.asarray(n_times, dtype=np.int64),
    )
    baseline = float(np.mean(np.maximum(payoffs[:, -1], 0.0)))
    diagnostics = {
        "folds": int(fold_count),
        "positive_counts": positive_counts.tolist(),
        "selected_lambdas": selected.tolist(),
        "oof_mse": [None if not np.isfinite(v) else float(v) for v in oof_mse],
        "oof_policy_payoff": float(np.mean(cashflow)),
        "terminal_only_payoff": baseline,
        "oof_worse_than_terminal": bool(np.mean(cashflow) + 1.0e-12 < baseline),
    }
    with open(os.path.join(output_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, separators=(",", ":"), sort_keys=True)


def _query_time(input_dir, history, n_times):
    with open(os.path.join(input_dir, "request.json"), "r", encoding="utf-8") as f:
        request = json.load(f)
    # Prefer an evaluator-supplied discrete index when present.  The history
    # length is a robust fallback for batched causal histories.
    for key in ("time_index", "time_idx", "exercise_index", "exercise_time_index"):
        if key in request:
            t = int(request[key])
            if 0 <= t < n_times - 1:
                return t
    if history.ndim >= 3:
        t = history.shape[1] - 1
        if 0 <= t < n_times - 1:
            return t
    raise ValueError("cannot determine a valid non-terminal query time")


def predict(model_dir, input_dir, output_dir):
    states = np.load(os.path.join(input_dir, "states.npy"), allow_pickle=False)
    payoff = np.load(os.path.join(input_dir, "immediate_payoffs.npy"), allow_pickle=False)
    history = np.load(os.path.join(input_dir, "history.npy"), allow_pickle=False)
    with np.load(os.path.join(model_dir, "model.npz"), allow_pickle=False) as model:
        n_times = int(model["n_times"])
        t = _query_time(input_dir, history, n_times)
        x = _features(states, payoff)
        if bool(model["is_constant"][t]):
            prediction = np.full(x.shape[0], model["constants"][t], dtype=np.float64)
        else:
            prediction = model["intercepts"][t] + (
                (x - model["means"][t]) / model["scales"][t]
            ) @ model["coefs"][t]
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=1.0e100, neginf=0.0)
        prediction = np.maximum(prediction, 0.0)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "predictions.npy"), prediction, allow_pickle=False)


def _main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--seed", required=True, type=int)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit(args.input, args.output, args.seed)
    else:
        predict(args.model, args.input, args.output)


if __name__ == "__main__":
    _main()
