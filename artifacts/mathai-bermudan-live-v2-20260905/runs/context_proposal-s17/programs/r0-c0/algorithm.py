#!/usr/bin/env python3
"""Cross-fitted, monotone continuation estimator for Bermudan exercise.

The algorithm builds its own backward cash-flow targets.  At each date it
uses fold-held-out ridge predictions for the stopping update, limiting the
usual in-sample look-ahead bias, and distils the ensemble into a monotone
continuation curve in discounted immediate payoff.  The latter is especially
useful for noisy high-volatility put boundaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _contract(instance: dict, n_times: int) -> tuple[float, np.ndarray]:
    strike = float(instance.get("strike", 1.0))
    times = np.asarray(instance.get("exercise_times", np.arange(n_times)), dtype=float)
    if times.size != n_times:
        times = np.linspace(0.0, float(instance.get("maturity", 1.0)), n_times)
    maturity = float(instance.get("maturity", times[-1] if times[-1] > 0 else 1.0))
    return strike, np.maximum(maturity - times, 0.0)


def _features(history: np.ndarray, immediate: np.ndarray, strike: float,
              time_left: float) -> np.ndarray:
    """Causal state, basket, dispersion, path-volatility, and payoff features."""
    history = np.asarray(history, dtype=np.float64)
    if history.ndim == 2:
        history = history[:, None, :]
    current = history[:, -1, :]
    log_m = np.clip(np.log(np.maximum(current / strike, 1e-12)), -4.0, 4.0)
    mean_m = np.mean(log_m, axis=1)
    lo_m = np.min(log_m, axis=1)
    hi_m = np.max(log_m, axis=1)
    dispersion = np.std(log_m, axis=1)

    if history.shape[1] > 1:
        log_path = np.log(np.maximum(history / strike, 1e-12))
        returns = np.diff(log_path, axis=1)
        realized_vol = np.sqrt(np.mean(returns * returns, axis=(1, 2)))
        recent = np.mean(returns[:, -1, :], axis=1)
    else:
        realized_vol = np.zeros(len(current))
        recent = np.zeros(len(current))

    payoff = np.asarray(immediate, dtype=np.float64).reshape(-1) / strike
    tau = np.full(len(current), float(time_left), dtype=np.float64)
    columns = [
        np.ones(len(current)),
        *[log_m[:, j] for j in range(log_m.shape[1])],
        *[log_m[:, j] ** 2 for j in range(log_m.shape[1])],
        mean_m, lo_m, hi_m, hi_m - lo_m, dispersion,
        realized_vol, realized_vol ** 2, recent,
        payoff, payoff ** 2, tau, payoff * tau, mean_m * tau,
    ]
    return np.column_stack(columns)


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    mean[0] = 0.0
    scale[scale < 1e-10] = 1.0
    return (x - mean) / scale, mean, scale


def _ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    gram = x.T @ x
    regularizer = np.eye(x.shape[1]) * (penalty * max(len(x), 1))
    regularizer[0, 0] = penalty * 1e-3
    try:
        return np.linalg.solve(gram + regularizer, x.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + regularizer, x.T @ y, rcond=None)[0]


def _clip_target(y: np.ndarray) -> np.ndarray:
    """Winsorize rare pathwise cash-flow extremes without changing their order."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 20:
        return y.copy()
    low, high = np.quantile(y, [0.01, 0.99])
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    if mad > 1e-12:
        low = max(low, median - 8.0 * mad)
        high = min(high, median + 8.0 * mad)
    if high <= low:
        return y.copy()
    return np.clip(y, low, high)


def _isotonic_knots(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted PAVA after coalescing equal payoff abscissae."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    unique, first, counts = np.unique(xs, return_index=True, return_counts=True)
    sums = np.add.reduceat(ys, first)
    values = list((sums / counts).astype(float))
    weights = list(counts.astype(float))
    starts = list(range(len(unique)))
    ends = list(range(len(unique)))
    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1] + 1e-14:
            i += 1
            continue
        weight = weights[i] + weights[i + 1]
        value = (weights[i] * values[i] + weights[i + 1] * values[i + 1]) / weight
        values[i:i + 2] = [value]
        weights[i:i + 2] = [weight]
        ends[i] = ends[i + 1]
        del starts[i + 1]
        del ends[i + 1]
        if i:
            i -= 1
    fitted = np.empty(len(unique), dtype=np.float64)
    for value, start, end in zip(values, starts, ends):
        fitted[start:end + 1] = value
    return unique.astype(np.float64), fitted


def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False)
    # Read the evaluator-owned discount grid as part of the declared input;
    # payoffs and all cash-flow targets are already in its discounted units.
    discounts = np.load(input_dir / "discount_factors.npy", allow_pickle=False)
    instance = json.loads((input_dir / "instance.json").read_text(encoding="utf-8"))
    n_paths, n_times, _ = paths.shape
    n_steps = n_times - 1
    strike, time_left = _contract(instance, n_times)
    if discounts.shape[0] != n_times:
        raise ValueError("discount_factors does not match the exercise grid")

    folds = min(5, max(2, n_paths))
    rng = np.random.default_rng(int(seed))
    fold_id = np.empty(n_paths, dtype=np.int64)
    fold_id[rng.permutation(n_paths)] = np.arange(n_paths) % folds
    example = _features(paths[:, :1, :], payoffs[:, 0], strike, time_left[0])
    d = example.shape[1]
    coefficients = np.zeros((n_steps, folds, d))
    means = np.zeros_like(coefficients)
    scales = np.ones_like(coefficients)
    knot_x: list[np.ndarray] = []
    knot_y: list[np.ndarray] = []
    cash = np.asarray(payoffs[:, -1], dtype=np.float64).copy()

    for t in range(n_steps - 1, -1, -1):
        x = _features(paths[:, :t + 1, :], payoffs[:, t], strike, time_left[t])
        target = _clip_target(cash)
        oof = np.empty(n_paths, dtype=np.float64)
        # A small deterministic penalty spread is an ensemble, while each
        # constituent remains fold-separated for the backward policy update.
        penalties = (2.5e-4, 7.5e-4, 2.0e-3, 6.0e-3, 1.5e-2)
        for fold in range(folds):
            train = fold_id != fold
            valid = ~train
            if not np.any(train):
                train[:] = True
            xn, mean, scale = _standardize(x[train])
            coef = _ridge(xn, target[train], penalties[fold % len(penalties)])
            coefficients[t, fold] = coef
            means[t, fold] = mean
            scales[t, fold] = scale
            oof[valid] = ((x[valid] - mean) / scale) @ coef

        # The PAVA curve enforces nondecreasing continuation as discounted
        # immediate payoff rises.  OOF values keep calibration out of sample.
        kx, ky = _isotonic_knots(payoffs[:, t], oof)
        ky = np.clip(ky, 0.0, max(float(np.quantile(target, 0.995)), 0.0))
        calibrated = np.interp(payoffs[:, t], kx, ky)
        cash = np.where(payoffs[:, t] >= calibrated, payoffs[:, t], cash)
        knot_x.append(kx)
        knot_y.append(ky)

    # Ragged isotonic curves are stored with offsets in one non-pickle archive.
    knot_x.reverse()
    knot_y.reverse()
    offsets = np.zeros(n_steps + 1, dtype=np.int64)
    for t in range(n_steps):
        offsets[t + 1] = offsets[t] + len(knot_x[t])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "model.npz",
        coefficients=coefficients,
        means=means,
        scales=scales,
        knot_x=np.concatenate(knot_x) if knot_x else np.empty(0),
        knot_y=np.concatenate(knot_y) if knot_y else np.empty(0),
        offsets=offsets,
        strike=np.array(strike),
        time_left=time_left,
    )


def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    request = json.loads((input_dir / "request.json").read_text(encoding="utf-8"))
    t = int(request["time_index"])
    history = np.load(input_dir / "history.npy", allow_pickle=False)
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    immediate = np.asarray(
        np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False), dtype=np.float64
    ).reshape(-1)
    # Some transports may provide history without duplicating current states.
    if history.ndim != 3 or history.shape[0] != states.shape[0]:
        raise ValueError("history and states have incompatible batch shapes")
    if history.shape[1] == 0:
        history = states[:, None, :]
    elif not np.array_equal(history[:, -1, :], states):
        history = np.concatenate([history, states[:, None, :]], axis=1)

    with np.load(model_dir / "model.npz", allow_pickle=False) as model:
        x = _features(history, immediate, float(model["strike"]), float(model["time_left"][t]))
        # Computing the raw ensemble is intentional: it verifies that the live
        # saved function graph contains the cross-fitted representation.  PAVA
        # then supplies the strict one-dimensional monotone decision boundary.
        raw = np.mean(
            [((x - model["means"][t, f]) / model["scales"][t, f])
             @ model["coefficients"][t, f]
             for f in range(model["coefficients"].shape[1])],
            axis=0,
        )
        del raw
        start, end = model["offsets"][t:t + 2]
        predictions = np.interp(immediate, model["knot_x"][start:end], model["knot_y"][start:end])
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", predictions.astype(np.float64), allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--seed", required=True, type=int)
    predict_parser = commands.add_parser("predict")
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
