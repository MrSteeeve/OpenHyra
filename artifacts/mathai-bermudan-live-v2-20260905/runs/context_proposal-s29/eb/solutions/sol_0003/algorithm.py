#!/usr/bin/env python3
"""Cross-fitted, robust k-nearest-neighbour Bermudan stopping policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FIXED_SEED = 29000087


def _features(history: np.ndarray, states: np.ndarray, immediate: np.ndarray) -> np.ndarray:
    """Causal level, return, dispersion, and path-summary coordinates."""
    h = np.asarray(history, dtype=np.float64)
    x = np.asarray(states, dtype=np.float64)
    if h.ndim == 2:
        h = h[:, :, None]
    if x.ndim == 1:
        x = x[:, None]
    tiny = 1e-12
    logh = np.log(np.maximum(h, tiny))
    logx = np.log(np.maximum(x, tiny))
    relative = logh - logh[:, :1, :]
    if h.shape[1] > 1:
        increments = np.diff(logh, axis=1)
        volatility = np.sqrt(np.mean(increments * increments, axis=1))
        trend = relative[:, -1, :] / float(h.shape[1] - 1)
    else:
        volatility = np.zeros_like(logx)
        trend = np.zeros_like(logx)
    running_mean = np.mean(relative, axis=1)
    cross_mean = np.mean(logx, axis=1, keepdims=True)
    cross_spread = (np.max(logx, axis=1) - np.min(logx, axis=1))[:, None]
    payoff = np.asarray(immediate, dtype=np.float64).reshape(-1, 1)
    return np.concatenate(
        [logx, relative[:, -1, :], running_mean, volatility, trend,
         cross_mean, cross_spread, payoff], axis=1
    )


def _robust_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(x, axis=0)
    q25, q75 = np.percentile(x, [25.0, 75.0], axis=0)
    scale = q75 - q25
    fallback = np.std(x, axis=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = np.clip((x - center) / scale, -8.0, 8.0)
    return z, center, scale


def _knn(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, k: int) -> np.ndarray:
    """Exact distance-weighted kNN, chunked to bound transient memory."""
    n_train = train_x.shape[0]
    if n_train == 0:
        return np.zeros(query_x.shape[0], dtype=np.float64)
    k = max(1, min(int(k), n_train))
    out = np.empty(query_x.shape[0], dtype=np.float64)
    train_norm = np.sum(train_x * train_x, axis=1)
    for start in range(0, query_x.shape[0], 128):
        q = query_x[start:start + 128]
        d2 = np.maximum(
            np.sum(q * q, axis=1)[:, None] + train_norm[None, :] - 2.0 * q @ train_x.T,
            0.0,
        )
        ids = np.argpartition(d2, k - 1, axis=1)[:, :k]
        near_d2 = np.take_along_axis(d2, ids, axis=1)
        near_y = train_y[ids]
        # A small local bandwidth prevents one nearly coincident point dominating.
        bandwidth = np.median(near_d2, axis=1, keepdims=True) + 1e-8
        weights = 1.0 / np.sqrt(near_d2 + 0.05 * bandwidth + 1e-10)
        out[start:start + len(q)] = np.sum(weights * near_y, axis=1) / np.sum(weights, axis=1)
    return out


def _calibrate_threshold(immediate: np.ndarray, continuation: np.ndarray,
                         future: np.ndarray, later: float) -> float:
    margin = immediate - continuation
    active = immediate > 1e-12
    if np.count_nonzero(active) < 16:
        return float(0.25 * later)
    payoff_scale = max(float(np.median(np.abs(future[active]))), 1e-8)
    qs = np.percentile(margin[active], [20, 35, 50, 65, 80])
    candidates = np.unique(np.r_[0.0, qs])
    candidates = np.clip(candidates, -0.12 * payoff_scale, 0.12 * payoff_scale)
    values = [np.mean(np.where(active & (margin >= q), immediate, future)) for q in candidates]
    raw = float(candidates[int(np.argmax(values))])
    # Shrink selection noise toward the neutral boundary and toward the next date.
    return 0.60 * raw + 0.25 * float(later)


def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    del seed  # The proposal specifies one fixed source of algorithmic randomness.
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.asarray(np.load(input_dir / "payoffs.npy", allow_pickle=False), dtype=np.float64)
    n_paths, n_times, _ = paths.shape
    n_steps = n_times - 1
    rng = np.random.default_rng(FIXED_SEED)
    fold = np.empty(n_paths, dtype=np.int16)
    fold[rng.permutation(n_paths)] = np.arange(n_paths) % min(4, n_paths)
    folds = int(np.max(fold)) + 1
    k = max(12, min(48, int(round(np.sqrt(n_paths)))))
    cash_flow = payoffs[:, -1].copy()
    all_x: list[np.ndarray] = [None] * n_steps  # type: ignore[list-item]
    centers: list[np.ndarray] = [None] * n_steps  # type: ignore[list-item]
    scales: list[np.ndarray] = [None] * n_steps  # type: ignore[list-item]
    thresholds = np.zeros(n_steps, dtype=np.float64)
    future_targets = np.empty((n_steps, n_paths), dtype=np.float64)
    later_threshold = 0.0

    for t in range(n_steps - 1, -1, -1):
        future_targets[t] = cash_flow
        raw = _features(paths[:, :t + 1, :], paths[:, t, :], payoffs[:, t])
        z, center, scale = _robust_scale(raw)
        oof = np.empty(n_paths, dtype=np.float64)
        for f in range(folds):
            valid = fold == f
            train = ~valid
            oof[valid] = _knn(z[train], cash_flow[train], z[valid], k)
        threshold = _calibrate_threshold(payoffs[:, t], oof, cash_flow, later_threshold)
        exercise = (payoffs[:, t] > 1e-12) & (payoffs[:, t] - oof >= threshold)
        cash_flow = np.where(exercise, payoffs[:, t], cash_flow)
        all_x[t], centers[t], scales[t] = z, center, scale
        thresholds[t] = threshold
        later_threshold = threshold

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "knn_policy.npz",
        train_x=np.stack(all_x),
        targets=future_targets,
        centers=np.stack(centers),
        scales=np.stack(scales),
        thresholds=thresholds,
        k=np.asarray(k, dtype=np.int64),
    )


def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    request = json.loads((input_dir / "request.json").read_text(encoding="utf-8"))
    t = int(request["time_index"])
    history = np.load(input_dir / "history.npy", allow_pickle=False)
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    immediate = np.asarray(np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False), dtype=np.float64)
    with np.load(model_dir / "knn_policy.npz", allow_pickle=False) as model:
        raw = _features(history, states, immediate)
        query = np.clip((raw - model["centers"][t]) / model["scales"][t], -8.0, 8.0)
        continuation = _knn(model["train_x"][t], model["targets"][t], query, int(model["k"]))
        decisions = (immediate > 1e-12) & (immediate - continuation >= model["thresholds"][t])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", decisions.astype(np.bool_), allow_pickle=False)


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
