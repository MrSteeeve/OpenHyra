#!/usr/bin/env python3
"""Cross-fitted extremely-randomized forests for direct Bermudan decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


N_TREES = 24
OOF_TREES = 10
MAX_DEPTH = 7
MIN_LEAF = 24


def _features(history: np.ndarray, states: np.ndarray, immediate: np.ndarray) -> np.ndarray:
    """Causal, scale-free summaries of the supplied prefix."""
    h = np.asarray(history, dtype=float)
    x = np.asarray(states, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if h.ndim == 2:
        h = h[:, :, None]
    eps = 1e-12
    base = np.maximum(np.abs(h[:, 0, :]), eps)
    ratio = x / base
    lo = np.min(h, axis=1) / base
    hi = np.max(h, axis=1) / base
    mean = np.mean(h / base[:, None, :], axis=1)
    if h.shape[1] > 1:
        lag = x / np.maximum(np.abs(h[:, -2, :]), eps) - 1.0
    else:
        lag = np.zeros_like(x)
    cross = np.column_stack((np.mean(ratio, axis=1), np.min(ratio, axis=1),
                             np.max(ratio, axis=1), np.std(ratio, axis=1)))
    g = np.asarray(immediate, dtype=float).reshape(-1, 1)
    return np.nan_to_num(np.concatenate((ratio, ratio * ratio, lo, hi, mean, lag,
                                          cross, g), axis=1), nan=0.0,
                         posinf=1e6, neginf=-1e6)


def _grow_tree(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    feat, cut, left, right, value = [], [], [], [], []

    def grow(rows: np.ndarray, depth: int) -> int:
        node = len(value)
        feat.append(-1); cut.append(0.0); left.append(-1); right.append(-1)
        value.append(float(np.mean(y[rows])) if rows.size else 0.0)
        if depth >= MAX_DEPTH or rows.size < 2 * MIN_LEAF:
            return node
        p = x.shape[1]
        candidates = rng.choice(p, size=min(p, max(3, int(np.sqrt(p)) + 1)), replace=False)
        best = None
        for j in candidates:
            z = x[rows, j]
            zlo, zhi = float(np.min(z)), float(np.max(z))
            if not np.isfinite(zlo + zhi) or zhi <= zlo:
                continue
            for threshold in rng.uniform(zlo, zhi, size=3):
                mask = z <= threshold
                nl = int(np.sum(mask)); nr = rows.size - nl
                if nl < MIN_LEAF or nr < MIN_LEAF:
                    continue
                yl, yr = y[rows[mask]], y[rows[~mask]]
                loss = float(np.var(yl) * nl + np.var(yr) * nr)
                if best is None or loss < best[0]:
                    best = (loss, int(j), float(threshold), rows[mask], rows[~mask])
        if best is not None:
            _, feat[node], cut[node], lrows, rrows = best
            left[node] = grow(lrows, depth + 1)
            right[node] = grow(rrows, depth + 1)
        return node

    grow(np.arange(x.shape[0]), 0)
    return {"f": feat, "c": cut, "l": left, "r": right, "v": value}


def _forest(x: np.ndarray, y: np.ndarray, count: int, rng: np.random.Generator) -> list[dict]:
    if x.shape[0] == 0:
        return [{"f": [-1], "c": [0.0], "l": [-1], "r": [-1], "v": [0.0]}]
    trees = []
    for _ in range(count):
        # Subsampling decorrelates the already-randomized trees without replacement.
        size = max(min(x.shape[0], MIN_LEAF * 2), int(0.82 * x.shape[0]))
        rows = rng.choice(x.shape[0], size=min(size, x.shape[0]), replace=False)
        trees.append(_grow_tree(x[rows], y[rows], rng))
    return trees


def _tree_predict(tree: dict, x: np.ndarray) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=float)
    for i in range(x.shape[0]):
        node = 0
        while tree["f"][node] >= 0:
            node = tree["l"][node] if x[i, tree["f"][node]] <= tree["c"][node] else tree["r"][node]
        out[i] = tree["v"][node]
    return out


def _predict_forest(trees: list[dict], x: np.ndarray) -> np.ndarray:
    return np.mean([_tree_predict(tree, x) for tree in trees], axis=0)


def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.asarray(np.load(input_dir / "payoffs.npy", allow_pickle=False), dtype=float)
    n, n_times = payoffs.shape
    # Search-specified randomness is fixed so model replay is independent of process state.
    rng = np.random.default_rng(41000123)
    fold = np.empty(n, dtype=int)
    fold[rng.permutation(n)] = np.arange(n) % 4
    cash = payoffs[:, -1].copy()
    forests: list[list[dict]] = [[] for _ in range(n_times - 1)]
    margins = np.zeros(n_times - 1, dtype=float)

    for t in range(n_times - 2, -1, -1):
        future_cash = cash.copy()
        g = payoffs[:, t]
        design = _features(paths[:, :t + 1, :], paths[:, t, :], g)
        eligible = g > 0.0
        oof = np.full(n, float(np.mean(future_cash)), dtype=float)
        for k in range(4):
            train = eligible & (fold != k)
            valid = fold == k
            if np.sum(train) >= 2 * MIN_LEAF:
                local = _forest(design[train], future_cash[train], OOF_TREES, rng)
                oof[valid] = _predict_forest(local, design[valid])

        # Select the conservative exercise buffer solely on held-out predictions.
        residual = future_cash[eligible] - oof[eligible]
        scale = float(np.median(np.abs(residual - np.median(residual)))) if residual.size else 0.0
        grid = np.array([0.0, 0.15, 0.30, 0.50, 0.75, 1.0]) * max(scale, 1e-10)
        values = []
        for margin in grid:
            exercise = eligible & (g > oof + margin)
            values.append(float(np.mean(np.where(exercise, g, future_cash))))
        chosen = float(grid[int(np.argmax(values))])
        # Earlier dates may be more conservative; the buffer shrinks monotonically.
        if t + 1 < margins.size:
            chosen = max(chosen, margins[t + 1])
        margins[t] = chosen
        exercise = eligible & (g > oof + chosen)
        cash = np.where(exercise, g, future_cash)

        train = eligible
        if np.sum(train) < 2 * MIN_LEAF:
            train = np.ones(n, dtype=bool)
        forests[t] = _forest(design[train], future_cash[train], N_TREES, rng)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.json").write_text(json.dumps({"margins": margins.tolist(), "forests": forests},
                                                       separators=(",", ":")), encoding="utf-8")


def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    model = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    request = json.loads((input_dir / "request.json").read_text(encoding="utf-8"))
    t = int(request["time_index"])
    history = np.load(input_dir / "history.npy", allow_pickle=False)
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    immediate = np.asarray(np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False), dtype=float)
    design = _features(history, states, immediate)
    continuation = _predict_forest(model["forests"][t], design)
    decisions = ((immediate > 0.0) & (immediate > continuation + model["margins"][t])).astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", decisions, allow_pickle=False)


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
