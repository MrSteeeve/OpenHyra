"""Offline fixture: filtered historical simulation (EWMA-standardised joint
residual bootstrap) with variance mean reversion; Q random walk."""

import argparse
import json
from pathlib import Path

import numpy as np


def forecast(history, query):
    rng = np.random.default_rng(int(query["seed"]))
    n, horizon = int(query["n_paths"]), int(query["horizon"])
    returns = np.asarray(history["returns"])
    returns = returns[np.all(np.isfinite(returns), axis=1)]
    variance = np.var(returns[:20], axis=0)
    standardised = np.empty_like(returns)
    for k, row in enumerate(returns):
        standardised[k] = row / np.sqrt(variance)
        variance = 0.94 * variance + 0.06 * row ** 2
    pool = standardised
    long_run = np.var(returns, axis=0)
    picks = rng.integers(0, len(pool), size=(n, horizon))
    decay = 0.97 ** np.arange(horizon)
    path_variance = long_run + (variance - long_run) * decay[:, None]
    paths = pool[picks] * np.sqrt(path_variance)[None, :, :]
    paths = np.clip(paths, -0.2, 0.2)
    q = np.empty((n, 3))
    for j, values in enumerate((history["iv_1m"], history["carry"][:, 0], history["carry"][:, 1])):
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        level = float(finite[-1]) if finite.size else (0.2 if j == 0 else 0.1)
        changes = finite[horizon:] - finite[:-horizon] if finite.size > horizon + 20 else np.array([0.02, -0.02])
        q[:, j] = level + rng.choice(changes[-504:], size=n)
    q[:, 0] = np.maximum(q[:, 0], 0.01)
    return paths, q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    with np.load(source / "history.npz", allow_pickle=False) as data:
        history = {name: data[name] for name in data.files}
    returns, q = forecast(history, json.loads((source / "query.json").read_text()))
    np.savez(Path(args.output) / "forecast.npz", returns=returns, q=q)


if __name__ == "__main__":
    main()
