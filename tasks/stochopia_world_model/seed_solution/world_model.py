"""Seed world model: EWMA Gaussian GBM with trailing correlation; Q random walk.

Contract (see TASK.md):
    python world_model.py --input IN --output OUT
reads IN/history.npz and IN/query.json, writes OUT/forecast.npz with
``returns`` (n_paths, horizon, 2) daily log returns and ``q`` (n_paths, 3).
"""

import argparse
import json
from pathlib import Path

import numpy as np

Q_DEFAULTS = (0.20, 0.10, 0.10)


def _last_finite(values, default):
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else float(default)


def _change_scale(values, lag, window, default):
    recent = values[-window:]
    changes = recent[lag:] - recent[:-lag]
    changes = changes[np.isfinite(changes)]
    return float(np.std(changes)) if changes.size >= 20 else float(default)


def forecast(history, query):
    rng = np.random.default_rng(int(query["seed"]))
    n, horizon = int(query["n_paths"]), int(query["horizon"])
    returns = np.asarray(history["returns"])
    returns = returns[np.all(np.isfinite(returns), axis=1)]
    variance = np.var(returns[:20], axis=0)
    for row in returns[20:]:
        variance = 0.94 * variance + 0.06 * row ** 2
    corr = np.corrcoef(returns[-252:].T)[0, 1]
    chol = np.linalg.cholesky(np.array([[1.0, corr], [corr, 1.0]]))
    paths = (rng.standard_normal((n, horizon, 2)) @ chol.T) * np.sqrt(variance)
    series = (history["iv_1m"], history["carry"][:, 0], history["carry"][:, 1])
    q = np.empty((n, 3))
    for j, values in enumerate(series):
        level = _last_finite(np.asarray(values), Q_DEFAULTS[j])
        scale = _change_scale(np.asarray(values), horizon, 126, 0.02)
        q[:, j] = level + scale * rng.standard_normal(n)
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
    query = json.loads((source / "query.json").read_text())
    returns, q = forecast(history, query)
    np.savez(Path(args.output) / "forecast.npz", returns=returns, q=q)


if __name__ == "__main__":
    main()
