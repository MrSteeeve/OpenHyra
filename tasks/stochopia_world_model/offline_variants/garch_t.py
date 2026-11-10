"""Offline fixture: GARCH(1,1) with Student-t shocks; IV mean-reverts (AR(1))."""

import argparse
import json
from pathlib import Path

import numpy as np

ALPHA, BETA, NU = 0.08, 0.90, 5.0


def _ar1_forecast(values, horizon, rng, n, default):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2 * horizon + 40:
        level = float(finite[-1]) if finite.size else default
        return level + 0.02 * rng.standard_normal(n)
    x, y = finite[:-horizon], finite[horizon:]
    slope, intercept = np.polyfit(x, y, 1)
    slope = float(np.clip(slope, 0.0, 1.0))
    intercept = float(np.mean(y) - slope * np.mean(x))
    residual = y - (intercept + slope * x)
    return intercept + slope * finite[-1] + np.std(residual) * rng.standard_normal(n)


def forecast(history, query):
    rng = np.random.default_rng(int(query["seed"]))
    n, horizon = int(query["n_paths"]), int(query["horizon"])
    returns = np.asarray(history["returns"])
    returns = returns[np.all(np.isfinite(returns), axis=1)]
    long_run = np.var(returns, axis=0)
    omega = long_run * (1.0 - ALPHA - BETA)
    variance = np.var(returns[:20], axis=0)
    for row in returns[20:]:
        variance = omega + ALPHA * row ** 2 + BETA * variance
    corr = np.corrcoef(returns[-252:].T)[0, 1]
    chol = np.linalg.cholesky(np.array([[1.0, corr], [corr, 1.0]]))
    scale = np.sqrt((NU - 2.0) / NU)
    paths = np.empty((n, horizon, 2))
    current = np.tile(variance, (n, 1))
    for step in range(horizon):
        z = rng.standard_normal((n, 2)) @ chol.T
        chi = rng.chisquare(NU, size=(n, 1))
        shock = z / np.sqrt(chi / NU) * scale
        paths[:, step] = np.sqrt(current) * shock
        current = omega + ALPHA * paths[:, step] ** 2 + BETA * current
    paths = np.clip(paths, -0.2, 0.2)
    q = np.column_stack([
        _ar1_forecast(history["iv_1m"], horizon, rng, n, 0.2),
        _ar1_forecast(history["carry"][:, 0], horizon, rng, n, 0.1),
        _ar1_forecast(history["carry"][:, 1], horizon, rng, n, 0.1),
    ])
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
