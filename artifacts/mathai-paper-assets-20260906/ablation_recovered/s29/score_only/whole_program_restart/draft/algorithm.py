"""Cross-fitted ridge least-squares Monte Carlo continuation estimator."""

import argparse
import json
import os

import numpy as np


RIDGES = np.asarray((1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                     1e-1, 3e-1, 1.0, 3.0), dtype=np.float64)
N_FOLDS = 8


def _features(states, immediate):
    """Compact state/payoff basis; scaling is learned separately per date."""
    x = np.asarray(states, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    p = np.asarray(immediate, dtype=np.float64).reshape(-1, 1)
    # Prices in this benchmark are positive.  The guard also makes malformed
    # or numerically underflowed inputs harmless rather than producing NaNs.
    logs = np.log(np.maximum(x, np.finfo(np.float64).tiny))
    mean = logs.mean(axis=1, keepdims=True)
    lo = logs.min(axis=1, keepdims=True)
    hi = logs.max(axis=1, keepdims=True)
    dispersion = logs.std(axis=1, keepdims=True)
    return np.concatenate((logs, p, mean, lo, hi, dispersion,
                           p * p, p * mean, p * lo, p * hi,
                           p * dispersion), axis=1)


def _standardize(x):
    center = np.median(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-10), scale, 1.0)
    return center, scale


def _design(x, center, scale):
    z = (x - center) / scale
    z = np.where(np.isfinite(z), z, 0.0)
    return np.concatenate((np.ones((len(z), 1)), z), axis=1)


def _ridge_gcv(a, y):
    """Select a positive ridge by generalized cross-validation and fit it."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) == 0:
        return np.zeros(a.shape[1], dtype=np.float64), RIDGES[0]
    if len(y) == 1:
        coef = np.zeros(a.shape[1], dtype=np.float64)
        coef[0] = y[0]
        return coef, RIDGES[0]

    penalty = np.ones(a.shape[1], dtype=np.float64)
    penalty[0] = 0.0
    gram = a.T @ a
    rhs = a.T @ y
    best_score = np.inf
    best_coef = None
    best_lam = RIDGES[0]
    n = float(len(y))
    for lam in RIDGES:
        system = gram + (float(lam) * n) * np.diag(penalty)
        coef = np.linalg.pinv(system, rcond=1e-11) @ rhs
        residual = y - a @ coef
        # Effective degrees of freedom, including the unpenalized intercept.
        df = float(np.trace(np.linalg.pinv(system, rcond=1e-11) @ gram))
        denom = max(1.0 - df / n, 1.0 / n)
        score = float(np.mean(residual * residual) / (denom * denom))
        if np.isfinite(score) and score < best_score:
            best_score, best_coef, best_lam = score, coef, lam
    if best_coef is None:
        best_coef = np.zeros(a.shape[1], dtype=np.float64)
        best_coef[0] = float(np.median(y))
    return best_coef, float(best_lam)


def _fit_regression(x_all, y, eligible):
    """Fit on eligible observations, with conservative sparse-data fallbacks."""
    idx = np.flatnonzero(eligible)
    if len(idx) == 0:
        center, scale = _standardize(x_all)
        coef = np.zeros(x_all.shape[1] + 1, dtype=np.float64)
        coef[0] = float(np.median(y)) if len(y) else 0.0
        return center, scale, coef, RIDGES[-1]
    center, scale = _standardize(x_all[idx])
    a = _design(x_all[idx], center, scale)
    coef, lam = _ridge_gcv(a, y[idx])
    return center, scale, coef, lam


def fit(input_dir, output_dir, seed):
    paths = np.load(os.path.join(input_dir, "training_paths.npy"),
                    allow_pickle=False)
    payoffs = np.load(os.path.join(input_dir, "payoffs.npy"),
                      allow_pickle=False).astype(np.float64, copy=False)
    if paths.ndim != 3 or payoffs.ndim != 2 or paths.shape[:2] != payoffs.shape:
        raise ValueError("training paths/payoffs have incompatible shapes")
    n, n_times, _ = paths.shape
    if n == 0 or n_times < 1:
        raise ValueError("empty training problem")

    rng = np.random.default_rng(int(seed))
    folds = np.empty(n, dtype=np.int64)
    folds[rng.permutation(n)] = np.arange(n, dtype=np.int64) % min(N_FOLDS, n)
    cashflow = payoffs[:, -1].copy()
    records = {}

    # Terminal exercise is evaluator-owned; models cover non-terminal dates.
    for t in range(n_times - 2, -1, -1):
        immediate = payoffs[:, t]
        future_targets = cashflow.copy()
        feats = _features(paths[:, t, :], immediate)
        oof = np.empty(n, dtype=np.float64)
        for fold in range(min(N_FOLDS, n)):
            held = folds == fold
            train = ~held
            eligible = train & (immediate > 0.0)
            center, scale, coef, _ = _fit_regression(feats, future_targets,
                                                     eligible)
            oof[held] = _design(feats[held], center, scale) @ coef

        exercise = (immediate > 0.0) & (immediate > oof)
        cashflow[exercise] = immediate[exercise]

        # Store a deployment model trained against the targets as they existed
        # before this date's decisions, exactly mirroring the OOF regressions.
        eligible = immediate > 0.0
        center, scale, coef, lam = _fit_regression(feats, future_targets,
                                                   eligible)
        clip_source = future_targets[eligible]
        if len(clip_source) < 4:
            clip_source = future_targets
        low, high = np.percentile(clip_source, (1.0, 99.0))
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = 0.0, 0.0
        if high < low:
            low, high = high, low
        records[t] = (center, scale, coef, float(low), float(high), lam)

    os.makedirs(output_dir, exist_ok=True)
    archive = {"n_times": np.asarray(n_times, dtype=np.int64)}
    metadata = {"schema": "crossfit-ridge-lsm.v1", "dates": {}}
    for t, (center, scale, coef, low, high, lam) in records.items():
        archive[f"center_{t}"] = center
        archive[f"scale_{t}"] = scale
        archive[f"coef_{t}"] = coef
        archive[f"clip_{t}"] = np.asarray((low, high), dtype=np.float64)
        metadata["dates"][str(t)] = {"ridge": lam}
    np.savez(os.path.join(output_dir, "model.npz"), **archive)
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, separators=(",", ":"))


def _request_index(request, history, n_times):
    for key in ("time_index", "exercise_index", "date_index", "step", "time"):
        value = request.get(key)
        if isinstance(value, (int, np.integer)):
            return int(value)
    # Causal history includes the current observation.
    if history.ndim >= 2:
        return int(history.shape[-2] - 1)
    raise ValueError("cannot determine exercise-date index")


def predict(model_dir, input_dir, output_dir):
    states = np.load(os.path.join(input_dir, "states.npy"), allow_pickle=False)
    immediate = np.load(os.path.join(input_dir, "immediate_payoffs.npy"),
                        allow_pickle=False).reshape(-1)
    history = np.load(os.path.join(input_dir, "history.npy"), allow_pickle=False)
    with open(os.path.join(input_dir, "request.json"), "r", encoding="utf-8") as f:
        request = json.load(f)
    with np.load(os.path.join(model_dir, "model.npz"), allow_pickle=False) as model:
        t = _request_index(request, history, int(model["n_times"]))
        key = f"coef_{t}"
        if key not in model.files:
            # Defensive terminal fallback; normally the evaluator never asks it.
            predictions = np.zeros(len(immediate), dtype=np.float64)
        else:
            feats = _features(states, immediate)
            predictions = _design(feats, model[f"center_{t}"],
                                  model[f"scale_{t}"]) @ model[key]
            bounds = model[f"clip_{t}"]
            predictions = np.clip(predictions, bounds[0], bounds[1])
            predictions = np.nan_to_num(predictions, nan=float(bounds[0]),
                                        posinf=float(bounds[1]),
                                        neginf=float(bounds[0]))
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "predictions.npy"),
            np.asarray(predictions, dtype=np.float64))


def _main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--seed", required=True, type=int)
    predict_parser = sub.add_parser("predict")
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
