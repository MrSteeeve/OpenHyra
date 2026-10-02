"""Cross-fitted, payoff-aware ensemble LSMC for Bermudan stopping.

Only NumPy and the Python standard library are used.  Training payoffs and
predictions are in the evaluator's discounted units, so no private market or
contract reconstruction is needed here.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


LAMBDAS = np.asarray([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
BUFFER_MULTIPLIERS = np.asarray([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])
N_FOLDS = 4
N_MODELS = 4


def _features(states, immediate):
    """Payoff-aware features with a fixed dimension for a given asset count."""
    x = np.asarray(states, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    p = np.asarray(immediate, dtype=np.float64).reshape(-1, 1)
    # Prices ought to be positive, but clipping makes malformed/extreme paths
    # harmless and keeps every prediction finite.
    z = np.log(np.maximum(x, 1e-12))
    cols = [z, z * z]
    if z.shape[1] > 1:
        cols.append(np.column_stack(
            [z[:, i] * z[:, j] for i in range(z.shape[1])
             for j in range(i + 1, z.shape[1])]
        ))
    extrema = np.column_stack((
        p[:, 0], p[:, 0] * p[:, 0],
        x.min(axis=1), x.max(axis=1), x.mean(axis=1),
        z.min(axis=1), z.max(axis=1), z.mean(axis=1),
    ))
    cols.append(extrema)
    return np.column_stack(cols)


def _linear_mask(n_assets):
    """Mask for the fail-safe strongly regularized linear specification."""
    d = 2 * n_assets + n_assets * (n_assets - 1) // 2 + 8
    mask = np.zeros(d, dtype=bool)
    mask[:n_assets] = True                  # log prices
    tail = 2 * n_assets + n_assets * (n_assets - 1) // 2
    mask[tail] = True                       # immediate payoff
    mask[tail + 2:tail + 5] = True          # price min/max/mean
    return mask


def _ridge(x, y, lam, mask=None):
    """Fit standardized ridge and return full-width (coef, mean, scale, bias)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    d = x.shape[1]
    use = np.ones(d, dtype=bool) if mask is None else np.asarray(mask, bool)
    xu = x[:, use]
    mean = xu.mean(axis=0)
    scale = xu.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    zs = np.clip((xu - mean) / scale, -12.0, 12.0)
    bias = float(y.mean())
    yc = y - bias
    gram = zs.T @ zs
    rhs = zs.T @ yc
    penalty = float(lam) * max(1, len(y))
    try:
        beta = np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram + penalty * np.eye(gram.shape[0])) @ rhs
    coef = np.zeros(d)
    full_mean = np.zeros(d)
    full_scale = np.ones(d)
    coef[use], full_mean[use], full_scale[use] = beta, mean, scale
    return coef, full_mean, full_scale, bias


def _apply(model, x):
    coef, mean, scale, bias = model
    z = np.clip((x - mean) / scale, -12.0, 12.0)
    return np.nan_to_num(bias + z @ coef, nan=bias, posinf=bias, neginf=bias)


def _choose_lambda(x, y, immediate, train_idx, valid_idx, fallback_mask):
    """Inner validation weighted toward the stopping boundary."""
    tr = train_idx[immediate[train_idx] > 0.0]
    va = valid_idx[immediate[valid_idx] > 0.0]
    if len(tr) < 24 or len(va) < 4:
        return 10.0
    scale = max(float(np.median(np.abs(y[va] - immediate[va]))),
                float(np.std(immediate[va])) * 0.1, 1e-8)
    weights = 0.15 + np.exp(-np.abs(y[va] - immediate[va]) / scale)
    losses = []
    for lam in LAMBDAS:
        pred = _apply(_ridge(x[tr], y[tr], lam), x[va])
        losses.append(float(np.average((pred - y[va]) ** 2, weights=weights)))
    return float(LAMBDAS[int(np.argmin(losses))])


def _scale(payoff):
    positive = np.asarray(payoff)[np.asarray(payoff) > 0.0]
    if len(positive):
        return max(float(np.median(positive)), float(np.std(positive)) * 0.25, 1e-8)
    return 1.0


def _select_buffer(immediate, continuation, future_value):
    """Cross-fitted primal selection with a conservative one-SE tie rule."""
    candidates = BUFFER_MULTIPLIERS * _scale(immediate)
    values = np.empty((len(candidates), len(immediate)), dtype=np.float64)
    for j, buffer in enumerate(candidates):
        exercise = (immediate > 0.0) & (immediate > continuation + buffer)
        values[j] = np.where(exercise, immediate, future_value)
    means = values.mean(axis=1)
    best = int(np.argmax(means))
    se = values[best].std(ddof=1) / np.sqrt(max(1, len(immediate))) if len(immediate) > 1 else 0.0
    eligible = np.flatnonzero(means >= means[best] - se)
    # Larger continuation buffer means fewer marginal early exercises.
    chosen = int(eligible[-1])
    return float(candidates[chosen]), values[chosen]


def fit(input_dir, output_dir, seed):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    paths = np.load(input_dir / "training_paths.npy", allow_pickle=False)
    payoffs = np.load(input_dir / "payoffs.npy", allow_pickle=False).astype(np.float64)
    # Reading these validates the supplied instance, while discounted payoffs
    # remain the single source of value units.
    np.load(input_dir / "discount_factors.npy", allow_pickle=False)
    with open(input_dir / "instance.json", "r", encoding="utf-8") as fh:
        json.load(fh)
    if paths.ndim != 3 or payoffs.shape != paths.shape[:2]:
        raise ValueError("incompatible training_paths/payoffs shapes")
    n, n_times, n_assets = paths.shape
    if n < N_FOLDS or n_times < 2:
        raise ValueError("need at least four paths and two exercise dates")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    fold_id = np.empty(n, dtype=np.int8)
    fold_id[order] = np.arange(n, dtype=np.int64) % N_FOLDS
    cashflow = payoffs[:, -1].copy()
    d = _features(paths[:, 0, :], payoffs[:, 0]).shape[1]
    coefs = np.zeros((n_times - 1, N_MODELS, d))
    means = np.zeros_like(coefs)
    scales = np.ones_like(coefs)
    biases = np.zeros((n_times - 1, N_MODELS))
    buffers = np.zeros(n_times - 1)
    lambdas = np.zeros(n_times - 1)
    fallback = np.zeros(n_times - 1, dtype=np.int8)
    linear_mask = _linear_mask(n_assets)

    for t in range(n_times - 2, -1, -1):
        immediate = payoffs[:, t]
        x = _features(paths[:, t, :], immediate)
        future_cashflow = cashflow.copy()
        oof = np.empty(n)
        fold_lams = []
        positive_count = int(np.count_nonzero(immediate > 0.0))
        use_fallback = positive_count < 24
        fallback[t] = use_fallback
        for held in range(N_FOLDS):
            test = np.flatnonzero(fold_id == held)
            valid_fold = (held + 1) % N_FOLDS
            valid = np.flatnonzero(fold_id == valid_fold)
            train = np.flatnonzero((fold_id != held) & (fold_id != valid_fold))
            lam = 10.0 if use_fallback else _choose_lambda(
                x, future_cashflow, immediate, train, valid, linear_mask)
            fold_lams.append(lam)
            outer_train = np.flatnonzero((fold_id != held) & (immediate > 0.0))
            if len(outer_train) == 0:
                outer_train = np.flatnonzero(fold_id != held)
            model = _ridge(x[outer_train], future_cashflow[outer_train], lam,
                           linear_mask if use_fallback else None)
            oof[test] = _apply(model, x[test])

        buffer, updated = _select_buffer(immediate, oof, future_cashflow)
        buffers[t] = buffer
        cashflow = updated
        selected_lam = float(np.median(fold_lams))
        lambdas[t] = selected_lam

        eligible = np.flatnonzero(immediate > 0.0)
        if len(eligible) == 0:
            eligible = np.arange(n)
        for m in range(N_MODELS):
            sample = rng.choice(eligible, size=len(eligible), replace=True)
            model = _ridge(x[sample], future_cashflow[sample], selected_lam,
                           linear_mask if use_fallback else None)
            coefs[t, m], means[t, m], scales[t, m], biases[t, m] = model

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "model.npz", coefs=coefs, means=means,
                        scales=scales, biases=biases, buffers=buffers,
                        lambdas=lambdas, fallback=fallback,
                        n_assets=np.asarray(n_assets), n_times=np.asarray(n_times))


def _request_time(request, history, n_times):
    for key in ("time_index", "exercise_time_index", "exercise_index",
                "date_index", "step_index", "current_index", "index", "time"):
        if key in request:
            value = request[key]
            if isinstance(value, (int, np.integer)):
                return int(value)
    if history.ndim >= 2:
        return int(history.shape[-2] - 1)
    raise ValueError("request has no recognizable exercise-time index")


def predict(model_dir, input_dir, output_dir):
    model_dir, input_dir, output_dir = Path(model_dir), Path(input_dir), Path(output_dir)
    states = np.load(input_dir / "states.npy", allow_pickle=False)
    immediate = np.load(input_dir / "immediate_payoffs.npy", allow_pickle=False).reshape(-1)
    history = np.load(input_dir / "history.npy", allow_pickle=False)
    with open(input_dir / "request.json", "r", encoding="utf-8") as fh:
        request = json.load(fh)
    with np.load(model_dir / "model.npz", allow_pickle=False) as data:
        n_times = int(data["n_times"])
        t = _request_time(request, history, n_times)
        if not 0 <= t < n_times - 1:
            raise ValueError("prediction requested outside a non-terminal exercise date")
        x = _features(states, immediate)
        predictions = []
        for m in range(data["coefs"].shape[1]):
            model = (data["coefs"][t, m], data["means"][t, m],
                     data["scales"][t, m], float(data["biases"][t, m]))
            predictions.append(_apply(model, x))
        continuation = np.median(np.vstack(predictions), axis=0) + float(data["buffers"][t])
    continuation = np.nan_to_num(continuation, nan=0.0, posinf=1e100, neginf=-1e100)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", continuation.astype(np.float64, copy=False))


def _main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--seed", required=True, type=int)
    pred_parser = sub.add_parser("predict")
    pred_parser.add_argument("--model", required=True)
    pred_parser.add_argument("--input", required=True)
    pred_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit(args.input, args.output, args.seed)
    else:
        predict(args.model, args.input, args.output)


if __name__ == "__main__":
    _main()
