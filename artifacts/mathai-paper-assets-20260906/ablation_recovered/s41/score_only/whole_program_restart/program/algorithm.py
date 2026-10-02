"""Cross-fitted least-squares Monte Carlo continuation estimator.

The evaluator supplies payoffs already expressed in the units in which exercise
decisions are made.  Consequently the backward recursion deliberately applies
no additional discount factor.
"""

import argparse
import json
import os

import numpy as np


RIDGES = np.asarray([0.01, 0.1, 1.0, 10.0, 100.0], dtype=np.float64)
N_FOLDS = 8


def _as_matrix(values):
    """Return one row per path and one column per state coordinate."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return values[:, None]
    return values.reshape(values.shape[0], -1)


def _features(states, immediate):
    """Compact curved basis specified by the intervention."""
    states = _as_matrix(states)
    immediate = np.asarray(immediate, dtype=np.float64).reshape(-1, 1)
    # Prices are positive in the benchmark.  Clipping also makes malformed or
    # numerically underflowed coordinates harmless rather than non-finite.
    logs = np.log(np.maximum(states, np.finfo(np.float64).tiny))
    columns = [logs, logs * logs]
    if logs.shape[1] > 1:
        products = [logs[:, i : i + 1] * logs[:, j : j + 1]
                    for i in range(logs.shape[1])
                    for j in range(i + 1, logs.shape[1])]
        columns.extend(products)
    columns.extend([immediate, immediate * logs])
    result = np.concatenate(columns, axis=1)
    return np.nan_to_num(result, nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def _standardize(x):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (x - mean) / scale, mean, scale


def _ridge(x, y, penalty):
    """Fit an unpenalized intercept and penalized standardized coefficients."""
    design = np.column_stack((np.ones(x.shape[0], dtype=np.float64), x))
    gram = design.T @ design
    regularizer = np.eye(design.shape[1], dtype=np.float64) * penalty
    regularizer[0, 0] = 0.0
    rhs = design.T @ y
    try:
        return np.linalg.solve(gram + regularizer, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + regularizer, rhs, rcond=None)[0]


def _fold_ids(n_paths, seed):
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n_paths)
    folds = np.empty(n_paths, dtype=np.int64)
    folds[order] = np.arange(n_paths, dtype=np.int64) % min(N_FOLDS, n_paths)
    return folds


def _cross_fitted_predictions(x, target, folds):
    """Return OOF predictions for every penalty and its aggregate OOF SSE."""
    predictions = np.empty((RIDGES.size, x.shape[0]), dtype=np.float64)
    for fold in np.unique(folds):
        validation = folds == fold
        training = ~validation
        train_x, mean, scale = _standardize(x[training])
        valid_x = (x[validation] - mean) / scale
        for index, penalty in enumerate(RIDGES):
            coef = _ridge(train_x, target[training], penalty)
            predictions[index, validation] = coef[0] + valid_x @ coef[1:]
    residual = predictions - target[None, :]
    errors = np.sum(residual * residual, axis=1)
    return predictions, errors


def fit(input_dir, output_dir, seed):
    paths = np.load(os.path.join(input_dir, "training_paths.npy"), allow_pickle=False)
    payoffs = np.load(os.path.join(input_dir, "payoffs.npy"), allow_pickle=False)
    # Loading validates the evaluator-owned timing input.  It is not multiplied
    # into targets because payoffs are already discounted by contract.
    np.load(os.path.join(input_dir, "discount_factors.npy"), allow_pickle=False)
    with open(os.path.join(input_dir, "instance.json"), "r", encoding="utf-8") as handle:
        json.load(handle)

    paths = np.asarray(paths, dtype=np.float64)
    payoffs = np.asarray(payoffs, dtype=np.float64)
    if paths.ndim < 3 or payoffs.ndim != 2:
        raise ValueError("unexpected training array rank")
    if paths.shape[0] != payoffs.shape[0] or paths.shape[1] != payoffs.shape[1]:
        raise ValueError("paths and payoffs must share path and time axes")
    n_paths, n_times = payoffs.shape
    if n_paths < 2 or n_times < 2:
        raise ValueError("at least two paths and two exercise times are required")

    folds = _fold_ids(n_paths, seed)
    realized = payoffs[:, -1].copy()
    feature_count = _features(paths[:, 0], payoffs[:, 0]).shape[1]
    means = np.empty((n_times - 1, feature_count), dtype=np.float64)
    scales = np.empty_like(means)
    coefficients = np.empty((n_times - 1, feature_count + 1), dtype=np.float64)
    penalties = np.empty(n_times - 1, dtype=np.float64)

    for time_index in range(n_times - 2, -1, -1):
        x = _features(paths[:, time_index], payoffs[:, time_index])
        if x.shape[1] != feature_count:
            raise ValueError("state dimension changes across exercise times")
        target = realized.copy()
        oof, errors = _cross_fitted_predictions(x, target, folds)
        choice = int(np.argmin(errors))
        penalties[time_index] = RIDGES[choice]
        continuation = np.maximum(oof[choice], 0.0)
        exercise = (payoffs[:, time_index] > 0.0) & (
            payoffs[:, time_index] >= continuation
        )
        realized = np.where(exercise, payoffs[:, time_index], realized)

        standardized, mean, scale = _standardize(x)
        means[time_index] = mean
        scales[time_index] = scale
        coefficients[time_index] = _ridge(standardized, target,
                                           penalties[time_index])

    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "model.npz"), means=means, scales=scales,
             coefficients=coefficients, penalties=penalties,
             n_times=np.asarray(n_times, dtype=np.int64))


def _request_time(request, history, n_models):
    for key in ("time_index", "exercise_index", "time_idx", "step_index", "step"):
        value = request.get(key)
        if isinstance(value, (int, np.integer)) and 0 <= int(value) < n_models:
            return int(value)
    if history.ndim >= 2:
        index = history.shape[1] - 1
        if 0 <= index < n_models:
            return index
    raise ValueError("cannot determine exercise-time index from request")


def predict(model_dir, input_dir, output_dir):
    history = np.load(os.path.join(input_dir, "history.npy"), allow_pickle=False)
    states = np.load(os.path.join(input_dir, "states.npy"), allow_pickle=False)
    immediate = np.load(os.path.join(input_dir, "immediate_payoffs.npy"), allow_pickle=False)
    with open(os.path.join(input_dir, "request.json"), "r", encoding="utf-8") as handle:
        request = json.load(handle)
    with np.load(os.path.join(model_dir, "model.npz"), allow_pickle=False) as model:
        means = model["means"]
        scales = model["scales"]
        coefficients = model["coefficients"]
    time_index = _request_time(request, np.asarray(history), coefficients.shape[0])
    x = _features(states, immediate)
    if x.shape[1] != means.shape[1]:
        raise ValueError("query state dimension differs from training")
    values = coefficients[time_index, 0] + (
        (x - means[time_index]) / scales[time_index]
    ) @ coefficients[time_index, 1:]
    values = np.maximum(np.nan_to_num(values, nan=0.0, posinf=1.0e100, neginf=0.0), 0.0)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "predictions.npy"), values.astype(np.float64))


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
