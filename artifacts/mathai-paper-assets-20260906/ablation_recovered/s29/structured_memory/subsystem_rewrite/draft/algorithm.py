"""Complete fit/predict seed for the open Python-program track.

This is deliberately just one ordinary starting program. Search candidates
may replace every function, model representation, objective, and update rule.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def features(states: np.ndarray, immediate: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    logged = np.log(np.maximum(states, 1e-12))
    columns = [np.ones(states.shape[0], dtype=np.float64), *[logged[:, index] for index in range(logged.shape[1])], *[logged[:, index] ** 2 for index in range(logged.shape[1])], np.mean(logged, axis=1), np.min(logged, axis=1), np.max(logged, axis=1), np.asarray(immediate, dtype=np.float64)]
    return np.column_stack(columns)

def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    paths = np.asarray(np.load(input_dir / 'training_paths.npy', allow_pickle=False), dtype=np.float64)
    payoffs = np.asarray(np.load(input_dir / 'payoffs.npy', allow_pickle=False), dtype=np.float64)
    if paths.ndim != 3 or payoffs.ndim != 2:
        raise ValueError('expected paths (n_paths,n_times,n_assets) and payoffs (n_paths,n_times)')
    n_paths, n_times, _ = paths.shape
    if payoffs.shape != (n_paths, n_times):
        raise ValueError('paths and payoffs have inconsistent shapes')
    raw = [features(paths[:, t, :], payoffs[:, t]) for t in range(n_times)]
    n_features = raw[0].shape[1]

    def normalization(indices):
        means = np.empty((n_times, n_features), dtype=np.float64)
        scales = np.empty_like(means)
        for t in range(n_times):
            sample = raw[t][indices]
            means[t] = np.mean(sample, axis=0)
            scales[t] = np.std(sample, axis=0)
            constant = scales[t] < 1e-10
            means[t, constant] = 0.0
            scales[t, constant] = 1.0
            scales[t] = np.maximum(scales[t], 1e-08)
        return (means, scales)

    def designs(indices, means, scales):
        return [(raw[t][indices] - means[t]) / scales[t] for t in range(n_times)]

    def ridge_initialization(indices, means, scales):
        xs = designs(indices, means, scales)
        gs = payoffs[indices]
        coefficients = np.zeros((n_times, n_features), dtype=np.float64)
        realized = gs[:, -1].copy()
        eye = np.eye(n_features, dtype=np.float64)
        for t in range(n_times - 2, -1, -1):
            x = xs[t]
            ridge = 0.001 * (1.0 + np.mean(np.sum(x * x, axis=1)))
            try:
                beta = np.linalg.solve(x.T @ x + ridge * eye, x.T @ realized)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(x.T @ x + ridge * eye, x.T @ realized, rcond=None)[0]
            beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
            coefficients[t] = beta
            continuation = x @ beta
            exercise = (gs[:, t] > 0.0) & (gs[:, t] >= continuation)
            realized = np.where(exercise, gs[:, t], realized)
        return coefficients

    def hard_values(indices, coefficients, means, scales):
        alive = np.ones(len(indices), dtype=bool)
        values = np.zeros(len(indices), dtype=np.float64)
        for t in range(n_times - 1):
            x = (raw[t][indices] - means[t]) / scales[t]
            g = payoffs[indices, t]
            exercise = alive & (g > 0.0) & (g >= x @ coefficients[t])
            values[exercise] = g[exercise]
            alive[exercise] = False
        values[alive] = payoffs[indices[alive], -1]
        return values

    def optimize(indices, initial, means, scales, temperature, shrinkage, iterations):
        xs = designs(indices, means, scales)
        gs = payoffs[indices]
        beta = initial.copy()
        first = np.zeros_like(beta)
        second = np.zeros_like(beta)
        best = beta.copy()
        best_objective = -np.inf
        stale = 0
        payoff_scale = max(float(np.mean(gs[gs > 0.0])) if np.any(gs > 0.0) else 1.0, 1e-08)
        penalty_weight = shrinkage * payoff_scale
        for iteration in range(1, iterations + 1):
            probabilities = []
            survival = []
            alive_probability = np.ones(len(indices), dtype=np.float64)
            for t in range(n_times - 1):
                margin = np.clip((gs[:, t] - xs[t] @ beta[t]) / temperature, -35.0, 35.0)
                p = 1.0 / (1.0 + np.exp(-margin))
                p = np.where(gs[:, t] > 0.0, p, 0.0)
                survival.append(alive_probability.copy())
                probabilities.append(p)
                alive_probability *= 1.0 - p
            value = gs[:, -1].copy()
            for t in range(n_times - 2, -1, -1):
                p = probabilities[t]
                value = p * gs[:, t] + (1.0 - p) * value
            delta = beta - initial
            objective = float(np.mean(value) - 0.5 * penalty_weight * np.mean(delta[:-1] ** 2))
            if objective > best_objective + 1e-10:
                best_objective = objective
                best = beta.copy()
                stale = 0
            else:
                stale += 1
            if stale >= 35:
                break
            future = gs[:, -1].copy()
            gradient = np.zeros_like(beta)
            for t in range(n_times - 2, -1, -1):
                p = probabilities[t]
                weight = -survival[t] * p * (1.0 - p) * (gs[:, t] - future) / temperature
                gradient[t] = xs[t].T @ weight / len(indices)
                future = p * gs[:, t] + (1.0 - p) * future
            gradient[:-1] -= penalty_weight * delta[:-1] / max((n_times - 1) * n_features, 1)
            norm = float(np.linalg.norm(gradient[:-1]))
            if norm > 10.0:
                gradient[:-1] *= 10.0 / norm
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            first_hat = first / (1.0 - 0.9 ** iteration)
            second_hat = second / (1.0 - 0.999 ** iteration)
            step = 0.03 * first_hat / (np.sqrt(second_hat) + 1e-08)
            beta[:-1] += np.clip(step[:-1], -0.08, 0.08)
            beta[:-1] = initial[:-1] + np.clip(beta[:-1] - initial[:-1], -4.0, 4.0)
        return best
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n_paths)
    fold_ids = np.empty(n_paths, dtype=np.int64)
    fold_ids[order] = np.arange(n_paths) % 4
    positive = payoffs[payoffs > 0.0]
    positive_scale = float(np.median(positive)) if positive.size else 1.0
    positive_scale = max(positive_scale, 1e-08)
    settings = [(fraction * positive_scale, shrinkage) for fraction in (0.02, 0.05, 0.1) for shrinkage in (0.01, 0.1, 1.0)]
    setting_totals = np.zeros(len(settings), dtype=np.float64)
    baseline_total = 0.0
    for fold in range(4):
        train = np.flatnonzero(fold_ids != fold)
        held_out = np.flatnonzero(fold_ids == fold)
        means, scales = normalization(train)
        initial = ridge_initialization(train, means, scales)
        baseline_total += float(np.sum(hard_values(held_out, initial, means, scales)))
        for setting_index, (temperature, shrinkage) in enumerate(settings):
            candidate = optimize(train, initial, means, scales, temperature, shrinkage, 140)
            setting_totals[setting_index] += float(np.sum(hard_values(held_out, candidate, means, scales)))
    best_index = int(np.argmax(setting_totals))
    all_indices = np.arange(n_paths)
    means, scales = normalization(all_indices)
    initial = ridge_initialization(all_indices, means, scales)
    coefficients = initial
    if setting_totals[best_index] > baseline_total + 1e-12:
        temperature, shrinkage = settings[best_index]
        coefficients = optimize(all_indices, initial, means, scales, temperature, shrinkage, 300)
    coefficients = np.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'model.npz', coefficients=coefficients, means=means, scales=scales)

def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    request = json.loads((input_dir / 'request.json').read_text(encoding='utf-8'))
    time_index = int(request['time_index'])
    states = np.load(input_dir / 'states.npy', allow_pickle=False)
    immediate = np.load(input_dir / 'immediate_payoffs.npy', allow_pickle=False)
    with np.load(model_dir / 'model.npz', allow_pickle=False) as model:
        design = features(states, immediate)
        normalized = (design - model['means'][time_index]) / model['scales'][time_index]
        predictions = normalized @ model['coefficients'][time_index]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'predictions.npy', np.asarray(predictions, dtype=np.float64), allow_pickle=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    fit_parser = subparsers.add_parser('fit')
    fit_parser.add_argument('--input', required=True)
    fit_parser.add_argument('--output', required=True)
    fit_parser.add_argument('--seed', required=True, type=int)
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('--model', required=True)
    predict_parser.add_argument('--input', required=True)
    predict_parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        fit(Path(args.input), Path(args.output), args.seed)
    else:
        predict(Path(args.model), Path(args.input), Path(args.output))
if __name__ == '__main__':
    main()
