"""Linear continuation policy with a fixed-pass robust backward update."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def features(states: np.ndarray, immediate: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    logged = np.log(np.maximum(states, 1e-12))
    columns = [np.ones(states.shape[0], dtype=np.float64), *[logged[:, index] for index in range(logged.shape[1])], *[logged[:, index] * 2 for index in range(logged.shape[1])], np.mean(logged, axis=1), np.min(logged, axis=1), np.max(logged, axis=1), np.asarray(immediate, dtype=np.float64)]
    return np.column_stack(columns)

def robust_ridge(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Perform three deterministic ridge/Huber IRLS solves."""
    feature_count = design.shape[1]
    weights = np.ones(target.shape[0], dtype=np.float64)
    coefficient = np.zeros(feature_count, dtype=np.float64)
    for _ in range(3):
        root_weight = np.sqrt(weights)
        weighted_design = design * root_weight[:, None]
        gram = weighted_design.T @ weighted_design
        gram.flat[::feature_count + 1] += 1e-6
        coefficient = np.linalg.solve(
            gram, weighted_design.T @ (target * root_weight)
        )
        residual = target - design @ coefficient
        center = np.median(residual)
        scale = max(
            float(1.4826 * np.median(np.abs(residual - center))), 1e-10
        )
        magnitude = np.abs(residual - center) / (1.345 * scale)
        weights = np.ones_like(magnitude)
        np.divide(1.0, magnitude, out=weights, where=magnitude > 1.0)
    return coefficient

def calibrate_boundary(immediate: np.ndarray, continuation: np.ndarray,
                       future_cash_flow: np.ndarray) -> float:
    """Select a shrunken continuation offset from a fixed small grid."""
    residual = future_cash_flow - continuation
    scale = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    if not np.isfinite(scale) or scale < 1e-10:
        return 0.0
    candidates = scale * np.array([-0.20, -0.10, 0.0, 0.10, 0.20])
    best_offset = 0.0
    best_value = float(np.mean(np.where(
        immediate >= continuation, immediate, future_cash_flow
    )))
    tolerance = 1e-10 * max(1.0, abs(best_value))
    for offset in candidates:
        exercise = immediate >= continuation + offset
        value = float(np.mean(np.where(exercise, immediate, future_cash_flow)))
        if value > best_value + tolerance:
            best_value = value
            best_offset = float(offset)
    return 0.75 * best_offset

def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    del seed
    paths = np.load(input_dir / 'training_paths.npy', allow_pickle=False)
    payoffs = np.load(input_dir / 'payoffs.npy', allow_pickle=False)
    n_steps = paths.shape[1] - 1
    feature_count = features(paths[:, 0, :], payoffs[:, 0]).shape[1]
    coefficients = np.zeros((n_steps, feature_count), dtype=np.float64)
    means = np.zeros_like(coefficients)
    scales = np.ones_like(coefficients)
    boundary_offsets = np.zeros(n_steps, dtype=np.float64)
    cash_flow = np.asarray(payoffs[:, -1], dtype=np.float64).copy()
    for time_index in range(n_steps - 1, -1, -1):
        design = features(paths[:, time_index, :], payoffs[:, time_index])
        mean = np.mean(design, axis=0)
        scale = np.std(design, axis=0)
        mean[0] = 0.0
        scale[scale < 1e-10] = 1.0
        normalized = (design - mean) / scale
        coefficient = robust_ridge(normalized, cash_flow)
        continuation = normalized @ coefficient
        boundary_offset = calibrate_boundary(
            payoffs[:, time_index], continuation, cash_flow
        )
        exercise = payoffs[:, time_index] >= continuation + boundary_offset
        cash_flow = np.where(exercise, payoffs[:, time_index], cash_flow)
        coefficients[time_index] = coefficient
        means[time_index] = mean
        scales[time_index] = scale
        boundary_offsets[time_index] = boundary_offset
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'model.npz', coefficients=coefficients, means=means,
             scales=scales, boundary_offsets=boundary_offsets)

def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    request = json.loads((input_dir / 'request.json').read_text(encoding='utf-8'))
    time_index = int(request['time_index'])
    states = np.load(input_dir / 'states.npy', allow_pickle=False)
    immediate = np.load(input_dir / 'immediate_payoffs.npy', allow_pickle=False)
    with np.load(model_dir / 'model.npz', allow_pickle=False) as model:
        design = features(states, immediate)
        normalized = (design - model['means'][time_index]) / model['scales'][time_index]
        predictions = (normalized @ model['coefficients'][time_index]
                       + model['boundary_offsets'][time_index])
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
