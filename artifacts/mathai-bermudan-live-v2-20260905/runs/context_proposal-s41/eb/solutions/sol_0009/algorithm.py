"""Ridge plus a trained MLP residual on Monte Carlo backward targets.

Both the least-squares fit and the network optimizer belong to this candidate.
No evaluator labels or prices are needed: future discounted path cash flows
supply the regression targets, with stopping decisions updated backward.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def features(states, immediate):
    z = np.log(np.maximum(np.asarray(states, float), 1e-12))
    return np.column_stack([np.ones(len(z)), z, z * z, np.mean(z, 1), immediate])

def fit(input_dir: Path, output_dir: Path, seed: int) -> None:
    import numpy as _np
    paths = _np.load(input_dir / 'training_paths.npy', allow_pickle=False)
    payoffs = _np.asarray(_np.load(input_dir / 'payoffs.npy', allow_pickle=False), dtype=float)
    n = len(paths)
    steps, width = (paths.shape[1] - 1, 16)
    d = features(paths[:, 0], payoffs[:, 0]).shape[1]
    rng = _np.random.default_rng(41000124)
    folds = _np.empty(n, dtype=_np.int8)
    folds[rng.permutation(n)] = _np.arange(n) % 4
    strengths = _np.asarray([0.0, 0.5, 1.5, 3.0])
    coefs = _np.zeros((steps, d))
    means = _np.zeros_like(coefs)
    scales = _np.ones_like(coefs)
    w1s = _np.zeros((steps, d, width))
    b1s = _np.zeros((steps, width))
    w2s = _np.zeros((steps, width))
    b2s = _np.zeros(steps)
    target_scales = _np.ones(steps)
    targets = _np.zeros((steps, n))
    losses = _np.zeros((steps, 2))
    updates = _np.zeros(steps)
    selected = _np.zeros(steps)
    validation_values = _np.zeros((steps, len(strengths)))

    def normalize(raw, rows):
        mean = raw[rows].mean(0)
        scale = raw[rows].std(0)
        mean[0] = 0.0
        scale[scale < 1e-10] = 1.0
        return (mean, scale)

    def boundary_ridge(x_train, y_train, exercise_train, strength):
        ridge = _np.eye(x_train.shape[1]) * 1e-05
        base = _np.linalg.solve(x_train.T @ x_train + ridge, x_train.T @ y_train)
        if strength == 0.0:
            return base
        margin = exercise_train - x_train @ base
        boundary_scale = max(1.4826 * float(_np.median(_np.abs(margin - _np.median(margin)))), 1e-06)
        weights = 1.0 + strength * _np.exp(-_np.abs(margin) / boundary_scale)
        sw = _np.sqrt(weights)
        xw = x_train * sw[:, None]
        return _np.linalg.solve(xw.T @ xw + ridge, xw.T @ (y_train * sw))
    cash = payoffs[:, -1].copy()
    for t in range(steps - 1, -1, -1):
        targets[t] = cash
        raw = features(paths[:, t], payoffs[:, t])
        oof_by_strength = _np.empty((len(strengths), n))
        for fold in range(4):
            train = folds != fold
            valid = ~train
            mean_f, scale_f = normalize(raw, train)
            x_train = (raw[train] - mean_f) / scale_f
            x_valid = (raw[valid] - mean_f) / scale_f
            for j, strength in enumerate(strengths):
                beta = boundary_ridge(x_train, cash[train], payoffs[train, t], float(strength))
                oof_by_strength[j, valid] = x_valid @ beta
        for j in range(len(strengths)):
            exercise = (payoffs[:, t] > 0.0) & (payoffs[:, t] >= oof_by_strength[j])
            validation_values[t, j] = _np.mean(_np.where(exercise, payoffs[:, t], cash))
        choice = int(_np.argmax(validation_values[t]))
        selected[t] = strengths[choice]
        continuation_oof = oof_by_strength[choice]
        exercise = (payoffs[:, t] > 0.0) & (payoffs[:, t] >= continuation_oof)
        cash = _np.where(exercise, payoffs[:, t], cash)
        mean, scale = normalize(raw, _np.ones(n, dtype=bool))
        x = (raw - mean) / scale
        coef = boundary_ridge(x, targets[t], payoffs[:, t], selected[t])
        residual = targets[t] - x @ coef
        target_scale = max(float(residual.std()), 1e-05)
        y = residual / target_scale
        w1 = rng.normal(0.0, 0.15, (d, width))
        initial = w1.copy()
        b1 = _np.zeros(width)
        w2 = rng.normal(0.0, 0.1, width)
        b2 = 0.0
        losses[t, 0] = _np.mean((_np.tanh(x @ w1 + b1) @ w2 + b2 - y) ** 2)
        for _ in range(64):
            h = _np.tanh(x @ w1 + b1)
            err = h @ w2 + b2 - y
            g = 2.0 * err / n
            hidden_g = g[:, None] * w2 * (1.0 - h * h)
            w1 -= 0.03 * (x.T @ hidden_g)
            b1 -= 0.03 * hidden_g.sum(0)
            w2 -= 0.03 * (h.T @ g)
            b2 -= 0.03 * g.sum()
        residual_pred = _np.tanh(x @ w1 + b1) @ w2 + b2
        losses[t, 1] = _np.mean((residual_pred - y) ** 2)
        updates[t] = _np.linalg.norm(w1 - initial)
        coefs[t], means[t], scales[t] = (coef, mean, scale)
        w1s[t], b1s[t], w2s[t], b2s[t] = (w1, b1, w2, b2)
        target_scales[t] = target_scale
    output_dir.mkdir(parents=True, exist_ok=True)
    _np.savez(output_dir / 'model.npz', coefficients=coefs, means=means, scales=scales, w1=w1s, b1=b1s, w2=w2s, b2=b2s, target_scales=target_scales)
    _np.savez(output_dir / 'training_trace.npz', backward_targets=targets, loss_before_after=losses, first_layer_update_norm=updates, boundary_strengths=selected, oof_policy_values=validation_values, folds=folds, seed=_np.array(41000124), requested_seed=_np.array(seed))

def predict(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    t = int(json.loads((input_dir / 'request.json').read_text())['time_index'])
    states = np.load(input_dir / 'states.npy', allow_pickle=False)
    immediate = np.load(input_dir / 'immediate_payoffs.npy', allow_pickle=False)
    with np.load(model_dir / 'model.npz', allow_pickle=False) as m:
        x = (features(states, immediate) - m['means'][t]) / m['scales'][t]
        residual = np.tanh(x @ m['w1'][t] + m['b1'][t]) @ m['w2'][t] + m['b2'][t]
        out = x @ m['coefficients'][t] + m['target_scales'][t] * residual
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'predictions.npy', out, allow_pickle=False)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--input', required=True)
    f.add_argument('--output', required=True)
    f.add_argument('--seed', type=int, required=True)
    p = sub.add_parser('predict')
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = parser.parse_args()
    if a.command == 'fit':
        fit(Path(a.input), Path(a.output), a.seed)
    else:
        predict(Path(a.model), Path(a.input), Path(a.output))
if __name__ == '__main__':
    main()
