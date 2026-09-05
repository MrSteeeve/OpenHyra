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
    rng = np.random.default_rng(seed)
    paths = np.load(input_dir / 'training_paths.npy', allow_pickle=False)
    payoffs = np.load(input_dir / 'payoffs.npy', allow_pickle=False)
    n = len(paths)
    steps = paths.shape[1] - 1
    width = 16
    d = features(paths[:, 0], payoffs[:, 0]).shape[1]
    order = rng.permutation(n)
    fold_id = np.empty(n, dtype=np.int8)
    fold_id[order] = np.arange(n, dtype=np.int64) % 4
    policy_targets = np.zeros((steps, n))
    oof_continuations = np.zeros((steps, n))
    policy_cash = payoffs[:, -1].astype(float, copy=True)
    eye = np.eye(d)
    for t in range(steps - 1, -1, -1):
        policy_targets[t] = policy_cash
        raw = features(paths[:, t], payoffs[:, t])
        for fold in range(4):
            held = fold_id == fold
            train = ~held
            if not np.any(held):
                continue
            mean = raw[train].mean(0)
            scale = raw[train].std(0)
            mean[0] = 0.0
            scale[scale < 1e-10] = 1.0
            x_train = (raw[train] - mean) / scale
            x_held = (raw[held] - mean) / scale
            coef = np.linalg.solve(x_train.T @ x_train + 0.02 * eye, x_train.T @ policy_cash[train])
            oof_continuations[t, held] = x_held @ coef
        policy_cash = np.where(payoffs[:, t] >= oof_continuations[t], payoffs[:, t], policy_cash)
    coefs = np.zeros((steps, d))
    means = np.zeros_like(coefs)
    scales = np.ones_like(coefs)
    w1s = np.zeros((steps, d, width))
    b1s = np.zeros((steps, width))
    w2s = np.zeros((steps, width))
    b2s = np.zeros(steps)
    target_scales = np.ones(steps)
    losses = np.zeros((steps, 2))
    updates = np.zeros(steps)
    for t in range(steps - 1, -1, -1):
        raw = features(paths[:, t], payoffs[:, t])
        mean = raw.mean(0)
        scale = raw.std(0)
        mean[0] = 0.0
        scale[scale < 1e-10] = 1.0
        x = (raw - mean) / scale
        target = policy_targets[t]
        coef = np.linalg.solve(x.T @ x + 0.002 * eye, x.T @ target)
        residual = target - x @ coef
        target_scale = max(float(residual.std()), 1e-05)
        y = residual / target_scale
        w1 = rng.normal(0.0, 0.1, (d, width))
        initial = w1.copy()
        b1 = np.zeros(width)
        w2 = rng.normal(0.0, 0.05, width)
        b2 = 0.0
        losses[t, 0] = np.mean((np.tanh(x @ w1 + b1) @ w2 + b2 - y) ** 2)
        for _ in range(48):
            h = np.tanh(x @ w1 + b1)
            err = h @ w2 + b2 - y
            g = 2.0 * err / n
            hidden_g = g[:, None] * w2 * (1.0 - h * h)
            gw1 = x.T @ hidden_g + 0.04 * w1
            gb1 = hidden_g.sum(0) + 0.04 * b1
            gw2 = h.T @ g + 0.04 * w2
            gb2 = g.sum() + 0.04 * b2
            w1 -= 0.015 * gw1
            b1 -= 0.015 * gb1
            w2 -= 0.015 * gw2
            b2 -= 0.015 * gb2
        correction = np.tanh(x @ w1 + b1) @ w2 + b2
        losses[t, 1] = np.mean((correction - y) ** 2)
        updates[t] = np.linalg.norm(w1 - initial)
        coefs[t], means[t], scales[t] = (coef, mean, scale)
        w1s[t], b1s[t], w2s[t], b2s[t] = (w1, b1, w2, b2)
        target_scales[t] = target_scale
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'model.npz', coefficients=coefs, means=means, scales=scales, w1=w1s, b1=b1s, w2=w2s, b2=b2s, target_scales=target_scales)
    np.savez(output_dir / 'training_trace.npz', backward_targets=policy_targets, oof_continuations=oof_continuations, loss_before_after=losses, first_layer_update_norm=updates, correction_passes=np.array(1), folds=np.array(4), seed=np.array(seed))

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
