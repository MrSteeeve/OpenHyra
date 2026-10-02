#!/usr/bin/env python3
"""Camera-ready statistics derived from the frozen tightness outputs (no new simulation).

Adds, for the six one-dimensional holdout puts:
  * a one-sided 95% upper confidence bound on each program's mean policy loss,
    mean((V* - L)/K) + 1.645 * sqrt(sum SE_L^2) / n, treating V* as exact
    (Richardson grid change <= 9e-6) and cells as independent;
  * the early-exercise premium V* - European, i.e. the policy loss of never
    exercising before maturity;
  * the Bonferroni z for simultaneous one-sided 95% coverage of n gaps.
Inputs: holdout_put_tightness_by_cell.csv, holdout_put_reference_values.csv.
Output: camera_ready_stats.json and ../tables/tightness_methods.tex.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
Z_ONE_SIDED_95 = NormalDist().inv_cdf(0.95)

# (label, method key, operator filter or None); PCA + Ridge and policy iteration
# fit the same regression as Ridge on 1-d puts and give identical cells.
ROWS = [
    (r"Ridge, PCA + Ridge, Pol.\ iter.$^\dagger$", "linear_ridge", None),
    ("Gated Ridge", "gated_ridge", None),
    ("MLP", "mlp", None),
    ("Ridge + MLP residual", "residual_hybrid", None),
    ("Threshold policy", "direct_decision", None),
    ("Original planner: best generated", "context_proposal", None),
    ("Original direct: best generated", "direct_generation", None),
    ("Score-only rewrites (one per seed)", "score_only", "subsystem_rewrite"),
]


def main() -> None:
    cells = list(csv.DictReader(open(HERE / "holdout_put_tightness_by_cell.csv")))
    refs = list(csv.DictReader(open(HERE / "holdout_put_reference_values.csv")))

    for twin in ("pca_ridge", "policy_iteration"):
        a = sorted((c["seed"], c["instance_id"], c["lower"]) for c in cells if c["method"] == twin)
        b = sorted((c["seed"], c["instance_id"], c["lower"]) for c in cells if c["method"] == "linear_ridge")
        assert a == b, f"{twin} cells differ from linear_ridge"

    v_star = [float(r["v_star"]) for r in refs]
    premium = [float(r["v_star"]) - float(r["european_put"]) for r in refs]
    out = {
        "mean_v_star": sum(v_star) / len(v_star),
        "early_exercise_premium": {f'{r["seed"]}/{r["instance_id"]}': p for r, p in zip(refs, premium)},
        "mean_early_exercise_premium": sum(premium) / len(premium),
        "cells_with_premium_below_4e-4": sum(p < 4e-4 for p in premium),
        "bonferroni_z_simultaneous_95": {n: NormalDist().inv_cdf(1 - 0.05 / (2 * n)) for n in (1, 2, 4, 6)},
        "methods": {},
    }

    lines = [
        "% Decomposition of the raw certificate width on the holdout puts: (U-L)/K = (V*-L)/K + (U-V*)/K. "
        "UCB = one-sided 95% upper confidence bound on mean policy loss (camera_ready_stats.py).",
        r"\begin{tabular}{l r rrrr r}",
        r"\toprule",
        r"Program & $n$ & $(\hat U-\hat L)/K$ & $(V^\star-\hat L)/K$ & UCB & $(\hat U-V^\star)/K$ & Within MC noise \\",
        r"\midrule",
    ]
    for label, method, operator in ROWS:
        rows = [c for c in cells if c["method"] == method and (operator is None or c["key"].endswith(operator))]
        n = len(rows)
        gap = sum(float(c["raw_gap"]) for c in rows) / n
        loss = sum(float(c["policy_suboptimality"]) for c in rows) / n
        loose = sum(float(c["dual_looseness"]) for c in rows) / n
        sem = math.sqrt(sum(float(c["lower_se"]) ** 2 for c in rows)) / n
        ucb = loss + Z_ONE_SIDED_95 * sem
        within = sum(abs(float(c["policy_suboptimality"])) <= 1.96 * float(c["lower_se"]) for c in rows)
        out["methods"][label] = dict(n=n, raw_gap=gap, policy_loss=loss, policy_loss_ucb95=ucb,
                                     dual_looseness=loose, within_noise=within,
                                     looseness_share=loose / gap)
        lines.append(f"{label} & {n} & {gap:.5f} & {loss:.5f} & {ucb:.5f} & {loose:.5f} & {within}/{n} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]

    (HERE / "camera_ready_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    (HERE.parent / "tables" / "tightness_methods.tex").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in out.items() if k != "methods"}, indent=2))
    for k, v in out["methods"].items():
        print(f"{k:60s} loss={v['policy_loss']:.5f} ucb={v['policy_loss_ucb95']:.5f} share={v['looseness_share']:.3f}")


if __name__ == "__main__":
    main()
