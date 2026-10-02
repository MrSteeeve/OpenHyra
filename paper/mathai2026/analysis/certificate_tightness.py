"""Certificate-tightness calibration on the one-dimensional Bermudan puts of the
frozen holdout panel.

For each 1-d put in the holdout suite we compute a high-precision reference value
V* with a Cox-Ross-Rubinstein binomial tree whose step grid contains every exercise
date (exercise allowed at the evaluator's dates only, including t=0 and maturity;
continuous dividend yield; discounting to time zero).  We then split the recorded
raw certificate width U_hat - L_hat of every frozen program into

    (V* - L_hat)   policy suboptimality   and   (U_hat - V*)   dual looseness.

Inputs are the frozen evaluator (instance derivation is verified by SHA-256 digest
against data/evaluation_cells.csv) and the frozen per-cell bounds.  No model calls.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "artifacts" / "mathai-paper-assets-20260906"
OUT = Path(__file__).resolve().parent


def load_frozen_evaluator():
    path = ASSETS / "frozen_runtime" / "tasks" / "bermudan_optimal_stopping" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("frozen_ev", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["frozen_ev"] = module
    spec.loader.exec_module(module)
    return module


def bermudan_put_crr(spot, strike, rate, dividend, vol, exercise_times, steps_per_interval):
    """CRR tree; exercise only at the given dates. Returns value at t=0."""
    times = np.asarray(exercise_times, dtype=float)
    intervals = np.diff(times)
    dt = intervals[0]  # uniform grid by construction
    if not np.allclose(intervals, dt):
        raise ValueError("non-uniform exercise grid")
    n_int = len(intervals)
    k = int(steps_per_interval)
    h = dt / k
    u = math.exp(vol * math.sqrt(h))
    d = 1.0 / u
    growth = math.exp((rate - dividend) * h)
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability out of range")
    disc = math.exp(-rate * h)
    N = n_int * k
    j = np.arange(N + 1)
    s_T = spot * u ** (N - j) * d ** j
    value = np.maximum(strike - s_T, 0.0)  # exercise at maturity (mandatory settlement)
    for step in range(N - 1, -1, -1):
        value = disc * (p * value[:-1] + (1.0 - p) * value[1:])
        if step % k == 0:  # an exercise date (including t=0)
            s = spot * u ** (step - np.arange(step + 1)) * d ** np.arange(step + 1)
            value = np.maximum(value, np.maximum(strike - s, 0.0))
    return float(value[0])


def main():
    ev = load_frozen_evaluator()
    cells = [r for r in csv.DictReader(open(ASSETS / "data" / "evaluation_cells.csv"))
             if r["stage"] == "holdout" and r["payoff_type"] == "put"]
    digest = {(r["seed"], r["instance_id"]): r["instance_sha256"] for r in cells}

    instances = {}
    for s, seed in (("17", 660017), ("29", 660029), ("41", 660041)):
        for inst in ev.derive_hidden_suite(seed, 4):
            if inst.payoff_type != "put":
                continue
            if ev._instance_digest(inst) != digest[(s, inst.instance_id)]:
                raise RuntimeError("instance digest mismatch")
            instances[(s, inst.instance_id)] = inst

    ref_rows = []
    vstar = {}
    for key, inst in instances.items():
        vals = {}
        for k in (250, 500, 1000):
            vals[k] = bermudan_put_crr(inst.spots[0], inst.strike, inst.rate, inst.dividends[0],
                                       inst.volatilities[0], inst.exercise_times, k)
        # Richardson on the two finest grids (CRR error ~ 1/N for Bermudan grids)
        v_ext = 2.0 * vals[1000] - vals[500]
        vstar[key] = v_ext
        eur = ev.black_scholes_european_price(inst.spots[0], inst.strike, inst.rate, inst.dividends[0],
                                              inst.volatilities[0], inst.maturity, "put")
        ref_rows.append({
            "seed": key[0], "instance_id": key[1], "spot": inst.spots[0], "strike": inst.strike,
            "rate": inst.rate, "dividend": inst.dividends[0], "vol": inst.volatilities[0],
            "maturity": inst.maturity, "exercise_dates": len(inst.exercise_times) - 1,
            "european_put": eur, "crr_250": vals[250], "crr_500": vals[500], "crr_1000": vals[1000],
            "v_star": v_ext, "grid_change_500_1000": vals[1000] - vals[500],
        })

    with open(OUT / "holdout_put_reference_values.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ref_rows[0].keys()))
        w.writeheader()
        w.writerows(ref_rows)

    out_rows = []
    for r in cells:
        key = (r["seed"], r["instance_id"])
        V = vstar[key]
        L = float(r["lower"]); U = float(r["upper"])
        K = float(r["strike"])
        out_rows.append({
            "seed": r["seed"], "instance_id": r["instance_id"], "method": r["method"], "key": r["key"],
            "v_star": V, "lower": L, "upper": U, "lower_se": float(r["lower_se"]), "upper_se": float(r["upper_se"]),
            "raw_gap": (U - L) / K, "policy_suboptimality": (V - L) / K, "dual_looseness": (U - V) / K,
            "looseness_share": (U - V) / (U - L) if U != L else float("nan"),
            "policy_z": (V - L) / float(r["lower_se"]) if float(r["lower_se"]) > 0 else float("nan"),
        })
    with open(OUT / "holdout_put_tightness_by_cell.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # summary per method over the six puts
    by_method = {}
    for r in out_rows:
        by_method.setdefault(r["method"], []).append(r)
    summary = []
    for m, rows in by_method.items():
        summary.append({
            "method": m, "n_cells": len(rows),
            "mean_raw_gap": np.mean([x["raw_gap"] for x in rows]),
            "mean_policy_suboptimality": np.mean([x["policy_suboptimality"] for x in rows]),
            "mean_dual_looseness": np.mean([x["dual_looseness"] for x in rows]),
            "mean_looseness_share": np.mean([x["looseness_share"] for x in rows]),
            "min_policy_suboptimality": min(x["policy_suboptimality"] for x in rows),
            "max_policy_suboptimality": max(x["policy_suboptimality"] for x in rows),
        })
    with open(OUT / "holdout_put_tightness_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    json.dump({"instances": ref_rows, "summary": summary}, open(OUT / "certificate_tightness.json", "w"), indent=1)

    print("reference values (V*), CRR grid check:")
    for r in ref_rows:
        print(f"  s{r['seed']} {r['instance_id']}: V*={r['v_star']:.6f}  crr500={r['crr_500']:.6f} crr1000={r['crr_1000']:.6f}  eur={r['european_put']:.6f}")
    print("per-method means over six puts:")
    for s in sorted(summary, key=lambda x: x["mean_policy_suboptimality"]):
        print(f"  {s['method']:<20} raw {s['mean_raw_gap']:.5f}  policy {s['mean_policy_suboptimality']:.5f}  dual {s['mean_dual_looseness']:.5f}  share {s['mean_looseness_share']:.3f}")


if __name__ == "__main__":
    main()
