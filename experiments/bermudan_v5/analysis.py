#!/usr/bin/env python3
"""Analyze Bermudan v5 experiment results for ICLR submission."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def load_eb_records(run_dir: Path) -> list[dict]:
    records_path = run_dir / "eb" / "records.jsonl"
    if not records_path.exists():
        return []
    records = []
    for line in records_path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def best_score(records: list[dict], direction: str = "max") -> float | None:
    scored = [r["score"] for r in records if r.get("score") is not None]
    if not scored:
        return None
    return max(scored) if direction == "max" else min(scored)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def analyze_h1(config: dict) -> dict:
    """H1: search beats baseline."""
    results = []
    for seed in config["seeds"]:
        run_dir = ROOT / "runs" / config["task"] / f"baseline_v5_full_seed{seed}"
        records = load_eb_records(run_dir)
        if not records:
            continue
        seed_record = records[0]
        best = best_score(records)
        if best is not None and seed_record.get("score") is not None:
            results.append({
                "seed": seed,
                "baseline_score": seed_record["score"],
                "best_score": best,
                "improvement": best - seed_record["score"],
            })
    improvements = [r["improvement"] for r in results]
    return {
        "hypothesis": "H1",
        "n_seeds": len(results),
        "mean_improvement": mean(improvements),
        "std_improvement": std(improvements),
        "all_positive": all(i > 0 for i in improvements) if improvements else False,
        "details": results,
    }


def analyze_h2(config: dict) -> dict:
    """H2: diversity (4 islands vs 1 island)."""
    results = []
    for seed in config["seeds"]:
        full_best = best_score(load_eb_records(
            ROOT / "runs" / config["task"] / f"baseline_v5_full_seed{seed}"))
        single_best = best_score(load_eb_records(
            ROOT / "runs" / config["task"] / f"ablation_single_island_seed{seed}"))
        if full_best is not None and single_best is not None:
            results.append({
                "seed": seed,
                "full_best": full_best,
                "single_best": single_best,
                "delta": full_best - single_best,
            })
    deltas = [r["delta"] for r in results]
    return {
        "hypothesis": "H2",
        "n_seeds": len(results),
        "mean_delta": mean(deltas),
        "std_delta": std(deltas),
        "diversity_wins": sum(1 for d in deltas if d > 0),
        "details": results,
    }


def analyze_h6(config: dict) -> dict:
    """H6: conditional algorithm advantage across instance slices."""
    results = []
    for seed in config["seeds"]:
        records = load_eb_records(
            ROOT / "runs" / config["task"] / f"baseline_v5_full_seed{seed}")
        algorithms = {}
        for r in records:
            per_instance = r.get("metrics", {}).get("per_instance_results")
            if per_instance and r.get("score") is not None:
                algo_type = r.get("description", "unknown")[:50]
                algorithms.setdefault(algo_type, []).append(r["id"])
        results.append({
            "seed": seed,
            "algorithm_count": len(algorithms),
            "algorithm_types": list(algorithms.keys())[:10],
        })
    return {"hypothesis": "H6", "n_seeds": len(results), "details": results}


def main():
    config_path = Path(__file__).parent / "experiment_config.json"
    config = json.loads(config_path.read_text())

    print(f"=== {config['experiment_name']} Analysis ===\n")

    h1 = analyze_h1(config)
    print(f"H1 (search beats baseline): improvement = {h1['mean_improvement']:.6f} +/- {h1['std_improvement']:.6f}, n={h1['n_seeds']}")

    h2 = analyze_h2(config)
    print(f"H2 (diversity helps): delta = {h2['mean_delta']:.6f} +/- {h2['std_delta']:.6f}, wins={h2['diversity_wins']}/{h2['n_seeds']}")

    h6 = analyze_h6(config)
    for d in h6["details"]:
        print(f"H6 seed={d['seed']}: {d['algorithm_count']} algorithm types")

    report = {"h1": h1, "h2": h2, "h6": h6}
    report_path = Path(__file__).parent / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
