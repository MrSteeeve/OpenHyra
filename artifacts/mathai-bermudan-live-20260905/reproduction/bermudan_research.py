"""Evaluator-backed paired prediction tests for the Bermudan research loop.

The primary evaluator score is untouched. These diagnostics condition on the
fixed public suite; three-seed comparisons remain exploratory search evidence.
"""
from __future__ import annotations
import math
from statistics import NormalDist


def _key(row):
    return (row.get("instance_id"), row.get("repeat", 0))


def paired_slice_effects(guided, control, target_slice=""):
    """Join common-random-number cells and compute within-path contrast SEs.

    If pathwise covariance was not exported, use SE(g)+SE(c), a conservative
    Cauchy-Schwarz bound. Never call max(SE(g),SE(c)) a paired standard error.
    """
    left = {_key(r): r for r in guided.get("summaries", [])}
    right = {_key(r): r for r in control.get("summaries", [])}
    cells = []
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        labels = {str(key[0]), "instance:" + str(key[0]), *a.get("slice_labels", [])}
        if target_slice and target_slice not in {"all", "public_suite"} and target_slice not in labels:
            continue
        av, bv = a.get("paired_pathwise_improvements"), b.get("paired_pathwise_improvements")
        path_hash = a.get("pricing_paths_sha256")
        if av is not None and bv is not None:
            if not path_hash or path_hash != b.get("pricing_paths_sha256") or len(av) != len(bv) or len(av) < 2:
                return {"status": "invalid_control", "reason": "pricing path identity/size mismatch", "cells": []}
            values = [float(x) - float(y) for x, y in zip(av, bv)]
            effect = sum(values) / len(values)
            se = math.sqrt(sum((x - effect) ** 2 for x in values) / (len(values) - 1) / len(values))
            method = "common_random_numbers_pathwise"
        else:
            if a.get("paired_normalized_improvement") is None or b.get("paired_normalized_improvement") is None:
                continue
            effect = a["paired_normalized_improvement"] - b["paired_normalized_improvement"]
            se = a["paired_normalized_standard_error"] + b["paired_normalized_standard_error"]
            method = "conservative_se_sum_without_covariance"
        cells.append({"instance_id": key[0], "repeat": key[1], "effect": effect, "standard_error": se,
                      "uncertainty_method": method, "pricing_paths_sha256": path_hash,
                      "behavior": {"guided_exercise_rate": a.get("candidate_exercise_rate"),
                                   "control_exercise_rate": b.get("candidate_exercise_rate"),
                                   "guided_stop_time_mean": a.get("candidate_stop_time_mean"),
                                   "control_stop_time_mean": b.get("candidate_stop_time_mean")}})
    if not cells:
        return {"status": "not_observed", "reason": "target slice has no common cells", "cells": []}
    n = len(cells)
    effect = sum(c["effect"] for c in cells) / n
    se = math.sqrt(sum(c["standard_error"] ** 2 for c in cells)) / n
    z = NormalDist().inv_cdf(.975)
    low, high = effect - z * se, effect + z * se
    verdict = "supported" if low > 0 else "refuted" if high < 0 else "inconclusive"
    return {"status": "observed", "target_slice": target_slice or "all", "cells": cells,
            "effect": effect, "standard_error": se, "ci95": [low, high],
            "prediction_verdict": verdict, "prediction_direction_correct": effect > 0,
            "test": "positive_guided_minus_control_mean_on_preregistered_slice",
            "next_action": "compose" if verdict == "supported" else "restart" if verdict == "refuted" else "revise",
            "scope": "fixed_public_suite_conditional_monte_carlo"}
