from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from typing import Any

import numpy as np

from schemas_v5 import BehaviorProfile


class BehaviorIndex:
    """Assign behavior cells and perform normalized nearest-neighbor search."""

    def __init__(self, bucket_boundaries: dict[str, list[float]]) -> None:
        missing = {"performance", "tail_risk"}.difference(bucket_boundaries)
        if missing:
            raise ValueError(f"missing bucket boundaries: {', '.join(sorted(missing))}")
        self.bucket_boundaries = {
            name: sorted(float(value) for value in boundaries)
            for name, boundaries in bucket_boundaries.items()
        }

    def assign_cell(self, profile: BehaviorProfile) -> tuple[int, int]:
        performance = float(profile.performance["paired_mean"])
        tail_risk = float(profile.outcome_distribution["cvar_95"])
        return (
            bisect_right(self.bucket_boundaries["performance"], performance),
            bisect_right(self.bucket_boundaries["tail_risk"], tail_risk),
        )

    def behavior_vector(self, profile: BehaviorProfile) -> list[float]:
        performance = profile.performance
        outcome = profile.outcome_distribution
        geometry = profile.policy_geometry
        sensitivity = profile.sensitivity
        robustness = profile.robustness
        compute = profile.compute

        vector = list(performance["per_instance_improvement"])
        vector.extend(geometry["exercise_rate_by_instance"])
        vector.extend(
            [
                performance["paired_mean"],
                performance["paired_standard_error"],
                outcome["mean_loss"],
                outcome["var_95"],
                outcome["cvar_95"],
                geometry["reference_boundary_agreement"],
                robustness["seed_instability"],
                compute["training_seconds"],
                compute["inference_microseconds_per_state"],
                compute["parameter_count"],
                sensitivity["moneyness"],
                sensitivity["volatility"],
                sensitivity["correlation"],
                sensitivity["time_to_maturity"],
                robustness["input_scale_invariance_error"],
                robustness["state_perturbation_lipschitz_proxy"],
                robustness["seed_instability"],
            ]
        )
        return [float(value) for value in vector]

    def nearest_neighbors(
        self,
        query: BehaviorProfile,
        candidates: list[BehaviorProfile],
        k: int = 5,
    ) -> list[tuple[int, float]]:
        if k <= 0 or not candidates:
            return []

        candidate_vectors = np.asarray(
            [self.behavior_vector(profile) for profile in candidates],
            dtype=float,
        )
        query_vector = np.asarray(self.behavior_vector(query), dtype=float)
        if candidate_vectors.ndim != 2 or candidate_vectors.shape[1] != query_vector.size:
            raise ValueError("all behavior vectors must have the same length")

        means = np.mean(candidate_vectors, axis=0)
        standard_deviations = np.std(candidate_vectors, axis=0)
        scales = np.where(standard_deviations == 0.0, 1.0, standard_deviations)
        normalized_candidates = (candidate_vectors - means) / scales
        normalized_query = (query_vector - means) / scales
        distances = np.linalg.norm(
            normalized_candidates - normalized_query,
            axis=1,
        )
        indices = sorted(
            range(len(candidates)),
            key=lambda index: (float(distances[index]), index),
        )[: min(k, len(candidates))]
        return [(index, float(distances[index])) for index in indices]

    def cell_diversity(self, profiles: list[BehaviorProfile]) -> dict[str, Any]:
        counts = Counter(self.assign_cell(profile) for profile in profiles)
        cell_counts = list(counts.values())
        return {
            "total_profiles": len(profiles),
            "occupied_cells": len(counts),
            "max_cell_count": max(cell_counts, default=0),
            "min_cell_count": min(cell_counts, default=0),
            "singleton_cells": sum(count == 1 for count in cell_counts),
        }
