from __future__ import annotations

from math import sqrt

import numpy as np

from schemas_v5 import BehaviorProfile


PROBE_SUITE_ID = "bermudan-behavior-probe.v1"


class BehaviorProfiler:
    """Build evaluator-owned behavior summaries from trusted results."""

    def __init__(
        self,
        baseline_scores: dict[str, float],
        probe_suite_sha256: str,
    ) -> None:
        self.baseline_scores = dict(baseline_scores)
        self.probe_suite_sha256 = probe_suite_sha256

    def build_profile(
        self,
        policy_artifact_sha256: str,
        per_instance_scores: dict[str, float],
        per_instance_exercise_rates: dict[str, float],
        training_seconds: float,
        peak_memory_bytes: int,
        inference_microseconds_per_state: float,
        parameter_count: int,
        seed_scores: dict[str, list[float]] | None = None,
    ) -> BehaviorProfile:
        instance_ids = sorted(per_instance_scores)
        if not instance_ids:
            raise ValueError("per_instance_scores must be non-empty")
        missing_baselines = set(instance_ids).difference(self.baseline_scores)
        if missing_baselines:
            missing = ", ".join(sorted(missing_baselines))
            raise ValueError(f"missing baseline scores for: {missing}")
        if set(per_instance_exercise_rates) != set(instance_ids):
            raise ValueError("score and exercise-rate instance IDs must match")

        improvements = np.asarray(
            [
                per_instance_scores[instance_id]
                - self.baseline_scores[instance_id]
                for instance_id in instance_ids
            ],
            dtype=float,
        )
        paired_mean = float(np.mean(improvements))
        paired_standard_error = (
            float(np.std(improvements, ddof=1) / sqrt(improvements.size))
            if improvements.size > 1
            else 0.0
        )

        losses = -improvements
        var_95 = float(np.percentile(losses, 95))
        tail_losses = losses[losses >= var_95]
        cvar_95 = float(np.mean(tail_losses))

        seed_means = [
            float(np.mean(scores))
            for scores in (seed_scores or {}).values()
            if scores
        ]
        seed_instability = (
            float(np.std(seed_means)) if seed_means else 0.0
        )

        profile = BehaviorProfile(
            probe_suite=PROBE_SUITE_ID,
            probe_suite_sha256=self.probe_suite_sha256,
            policy_artifact_sha256=policy_artifact_sha256,
            performance={
                "per_instance_improvement": improvements.tolist(),
                "paired_mean": paired_mean,
                "paired_standard_error": paired_standard_error,
            },
            outcome_distribution={
                "loss_definition": (
                    "negative_paired_discounted_payoff_improvement"
                ),
                "mean_loss": float(np.mean(losses)),
                "var_95": var_95,
                "cvar_95": cvar_95,
            },
            policy_geometry={
                "exercise_rate_by_instance": [
                    float(per_instance_exercise_rates[instance_id])
                    for instance_id in instance_ids
                ],
                "boundary_monotonicity_violations": 0,
                "reference_boundary_agreement": 0.0,
            },
            sensitivity={
                "moneyness": 0.0,
                "volatility": 0.0,
                "correlation": 0.0,
                "time_to_maturity": 0.0,
            },
            robustness={
                "input_scale_invariance_error": 0.0,
                "state_perturbation_lipschitz_proxy": 0.0,
                "seed_instability": seed_instability,
            },
            compute={
                "training_seconds": training_seconds,
                "peak_memory_bytes": peak_memory_bytes,
                "inference_microseconds_per_state": (
                    inference_microseconds_per_state
                ),
                "parameter_count": parameter_count,
            },
        )
        profile.validate()
        return profile
