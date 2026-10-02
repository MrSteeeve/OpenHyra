from __future__ import annotations

import numpy as np

from behavior_index import BehaviorIndex
from behavior_profiler import PROBE_SUITE_ID, BehaviorProfiler
from schemas_v5 import BehaviorProfile


PROBE_SHA = "a" * 64
POLICY_SHA = "b" * 64
BOUNDARIES = {
    "performance": [-0.01, 0.0, 0.01, 0.03],
    "tail_risk": [0.005, 0.01, 0.02],
}


def make_profiler(count: int = 4) -> BehaviorProfiler:
    return BehaviorProfiler(
        {f"instance_{index}": 1.0 for index in range(count)},
        PROBE_SHA,
    )


def make_profile(improvement: float, count: int = 4) -> BehaviorProfile:
    profiler = make_profiler(count)
    return profiler.build_profile(
        POLICY_SHA,
        {f"instance_{index}": 1.0 + improvement for index in range(count)},
        {f"instance_{index}": 0.2 + index * 0.1 for index in range(count)},
        training_seconds=5.0 + improvement,
        peak_memory_bytes=1_000,
        inference_microseconds_per_state=2.0 + improvement,
        parameter_count=100,
    )


def test_build_profile_basic() -> None:
    profile = make_profiler().build_profile(
        POLICY_SHA,
        {
            "instance_3": 1.04,
            "instance_1": 1.02,
            "instance_0": 1.01,
            "instance_2": 1.03,
        },
        {
            "instance_3": 0.4,
            "instance_1": 0.2,
            "instance_0": 0.1,
            "instance_2": 0.3,
        },
        training_seconds=12.5,
        peak_memory_bytes=4096,
        inference_microseconds_per_state=3.5,
        parameter_count=128,
    )

    improvements = np.array([0.01, 0.02, 0.03, 0.04])
    losses = -improvements
    assert profile.schema == "openhyra-behavior-profile.v1"
    assert profile.probe_suite == PROBE_SUITE_ID
    assert profile.probe_suite_sha256 == PROBE_SHA
    assert profile.policy_artifact_sha256 == POLICY_SHA
    assert np.allclose(profile.performance["per_instance_improvement"], improvements)
    assert np.isclose(profile.performance["paired_mean"], np.mean(improvements))
    assert np.isclose(
        profile.performance["paired_standard_error"],
        np.std(improvements, ddof=1) / np.sqrt(improvements.size),
    )
    assert profile.outcome_distribution["loss_definition"] == (
        "negative_paired_discounted_payoff_improvement"
    )
    assert np.isclose(profile.outcome_distribution["mean_loss"], np.mean(losses))
    assert np.isclose(
        profile.outcome_distribution["var_95"],
        np.percentile(losses, 95),
    )
    assert np.isclose(profile.outcome_distribution["cvar_95"], -0.01)
    assert profile.policy_geometry["exercise_rate_by_instance"] == [0.1, 0.2, 0.3, 0.4]
    assert profile.policy_geometry["boundary_monotonicity_violations"] == 0
    assert profile.policy_geometry["reference_boundary_agreement"] == 0.0
    assert profile.sensitivity == {
        "moneyness": 0.0,
        "volatility": 0.0,
        "correlation": 0.0,
        "time_to_maturity": 0.0,
    }
    assert profile.robustness == {
        "input_scale_invariance_error": 0.0,
        "state_perturbation_lipschitz_proxy": 0.0,
        "seed_instability": 0.0,
    }
    assert profile.compute == {
        "training_seconds": 12.5,
        "peak_memory_bytes": 4096,
        "inference_microseconds_per_state": 3.5,
        "parameter_count": 128,
    }


def test_improvement_calculation() -> None:
    profiler = BehaviorProfiler({"b": 2.0, "a": 1.0}, PROBE_SHA)
    profile = profiler.build_profile(
        POLICY_SHA,
        {"b": 1.5, "a": 1.25},
        {"b": 0.3, "a": 0.2},
        1.0,
        100,
        2.0,
        10,
    )
    assert profile.performance["per_instance_improvement"] == [0.25, -0.5]


def test_loss_distribution() -> None:
    profiler = BehaviorProfiler({str(index): 0.0 for index in range(4)}, PROBE_SHA)
    profile = profiler.build_profile(
        POLICY_SHA,
        {"0": -1.0, "1": 0.0, "2": 1.0, "3": 2.0},
        {str(index): 0.1 * index for index in range(4)},
        1.0,
        100,
        2.0,
        10,
    )
    losses = np.array([1.0, 0.0, -1.0, -2.0])
    var_95 = float(np.percentile(losses, 95))
    assert np.isclose(profile.outcome_distribution["mean_loss"], np.mean(losses))
    assert np.isclose(profile.outcome_distribution["var_95"], var_95)
    assert np.isclose(profile.outcome_distribution["cvar_95"], 1.0)


def test_seed_instability() -> None:
    profiler = make_profiler()
    profile = profiler.build_profile(
        POLICY_SHA,
        {f"instance_{index}": 1.1 for index in range(4)},
        {f"instance_{index}": 0.2 for index in range(4)},
        1.0,
        100,
        2.0,
        10,
        seed_scores={"seed_1": [1.0, 1.1], "seed_2": [1.4, 1.5]},
    )
    assert profile.robustness["seed_instability"] > 0.0


def test_no_seed_scores() -> None:
    assert make_profile(0.01).robustness["seed_instability"] == 0.0


def test_assign_cell() -> None:
    profile = make_profile(-0.02)
    assert BehaviorIndex(BOUNDARIES).assign_cell(profile) == (0, 3)


def test_behavior_vector_length() -> None:
    instance_count = 4
    vector = BehaviorIndex(BOUNDARIES).behavior_vector(make_profile(0.01, instance_count))
    assert len(vector) == 2 * instance_count + 10 + 4 + 3


def test_nearest_neighbors() -> None:
    candidates = [make_profile(value) for value in [-0.04, -0.01, 0.02, 0.05, 0.09]]
    query = make_profile(0.021)
    neighbors = BehaviorIndex(BOUNDARIES).nearest_neighbors(query, candidates, k=3)
    assert neighbors[0][0] == 2
    assert len(neighbors) == 3
    assert neighbors == sorted(neighbors, key=lambda item: item[1])


def test_cell_diversity() -> None:
    profiles = (
        [make_profile(0.02) for _ in range(4)]
        + [make_profile(0.005) for _ in range(3)]
        + [make_profile(-0.015) for _ in range(2)]
        + [make_profile(-0.03)]
    )
    assert BehaviorIndex(BOUNDARIES).cell_diversity(profiles) == {
        "total_profiles": 10,
        "occupied_cells": 4,
        "max_cell_count": 4,
        "min_cell_count": 1,
        "singleton_cells": 1,
    }


def test_profile_validates() -> None:
    profile = make_profile(0.01)
    assert isinstance(profile, BehaviorProfile)
    validate = getattr(profile, "validate", None)
    if validate is not None:
        assert validate() is None
