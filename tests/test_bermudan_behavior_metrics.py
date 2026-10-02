"""Small behavior descriptors emitted by the Bermudan evaluator."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from tasks.bermudan_optimal_stopping import evaluator


def _search_request(*, repeats: int = 2) -> dict:
    return {
        "schema": evaluator.REQUEST_SCHEMA,
        "stage": "search",
        "task": evaluator.TASK_NAME,
        "protocol": evaluator.TASK_PROTOCOL,
        "seed": 20260903,
        "suite_id": "behavior-metrics-smoke",
        "config": {
            "instance_count": 1,
            "repeats": repeats,
            "training_paths": 64,
            "pricing_paths": 128,
            "ridge_alpha": 1e-6,
        },
    }


def test_policy_behavior_descriptor_uses_early_exercise_rate() -> None:
    descriptor = evaluator._policy_behavior_metrics(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([0, 1, 3, 3], dtype=np.int64),
        4,
    )

    # Two of four paths stop before maturity; terminal settlement is not an
    # exercise event for the scalar rate.
    assert descriptor["exercise_rate"] == 0.5
    assert descriptor["exercise_rate_by_time"] == [0.25, 0.25, 0.0, 0.5]
    assert math.isclose(descriptor["stop_time_mean"], 7.0 / 12.0)
    assert math.isclose(descriptor["stop_time_std"], 0.4330127019, rel_tol=1e-9)
    assert descriptor["finite"] is True
    assert descriptor["valid_stop_rate"] == 1.0


def test_search_emits_per_instance_behavior_and_repeat_stability() -> None:
    _score, metrics, _normalized, _evidence = evaluator.evaluate_submission(
        evaluator.BASELINE_PROGRAM,
        _search_request(repeats=2),
    )

    instance_id = "public-put-atm"
    assert set(metrics["per_instance_scores"]) == {instance_id}
    assert set(metrics["baseline_scores"]) == {instance_id}
    assert set(metrics["per_instance_exercise_rates"]) == {instance_id}
    assert set(metrics["per_instance_exercise_rate_std"]) == {instance_id}
    assert metrics["per_instance_finite_rates"][instance_id] == 1.0
    assert metrics["behavior_finite"] is True
    assert 0.0 <= metrics["per_instance_exercise_rates"][instance_id] <= 1.0
    assert metrics["per_instance_exercise_rate_std"][instance_id] >= 0.0

    rows = metrics["summaries"]
    assert len(rows) == 2
    for row in rows:
        assert row["candidate_finite"] is True
        assert row["baseline_finite"] is True
        assert 0.0 <= row["candidate_exercise_rate"] <= 1.0
        assert len(row["candidate_exercise_rate_by_time"]) == 6
        assert math.isclose(
            sum(row["candidate_exercise_rate_by_time"]), 1.0,
            rel_tol=1e-12,
        )


def test_invalid_stopping_indices_are_marked_without_nonfinite_metrics() -> None:
    descriptor = evaluator._policy_behavior_metrics(
        np.asarray([1.0, 2.0]),
        np.asarray([0.0, 2.0]),
        2,
    )
    assert descriptor["finite"] is False
    assert descriptor["valid_stop_rate"] == 0.0
    assert all(math.isfinite(value) for value in descriptor["exercise_rate_by_time"])


def test_algorithm_bundle_search_exposes_the_same_behavior_projection() -> None:
    """The open Python path gets descriptors from the trusted evaluator too."""
    manifest = {
        "schema": "continuation-expression.v1",
        "runner_type": "expression",
        "inference_config": {
            "input_dim": "n_assets",
            "output_dim": 1,
            "output_clip": [-1_000_000.0, 1_000_000.0],
        },
        "output_semantics": "discounted_continuation_value_t0",
        "normalization": "none",
        "weight_pattern": "step_{:03d}.json",
    }
    train_source = """\
import argparse, json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--seed', required=True)
args = parser.parse_args()
steps = np.load(Path(args.input) / 'training_paths.npy', allow_pickle=False).shape[1] - 1
for index in range(steps):
    (Path(args.output) / f'step_{index:03d}.json').write_text(
        json.dumps({'op': 'constant', 'value': 0.0})
    )
"""
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "candidate"
        source.mkdir()
        (source / "manifest.json").write_text(json.dumps(manifest))
        (source / "train.py").write_text(train_source)
        _score, metrics, _normalized, _evidence = evaluator.evaluate_submission(
            manifest,
            _search_request(repeats=1),
            candidate_source_dir=source,
        )

    assert metrics["candidate_kind"] == "algorithm_bundle"
    assert metrics["runner_type"] == "expression"
    assert metrics["training_cell_count"] == 1
    assert metrics["per_instance_exercise_rates"]
    assert metrics["per_instance_finite_rates"] == {"public-put-atm": 1.0}
