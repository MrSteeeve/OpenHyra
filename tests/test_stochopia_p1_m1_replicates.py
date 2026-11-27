"""P2 metric and rollout settings are explicit in workers and cache identities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stochopia.neural.lob_training import P1_TRAIN_SEEDS
from stochopia.neural.primitives import seed_lob_source
from tasks.stochopia_p1 import market_eval, market_worker
from tasks.stochopia_p1.store import EvaluationRequest, Json, ModelRef


SYMBOL = "300001.SZ"
TRAIN = tuple(range(1, 37))
DEV = tuple(range(38, 50))
TEST = tuple(range(51, 63))


def _manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({
        "train_dates": TRAIN, "dev_dates": DEV, "test_dates": TEST, "symbols": [SYMBOL],
    }), encoding="utf-8")


def test_p2_worker_argv_carries_replica_schema_batch_and_state_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = []

    def worker(command: list[str], **kwargs: object) -> Json:
        commands.append(command)
        return {"returncode": 0, "timed_out": False, "stdout": "", "stderr": ""}

    monkeypatch.setattr(market_eval, "run_worker_subprocess", worker)
    market_eval._run_worker(tmp_path / "old", tmp_path, tmp_path, "evaluate", 60)
    market_eval._run_worker(
        tmp_path / "new", tmp_path, tmp_path, "evaluate", 60,
        num_replicates=8, metric_schema_version="p2-m1", rollout_batch_windows=3,
    )
    assert commands[0][8:] == ["--device", "cpu", "--device-id", "cpu"]
    assert commands[1][12:] == [
        "--num-replicates", "8", "--metric-schema-version", "p2-m1",
        "--rollout-batch-windows", "3", "--max-state-dim", "512",
    ]


def test_p2_profiles_are_separate_by_schema_k_and_batch_but_share_fit_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest(tmp_path)
    cache = tmp_path / "run-cache"
    model = ModelRef("seed", seed_lob_source())
    seen = []

    def measure(
        self: market_eval.MarketEvaluator, model: ModelRef, request: EvaluationRequest, *,
        key: str, candidate: bool,
    ) -> Json:
        seen.append((key, self.metric_schema_version, self.num_replicates, self.options["rollout_batch_windows"]))
        return {"evidence": dict(self.options), "day_metrics": {str(date): {} for date in DEV}}

    monkeypatch.setattr(market_eval.MarketEvaluator, "_measure", measure)
    variants = [("p1", 1, 2), ("p2-m1", 1, 2), ("p2-m1", 4, 2), ("p2-m1", 8, 2), ("p2-m1", 8, 3)]
    for schema, k, batch in variants:
        evaluator = market_eval.MarketEvaluator(
            tmp_path, SYMBOL, cache_root=cache, metric_schema_version=schema,
            num_replicates=k, rollout_batch_windows=batch,
        )
        first = evaluator.seed_profile(model, rollout_seed=12300)
        assert evaluator.seed_profile(model, rollout_seed=12300) == first
        assert first["evidence"]["num_replicates"] == k
    assert [row[1:] for row in seen] == variants
    assert len({row[0] for row in seen}) == 1  # one fit per device/source/seed panel
    assert len(list((cache / SYMBOL / "dev").rglob("profile.json"))) == len(variants)


@pytest.mark.parametrize("kwargs", [
    {"num_replicates": 0},
    {"num_replicates": 4},
    {"num_replicates": True},
    {"metric_schema_version": "unknown"},
    {"rollout_batch_windows": 0},
    {"max_state_dim": 512},  # P1 remains sealed at its legacy cap
])
def test_invalid_evaluation_protocol_rejected_before_launch(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        market_worker.evaluation_options(**kwargs)
