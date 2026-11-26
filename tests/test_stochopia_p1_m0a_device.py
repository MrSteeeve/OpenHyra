"""Check explicit device identity and cache compatibility without a GPU."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stochopia.neural.lob_training import P1_TRAIN_SEEDS
from stochopia.neural.primitives import seed_lob_source
from tasks.stochopia_p1 import market_eval, market_worker
from tasks.stochopia_p1.store import EvaluationRequest, Json, ModelRef


SYMBOL = "000001.SZ"
TRAIN = tuple(range(1, 37))
DEV = tuple(range(38, 50))
TEST = tuple(range(51, 63))


def _manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({
        "train_dates": TRAIN, "dev_dates": DEV, "test_dates": TEST,
        "symbols": [SYMBOL, "300001.SZ"],
    }), encoding="utf-8")


def _request() -> EvaluationRequest:
    model = ModelRef("seed", seed_lob_source())
    return EvaluationRequest("trial", "no_llm", model, model, TRAIN, DEV, P1_TRAIN_SEEDS, 12_300, "dev")


def _legacy_key(model: ModelRef) -> str:
    source_digest = sha256(model.source.encode("utf-8")).hexdigest()
    return f"{source_digest}-{'-'.join(map(str, P1_TRAIN_SEEDS))}"


def _mock_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(
        uuid=f"GPU-{index}", name="test", major=8, minor=6, total_memory=1024,
    ))


@pytest.mark.parametrize(("raw", "canonical"), [
    ("cpu", "cpu"), ("CPU:0", "cpu"), (" cuda ", "cuda:0"), ("cuda:1", "cuda:1"),
])
def test_device_aliases_normalize(raw: str, canonical: str) -> None:
    assert market_worker.normalize_device(raw) == canonical


@pytest.mark.parametrize("raw", ["", "cpu:1", "cuda:-1", "meta", "gpu", "../cuda"])
def test_invalid_worker_devices_fail_clearly(raw: str) -> None:
    with pytest.raises(ValueError, match="device"):
        market_worker.normalize_device(raw)


def test_cuda_cache_identity_includes_visible_gpu_and_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cuda(monkeypatch)
    first = market_worker.device_cache_identity("cuda")
    assert first == market_worker.device_cache_identity("cuda:0")
    assert first != market_worker.device_cache_identity("cuda:1")
    assert first != market_worker.device_cache_identity("cpu")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(
        uuid="GPU-different", name="test", major=8, minor=6, total_memory=1024,
    ))
    assert first != market_worker.device_cache_identity("cuda:0")
    with pytest.raises(ValueError, match="identity differs"):
        market_worker.main("fit", Path("candidate.py"), Path("input"), Path("output"), 103,
                           device="cuda:0", device_id=first)
    with pytest.raises(ValueError, match="not available"):
        market_worker.device_cache_identity("cuda:2")


def test_cpu_profile_cache_migration_does_not_reuse_for_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest(tmp_path)
    _mock_cuda(monkeypatch)
    request = _request()
    cache = tmp_path / "run-cache"
    original = cache / SYMBOL / "dev" / str(request.rollout_seed) / _legacy_key(request.parent) / "profile.json"
    original.parent.mkdir(parents=True)
    payload = {"evidence": {"val_nll_total": 5.0}, "day_metrics": {}}
    original.write_text(json.dumps(payload), encoding="utf-8")
    unchanged = original.read_bytes()
    calls: list[str] = []

    def measure(
        self: market_eval.MarketEvaluator, model: ModelRef, request: EvaluationRequest, *,
        key: str, candidate: bool,
    ) -> Json:
        calls.append(key)
        return {"evidence": {"val_nll_total": 6.0}, "day_metrics": {}}

    monkeypatch.setattr(market_eval.MarketEvaluator, "_measure", measure)
    cpu = market_eval.MarketEvaluator(tmp_path, SYMBOL, cache_root=cache)
    assert cpu.seed_profile(request.parent, rollout_seed=request.rollout_seed) == payload
    assert calls == []
    migrated = original.parent.with_name(f"{original.parent.name}-device-cpu") / "profile.json"
    assert json.loads(migrated.read_text(encoding="utf-8")) == payload
    assert original.read_bytes() == unchanged and list(original.parent.iterdir()) == [original]

    gpu = market_eval.MarketEvaluator(tmp_path, SYMBOL, cache_root=cache, device="cuda")
    measured = gpu.seed_profile(request.parent, rollout_seed=request.rollout_seed)
    assert measured["evidence"]["val_nll_total"] == 6.0
    assert gpu.seed_profile(request.parent, rollout_seed=request.rollout_seed) == measured
    assert calls == [f"{_legacy_key(request.parent)}-device-{gpu.device_id}"]
    assert gpu.device == "cuda:0"

    # A changed fit protocol may not adopt the original P1 CPU cache.
    monkeypatch.setattr(market_eval, "P1_FIT_PROTOCOL", replace(market_eval.P1_FIT_PROTOCOL, steps=7))
    assert cpu._legacy_cpu_key(f"{_legacy_key(request.parent)}-device-cpu") is None


def test_fit_cache_separates_device_and_never_writes_legacy_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest(tmp_path)
    _mock_cuda(monkeypatch)
    request = _request()
    cache = tmp_path / "run-cache"
    old_key = _legacy_key(request.parent)
    legacy = cache / SYMBOL / "train" / old_key
    legacy.mkdir(parents=True)
    records = [{"seed": seed, "seconds": 1.0} for seed in P1_TRAIN_SEEDS]
    (legacy / "fit.json").write_text(json.dumps({"fits": records}), encoding="utf-8")
    for seed in P1_TRAIN_SEEDS:
        (legacy / f"{seed}.pt").write_bytes(f"cpu-{seed}".encode())
    before = {path.name: path.read_bytes() for path in legacy.iterdir()}
    calls: list[str] = []

    def fit(
        self: market_eval.MarketEvaluator, work: Path, train: Path, source: Path,
        weights: Path, *, candidate: bool,
    ) -> list[Json]:
        calls.append(self.device)
        for seed in P1_TRAIN_SEEDS:
            (weights / f"{seed}.pt").write_bytes(f"{self.device}-{seed}".encode())
        return records

    monkeypatch.setattr(market_eval.MarketEvaluator, "_fit_into", fit)
    cpu = market_eval.MarketEvaluator(tmp_path, SYMBOL, cache_root=cache, device="cpu:0")
    cpu_out = tmp_path / "cpu"
    cpu._weights(tmp_path, tmp_path, tmp_path, cpu_out, key=f"{old_key}-device-{cpu.device_id}", candidate=True)
    assert calls == []
    assert (cpu_out / f"{P1_TRAIN_SEEDS[0]}.pt").read_bytes() == b"cpu-103"
    assert (legacy.parent / f"{old_key}-device-cpu" / "fit.json").is_file()
    assert {path.name: path.read_bytes() for path in legacy.iterdir()} == before

    gpu = market_eval.MarketEvaluator(tmp_path, SYMBOL, cache_root=cache, device="cuda:0")
    gpu_out = tmp_path / "gpu"
    gpu_key = f"{old_key}-device-{gpu.device_id}"
    gpu._weights(tmp_path, tmp_path, tmp_path, gpu_out, key=gpu_key, candidate=True)
    assert calls == ["cuda:0"]
    assert (gpu_out / f"{P1_TRAIN_SEEDS[0]}.pt").read_bytes() == b"cuda:0-103"
    assert (legacy.parent / gpu_key / "fit.json").is_file()
    assert {path.name: path.read_bytes() for path in legacy.iterdir()} == before
