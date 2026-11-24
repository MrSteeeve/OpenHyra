"""Exercise the real evaluator adapter and data boundary without market runs."""

from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from stochopia.data import SZSEDaySlice, save_day_slice
from stochopia.neural.lob_training import P1_TRAIN_SEEDS
from stochopia.neural.primitives import SeedGRULOBModel, seed_lob_source
from tasks.stochopia_p1 import market_eval
from tasks.stochopia_p1.no_llm_mutator import mutate
from tasks.stochopia_p1.store import CandidateEvaluationFailed, EvaluationRequest, Json, ModelRef, NonfiniteCandidateLoss


SYMBOL = "000001.SZ"
TRAIN = tuple(range(1, 37))
DEV = tuple(range(38, 50))
TEST = tuple(range(51, 63))


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    (tmp_path / "manifest.json").write_text(json.dumps({
        "train_dates": TRAIN, "dev_dates": DEV, "test_dates": TEST,
        "symbols": [SYMBOL, "300001.SZ"],
    }))
    for day in (*TRAIN, *DEV):
        cache = tmp_path / "cache" / str(day)
        cache.mkdir(parents=True)
        (cache / f"{SYMBOL}.npz").write_bytes(str(day).encode())
        (cache / "300001.SZ.npz").write_bytes(b"different symbol")
        if day in TRAIN:
            quote = tmp_path / "raw" / str(day) / "行情.parquet"
            quote.parent.mkdir(parents=True)
            pq.write_table(pa.table({
                "wind_code": [SYMBOL, "300001.SZ"], "cum_volume": [100, 200],
                "unused_field": [1, 2],
            }), quote)
    return tmp_path


def _request() -> EvaluationRequest:
    return EvaluationRequest(
        "trial", "no_llm", ModelRef("seed", seed_lob_source()),
        ModelRef("trial", mutate(1, family="lru").source),
        TRAIN, DEV, P1_TRAIN_SEEDS, 12_300, "dev",
    )


# --- Dataset and worker boundaries ---

def test_market_adapter_dispatches_the_same_plan_and_only_allowed_inputs(
    archive: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    gated_sources: list[str] = []
    calls: list[tuple[str, str, int | None]] = []
    eval_inputs: list[Path] = []

    def gate(source: str) -> None:
        gated_sources.append(source)

    def worker(
        command: list[str], *, source_dir: Path, input_dir: Path, output_dir: Path,
        tmp_dir: Path, runtime_roots: tuple[Path, ...], timeout_s: float, memory_bytes: int,
    ) -> Json:
        mode = command[3]
        source = (source_dir / "candidate.py").read_text()
        role = "parent" if source == request.parent.source else "candidate"
        model = request.parent if role == "parent" else request.candidate
        assert gated_sources[-1] == model.source
        assert (source_dir / "candidate.py").read_text() == model.source
        assert (source_dir / "experiments" / "p1_m2_baseline.py").is_file()
        assert len(list(source_dir.rglob("*.npz"))) == 0
        assert not any(tmp_dir.iterdir()) and runtime_roots and memory_bytes > 0
        plan = json.loads((input_dir / "plan.json").read_bytes())
        assert plan["symbol"] == SYMBOL
        assert plan["train_dates"] == list(TRAIN)
        expected_dates = TRAIN if mode == "fit" else (*TRAIN, *DEV)
        npz = list(input_dir.rglob("*.npz"))
        assert {path.name for path in npz} == {f"{SYMBOL}.npz"}
        assert {int(path.parent.name) for path in npz} == set(expected_dates)
        if mode == "fit":
            seed = int(command[-1])
            calls.append((role, mode, seed))
            assert timeout_s == 60
            assert not (input_dir / "raw").exists()
            assert not (input_dir / "weights").exists()
            assert set(plan) == {"symbol", "train_dates"}
            (output_dir / "weights.pt").write_bytes(f"{role}-{seed}".encode())
        else:
            calls.append((role, mode, None))
            eval_inputs.append(input_dir)
            assert mode == "evaluate" and timeout_s == 700
            assert input_dir.parent == source_dir.parent
            assert plan["rollout_seed"] == request.rollout_seed
            assert plan["eval_dates"] == list(DEV)
            quotes = list((input_dir / "raw").rglob("*.parquet"))
            assert {int(path.parent.name) for path in quotes} == set(TRAIN)
            table = pq.read_table(quotes[0])
            assert table.to_pydict() == {"wind_code": [SYMBOL], "cum_volume": [100]}
            for seed in P1_TRAIN_SEEDS:
                assert (input_dir / "weights" / f"{seed}.pt").read_bytes() == f"{role}-{seed}".encode()
            nll = 2.0 if role == "parent" else 1.0
            (output_dir / "result.json").write_text(json.dumps({
                "day_metrics": {day: {"val_nll_total": nll, "lob_impact_score": None} for day in DEV},
                "evidence": {"lob_impact_score": 0.5},
            }))
        return {"status": "ok"}

    monkeypatch.setattr(market_eval, "CandidateGate", lambda: gate)
    monkeypatch.setattr(market_eval, "run_training_sandbox", worker)
    evaluate = market_eval.MarketEvaluator(archive, SYMBOL, evaluation_timeout_s=700)
    result = evaluate(request)

    expected: list[tuple[str, str, int | None]] = []
    for role in ("parent", "candidate"):
        expected.extend((role, "fit", seed) for seed in P1_TRAIN_SEEDS)
        expected.append((role, "evaluate", None))
    assert calls == expected
    assert gated_sources == [request.parent.source, request.candidate.source]
    assert len(set(eval_inputs)) == 2
    assert result.parent_rows[DEV[0]] == {"val_nll_total": 2.0, "lob_impact_score": None}
    assert result.candidate_rows[DEV[0]] == {"val_nll_total": 1.0, "lob_impact_score": None}
    assert result.candidate_evidence["lob_impact_score"] == 0.5


def test_market_adapter_rejects_unregistered_dates_and_seeds_before_gate(
    archive: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_gate() -> None:
        pytest.fail("unregistered request attempted to load the model")

    monkeypatch.setattr(market_eval, "CandidateGate", unexpected_gate)
    evaluate = market_eval.MarketEvaluator(archive, SYMBOL)
    request = _request()
    with pytest.raises(ValueError, match="registered evaluator phase"):
        evaluate(replace(request, split="test", eval_dates=TEST))
    with pytest.raises(ValueError, match="must agree with the P1 registration"):
        evaluate(replace(request, eval_dates=TEST))
    with pytest.raises(ValueError, match="must agree with the P1 registration"):
        evaluate(replace(request, train_seeds=(137, 211, 103)))


@pytest.mark.parametrize(("candidate", "outcome", "error"), [
    (True, {"status": "timeout"}, CandidateEvaluationFailed),
    (True, {"status": "crash", "returncode": 65}, CandidateEvaluationFailed),
    (True, {"status": "crash", "returncode": 66}, NonfiniteCandidateLoss),
    (True, {"status": "crash", "returncode": None}, market_eval.MarketEvaluationError),
    (False, {"status": "timeout"}, market_eval.MarketEvaluationError),
])
def test_only_known_candidate_failures_are_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: bool, outcome: Json,
    error: type[RuntimeError],
) -> None:
    monkeypatch.setattr(market_eval, "run_training_sandbox", lambda *args, **kwargs: {
        **outcome, "log_tail": "worker failed",
    })
    with pytest.raises(error, match="fit worker"):
        market_eval._run_worker(tmp_path / "fit", tmp_path, tmp_path, "fit", 60,
                                P1_TRAIN_SEEDS[0], candidate=candidate)


def test_cached_models_reuse_each_profile_only_within_the_registered_split(
    archive: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def measure(
        self: market_eval.MarketEvaluator, model: ModelRef, request: EvaluationRequest, *,
        key: str, candidate: bool,
    ) -> Json:
        calls.append((request.split, model.model_id))
        return {"evidence": {"val_nll_total": 4.5},
                "day_metrics": {str(date): {} for date in request.eval_dates},
                "market_bootstrap": {"split": request.split}}

    monkeypatch.setattr(market_eval.MarketEvaluator, "_measure", measure)
    cache = archive / "run-cache"
    request = _request()
    dev = market_eval.MarketEvaluator(archive, SYMBOL, cache_root=cache)
    dev.seed_profile(request.parent, rollout_seed=request.rollout_seed)
    assert dev(request).candidate_market == {"split": "dev"}
    assert dev(request).parent_market == {"split": "dev"}
    with pytest.raises(ValueError, match="frozen test dates"):
        market_eval.MarketEvaluator(archive, SYMBOL, authorized_test_dates=DEV)
    test = market_eval.MarketEvaluator(archive, SYMBOL, cache_root=cache, authorized_test_dates=TEST)
    with pytest.raises(ValueError, match="registered evaluator phase"):
        test(request)
    holdout = replace(request, split="test", eval_dates=TEST)
    assert test(holdout).candidate_market == {"split": "test"}
    assert test(holdout).parent_market == {"split": "test"}
    assert calls == [("dev", "seed"), ("dev", "trial"), ("test", "seed"), ("test", "trial")]


# --- Native worker smoke ---

def test_native_worker_fits_one_minibatch_from_a_synthetic_npz(tmp_path: Path) -> None:
    # Only the test's staged copy uses a reduced fit budget. The native entry,
    # import staging, NPZ loading, encoding, optimizer, and export stay real.
    source = tmp_path / "source"
    market_eval._stage_runtime(source, _request().parent)
    training = source / "stochopia/neural/lob_training.py"
    with training.open("a", encoding="utf-8") as handle:
        handle.write("""

from dataclasses import replace

full_fit = fit_lob_world_model


def fit_lob_world_model(factory, days, seed):
    return full_fit(factory, days, seed, protocol=replace(
        P1_FIT_PROTOCOL, subseq_len=4, burn_in=1, batch_size=1, steps=1,
    ))
""")
    events = np.arange(8, dtype=np.int64)
    book_ticks = np.zeros((len(events), 2, 10), dtype=np.int64)
    book_ticks[:, 0, 0] = 99
    book_ticks[:, 1, 0] = 101
    book_shares = np.where(book_ticks > 0, 100, 0)
    day = SZSEDaySlice(
        wind_code=SYMBOL, trading_date=1, prev_close_ticks=100, session_bounds=((0, len(events)),),
        time_ms=34_200_000 + events * 10, seq_id=events, dt_ms=np.full(len(events), 10),
        action_type=np.zeros_like(events), rel_ticks=np.full(len(events), -1),
        volume_shares=np.full(len(events), 100), cancel_age_ms=np.full(len(events), -1),
        pre_book_ticks=book_ticks, pre_book_shares=book_shares, mid_ticks_x2=np.full(len(events), 200),
        spread_ticks=np.full(len(events), 2),
    )
    inputs = tmp_path / "train"
    save_day_slice(inputs / "cache" / "1" / f"{SYMBOL}.npz", day)
    (inputs / "plan.json").write_text(json.dumps({"symbol": SYMBOL, "train_dates": [1]}))

    output = market_eval._run_worker(tmp_path / "fit", source, inputs, "fit", 60, P1_TRAIN_SEEDS[0])

    weights = market_eval.read_regular_file(output / "weights.pt", 1024 * 1024)
    state = torch.load(BytesIO(weights), map_location="cpu", weights_only=True)
    SeedGRULOBModel().load_state_dict(state)
    assert all(torch.isfinite(weight).all().item() for weight in state.values())
