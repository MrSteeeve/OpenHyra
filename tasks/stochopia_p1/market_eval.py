"""Bind registered development and frozen requests to isolated market evaluation."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from importlib.util import find_spec
from math import isfinite
from pathlib import Path
from time import perf_counter

import fcntl

from sandbox import (
    TRAINING_SANDBOX_DEFAULT_OUTPUT_ENTRIES, TRAINING_SANDBOX_DEFAULT_TOTAL_OUTPUT_BYTES,
    read_regular_file, run_training_sandbox,
)
from stochopia.common import atomic_write_json
from stochopia.neural.lob_training import P1_FIT_PROTOCOL, P1_TRAIN_SEEDS

from .gate import CandidateGate, _runtime_roots
from .market_worker import MODEL_FAILURE_EXIT, NONFINITE_LOSS_EXIT
from .store import (
    CandidateEvaluationFailed, EvaluationRequest, Json, ModelRef, NonfiniteCandidateLoss,
    PairedEvaluation, validate_dates,
)


_RUNTIME_FILES = (
    "stochopia/__init__.py",
    "stochopia/common.py",
    "stochopia/data/__init__.py",
    "stochopia/data/day_cache.py",
    "stochopia/data/szse_l2.py",
    "stochopia/harness/__init__.py",
    "stochopia/harness/lob_rollout.py",
    "stochopia/measure/__init__.py",
    "stochopia/measure/impact.py",
    "stochopia/neural/__init__.py",
    "stochopia/neural/lob_interface.py",
    "stochopia/neural/lob_training.py",
    "stochopia/neural/primitives.py",
    "experiments/p1_m2_baseline.py",
    "experiments/p1_m2_metrics.py",
    "experiments/p1_m3_evidence.py",
    "experiments/p1_m4_market_stats.py",
)
_WEIGHTS_LIMIT = 64 * 1024**2
_PROFILE_LIMIT = 2 * 1024**2
_MEMORY_LIMIT = 4 * 1024**3


class MarketEvaluationError(RuntimeError):
    """The production evaluation could not finish due to a shared or unexpected error."""


# --- Sandboxed worker staging ---

def _stage_runtime(source_dir: Path, model: ModelRef) -> None:
    location = find_spec("stochopia")
    if location is None or location.origin is None:
        raise RuntimeError("The P1 market evaluator requires the Stochopia checkout")
    checkout = Path(location.origin).resolve().parents[1]
    for filename in _RUNTIME_FILES:
        output = source_dir / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkout / filename, output)
    (source_dir / "experiments" / "__init__.py").touch()
    for filename in ("gate_worker.py", "market_worker.py"):
        shutil.copyfile(Path(__file__).with_name(filename), source_dir / filename)
    (source_dir / "candidate.py").write_text(model.source, encoding="utf-8")


def _run_worker(
    work: Path, source_dir: Path, input_dir: Path, mode: str, timeout: float, seed: int | None = None,
    *, candidate: bool = False,
) -> Path:
    output_dir = work / "output"
    scratch = work / "scratch"
    for directory in (output_dir, scratch):
        directory.mkdir(parents=True)
    command = [sys.executable, "-I", str(source_dir / "market_worker.py"),
               mode, str(input_dir), str(output_dir)]
    if seed is not None:
        command.append(str(seed))
    result = run_training_sandbox(
        command, source_dir=source_dir, input_dir=input_dir, output_dir=output_dir,
        tmp_dir=scratch, runtime_roots=_runtime_roots(), timeout_s=timeout,
        memory_bytes=_MEMORY_LIMIT,
    )
    if result["status"] != "ok":
        trace = result["log_tail"].replace(str(work.parents[1]), "<sandbox>")
        detail = f"{mode} worker {result['status']}\n{trace[-2500:]}"
        if candidate and result["status"] == "crash" and result["returncode"] == NONFINITE_LOSS_EXIT:
            raise NonfiniteCandidateLoss(detail)
        expected = result["status"] == "timeout"
        if result["status"] == "crash":
            expected = result["returncode"] == MODEL_FAILURE_EXIT
        if result["status"] == "resource_limit":
            expected = (
                result["peak_memory_bytes"] > _MEMORY_LIMIT
                or result["peak_writable_bytes"] > TRAINING_SANDBOX_DEFAULT_TOTAL_OUTPUT_BYTES
                or result["peak_writable_entries"] > TRAINING_SANDBOX_DEFAULT_OUTPUT_ENTRIES
            )
        if candidate and expected:
            raise CandidateEvaluationFailed(detail)
        raise MarketEvaluationError(detail)
    return output_dir


@contextmanager
def _exclusive(directory: Path) -> Iterator[None]:
    """Serialize duplicate model work for one study and one cache directory."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


# --- Registered market data and model caches ---

class MarketEvaluator:
    """Run the fixed P1 protocol for one symbol from its registered dates.

    Each fit sees only training caches. Evaluation sees the registered split
    and read-only checkpoints. Separate cache keys keep train-only weights
    reusable across phases without reusing development readouts in the verifier.
    """

    def __init__(
        self, market_root: Path, symbol: str, *, evaluation_timeout_s: float = 600.0,
        cache_root: Path | None = None, authorized_test_dates: tuple[int, ...] | None = None,
    ) -> None:
        self.root = market_root.expanduser().resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.train_dates = tuple(manifest["train_dates"])
        self.dev_dates = tuple(manifest["dev_dates"])
        test_dates = tuple(manifest["test_dates"])
        if any(type(date) is not int for date in (*self.train_dates, *self.dev_dates, *test_dates)):
            raise ValueError("registered market dates must be integers")
        validate_dates(self.train_dates, "train_dates")
        validate_dates(self.dev_dates, "dev_dates")
        validate_dates(test_dates, "test_dates")
        if (len(self.train_dates) != 36 or len(self.dev_dates) != 12
                or self.train_dates[-1] >= self.dev_dates[0]
                or len(test_dates) != 12 or self.dev_dates[-1] >= test_dates[0]):
            raise ValueError("market development requires a disjoint registered 36/12 date split")
        if symbol not in manifest["symbols"] or symbol not in ("000001.SZ", "300001.SZ"):
            raise ValueError("symbol must be one of the registered P1 instruments")
        if not isfinite(evaluation_timeout_s) or evaluation_timeout_s <= 0:
            raise ValueError("evaluation timeout must be finite and positive")
        if authorized_test_dates is not None and tuple(authorized_test_dates) != test_dates:
            raise ValueError("frozen test dates must equal the registered test split")
        self.symbol = symbol
        self.authorized_test_dates = authorized_test_dates
        self.evaluation_timeout_s = evaluation_timeout_s
        self.cache_root = None if cache_root is None else cache_root.resolve() / symbol

    def _registered_file(self, relative: Path) -> Path:
        path = self.root / relative
        if path.resolve() != path:
            raise ValueError(f"registered market file cannot redirect through a symlink: {relative}")
        return path

    def _stage_caches(self, output: Path, dates: tuple[int, ...]) -> None:
        for date in dates:
            relative = Path("cache") / str(date) / f"{self.symbol}.npz"
            destination = output / relative
            destination.parent.mkdir(parents=True)
            shutil.copyfile(self._registered_file(relative), destination)

    def _stage_volume(self, output: Path) -> None:
        import pyarrow.parquet as pq

        for date in self.train_dates:
            relative = Path("raw") / str(date) / "行情.parquet"
            destination = output / relative
            destination.parent.mkdir(parents=True)
            table = pq.read_table(
                self._registered_file(relative), columns=["wind_code", "cum_volume"],
                filters=[("wind_code", "==", self.symbol)],
            )
            pq.write_table(table, destination)

    def _fit_into(
        self, work: Path, train: Path, source: Path, weights: Path, *, candidate: bool,
    ) -> list[Json]:
        records = []
        for seed in P1_TRAIN_SEEDS:
            started = perf_counter()
            output = _run_worker(work / f"fit-{seed}", source, train, "fit",
                                 P1_FIT_PROTOCOL.max_train_seconds, seed, candidate=candidate)
            checkpoint = read_regular_file(output / "weights.pt", _WEIGHTS_LIMIT, label="P1 model weights")
            (weights / f"{seed}.pt").write_bytes(checkpoint)
            record = output / "fit.json"
            if record.is_file():
                records.append(json.loads(read_regular_file(record, 4096, label="P1 fit record")))
            else:
                records.append({"seed": seed, "seconds": perf_counter() - started})
        return records

    def _weights(
        self, work: Path, train: Path, source: Path, weights: Path, *, key: str, candidate: bool,
    ) -> list[Json]:
        weights.mkdir()
        if self.cache_root is None:
            return self._fit_into(work, train, source, weights, candidate=candidate)
        cache = self.cache_root / "train" / key
        with _exclusive(cache):
            ready = cache / "fit.json"
            if not ready.exists():
                records = self._fit_into(work, train, source, weights, candidate=candidate)
                for seed in P1_TRAIN_SEEDS:
                    shutil.copyfile(weights / f"{seed}.pt", cache / f"{seed}.pt")
                atomic_write_json(ready, {"fits": records})
                return records
            for seed in P1_TRAIN_SEEDS:
                shutil.copyfile(cache / f"{seed}.pt", weights / f"{seed}.pt")
            return json.loads(ready.read_text(encoding="utf-8"))["fits"]

    def _measure(
        self, model: ModelRef, request: EvaluationRequest, *, key: str, candidate: bool,
    ) -> Json:
        diagnostic = CandidateGate()(model.source)
        if diagnostic is not None:
            raise MarketEvaluationError(f"Model {model.model_id} failed the zero-data gate\n{diagnostic}")

        with tempfile.TemporaryDirectory(prefix="stochopia-p1-market-") as directory:
            root = Path(directory).resolve()
            source = root / "source"
            train = root / "train-input"
            evaluation = root / "evaluation-input"
            _stage_runtime(source, model)
            self._stage_caches(train, self.train_dates)
            atomic_write_json(train / "plan.json", {"symbol": self.symbol, "train_dates": self.train_dates})
            shutil.copytree(train / "cache", evaluation / "cache")
            self._stage_caches(evaluation, request.eval_dates)
            self._stage_volume(evaluation)
            atomic_write_json(evaluation / "plan.json", {
                "symbol": self.symbol, "train_dates": self.train_dates,
                "eval_dates": request.eval_dates, "rollout_seed": request.rollout_seed,
            })
            records = self._weights(root, train, source, evaluation / "weights", key=key, candidate=candidate)
            output = _run_worker(root / "measure", source, evaluation, "evaluate", self.evaluation_timeout_s,
                                 candidate=candidate)
            result = json.loads(read_regular_file(output / "result.json", _PROFILE_LIMIT, label="P1 market profile"))
            result["evidence"]["fit_seconds_by_seed"] = {str(row["seed"]): row["seconds"] for row in records}
            result["evidence"]["evaluation_seconds"] = result.get("evaluation_seconds")
            return result

    def _profile(self, model: ModelRef, request: EvaluationRequest, *, candidate: bool) -> Json:
        # Identical executable source on the same stock and seed panel has
        # the same deterministic fit regardless of its experiment label.
        source_sha256 = sha256(model.source.encode("utf-8")).hexdigest()
        train_seeds = "-".join(map(str, request.train_seeds))
        key = f"{source_sha256}-{train_seeds}"
        if self.cache_root is None:
            return self._measure(model, request, key=key, candidate=candidate)
        cache = self.cache_root / request.split / str(request.rollout_seed) / key
        with _exclusive(cache):
            ready = cache / "profile.json"
            if ready.exists():
                return json.loads(ready.read_text(encoding="utf-8"))
            result = self._measure(model, request, key=key, candidate=candidate)
            atomic_write_json(ready, result)
            return result

    def _check_request(self, request: EvaluationRequest) -> None:
        if request.split == "dev" and self.authorized_test_dates is None:
            expected = self.dev_dates
        elif request.split == "test" and self.authorized_test_dates is not None:
            expected = self.authorized_test_dates
        else:
            raise ValueError("request split disagrees with the registered evaluator phase")
        if (request.train_dates != self.train_dates or request.eval_dates != expected
                or request.train_seeds != P1_TRAIN_SEEDS):
            raise ValueError("request dates and seeds must agree with the P1 registration")

    def seed_profile(self, model: ModelRef, *, rollout_seed: int) -> Json:
        """Measure the common seed on the registered development split."""
        request = EvaluationRequest("seed-dev", "shared", model, model, self.train_dates, self.dev_dates,
                                    P1_TRAIN_SEEDS, rollout_seed, "dev")
        self._check_request(request)
        return self._profile(model, request, candidate=False)

    def __call__(self, request: EvaluationRequest) -> PairedEvaluation:
        self._check_request(request)
        parent = self._profile(request.parent, request, candidate=False)
        if request.parent.source == request.candidate.source:
            candidate = parent
        else:
            candidate = self._profile(request.candidate, request, candidate=True)

        paired = PairedEvaluation(
            {int(day): row for day, row in parent["day_metrics"].items()},
            {int(day): row for day, row in candidate["day_metrics"].items()},
            parent["evidence"], candidate["evidence"],
            parent_market=parent.get("market_bootstrap"), candidate_market=candidate.get("market_bootstrap"),
        )
        paired.validate(request)
        return paired
