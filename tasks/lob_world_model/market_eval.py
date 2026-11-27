"""Bind registered development and frozen requests to isolated market evaluation."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from math import isfinite
from pathlib import Path
from time import perf_counter

import fcntl

from sandbox import run_worker_subprocess
from stochopia.common import atomic_write_json
from stochopia.measure.impact import LEGACY_METRIC_SCHEMA
from stochopia.neural.lob_training import FitProtocol, DEFAULT_FIT_PROTOCOL, DEFAULT_TRAIN_SEEDS

from .gate import CandidateGate, _runtime_pythonpath
from .market_worker import (
    MODEL_FAILURE_EXIT, NONFINITE_LOSS_EXIT, device_cache_identity, evaluation_options, normalize_device,
)
from .store import (
    CandidateEvaluationFailed, EvaluationRequest, Json, ModelRef, NonfiniteCandidateLoss,
    PairedEvaluation, validate_dates,
)


_WORKER_SCRIPT = Path(__file__).resolve().with_name("market_worker.py")
# Only the exact frozen legacy fit parameters produced pre-device CPU caches.
_LEGACY_CPU_FIT = FitProtocol(
    subseq_len=256, burn_in=64, batch_size=16, steps=1_200, start_lr=0.002,
    end_lr=0.0001, weight_decay=0.0001, grad_clip=1.0, max_train_seconds=180.0,
)


class MarketEvaluationError(RuntimeError):
    """The production evaluation could not finish due to a shared or unexpected error."""


# --- Per-run inputs and worker execution ---

def _stage_candidate(source_dir: Path, model: ModelRef) -> None:
    """Write candidate source once; workers import the shared research runtime."""
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "candidate.py").write_text(model.source, encoding="utf-8")


def _run_worker(
    work: Path, source_dir: Path, input_dir: Path, mode: str, timeout: float, seed: int | None = None,
    *, candidate: bool = False, device: str = "cpu", device_id: str | None = None,
    num_replicates: int = 1, metric_schema_version: str = LEGACY_METRIC_SCHEMA, rollout_batch_windows: int = 2,
    max_state_dim: int | None = None,
) -> Path:
    output_dir = work / "output"
    output_dir.mkdir(parents=True)
    command = [
        sys.executable, "-s", "-B", str(_WORKER_SCRIPT), mode,
        str(source_dir / "candidate.py"), str(input_dir), str(output_dir),
        "--device", device, "--device-id", device_id or device_cache_identity(device),
    ]
    options = evaluation_options(
        num_replicates=num_replicates, metric_schema_version=metric_schema_version,
        rollout_batch_windows=rollout_batch_windows, max_state_dim=max_state_dim,
    )
    if metric_schema_version != LEGACY_METRIC_SCHEMA:
        command.extend([
            "--num-replicates", str(options["num_replicates"]),
            "--metric-schema-version", str(options["metric_schema_version"]),
            "--rollout-batch-windows", str(options["rollout_batch_windows"]),
            "--max-state-dim", str(options["max_state_dim"]),
        ])
    if seed is not None:
        command.append(str(seed))
    result = run_worker_subprocess(
        command, work_dir=work, timeout_s=timeout,
        extra_env={"PYTHONPATH": _runtime_pythonpath()},
    )
    if result["returncode"] != 0 or result["timed_out"] or result.get("cancelled"):
        status = "timeout" if result["timed_out"] else "crash"
        if result.get("cancelled"):
            status = "cancelled"
        trace = (result["stdout"] + "\n" + result["stderr"]).replace(str(work.parents[1]), "<worker>")
        detail = f"{mode} worker {status}\n{trace[-2500:]}"
        if candidate and status == "crash" and result["returncode"] == NONFINITE_LOSS_EXIT:
            raise NonfiniteCandidateLoss(detail)
        expected = status == "timeout" or (status == "crash" and result["returncode"] == MODEL_FAILURE_EXIT)
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
    """Evaluate a registered market protocol for one symbol from its registered dates.

    Each fit sees only training caches. Evaluation sees the registered split
    and read-only checkpoints. Separate cache keys keep train-only weights
    reusable across phases without reusing development readouts in the verifier.
    Worker isolation from the host filesystem/network is a deployment concern;
    separate work directories and registered inputs are the experiment boundary.
    """

    def __init__(
        self, market_root: Path, symbol: str, *, evaluation_timeout_s: float = 1_800.0,
        cache_root: Path | None = None, authorized_test_dates: tuple[int, ...] | None = None,
        expected_dev_days: int = 12, expected_test_days: int = 12,
        device: str = "cpu", num_replicates: int = 1, metric_schema_version: str = LEGACY_METRIC_SCHEMA,
        rollout_batch_windows: int = 2, max_state_dim: int | None = None,
    ) -> None:
        if (expected_dev_days, expected_test_days) not in ((12, 12), (24, 20)):
            raise ValueError("expected a registered 12/12 or 24/20 development/test split")
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
        if (len(self.train_dates) != 36 or len(self.dev_dates) != expected_dev_days
                or self.train_dates[-1] >= self.dev_dates[0]
                or len(test_dates) != expected_test_days or self.dev_dates[-1] >= test_dates[0]):
            raise ValueError(
                f"market development requires a disjoint registered 36/{expected_dev_days}/{expected_test_days} date split"
            )
        replacement_dates: tuple[int, ...] = ()
        if expected_test_days == 20:
            replacement_dates = tuple(manifest.get("replacement_test_dates", ()))
            if any(type(date) is not int for date in replacement_dates):
                raise ValueError("registered replacement test dates must be integers")
            validate_dates(replacement_dates, "replacement_test_dates")
            if len(replacement_dates) != 10 or test_dates[-1] >= replacement_dates[0]:
                raise ValueError("replacement test dates must be ten subsequent registered dates")
        if symbol not in manifest["symbols"] or symbol not in ("000001.SZ", "300001.SZ"):
            raise ValueError("symbol must be one of the registered instruments")
        if not isfinite(evaluation_timeout_s) or evaluation_timeout_s <= 0:
            raise ValueError("evaluation timeout must be finite and positive")
        if authorized_test_dates is not None:
            authorized_test_dates = tuple(authorized_test_dates)
            if (len(authorized_test_dates) != expected_test_days
                    or any(type(date) is not int for date in authorized_test_dates)
                    or not set(authorized_test_dates).issubset((*test_dates, *replacement_dates))):
                raise ValueError("frozen test dates must equal the registered test split or its authorized replacements")
            validate_dates(authorized_test_dates, "authorized_test_dates")
        self.symbol = symbol
        self.authorized_test_dates = authorized_test_dates
        self.evaluation_timeout_s = evaluation_timeout_s
        self.cache_root = None if cache_root is None else cache_root.resolve() / symbol
        self.device = normalize_device(device)
        self.device_id = device_cache_identity(self.device)
        self.options = evaluation_options(
            num_replicates=num_replicates, metric_schema_version=metric_schema_version,
            rollout_batch_windows=rollout_batch_windows, max_state_dim=max_state_dim,
        )
        self.metric_schema_version = metric_schema_version
        self.num_replicates = num_replicates

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
        for seed in DEFAULT_TRAIN_SEEDS:
            started = perf_counter()
            output = _run_worker(work / f"fit-{seed}", source, train, "fit",
                                 DEFAULT_FIT_PROTOCOL.max_train_seconds, seed, candidate=candidate,
                                 device=self.device, device_id=self.device_id)
            shutil.copyfile(output / "weights.pt", weights / f"{seed}.pt")
            record = output / "fit.json"
            if record.is_file():
                records.append(json.loads(record.read_text(encoding="utf-8")))
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
                legacy_key = self._legacy_cpu_key(key)
                legacy = self.cache_root / "train" / legacy_key if legacy_key is not None else None
                if legacy is not None and (legacy / "fit.json").is_file():
                    # The old ready marker was committed after all CPU fit
                    # checkpoints. Never write to the old legacy key.
                    records = json.loads((legacy / "fit.json").read_text(encoding="utf-8"))["fits"]
                    for seed in DEFAULT_TRAIN_SEEDS:
                        shutil.copyfile(legacy / f"{seed}.pt", weights / f"{seed}.pt")
                else:
                    records = self._fit_into(work, train, source, weights, candidate=candidate)
                for seed in DEFAULT_TRAIN_SEEDS:
                    shutil.copyfile(weights / f"{seed}.pt", cache / f"{seed}.pt")
                atomic_write_json(ready, {"fits": records})
                return records
            for seed in DEFAULT_TRAIN_SEEDS:
                shutil.copyfile(cache / f"{seed}.pt", weights / f"{seed}.pt")
            return json.loads(ready.read_text(encoding="utf-8"))["fits"]

    def _measure(
        self, model: ModelRef, request: EvaluationRequest, *, key: str, candidate: bool,
    ) -> Json:
        gate = CandidateGate() if self.metric_schema_version == LEGACY_METRIC_SCHEMA else CandidateGate(
            max_state_dim=int(self.options["max_state_dim"]),
        )
        diagnostic = gate(model.source)
        if diagnostic is not None:
            raise MarketEvaluationError(f"Model {model.model_id} failed the zero-data gate\n{diagnostic}")

        with tempfile.TemporaryDirectory(prefix="lob-world-model-market-") as directory:
            root = Path(directory).resolve()
            source = root / "source"
            train = root / "train-input"
            evaluation = root / "evaluation-input"
            _stage_candidate(source, model)
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
                                 candidate=candidate, device=self.device, device_id=self.device_id,
                                 **self.options)
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            result["evidence"]["fit_seconds_by_seed"] = {str(row["seed"]): row["seconds"] for row in records}
            result["evidence"]["evaluation_seconds"] = result.get("evaluation_seconds")
            return result

    def _profile(self, model: ModelRef, request: EvaluationRequest, *, candidate: bool) -> Json:
        # Identical executable source on the same stock and seed panel has
        # the same deterministic fit regardless of its experiment label.
        source_sha256 = sha256(model.source.encode("utf-8")).hexdigest()
        train_seeds = "-".join(map(str, request.train_seeds))
        key = f"{source_sha256}-{train_seeds}-device-{self.device_id}"
        if self.cache_root is None:
            return self._measure(model, request, key=key, candidate=candidate)
        profile_key = self._evaluation_key(key)
        cache = self._profile_cache(request, profile_key)
        with _exclusive(cache):
            ready = cache / "profile.json"
            if ready.exists():
                return json.loads(ready.read_text(encoding="utf-8"))
            legacy_key = self._legacy_cpu_key(key) if self.metric_schema_version == LEGACY_METRIC_SCHEMA else None
            legacy = self._profile_cache(request, legacy_key) if legacy_key is not None else None
            if legacy is not None and (legacy / "profile.json").is_file():
                result = json.loads((legacy / "profile.json").read_text(encoding="utf-8"))
            else:
                result = self._measure(model, request, key=key, candidate=candidate)
            atomic_write_json(ready, result)
            return result

    def _evaluation_key(self, train_key: str) -> str:
        """Separate evaluation schemas/K while allowing identical CPU fits to be reused.

        Include batch width and state cap so a changed numerical protocol or
        admissibility check cannot accidentally return an old evaluation.
        """
        if self.metric_schema_version == LEGACY_METRIC_SCHEMA:
            return train_key
        return (f"{train_key}-schema-{self.metric_schema_version}-rollout-v2"
                f"-k{self.num_replicates}-batch{self.options['rollout_batch_windows']}"
                f"-cap{self.options['max_state_dim']}")

    def _legacy_cpu_key(self, key: str) -> str | None:
        """Permit read-only migration of the known frozen CPU protocol."""
        if self.device != "cpu" or DEFAULT_FIT_PROTOCOL != _LEGACY_CPU_FIT or not key.endswith("-device-cpu"):
            return None
        return key.removesuffix("-device-cpu")

    def _profile_cache(self, request: EvaluationRequest, key: str) -> Path:
        if self.cache_root is None:
            raise ValueError("profile cache requires a cache root")
        cache = self.cache_root / request.split / str(request.rollout_seed) / key
        if len(self.dev_dates) == 24:
            # A registered holdout replacement can change this panel without
            # changing the fitted weights, candidate source or rollout seed.
            cache /= "-".join(map(str, request.eval_dates))
        return cache

    def _check_request(self, request: EvaluationRequest) -> None:
        if request.split == "dev" and self.authorized_test_dates is None:
            expected = self.dev_dates
        elif request.split == "test" and self.authorized_test_dates is not None:
            expected = self.authorized_test_dates
        else:
            raise ValueError("request split disagrees with the registered evaluator phase")
        if (request.train_dates != self.train_dates or request.eval_dates != expected
                or request.train_seeds != DEFAULT_TRAIN_SEEDS):
            raise ValueError("request dates and seeds must agree with the market registration")

    def seed_profile(self, model: ModelRef, *, rollout_seed: int) -> Json:
        """Measure the common seed on the registered development split."""
        request = EvaluationRequest("seed-dev", "shared", model, model, self.train_dates, self.dev_dates,
                                    DEFAULT_TRAIN_SEEDS, rollout_seed, "dev")
        self._check_request(request)
        return self._profile(model, request, candidate=False)

    def verification_profile(
        self, model: ModelRef, request: EvaluationRequest, *, candidate: bool,
    ) -> Json:
        """Measure one frozen test source, reusing only this stock's model cache."""
        self._check_request(request)
        if request.split != "test" or self.cache_root is None:
            raise ValueError("closing verification requires a registered holdout and durable profile cache")
        if model.source not in (request.parent.source, request.candidate.source):
            raise ValueError("verification profile must belong to the frozen pair")
        return self._profile(model, request, candidate=candidate)

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
