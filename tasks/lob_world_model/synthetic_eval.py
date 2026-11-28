"""Subprocess bridge for matched open-candidate evaluations on anonymous synthetic worlds."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any

from experiments.synthetic_tau_z.design import SEEDS
from stochopia.common import atomic_write_json
from worker_subprocess import run_worker_subprocess

from .gate import CandidateGate, _runtime_pythonpath
from .market_eval import _exclusive
from .market_worker import MODEL_FAILURE_EXIT, NONFINITE_LOSS_EXIT
from .synthetic_data import DATA_PROTOCOL, ENV_IDS, prepare_development, materialize_holdout
from .synthetic_stats import METRICS, paired_summary

Json = dict[str, Any]
_WORKER = Path(__file__).resolve().with_name("synthetic_worker.py")


class _CandidateFailure(RuntimeError):
    def __init__(self, message: str, *, status: str = "EVALUATION_FAILED") -> None:
        super().__init__(message)
        self.status = status


def _json(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _run(work: Path, mode: str, source: str, files: dict[str, Path], timeout_s: float) -> Path:
    """Only copy authorized phase inputs to the isolated worker directory."""
    work.mkdir()
    candidate = work / "candidate.py"
    candidate.write_text(source, encoding="utf-8")
    inputs = work / "inputs"
    inputs.mkdir()
    for name, original in files.items():
        shutil.copyfile(original, inputs / name)
    output = work / "output"
    result = run_worker_subprocess(
        [sys.executable, "-s", "-B", str(_WORKER), mode, str(candidate), str(inputs), str(output)],
        work_dir=work, timeout_s=timeout_s, extra_env={"PYTHONPATH": _runtime_pythonpath()},
    )
    if result["returncode"] != 0 or result["timed_out"] or result["cancelled"]:
        if result["cancelled"]:
            raise RuntimeError("synthetic worker cancelled")
        trace = (result["stdout"] + "\n" + result["stderr"]).replace(str(work), "<worker>")
        detail = f"{mode} failed: {trace[-1800:]}"
        if result["returncode"] == NONFINITE_LOSS_EXIT and not result["timed_out"]:
            raise _CandidateFailure(detail, status="NONFINITE_LOSS")
        if result["timed_out"] or result["returncode"] == MODEL_FAILURE_EXIT:
            raise _CandidateFailure(detail)
        raise RuntimeError(f"unexpected synthetic worker failure ({result['returncode']}): {detail}")
    return output


class SyntheticEvaluator:
    """Three fixed-seed fits, one shared baseline cache, and a separately authorized holdout."""

    def __init__(self, root: Path, env_id: str, *, timeout_s: float = 420.0) -> None:
        if env_id not in ENV_IDS or timeout_s <= 0:
            raise ValueError("unknown anonymous environment or invalid worker timeout")
        self.root = root.resolve() / env_id
        self.env_id = env_id
        self.timeout_s = timeout_s
        self.gate = CandidateGate(max_state_dim=60)
        self.data = self.root / "dataset"
        self.authorization = self.root / "holdout.authorization.json"

    def prepare_dev(self) -> Json:
        with _exclusive(self.data):
            if self.authorization.exists() and not (self.data / "development.json").exists():
                raise ValueError("cannot create development history after holdout authorization")
            self.development = prepare_development(self.data, self.env_id)
        return self.development["z"]

    def _weights(self, cache: Path, work: Path, source: str) -> Json:
        receipt = cache / "fit.json"
        if not receipt.exists():
            output = _run(work, "fit", source, {"train.pkl": self.data / "train.pkl"}, self.timeout_s)
            fit = _json(output / "fit.json")
            cache.mkdir(exist_ok=True)
            hashes = {}
            for seed in SEEDS:
                name = f"{seed}.pt"
                shutil.copyfile(output / name, cache / name)
                hashes[str(seed)] = _digest(cache / name)
            fit["training_checkpoint_hashes"] = hashes
            atomic_write_json(receipt, fit)
        fit = _json(receipt)
        for seed in SEEDS:
            if _digest(cache / f"{seed}.pt") != fit["training_checkpoint_hashes"][str(seed)]:
                raise ValueError("registered training checkpoint fingerprint mismatch")
        return fit

    def evaluate(self, source: str, *, split: str = "dev") -> Json:
        digest = sha256(source.encode("utf-8")).hexdigest()
        cache = self.root / "models" / digest
        base = {"env_id": self.env_id, "source_sha256": digest, "split": split}
        if split == "dev":
            if self.authorization.exists():
                raise ValueError("development evaluation is closed after the freeze")
            if not hasattr(self, "development"):
                self.prepare_dev()
        elif split == "holdout_test":
            if not self.authorization.exists() or not (self.data / "holdout.json").exists():
                raise ValueError("holdout is not materialized by the frozen verifier")
            if digest not in _json(self.authorization)["approved"]:
                raise ValueError("candidate was not frozen for holdout evaluation")
        else:
            raise ValueError("unknown split")
        path = cache / f"{split}.json"
        with _exclusive(cache):
            if path.exists():
                return _json(path)
            if split == "dev":
                failure = self.gate(source)
                if failure is not None:
                    result = {**base, "status": "GATE_REJECTED", "error": failure}
                    atomic_write_json(path, result)
                    return result
            if split != "dev" and not (cache / "fit.json").exists():
                raise ValueError("holdout must reuse already fitted checkpoints")
            try:
                with TemporaryDirectory(prefix="synthetic-worker-") as directory:
                    work = Path(directory)
                    fit = self._weights(cache, work / "fit", source)
                    files = {f"{seed}.pt": cache / f"{seed}.pt" for seed in SEEDS}
                    files["panel.pkl"] = self.data / f"{split}.pkl"
                    output = _run(work / "measure", "measure", source, files, self.timeout_s)
                    measured = _json(output / "result.json")
                    if (fit["params"], fit["state"]) != (measured["params"], measured["state"]):
                        raise _CandidateFailure("checkpoint model capacity mismatch")
                    result = {**base, **fit, **measured, "status": "VALID"}
            except _CandidateFailure as exc:
                result = {**base, "status": exc.status, "error": str(exc)}
            atomic_write_json(path, result)
            return result

    def open_holdout(self, freeze_path: Path) -> Json:
        """Seal the approved candidate set before any independent holdout generation."""
        freeze_sha256 = _digest(freeze_path)
        approved = _json(freeze_path)["approved"]
        if set(approved) != set(ENV_IDS) or not all(approved.values()):
            raise ValueError("both environments require a frozen digest set")
        for environment, digests in approved.items():
            for digest in digests:
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("expected frozen hexadecimal source digest")
                model = self.root.parent / environment / "models" / digest
                if not (model / "dev.json").exists() or _json(model / "dev.json")["status"] != "VALID":
                    raise ValueError("holdout requires a successful development evaluation in both environments")
        authorization = {"freeze_sha256": freeze_sha256, "approved": sorted(set(approved[self.env_id]))}
        with _exclusive(self.data):
            if self.authorization.exists() and _json(self.authorization) != authorization:
                raise ValueError("holdout approved candidates were changed after the freeze")
            atomic_write_json(self.authorization, authorization)
            return materialize_holdout(self.data, self.env_id, freeze_sha256=freeze_sha256)
