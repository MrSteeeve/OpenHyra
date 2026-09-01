"""Experimental per-instance training bridge for Bermudan MLP policies.

This module is intentionally additive: the default task and evaluator do not
call it.  A trusted caller gives one validated contract and only its training
paths.  The candidate process never receives an evaluation request or any
pricing, outer, or inner audit paths.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sandbox import run_training_sandbox
from tasks.bermudan_optimal_stopping.evaluator import BSInstance, discounted_rewards
from tasks.bermudan_optimal_stopping.policy_artifact import (
    MLPContinuationRunner,
    load_policy_artifact,
    load_policy_manifest,
)


TRAINING_INSTANCE_SCHEMA = "openhyra-bermudan-training-instance.v1"
TRAINING_INPUT_BUNDLE_SCHEMA = "openhyra-bermudan-training-inputs.v1"
TRAINING_INPUT_FILES = frozenset({
    "training_paths.npy",
    "payoffs.npy",
    "discount_factors.npy",
    "instance.json",
})
MAX_TRAINING_INPUT_BYTES = 256 * 1024 * 1024
MAX_TRAINING_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class TrainingInputBundle:
    """Hashes of the exact trusted bytes exposed to one training process."""

    schema: str
    file_sha256: tuple[tuple[str, str], ...]
    bundle_sha256: str
    total_bytes: int


@dataclass(frozen=True)
class TrainingCellResult:
    """One fail-closed sandbox outcome and its trusted policy, if any."""

    status: str
    returncode: int | None
    isolation: str
    log_tail: str
    wall_seconds: float
    peak_memory_bytes: int
    output_entries: int
    output_bytes: int
    train_seed: int
    input_file_sha256: tuple[tuple[str, str], ...]
    input_bundle_sha256: str
    policy_file_sha256: tuple[tuple[str, str], ...] | None
    policy_artifact_sha256: str | None
    runner: MLPContinuationRunner | None


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("training instance must be canonical-JSON serializable") from exc


def _bundle_sha256(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(TRAINING_INPUT_BUNDLE_SCHEMA.encode("utf-8") + b"\0")
    for name in sorted(files):
        name_bytes = name.encode("utf-8")
        data = files[name]
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value, dtype=np.float64), allow_pickle=False)
    return stream.getvalue()


def _strict_seed(seed: Any) -> int:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= MAX_TRAINING_SEED
    ):
        raise ValueError("train_seed must be a non-negative 63-bit integer")
    return seed


def _validated_instance(instance: Any) -> BSInstance:
    """Reconstruct the evaluator type instead of trusting a lookalike object."""
    try:
        validated = BSInstance(
            instance_id=instance.instance_id,
            payoff_type=instance.payoff_type,
            spots=tuple(instance.spots),
            strike=instance.strike,
            rate=instance.rate,
            dividends=tuple(instance.dividends),
            volatilities=tuple(instance.volatilities),
            correlation=tuple(tuple(row) for row in instance.correlation),
            maturity=instance.maturity,
            exercise_times=tuple(instance.exercise_times),
            weights=None if instance.weights is None else tuple(instance.weights),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("instance must be a valid Bermudan BSInstance") from exc
    if (
        not isinstance(validated.instance_id, str)
        or not validated.instance_id
        or len(validated.instance_id) > 128
        or any(not (character.isalnum() or character in "_.-")
               for character in validated.instance_id)
    ):
        raise ValueError("instance_id must be bounded path-independent text")
    return validated


def training_instance_payload(instance: Any) -> dict[str, Any]:
    """Return the exact candidate-visible contract schema."""
    value = _validated_instance(instance)
    return {
        "schema": TRAINING_INSTANCE_SCHEMA,
        "payoff_type": value.payoff_type,
        "dimension": value.dimension,
        "spots": [float(item) for item in value.spots],
        "strike": float(value.strike),
        "rate": float(value.rate),
        "dividends": [float(item) for item in value.dividends],
        "volatilities": [float(item) for item in value.volatilities],
        "correlation": [
            [float(item) for item in row] for row in value.correlation
        ],
        "maturity": float(value.maturity),
        "exercise_times": [float(item) for item in value.exercise_times],
        "weights": (
            None if value.weights is None
            else [float(item) for item in value.weights]
        ),
    }


def _validated_training_paths(
    training_paths: Any, instance: BSInstance,
) -> np.ndarray:
    raw = np.asarray(training_paths)
    if (
        not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError("training_paths must contain real numeric values")
    paths = np.ascontiguousarray(raw, dtype=np.float64)
    expected_tail = (len(instance.exercise_times), instance.dimension)
    if paths.ndim != 3 or paths.shape[0] < 1 or paths.shape[1:] != expected_tail:
        raise ValueError(
            "training_paths must have shape "
            f"(n_paths, {expected_tail[0]}, {expected_tail[1]})"
        )
    if paths.nbytes > MAX_TRAINING_INPUT_BYTES:
        raise ValueError("training_paths exceeds the trusted input byte limit")
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0.0):
        raise ValueError("training_paths must contain finite positive asset states")
    return paths


def _empty_real_directory(path: Path, label: str) -> Path:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a real directory")
    if any(path.iterdir()):
        raise ValueError(f"{label} must be empty")
    return path.resolve(strict=True)


def write_training_input_bundle(
    instance: Any,
    training_paths: Any,
    input_dir: str | os.PathLike[str],
) -> TrainingInputBundle:
    """Write the only four files a candidate may see for one training cell."""
    validated_instance = _validated_instance(instance)
    paths = _validated_training_paths(training_paths, validated_instance)
    discounts = np.ascontiguousarray(
        np.exp(
            -validated_instance.rate
            * np.asarray(validated_instance.exercise_times, dtype=np.float64)
        ),
        dtype=np.float64,
    )
    payoffs = np.ascontiguousarray(
        discounted_rewards(paths, validated_instance), dtype=np.float64,
    )
    files = {
        "training_paths.npy": _npy_bytes(paths),
        "payoffs.npy": _npy_bytes(payoffs),
        "discount_factors.npy": _npy_bytes(discounts),
        "instance.json": _canonical_json_bytes(
            training_instance_payload(validated_instance)
        ),
    }
    total = sum(len(data) for data in files.values())
    if total > MAX_TRAINING_INPUT_BYTES:
        raise ValueError("training input bundle exceeds the trusted byte limit")

    root = _empty_real_directory(Path(input_dir), "input_dir")
    for name in sorted(files):
        target = root / name
        target.write_bytes(files[name])
        target.chmod(0o400)
    root.chmod(0o500)
    if {entry.name for entry in os.scandir(root)} != TRAINING_INPUT_FILES:
        raise RuntimeError("trusted training input file set is incomplete")
    hashes = tuple(
        (name, hashlib.sha256(files[name]).hexdigest()) for name in sorted(files)
    )
    return TrainingInputBundle(
        schema=TRAINING_INPUT_BUNDLE_SCHEMA,
        file_sha256=hashes,
        bundle_sha256=_bundle_sha256(files),
        total_bytes=total,
    )


def _create_cell_directories(
    cell_dir: str | os.PathLike[str],
) -> tuple[Path, Path, Path, Path]:
    root = Path(cell_dir)
    if root.exists() or root.is_symlink():
        raise FileExistsError("training cell directory must be fresh")
    parent = root.parent
    try:
        parent_info = os.lstat(parent)
    except FileNotFoundError as exc:
        raise ValueError("training cell parent directory must already exist") from exc
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise ValueError("training cell parent must be a real directory")
    root.mkdir(mode=0o700)
    input_dir = root / "input"
    output_dir = root / "output"
    tmp_dir = root / "tmp"
    for directory in (input_dir, output_dir, tmp_dir):
        directory.mkdir(mode=0o700)
    return root.resolve(strict=True), input_dir, output_dir, tmp_dir


def run_per_instance_training(
    *,
    instance: Any,
    training_paths: Any,
    candidate_source_dir: str | os.PathLike[str],
    cell_dir: str | os.PathLike[str],
    train_seed: int,
    runtime_roots: Iterable[str | os.PathLike[str]],
    timeout_s: float = 60,
    cpu_seconds: float | None = None,
    memory_bytes: int = 1024 * 1024 * 1024,
    file_size_bytes: int = 64 * 1024 * 1024,
    externally_isolated: bool = False,
    cancel_event: Any = None,
    python_executable: str | os.PathLike[str] = sys.executable,
) -> TrainingCellResult:
    """Train and trusted-load one fresh per-instance policy cell.

    There is deliberately no evaluation-request or evaluation-path argument.
    Only the sealed training bundle is mounted as candidate input.
    """
    seed = _strict_seed(train_seed)
    validated_instance = _validated_instance(instance)
    source = Path(candidate_source_dir)
    manifest = load_policy_manifest(source / "manifest.json")
    _root, input_dir, output_dir, tmp_dir = _create_cell_directories(cell_dir)
    input_bundle = write_training_input_bundle(
        validated_instance, training_paths, input_dir,
    )
    command = [
        os.fspath(python_executable),
        "train.py",
        "--input",
        str(input_dir.resolve(strict=True)),
        "--output",
        str(output_dir.resolve(strict=True)),
        "--seed",
        str(seed),
    ]
    sandbox_result = run_training_sandbox(
        command,
        source_dir=source,
        input_dir=input_dir,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        runtime_roots=runtime_roots,
        timeout_s=timeout_s,
        cpu_seconds=cpu_seconds,
        memory_bytes=memory_bytes,
        file_size_bytes=file_size_bytes,
        externally_isolated=externally_isolated,
        cancel_event=cancel_event,
    )
    common = {
        "returncode": sandbox_result["returncode"],
        "isolation": sandbox_result["isolation"],
        "log_tail": sandbox_result["log_tail"],
        "wall_seconds": float(sandbox_result["wall_seconds"]),
        "peak_memory_bytes": int(sandbox_result["peak_memory_bytes"]),
        "output_entries": int(sandbox_result["output_entries"]),
        "output_bytes": int(sandbox_result["output_bytes"]),
        "train_seed": seed,
        "input_file_sha256": input_bundle.file_sha256,
        "input_bundle_sha256": input_bundle.bundle_sha256,
    }
    if sandbox_result["status"] != "ok":
        return TrainingCellResult(
            status=sandbox_result["status"],
            policy_file_sha256=None,
            policy_artifact_sha256=None,
            runner=None,
            **common,
        )

    try:
        artifact = load_policy_artifact(
            manifest,
            output_dir,
            n_exercise_times=len(validated_instance.exercise_times),
            input_dim=validated_instance.dimension,
        )
    except ValueError as exc:
        note = f"trusted policy artifact rejected: {exc}"
        log_tail = (common["log_tail"] + "\n" + note).strip()
        return TrainingCellResult(
            status="invalid_artifact",
            policy_file_sha256=None,
            policy_artifact_sha256=None,
            runner=None,
            **{**common, "log_tail": log_tail},
        )
    runner = MLPContinuationRunner(artifact)
    return TrainingCellResult(
        status="ok",
        policy_file_sha256=artifact.file_sha256,
        policy_artifact_sha256=artifact.bundle_sha256,
        runner=runner,
        **common,
    )
