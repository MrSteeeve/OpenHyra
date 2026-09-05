"""Per-instance training bridge for trusted continuation-policy protocols.

The legacy feature task does not call this bridge; executable-program tasks
do. A trusted caller gives one validated contract and only its training paths.
The candidate process never receives an evaluation request or any pricing,
outer, or inner audit paths.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sandbox import read_regular_file, run_training_sandbox
from tasks.bermudan_optimal_stopping.evaluator import (
    BSInstance,
    discounted_rewards,
    payoff,
)
from tasks.bermudan_optimal_stopping.policy_protocols import (
    ContinuationRunner,
    PROTOCOL_OUTPUT_CLIP,
    PythonProgramManifest,
    load_candidate_manifest,
    load_continuation_runner,
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
PROGRAM_PREDICTION_REQUEST_SCHEMA = "openhyra-python-program-predict.v1"
PROGRAM_ENTRYPOINT = "algorithm.py"
DEFAULT_PREDICTION_TIMEOUT_S = 5.0
DEFAULT_PREDICTION_MEMORY_BYTES = 1024 * 1024 * 1024
DEFAULT_PREDICTION_FILE_SIZE_BYTES = 16 * 1024 * 1024


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
    runner: ContinuationRunner | "SandboxedPythonProgramRunner" | None
    research_fallback: bool = False


@dataclass(frozen=True)
class PythonProgramModelArtifact:
    """Opaque candidate model bytes; trusted code never interprets them."""

    manifest: PythonProgramManifest
    files: tuple[tuple[str, bytes], ...]


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


def _load_python_program_model(
    output_dir: Path,
    manifest: PythonProgramManifest,
    *,
    max_bytes: int,
) -> PythonProgramModelArtifact:
    """Keep the candidate's arbitrary model tree as opaque bytes."""
    files: list[tuple[str, bytes]] = []
    total = 0
    pending = [output_dir]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ValueError(f"could not inspect Python program model: {exc}") from exc
        for entry in children:
            path = Path(entry.path)
            relative = path.relative_to(output_dir).as_posix()
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(
                    f"Python program model entry {relative} must not be a symbolic link"
                )
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(
                    f"Python program model entry {relative} must be a regular file"
                )
            remaining = max_bytes - total
            data = read_regular_file(
                path,
                max(0, remaining),
                label=f"Python program model file {relative}",
            )
            total += len(data)
            if total > max_bytes:
                raise ValueError("Python program model exceeds the byte limit")
            files.append((relative, data))
    if not files:
        raise ValueError("Python program fit produced no model files")
    return PythonProgramModelArtifact(
        manifest=manifest,
        files=tuple(sorted(files)),
    )


def _python_model_artifact_sha256(artifact: PythonProgramModelArtifact) -> str:
    """Hash the opaque model tree with a stable manifest/framing envelope."""
    digest = hashlib.sha256()
    manifest_payload = {
        "schema": artifact.manifest.schema,
        "interface": artifact.manifest.interface,
    }
    manifest_bytes = _canonical_json_bytes(manifest_payload)
    digest.update(b"openhyra-python-model.v1\0")
    digest.update(len(manifest_bytes).to_bytes(4, "big"))
    digest.update(manifest_bytes)
    for name, data in artifact.files:
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _seal_readonly_tree(root: Path) -> None:
    directories = [root]
    for current, child_directories, filenames in os.walk(root):
        current_path = Path(current)
        directories.extend(current_path / name for name in child_directories)
        for name in filenames:
            (current_path / name).chmod(0o400)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        directory.chmod(0o500)


def _write_python_prediction_input(
    input_dir: Path,
    artifact: PythonProgramModelArtifact,
    *,
    instance: BSInstance,
    time_index: int,
    states: Any,
    history: Any,
    immediate_payoffs: Any,
) -> tuple[Path, Path, tuple[int, ...]]:
    """Write the exact causal query exposed to one prediction process."""
    state_raw = np.asarray(states)
    if (
        not np.issubdtype(state_raw.dtype, np.number)
        or np.issubdtype(state_raw.dtype, np.complexfloating)
        or np.issubdtype(state_raw.dtype, np.bool_)
    ):
        raise ValueError("prediction states must contain real numeric values")
    state = np.ascontiguousarray(state_raw, dtype=np.float64)
    if state.ndim < 2 or state.shape[-1] != instance.dimension:
        raise ValueError(
            f"prediction states must have shape (..., {instance.dimension})"
        )
    if not np.all(np.isfinite(state)) or np.any(state <= 0.0):
        raise ValueError("prediction states must be finite positive asset states")
    leading_shape = state.shape[:-1]

    history_raw = np.asarray(history)
    if (
        not np.issubdtype(history_raw.dtype, np.number)
        or np.issubdtype(history_raw.dtype, np.complexfloating)
        or np.issubdtype(history_raw.dtype, np.bool_)
    ):
        raise ValueError("prediction history must contain real numeric values")
    history_array = np.ascontiguousarray(history_raw, dtype=np.float64)
    expected_history_shape = (*leading_shape, time_index + 1, instance.dimension)
    if history_array.shape != expected_history_shape:
        raise ValueError(
            "prediction history must have shape " + str(expected_history_shape)
        )
    if not np.all(np.isfinite(history_array)) or np.any(history_array <= 0.0):
        raise ValueError("prediction history must contain finite positive states")
    if not np.array_equal(history_array[..., -1, :], state):
        raise ValueError("prediction states must equal the final history state")

    payoff_raw = np.asarray(immediate_payoffs)
    if (
        not np.issubdtype(payoff_raw.dtype, np.number)
        or np.issubdtype(payoff_raw.dtype, np.complexfloating)
        or np.issubdtype(payoff_raw.dtype, np.bool_)
    ):
        raise ValueError("immediate_payoffs must contain real numeric values")
    immediate = np.ascontiguousarray(payoff_raw, dtype=np.float64)
    if immediate.shape != leading_shape:
        raise ValueError(
            f"immediate_payoffs must have shape {leading_shape}"
        )
    if not np.all(np.isfinite(immediate)) or np.any(immediate < 0.0):
        raise ValueError("immediate_payoffs must be finite and non-negative")

    model_dir = input_dir / "model"
    query_dir = input_dir / "query"
    model_dir.mkdir(mode=0o700)
    query_dir.mkdir(mode=0o700)
    for relative, data in artifact.files:
        destination = model_dir / Path(relative)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_bytes(data)
    (query_dir / "states.npy").write_bytes(_npy_bytes(state))
    (query_dir / "history.npy").write_bytes(_npy_bytes(history_array))
    (query_dir / "immediate_payoffs.npy").write_bytes(_npy_bytes(immediate))
    (query_dir / "request.json").write_bytes(_canonical_json_bytes({
        "schema": PROGRAM_PREDICTION_REQUEST_SCHEMA,
        "interface": artifact.manifest.interface,
        "time_index": time_index,
        "instance": training_instance_payload(instance),
    }))
    _seal_readonly_tree(input_dir)
    return model_dir, query_dir, leading_shape


class SandboxedPythonProgramRunner:
    """Execute every candidate prediction in a fresh, causal sandbox call."""

    runner_type = "python_program"

    def __init__(
        self,
        *,
        artifact: PythonProgramModelArtifact,
        source_dir: Path,
        instance: BSInstance,
        prediction_parent: Path,
        runtime_roots: Iterable[str | os.PathLike[str]],
        timeout_s: float,
        cpu_seconds: float | None,
        memory_bytes: int,
        file_size_bytes: int,
        externally_isolated: bool,
        cancel_event: Any,
        python_executable: str | os.PathLike[str],
    ) -> None:
        self.artifact = artifact
        self.source_dir = source_dir
        self.instance = instance
        self.policy_interface = artifact.manifest.interface
        self.prediction_parent = prediction_parent
        self.runtime_roots = tuple(runtime_roots)
        self.timeout_s = timeout_s
        self.cpu_seconds = cpu_seconds
        self.memory_bytes = memory_bytes
        self.file_size_bytes = file_size_bytes
        self.externally_isolated = externally_isolated
        self.cancel_event = cancel_event
        self.python_executable = os.fspath(python_executable)
        self.prediction_calls = 0
        self.prediction_wall_seconds = 0.0

    def _predict(
        self,
        time_index: int,
        states: Any,
        *,
        history: Any,
        immediate_payoffs: Any,
    ) -> np.ndarray:
        if isinstance(time_index, bool) or not isinstance(time_index, int):
            raise ValueError("time_index must be an integer")
        if not 0 <= time_index < len(self.instance.exercise_times) - 1:
            raise ValueError("time_index is outside the non-terminal exercise grid")
        call_index = self.prediction_calls
        self.prediction_calls += 1
        call_root, input_dir, output_dir, tmp_dir = _create_cell_directories(
            self.prediction_parent / f"call-{call_index:06d}"
        )
        try:
            model_dir, query_dir, leading_shape = _write_python_prediction_input(
                input_dir,
                self.artifact,
                instance=self.instance,
                time_index=time_index,
                states=states,
                history=history,
                immediate_payoffs=immediate_payoffs,
            )
            command = [
                self.python_executable,
                PROGRAM_ENTRYPOINT,
                "predict",
                "--model",
                str(model_dir.resolve(strict=True)),
                "--input",
                str(query_dir.resolve(strict=True)),
                "--output",
                str(output_dir.resolve(strict=True)),
            ]
            result = run_training_sandbox(
                command,
                source_dir=self.source_dir,
                input_dir=input_dir,
                output_dir=output_dir,
                tmp_dir=tmp_dir,
                runtime_roots=self.runtime_roots,
                timeout_s=self.timeout_s,
                cpu_seconds=self.cpu_seconds,
                memory_bytes=self.memory_bytes,
                file_size_bytes=self.file_size_bytes,
                max_total_output_bytes=self.file_size_bytes,
                externally_isolated=self.externally_isolated,
                cancel_event=self.cancel_event,
            )
            self.prediction_wall_seconds += float(result["wall_seconds"])
            if result["status"] != "ok":
                note = str(result["log_tail"]).strip() or "candidate prediction failed"
                raise ValueError(
                    f"candidate prediction failed: {result['status']}: {note[-1000:]}"
                )
            entries = list(os.scandir(output_dir))
            if len(entries) != 1 or entries[0].name != "predictions.npy":
                raise ValueError(
                    "candidate prediction output must contain exactly predictions.npy"
                )
            data = read_regular_file(
                output_dir / "predictions.npy",
                self.file_size_bytes,
                label="predictions.npy",
            )
            try:
                prediction = np.load(io.BytesIO(data), allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ValueError("predictions.npy is not a valid NumPy array") from exc
            if not isinstance(prediction, np.ndarray) or prediction.shape != leading_shape:
                raise ValueError(
                    f"predictions.npy must have shape {leading_shape}"
                )
            if self.policy_interface == "decision":
                if prediction.dtype == np.dtype(np.bool_):
                    return np.asarray(prediction, dtype=np.bool_)
                if (
                    np.issubdtype(prediction.dtype, np.number)
                    and not np.issubdtype(prediction.dtype, np.complexfloating)
                ):
                    numeric = np.asarray(prediction, dtype=np.float64)
                    if np.all(np.isfinite(numeric)) and np.all(
                        (numeric == 0.0) | (numeric == 1.0)
                    ):
                        return numeric.astype(np.bool_)
                raise ValueError(
                    "decision predictions must contain booleans or exact zero/one values"
                )
            if (
                not np.issubdtype(prediction.dtype, np.number)
                or np.issubdtype(prediction.dtype, np.complexfloating)
                or np.issubdtype(prediction.dtype, np.bool_)
            ):
                raise ValueError("continuation predictions must be real numeric values")
            values = np.asarray(prediction, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError("continuation predictions contain NaN or infinity")
            return np.clip(values, *PROTOCOL_OUTPUT_CLIP)
        finally:
            shutil.rmtree(call_root, ignore_errors=True)

    def _default_query(
        self,
        time_index: int,
        states: Any,
        history: Any | None,
        immediate_payoffs: Any | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(states, dtype=np.float64)
        if history is None:
            if time_index != 0:
                raise ValueError("history is required after the first exercise time")
            history = np.expand_dims(state, axis=-2)
        if immediate_payoffs is None:
            immediate_payoffs = payoff(state, self.instance) * math.exp(
                -self.instance.rate * self.instance.exercise_times[time_index]
            )
        return np.asarray(history), np.asarray(immediate_payoffs)

    def continuation(
        self,
        time_index: int,
        states: np.ndarray,
        instance: Any | None = None,
        *,
        history: Any | None = None,
        immediate_payoffs: Any | None = None,
    ) -> np.ndarray:
        if self.policy_interface != "continuation":
            raise ValueError("Python program exposes direct decisions, not continuation")
        history, immediate_payoffs = self._default_query(
            time_index, states, history, immediate_payoffs,
        )
        return self._predict(
            time_index,
            states,
            history=history,
            immediate_payoffs=immediate_payoffs,
        )

    def decision(
        self,
        time_index: int,
        states: np.ndarray,
        instance: Any | None = None,
        *,
        history: Any | None = None,
        immediate_payoffs: Any | None = None,
    ) -> np.ndarray:
        if self.policy_interface != "decision":
            raise ValueError("Python program exposes continuation, not decisions")
        history, immediate_payoffs = self._default_query(
            time_index, states, history, immediate_payoffs,
        )
        return self._predict(
            time_index,
            states,
            history=history,
            immediate_payoffs=immediate_payoffs,
        )


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
    prediction_timeout_s: float = DEFAULT_PREDICTION_TIMEOUT_S,
    prediction_cpu_seconds: float | None = None,
    prediction_memory_bytes: int = DEFAULT_PREDICTION_MEMORY_BYTES,
    prediction_file_size_bytes: int = DEFAULT_PREDICTION_FILE_SIZE_BYTES,
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
    runtime_root_tuple = tuple(runtime_roots)
    # Parse the manifest before starting an untrusted process so unsupported
    # runner protocols fail deterministically and do not consume a training
    # cell.  The same validated manifest is used for artifact loading below.
    manifest = load_candidate_manifest(source / "manifest.json")
    is_python_program = isinstance(manifest, PythonProgramManifest)
    entrypoint = PROGRAM_ENTRYPOINT if is_python_program else "train.py"
    try:
        entrypoint_info = os.lstat(source / entrypoint)
    except FileNotFoundError as exc:
        raise ValueError(f"candidate {entrypoint} not found") from exc
    if stat.S_ISLNK(entrypoint_info.st_mode) or not stat.S_ISREG(entrypoint_info.st_mode):
        raise ValueError(f"candidate {entrypoint} must be a regular file")
    root, input_dir, output_dir, tmp_dir = _create_cell_directories(cell_dir)
    input_bundle = write_training_input_bundle(
        validated_instance, training_paths, input_dir,
    )
    command = [
        os.fspath(python_executable),
        entrypoint,
    ]
    if is_python_program:
        command.append("fit")
    command.extend([
        "--input",
        str(input_dir.resolve(strict=True)),
        "--output",
        str(output_dir.resolve(strict=True)),
        "--seed",
        str(seed),
    ])
    sandbox_result = run_training_sandbox(
        command,
        source_dir=source,
        input_dir=input_dir,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        runtime_roots=runtime_root_tuple,
        timeout_s=timeout_s,
        cpu_seconds=cpu_seconds,
        memory_bytes=memory_bytes,
        file_size_bytes=file_size_bytes,
        max_total_output_bytes=(
            file_size_bytes if is_python_program else 64 * 1024 * 1024
        ),
        externally_isolated=externally_isolated,
        cancel_event=cancel_event,
    )
    common = {
        "returncode": sandbox_result["returncode"],
        "isolation": sandbox_result["isolation"],
        "research_fallback": bool(sandbox_result.get("research_fallback", False)),
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
        if is_python_program:
            artifact = _load_python_program_model(
                output_dir,
                manifest,
                max_bytes=file_size_bytes,
            )
            prediction_parent = root / "prediction_calls"
            prediction_parent.mkdir(mode=0o700)
            runner = SandboxedPythonProgramRunner(
                artifact=artifact,
                source_dir=source.resolve(strict=True),
                instance=validated_instance,
                prediction_parent=prediction_parent,
                runtime_roots=runtime_root_tuple,
                timeout_s=prediction_timeout_s,
                cpu_seconds=prediction_cpu_seconds,
                memory_bytes=prediction_memory_bytes,
                file_size_bytes=prediction_file_size_bytes,
                externally_isolated=externally_isolated,
                cancel_event=cancel_event,
                python_executable=python_executable,
            )
        else:
            runner = load_continuation_runner(
                manifest,
                output_dir,
                n_exercise_times=len(validated_instance.exercise_times),
                input_dim=validated_instance.dimension,
                instance=validated_instance,
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
    return TrainingCellResult(
        status="ok",
        policy_file_sha256=(
            tuple((name, hashlib.sha256(data).hexdigest()) for name, data in artifact.files)
            if is_python_program else runner.artifact.file_sha256
        ),
        policy_artifact_sha256=(
            _python_model_artifact_sha256(artifact)
            if is_python_program else runner.artifact.bundle_sha256
        ),
        runner=runner,
        **common,
    )
