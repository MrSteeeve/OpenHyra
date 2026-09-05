"""Trusted loader and runner for frozen Bermudan continuation policies.

The candidate-facing training process may create these files, but this module
is evaluator-owned.  It treats the output directory as untrusted input, loads
only a small data-only format, and never imports candidate code.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sandbox import read_regular_file as _read_regular_file


POLICY_SCHEMA = "openhyra-policy-spec.v1"
OUTPUT_SEMANTICS = "discounted_continuation_value_t0"
WEIGHT_PATTERN = "step_{:03d}.npy"
NORMALIZATION_MODE = "per_step"
PROTOCOL_OUTPUT_CLIP = (-1_000_000.0, 1_000_000.0)
SUPPORTED_ACTIVATIONS = frozenset({"relu", "tanh"})

MAX_MANIFEST_BYTES = 64 * 1024
MAX_NORMALIZATION_BYTES = 256 * 1024
MAX_STEP_FILE_BYTES = 1024 * 1024
MAX_ARTIFACT_BUNDLE_BYTES = 8 * 1024 * 1024
MAX_EXERCISE_TIMES = 1000
MAX_INPUT_DIM = 4096
MAX_HIDDEN_LAYERS = 8
MAX_LAYER_WIDTH = 4096
MAX_PARAMETERS_PER_STEP = (MAX_STEP_FILE_BYTES - 256) // 8
NORMALIZATION_EPSILON = 1e-10


@dataclass(frozen=True)
class MLPInferenceConfig:
    """Validated, closed inference configuration for the v1 MLP runner."""

    input_dim: int | str
    layers: tuple[int, ...]
    activation: str
    output_dim: int
    output_clip: tuple[float, float]


@dataclass(frozen=True)
class PolicyManifest:
    """Immutable normalized form of ``openhyra-policy-spec.v1``."""

    schema: str
    runner_type: str
    inference_config: MLPInferenceConfig
    output_semantics: str
    normalization: str
    weight_pattern: str


@dataclass(frozen=True)
class NormalizationStats:
    """Read-only input normalization for one exercise time."""

    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class DenseLayer:
    """One immutable evaluator-owned dense layer, stored output by input."""

    weights: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class MLPStep:
    """The fully split MLP parameters for one non-terminal exercise time."""

    layers: tuple[DenseLayer, ...]


@dataclass(frozen=True)
class PolicyArtifact:
    """Validated policy bytes, immutable parameters, and their provenance."""

    manifest: PolicyManifest
    input_dim: int
    normalizations: tuple[NormalizationStats, ...]
    steps: tuple[MLPStep, ...]
    file_sha256: tuple[tuple[str, str], ...]
    bundle_sha256: str
    parameter_count_per_step: int


def _strict_keys(
    value: Any,
    *,
    required: set[str],
    allowed: set[str],
    path: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{path} is missing field(s): {', '.join(missing)}")


def _strict_int(value: Any, *, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{path} must be in [{minimum}, {maximum}]")
    return value


def _strict_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _load_json_bytes(data: bytes, *, label: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _manifest_payload(manifest: PolicyManifest) -> dict[str, Any]:
    config = manifest.inference_config
    # Continuation protocol adapters (linear/expression) intentionally share
    # the provenance helper but do not have MLP ``layers``/``activation``.
    # Keep the historical MLP payload byte-for-byte unchanged while allowing
    # the generic evaluator to canonicalize those adapters without importing
    # their module here (and without creating a circular import).
    if not hasattr(config, "layers"):
        return {
            "schema": manifest.schema,
            "runner_type": manifest.runner_type,
            "inference_config": {
                "input_dim": config.input_dim,
                "output_dim": config.output_dim,
                "output_clip": list(config.output_clip),
            },
            "output_semantics": manifest.output_semantics,
            "normalization": manifest.normalization,
            "weight_pattern": manifest.weight_pattern,
        }
    return {
        "schema": manifest.schema,
        "runner_type": manifest.runner_type,
        "inference_config": {
            "input_dim": config.input_dim,
            "layers": list(config.layers),
            "activation": config.activation,
            "output_dim": config.output_dim,
            "output_clip": list(config.output_clip),
        },
        "output_semantics": manifest.output_semantics,
        "normalization": manifest.normalization,
        "weight_pattern": manifest.weight_pattern,
    }


def validate_policy_manifest(raw: Any) -> PolicyManifest:
    """Validate the exact v1 manifest schema and return an immutable value."""
    if isinstance(raw, PolicyManifest):
        # A frozen dataclass is an immutable container, not a trust token: it
        # can be constructed directly (or forged with object.__setattr__).  Run
        # every field back through the same closed schema used for JSON input.
        try:
            raw = _manifest_payload(raw)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("PolicyManifest has an invalid internal structure") from exc
    fields = {
        "schema",
        "runner_type",
        "inference_config",
        "output_semantics",
        "normalization",
        "weight_pattern",
    }
    _strict_keys(raw, required=fields, allowed=fields, path="policy manifest")
    if raw["schema"] != POLICY_SCHEMA:
        raise ValueError(f"policy manifest schema must be {POLICY_SCHEMA}")
    if raw["runner_type"] != "mlp":
        raise ValueError("policy manifest runner_type must be mlp")
    if raw["output_semantics"] != OUTPUT_SEMANTICS:
        raise ValueError(
            f"policy manifest output_semantics must be {OUTPUT_SEMANTICS}"
        )
    if raw["normalization"] != NORMALIZATION_MODE:
        raise ValueError(f"policy manifest normalization must be {NORMALIZATION_MODE}")
    if raw["weight_pattern"] != WEIGHT_PATTERN:
        raise ValueError(f"policy manifest weight_pattern must be {WEIGHT_PATTERN}")

    config = raw["inference_config"]
    config_fields = {
        "input_dim", "layers", "activation", "output_dim", "output_clip",
    }
    _strict_keys(
        config,
        required=config_fields,
        allowed=config_fields,
        path="policy manifest.inference_config",
    )
    declared_input_dim = config["input_dim"]
    if declared_input_dim == "n_assets":
        input_dim: int | str = "n_assets"
    else:
        input_dim = _strict_int(
            declared_input_dim,
            path="policy manifest.inference_config.input_dim",
            minimum=1,
            maximum=MAX_INPUT_DIM,
        )

    raw_layers = config["layers"]
    if not isinstance(raw_layers, list):
        raise ValueError("policy manifest.inference_config.layers must be an array")
    if len(raw_layers) > MAX_HIDDEN_LAYERS:
        raise ValueError(
            "policy manifest.inference_config.layers exceeds the hidden-layer limit"
        )
    layers = tuple(
        _strict_int(
            width,
            path=f"policy manifest.inference_config.layers[{index}]",
            minimum=1,
            maximum=MAX_LAYER_WIDTH,
        )
        for index, width in enumerate(raw_layers)
    )
    activation = config["activation"]
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            "policy manifest.inference_config.activation must be one of: "
            + ", ".join(sorted(SUPPORTED_ACTIVATIONS))
        )
    output_dim = _strict_int(
        config["output_dim"],
        path="policy manifest.inference_config.output_dim",
        minimum=1,
        maximum=1,
    )
    raw_clip = config["output_clip"]
    if not isinstance(raw_clip, list) or len(raw_clip) != 2:
        raise ValueError(
            "policy manifest.inference_config.output_clip must be a two-number array"
        )
    output_clip = tuple(
        _strict_number(value, path=f"policy manifest.inference_config.output_clip[{index}]")
        for index, value in enumerate(raw_clip)
    )
    if output_clip != PROTOCOL_OUTPUT_CLIP:
        raise ValueError(
            "policy manifest.inference_config.output_clip must equal the protocol "
            f"limit {list(PROTOCOL_OUTPUT_CLIP)}"
        )

    return PolicyManifest(
        schema=POLICY_SCHEMA,
        runner_type="mlp",
        inference_config=MLPInferenceConfig(
            input_dim=input_dim,
            layers=layers,
            activation=activation,
            output_dim=output_dim,
            output_clip=PROTOCOL_OUTPUT_CLIP,
        ),
        output_semantics=OUTPUT_SEMANTICS,
        normalization=NORMALIZATION_MODE,
        weight_pattern=WEIGHT_PATTERN,
    )


def load_policy_manifest(path: str | os.PathLike[str]) -> PolicyManifest:
    """Safely read and validate a file named exactly ``manifest.json``."""
    manifest_path = Path(path)
    if manifest_path.name != "manifest.json":
        raise ValueError("policy manifest file must be named manifest.json")
    data = _read_regular_file(
        manifest_path, MAX_MANIFEST_BYTES, label="manifest.json",
    )
    return validate_policy_manifest(_load_json_bytes(data, label="manifest.json"))


def _coerce_manifest(
    value: PolicyManifest | Mapping[str, Any] | str | os.PathLike[str],
) -> tuple[PolicyManifest, bytes]:
    if isinstance(value, PolicyManifest):
        manifest = validate_policy_manifest(value)
        return manifest, _canonical_json_bytes(_manifest_payload(manifest))
    if isinstance(value, Mapping):
        manifest = validate_policy_manifest(dict(value))
        return manifest, _canonical_json_bytes(_manifest_payload(manifest))
    path = Path(value)
    if path.name != "manifest.json":
        raise ValueError("policy manifest file must be named manifest.json")
    data = _read_regular_file(path, MAX_MANIFEST_BYTES, label="manifest.json")
    manifest = validate_policy_manifest(_load_json_bytes(data, label="manifest.json"))
    return manifest, _canonical_json_bytes(_manifest_payload(manifest))


def _resolved_input_dim(manifest: PolicyManifest, input_dim: int | None) -> int:
    declared = manifest.inference_config.input_dim
    if input_dim is not None:
        resolved = _strict_int(
            input_dim, path="input_dim", minimum=1, maximum=MAX_INPUT_DIM,
        )
    elif isinstance(declared, int):
        resolved = declared
    else:
        raise ValueError("input_dim is required when manifest input_dim is n_assets")
    if isinstance(declared, int) and declared != resolved:
        raise ValueError("input_dim does not match policy manifest.inference_config.input_dim")
    return resolved


def _parameter_shapes(
    input_dim: int, hidden_layers: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], tuple[int]], ...]:
    widths = (input_dim, *hidden_layers, 1)
    return tuple(
        ((output_width, input_width), (output_width,))
        for input_width, output_width in zip(widths, widths[1:])
    )


def _parameter_count(
    shapes: tuple[tuple[tuple[int, int], tuple[int]], ...],
) -> int:
    return sum(
        weight_shape[0] * weight_shape[1] + bias_shape[0]
        for weight_shape, bias_shape in shapes
    )


def _validate_artifact_directory(root: Path, expected_names: set[str]) -> None:
    try:
        info = os.lstat(root)
    except FileNotFoundError as exc:
        raise ValueError("policy artifact directory not found") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("policy artifact root must be a real directory")
    actual_names = {entry.name for entry in os.scandir(root)}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing:
        raise ValueError(f"policy artifact is missing file(s): {', '.join(missing)}")
    if unexpected:
        raise ValueError(
            f"policy artifact has unexpected file(s): {', '.join(unexpected)}"
        )


def _load_canonical_flat_weights(
    data: bytes, *, expected_count: int, label: str,
) -> np.ndarray:
    header_stream = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(header_stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                header_stream,
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                header_stream,
            )
        else:
            raise ValueError(f"{label} uses unsupported NPY format version {version}")
    except (EOFError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(label):
            raise
        raise ValueError(f"{label} is not a valid canonical NPY file") from exc
    expected_dtype = np.dtype(np.float64)
    if shape != (expected_count,):
        raise ValueError(
            f"{label} shape must be ({expected_count},), got {shape}"
        )
    if fortran_order:
        raise ValueError(f"{label} must use C order")
    if dtype.hasobject or dtype != expected_dtype or not dtype.isnative:
        raise ValueError(f"{label} dtype must be native float64")

    stream = io.BytesIO(data)
    try:
        loaded = np.load(stream, allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise ValueError(f"{label} could not be safely loaded: {exc}") from exc
    if not isinstance(loaded, np.ndarray):
        close = getattr(loaded, "close", None)
        if close is not None:
            close()
        raise ValueError(f"{label} must contain one NPY array, not an archive")
    if stream.tell() != len(data):
        raise ValueError(f"{label} contains trailing data")
    if (
        loaded.ndim != 1
        or loaded.shape != (expected_count,)
        or loaded.dtype != expected_dtype
        or not loaded.dtype.isnative
        or not loaded.flags.c_contiguous
    ):
        raise ValueError(f"{label} must be a canonical 1-D native float64 array")
    if not np.all(np.isfinite(loaded)):
        raise ValueError(f"{label} contains NaN or infinity")
    return loaded


def _immutable_array(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return an array backed by immutable bytes, not merely a writable owner."""
    data = np.asarray(value, dtype=np.float64).reshape(shape).tobytes(order="C")
    return np.frombuffer(data, dtype=np.float64).reshape(shape)


def _split_step(
    flat: np.ndarray,
    shapes: tuple[tuple[tuple[int, int], tuple[int]], ...],
) -> MLPStep:
    layers: list[DenseLayer] = []
    offset = 0
    for weight_shape, bias_shape in shapes:
        weight_count = weight_shape[0] * weight_shape[1]
        weights = _immutable_array(
            flat[offset:offset + weight_count], weight_shape,
        )
        offset += weight_count
        bias = _immutable_array(flat[offset:offset + bias_shape[0]], bias_shape)
        offset += bias_shape[0]
        layers.append(DenseLayer(weights=weights, bias=bias))
    if offset != flat.size:  # pragma: no cover - derived count makes this total
        raise ValueError("internal weight splitting mismatch")
    return MLPStep(layers=tuple(layers))


def _validate_normalizations(
    data: bytes, *, step_count: int, input_dim: int,
) -> tuple[NormalizationStats, ...]:
    raw = _load_json_bytes(data, label="normalization.json")
    _strict_keys(raw, required={"steps"}, allowed={"steps"}, path="normalization")
    steps = raw["steps"]
    if not isinstance(steps, list) or len(steps) != step_count:
        raise ValueError(
            f"normalization.steps must contain exactly {step_count} entries"
        )
    result: list[NormalizationStats] = []
    for step_index, entry in enumerate(steps):
        path = f"normalization.steps[{step_index}]"
        _strict_keys(
            entry,
            required={"mean", "scale"},
            allowed={"mean", "scale"},
            path=path,
        )
        mean_raw, scale_raw = entry["mean"], entry["scale"]
        if not isinstance(mean_raw, list) or len(mean_raw) != input_dim:
            raise ValueError(f"{path}.mean length must equal input_dim ({input_dim})")
        if not isinstance(scale_raw, list) or len(scale_raw) != input_dim:
            raise ValueError(f"{path}.scale length must equal input_dim ({input_dim})")
        mean = np.asarray(
            [_strict_number(value, path=f"{path}.mean[{index}]")
             for index, value in enumerate(mean_raw)],
            dtype=np.float64,
        )
        scale = np.asarray(
            [_strict_number(value, path=f"{path}.scale[{index}]")
             for index, value in enumerate(scale_raw)],
            dtype=np.float64,
        )
        if np.any(scale <= NORMALIZATION_EPSILON):
            raise ValueError(
                f"{path}.scale values must be greater than {NORMALIZATION_EPSILON}"
            )
        result.append(
            NormalizationStats(
                mean=_immutable_array(mean, (input_dim,)),
                scale=_immutable_array(scale, (input_dim,)),
            )
        )
    return tuple(result)


def _bundle_hash(files: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(b"openhyra-policy-bundle.v1\0")
    for name in sorted(files):
        name_bytes = name.encode("utf-8")
        data = files[name]
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def load_policy_artifact(
    manifest: PolicyManifest | Mapping[str, Any] | str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    *,
    n_exercise_times: int,
    input_dim: int | None = None,
) -> PolicyArtifact:
    """Load one untrusted per-instance artifact into immutable trusted data.

    ``manifest`` is either a previously validated manifest, a manifest mapping,
    or a safely read ``manifest.json`` path.  The output directory must contain
    exactly ``normalization.json`` and one canonical NPY file for every
    non-terminal exercise time.
    """
    validated_manifest, manifest_bytes = _coerce_manifest(manifest)
    time_count = _strict_int(
        n_exercise_times,
        path="n_exercise_times",
        minimum=2,
        maximum=MAX_EXERCISE_TIMES,
    )
    step_count = time_count - 1
    resolved_input_dim = _resolved_input_dim(validated_manifest, input_dim)
    shapes = _parameter_shapes(
        resolved_input_dim, validated_manifest.inference_config.layers,
    )
    parameter_count = _parameter_count(shapes)
    if parameter_count > MAX_PARAMETERS_PER_STEP:
        raise ValueError(
            "policy MLP exceeds the per-step parameter limit: "
            f"{parameter_count} > {MAX_PARAMETERS_PER_STEP}"
        )

    expected_step_names = [WEIGHT_PATTERN.format(index) for index in range(step_count)]
    expected_names = {"normalization.json", *expected_step_names}
    root = Path(artifact_dir)
    _validate_artifact_directory(root, expected_names)

    files: dict[str, bytes] = {"manifest.json": manifest_bytes}
    total_bytes = len(manifest_bytes)
    normalization_bytes = _read_regular_file(
        root / "normalization.json",
        min(MAX_NORMALIZATION_BYTES, MAX_ARTIFACT_BUNDLE_BYTES - total_bytes),
        label="normalization.json",
    )
    files["normalization.json"] = normalization_bytes
    total_bytes += len(normalization_bytes)
    for name in expected_step_names:
        remaining = MAX_ARTIFACT_BUNDLE_BYTES - total_bytes
        data = _read_regular_file(
            root / name, min(MAX_STEP_FILE_BYTES, max(0, remaining)), label=name,
        )
        files[name] = data
        total_bytes += len(data)

    normalizations = _validate_normalizations(
        normalization_bytes,
        step_count=step_count,
        input_dim=resolved_input_dim,
    )
    steps = tuple(
        _split_step(
            _load_canonical_flat_weights(
                files[name], expected_count=parameter_count, label=name,
            ),
            shapes,
        )
        for name in expected_step_names
    )
    hashes = tuple(
        (name, hashlib.sha256(files[name]).hexdigest()) for name in sorted(files)
    )
    return PolicyArtifact(
        manifest=validated_manifest,
        input_dim=resolved_input_dim,
        normalizations=normalizations,
        steps=steps,
        file_sha256=hashes,
        bundle_sha256=_bundle_hash(files),
        parameter_count_per_step=parameter_count,
    )


@dataclass(frozen=True, slots=True)
class MLPContinuationRunner:
    """Deterministic and stateless evaluator-owned continuation runner."""

    artifact: PolicyArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, PolicyArtifact):
            raise TypeError("artifact must be a validated PolicyArtifact")

    def continuation(
        self,
        time_index: int,
        states: np.ndarray,
        instance: Any | None = None,
    ) -> np.ndarray:
        """Return t0-discounted continuation values with the protocol clip.

        Each sample follows an identical scalar float64 operation sequence.
        In particular, affine reductions never enter a batch-shaped BLAS call,
        so splitting or reordering a batch cannot change reduction grouping.

        ``instance`` is accepted for the common continuation-runner protocol.
        The MLP wire format already fixes its input semantics, so the argument
        is intentionally ignored; expression runners use it for payoff-aware
        terminals.
        """
        if isinstance(time_index, bool) or not isinstance(time_index, int):
            raise ValueError("time_index must be an integer")
        if not 0 <= time_index < len(self.artifact.steps):
            raise ValueError("time_index is outside the non-terminal exercise grid")
        raw_states = np.asarray(states)
        if (
            not np.issubdtype(raw_states.dtype, np.number)
            or np.issubdtype(raw_states.dtype, np.complexfloating)
            or np.issubdtype(raw_states.dtype, np.bool_)
        ):
            raise ValueError("states must contain real numeric values")
        state = np.asarray(raw_states, dtype=np.float64)
        if state.ndim < 1 or state.shape[-1] != self.artifact.input_dim:
            raise ValueError(
                f"states must have shape (..., {self.artifact.input_dim})"
            )
        if not np.all(np.isfinite(state)):
            raise ValueError("states must contain only finite values")

        leading_shape = state.shape[:-1]
        flat = state.reshape(-1, self.artifact.input_dim)
        normalization = self.artifact.normalizations[time_index]
        layers = self.artifact.steps[time_index].layers
        activation = self.artifact.manifest.inference_config.activation
        output = np.empty(flat.shape[0], dtype=np.float64)
        lower, upper = PROTOCOL_OUTPUT_CLIP

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for sample_index in range(flat.shape[0]):
                current = np.empty(self.artifact.input_dim, dtype=np.float64)
                for input_index in range(self.artifact.input_dim):
                    difference = np.float64(
                        flat[sample_index, input_index]
                        - normalization.mean[input_index]
                    )
                    current[input_index] = np.float64(
                        difference / normalization.scale[input_index]
                    )
                if not np.all(np.isfinite(current)):
                    raise ValueError(
                        "normalized states are outside the finite numeric domain"
                    )

                for layer_index, layer in enumerate(layers):
                    next_values = np.empty(layer.bias.shape[0], dtype=np.float64)
                    for output_index in range(layer.bias.shape[0]):
                        accumulator = np.float64(0.0)
                        for input_index in range(current.shape[0]):
                            product = np.float64(
                                current[input_index]
                                * layer.weights[output_index, input_index]
                            )
                            accumulator = np.float64(accumulator + product)
                        next_values[output_index] = np.float64(
                            accumulator + layer.bias[output_index]
                        )
                    current = next_values
                    is_output = layer_index == len(layers) - 1
                    if not is_output:
                        if activation == "tanh":
                            current = np.tanh(current)
                        else:
                            current = np.maximum(current, np.float64(0.0))
                        if not np.all(np.isfinite(current)):
                            raise ValueError(
                                "MLP hidden layer produced NaN or infinity"
                            )

                value = current[0]
                if np.isnan(value):
                    raise ValueError("MLP output produced NaN")
                if value < lower:
                    value = np.float64(lower)
                elif value > upper:
                    value = np.float64(upper)
                if not np.isfinite(value):  # pragma: no cover - comparisons clip infinity
                    raise ValueError("MLP output is not finite after protocol clipping")
                output[sample_index] = value
        return output.reshape(leading_shape)
