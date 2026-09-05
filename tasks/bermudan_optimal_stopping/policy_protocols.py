"""Policy-protocol dispatch for legacy artifacts and open Python programs.

The historical task accepts data-only Linear, Expression, and MLP artifacts
through small trusted runners. The additive ``python_program`` task instead
executes a candidate-owned ``algorithm.py`` through the fit/predict process
contract; its manifest declares only continuation versus direct decision.

The legacy artifact protocols are:

``continuation-linear.v1``
    Per-exercise-date affine continuation values.  The artifact contains one
    flat ``float64`` coefficient vector per non-terminal date and optional
    per-date input normalization.

``continuation-expression.v1``
    A bounded scalar expression per non-terminal date.  Expressions use the
    same finance-aware, strike-normalized terminals as the legacy Feature IR.
    The expression is interpreted in normalized *exercise-time* units; the
    trusted runner converts it to the common time-zero discounted currency
    units used by the evaluator before returning it.

The existing ``openhyra-policy-spec.v1`` MLP loader remains the canonical MLP
implementation in :mod:`policy_artifact`; ``load_continuation_runner`` routes
to it without changing its wire format. These legacy runners are deterministic,
stateless, and expose the small common interface::

    runner.continuation(time_index, states, instance=None)

``instance`` is optional for linear/MLP policies and required by expression
terminals that need contract semantics (payoff type, strike, weights, and
exercise times). No legacy continuation runner makes the exercise decision.
Python-program candidates use the separate process runner and may return a
direct decision explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .policy_artifact import (
    MAX_ARTIFACT_BUNDLE_BYTES,
    MAX_INPUT_DIM,
    MAX_NORMALIZATION_BYTES,
    MAX_PARAMETERS_PER_STEP,
    MAX_STEP_FILE_BYTES,
    NORMALIZATION_EPSILON,
    OUTPUT_SEMANTICS,
    POLICY_SCHEMA,
    PROTOCOL_OUTPUT_CLIP,
    WEIGHT_PATTERN,
    MLPContinuationRunner,
    PolicyManifest,
    _bundle_hash as _mlp_bundle_hash,
    _canonical_json_bytes as _mlp_canonical_json_bytes,
    _coerce_manifest as _coerce_mlp_manifest,
    _load_canonical_flat_weights,
    _load_json_bytes,
    _read_regular_file,
    _resolved_input_dim,
    _strict_int,
    _strict_keys,
    _strict_number,
    _validate_normalizations,
    load_policy_artifact,
    validate_policy_manifest,
)


LINEAR_SCHEMA = "continuation-linear.v1"
EXPRESSION_SCHEMA = "continuation-expression.v1"
PYTHON_PROGRAM_SCHEMA = "openhyra-python-program.v1"
LINEAR_RUNNER_TYPE = "linear"
EXPRESSION_RUNNER_TYPE = "expression"
PYTHON_PROGRAM_RUNNER_TYPE = "python_program"
PYTHON_PROGRAM_INTERFACES = frozenset({"continuation", "decision"})
LINEAR_WEIGHT_PATTERN = WEIGHT_PATTERN
EXPRESSION_WEIGHT_PATTERN = "step_{:03d}.json"
NORMALIZATION_NONE = "none"
NORMALIZATION_PER_STEP = "per_step"

# The expression evaluator deliberately mirrors the bounded legacy IR rather
# than accepting Python source.  Keeping this list here avoids importing the
# evaluator (which would create a cycle once the evaluator dispatches here).
EXPRESSION_UNARY_OPS = frozenset(
    {
        "abs",
        "square",
        "cube",
        "sqrt_abs",
        "log1p_abs",
        "exp_neg_abs",
        "reciprocal_one_plus_abs",
    }
)
EXPRESSION_BINARY_OPS = frozenset(
    {"add", "subtract", "multiply", "divide_safe", "minimum", "maximum"}
)
EXPRESSION_TERMINALS = frozenset(
    {
        "time",
        "time_to_maturity",
        "spot",
        "mean_spot",
        "max_spot",
        "min_spot",
        "basket_spot",
        "underlying",
        "intrinsic",
    }
)
EXPRESSION_MAX_NODES = 128
EXPRESSION_MAX_DEPTH = 8
EXPRESSION_MAX_CONSTANT = 10.0
EXPRESSION_MAX_ABS = 1_000_000.0


class ContinuationRunner(Protocol):
    """Common runner surface consumed by the trusted evaluator."""

    def continuation(
        self,
        time_index: int,
        states: np.ndarray,
        instance: Any | None = None,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ContinuationInferenceConfig:
    input_dim: int | str
    output_dim: int
    output_clip: tuple[float, float]


@dataclass(frozen=True)
class ContinuationManifest:
    """Normalized manifest shared by linear and expression protocols."""

    schema: str
    runner_type: str
    inference_config: ContinuationInferenceConfig
    output_semantics: str
    normalization: str
    weight_pattern: str


@dataclass(frozen=True)
class PythonProgramManifest:
    """Minimal interface declaration for an executable ``algorithm.py``.

    Entrypoint names and CLI verbs are evaluator-owned.  The candidate may
    select only whether ``predictions.npy`` contains continuation values or
    direct exercise decisions.
    """

    schema: str
    interface: str

    @property
    def runner_type(self) -> str:
        return PYTHON_PROGRAM_RUNNER_TYPE


@dataclass(frozen=True)
class LinearStep:
    """One normalized affine continuation model, ``x @ coefficients + bias``."""

    coefficients: np.ndarray
    bias: float


@dataclass(frozen=True)
class LinearPolicyArtifact:
    manifest: ContinuationManifest
    input_dim: int
    normalizations: tuple[Any, ...]
    steps: tuple[LinearStep, ...]
    file_sha256: tuple[tuple[str, str], ...]
    bundle_sha256: str


@dataclass(frozen=True)
class ExpressionPolicyArtifact:
    manifest: ContinuationManifest
    input_dim: int
    expressions: tuple[dict[str, Any], ...]
    normalizations: tuple[Any, ...]
    file_sha256: tuple[tuple[str, str], ...]
    bundle_sha256: str


def _manifest_payload(manifest: ContinuationManifest) -> dict[str, Any]:
    config = manifest.inference_config
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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_continuation_manifest(raw: Any, *, expected_runner: str | None = None) -> ContinuationManifest:
    """Validate a linear/expression manifest without accepting extensions."""
    if isinstance(raw, ContinuationManifest):
        raw = _manifest_payload(raw)
    fields = {
        "schema",
        "runner_type",
        "inference_config",
        "output_semantics",
        "normalization",
        "weight_pattern",
    }
    _strict_keys(raw, required=fields, allowed=fields, path="policy manifest")
    schema = raw["schema"]
    runner_type = raw["runner_type"]
    expected_schema = {
        LINEAR_RUNNER_TYPE: LINEAR_SCHEMA,
        EXPRESSION_RUNNER_TYPE: EXPRESSION_SCHEMA,
    }.get(runner_type)
    if expected_runner is not None and runner_type != expected_runner:
        raise ValueError(f"policy manifest runner_type must be {expected_runner}")
    if expected_schema is None or schema != expected_schema:
        raise ValueError(
            "policy manifest schema/runner_type pair is not a supported continuation protocol"
        )
    if raw["output_semantics"] != OUTPUT_SEMANTICS:
        raise ValueError(
            f"policy manifest output_semantics must be {OUTPUT_SEMANTICS}"
        )

    if runner_type == LINEAR_RUNNER_TYPE:
        expected_normalization = NORMALIZATION_PER_STEP
        expected_pattern = LINEAR_WEIGHT_PATTERN
    else:
        expected_normalization = {NORMALIZATION_NONE, NORMALIZATION_PER_STEP}
        expected_pattern = EXPRESSION_WEIGHT_PATTERN
    normalization = raw["normalization"]
    if normalization not in (
        expected_normalization
        if isinstance(expected_normalization, set)
        else {expected_normalization}
    ):
        allowed_text = (
            ", ".join(sorted(expected_normalization))
            if isinstance(expected_normalization, set)
            else str(expected_normalization)
        )
        raise ValueError(f"policy manifest normalization must be one of: {allowed_text}")
    if raw["weight_pattern"] != expected_pattern:
        raise ValueError(f"policy manifest weight_pattern must be {expected_pattern}")

    config = raw["inference_config"]
    config_fields = {"input_dim", "output_dim", "output_clip"}
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
        _strict_number(value, path=f"policy manifest.inference_config.output_clip[{i}]")
        for i, value in enumerate(raw_clip)
    )
    if output_clip != PROTOCOL_OUTPUT_CLIP:
        raise ValueError(
            "policy manifest.inference_config.output_clip must equal the protocol limit "
            f"{list(PROTOCOL_OUTPUT_CLIP)}"
        )
    return ContinuationManifest(
        schema=schema,
        runner_type=runner_type,
        inference_config=ContinuationInferenceConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            output_clip=PROTOCOL_OUTPUT_CLIP,
        ),
        output_semantics=OUTPUT_SEMANTICS,
        normalization=normalization,
        weight_pattern=expected_pattern,
    )


def validate_continuation_manifest(raw: Any) -> ContinuationManifest | PolicyManifest:
    """Validate any registered continuation manifest.

    The legacy MLP schema is returned as its existing ``PolicyManifest`` type;
    linear and expression manifests use ``ContinuationManifest``.
    """
    if isinstance(raw, PolicyManifest):
        return validate_policy_manifest(raw)
    if isinstance(raw, ContinuationManifest):
        return _validate_continuation_manifest(_manifest_payload(raw))
    if isinstance(raw, Mapping) and raw.get("schema") == POLICY_SCHEMA:
        return validate_policy_manifest(dict(raw))
    if isinstance(raw, Mapping):
        return _validate_continuation_manifest(raw)
    raise ValueError("policy manifest must be a supported manifest object")


def validate_python_program_manifest(raw: Any) -> PythonProgramManifest:
    """Validate the minimal open-program manifest without executable knobs."""
    if isinstance(raw, PythonProgramManifest):
        raw = {"schema": raw.schema, "interface": raw.interface}
    fields = {"schema", "interface"}
    _strict_keys(
        raw,
        required=fields,
        allowed=fields,
        path="python program manifest",
    )
    if raw["schema"] != PYTHON_PROGRAM_SCHEMA:
        raise ValueError(
            f"python program manifest schema must be {PYTHON_PROGRAM_SCHEMA}"
        )
    interface = raw["interface"]
    if interface not in PYTHON_PROGRAM_INTERFACES:
        raise ValueError(
            "python program manifest interface must be continuation or decision"
        )
    return PythonProgramManifest(
        schema=PYTHON_PROGRAM_SCHEMA,
        interface=interface,
    )


def validate_candidate_manifest(
    raw: Any,
) -> ContinuationManifest | PolicyManifest | PythonProgramManifest:
    """Validate either a frozen continuation runner or an open program.

    The legacy ``validate_continuation_manifest`` remains continuation-only.
    Executable programs therefore require an explicit opt-in at the caller.
    """
    if isinstance(raw, PythonProgramManifest):
        return validate_python_program_manifest(raw)
    if isinstance(raw, Mapping) and raw.get("schema") == PYTHON_PROGRAM_SCHEMA:
        return validate_python_program_manifest(raw)
    return validate_continuation_manifest(raw)


def _coerce_continuation_manifest(
    value: Any,
) -> tuple[ContinuationManifest | PolicyManifest, bytes]:
    if isinstance(value, (ContinuationManifest, PolicyManifest)):
        validated = validate_continuation_manifest(value)
        if isinstance(validated, PolicyManifest):
            return validated, _mlp_canonical_json_bytes(
                {
                    "schema": validated.schema,
                    "runner_type": validated.runner_type,
                    "inference_config": {
                        "input_dim": validated.inference_config.input_dim,
                        "layers": list(validated.inference_config.layers),
                        "activation": validated.inference_config.activation,
                        "output_dim": validated.inference_config.output_dim,
                        "output_clip": list(validated.inference_config.output_clip),
                    },
                    "output_semantics": validated.output_semantics,
                    "normalization": validated.normalization,
                    "weight_pattern": validated.weight_pattern,
                }
            )
        return validated, _canonical_json_bytes(_manifest_payload(validated))
    if isinstance(value, Mapping):
        validated = validate_continuation_manifest(dict(value))
        if isinstance(validated, PolicyManifest):
            # Use the existing loader's canonical representation for MLP.
            _, data = _coerce_mlp_manifest(validated)
            return validated, data
        return validated, _canonical_json_bytes(_manifest_payload(validated))
    path = Path(value)
    if path.name != "manifest.json":
        raise ValueError("policy manifest file must be named manifest.json")
    # MLP and continuation manifests share the same filename.  Read once and
    # choose the parser by schema; the existing MLP loader still owns its
    # detailed validation.
    data = _read_regular_file(path, 64 * 1024, label="manifest.json")
    raw = _load_json_bytes(data, label="manifest.json")
    validated = validate_continuation_manifest(raw)
    if isinstance(validated, PolicyManifest):
        _, canonical = _coerce_mlp_manifest(validated)
        return validated, canonical
    return validated, _canonical_json_bytes(_manifest_payload(validated))


def load_continuation_manifest(path: str | os.PathLike[str]) -> ContinuationManifest | PolicyManifest:
    """Read and validate a manifest file for a registered protocol."""
    return _coerce_continuation_manifest(path)[0]


def load_candidate_manifest(
    path: str | os.PathLike[str],
) -> ContinuationManifest | PolicyManifest | PythonProgramManifest:
    """Read one strict candidate manifest and dispatch by schema."""
    manifest_path = Path(path)
    if manifest_path.name != "manifest.json":
        raise ValueError("policy manifest file must be named manifest.json")
    data = _read_regular_file(manifest_path, 64 * 1024, label="manifest.json")
    raw = _load_json_bytes(data, label="manifest.json")
    return validate_candidate_manifest(raw)


def canonical_manifest_payload(
    manifest: ContinuationManifest | PolicyManifest | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the normalized, JSON-ready manifest payload used in provenance.

    Callers should use this instead of reaching into protocol-specific
    dataclasses.  Unknown fields have already been rejected by validation.
    """
    validated = validate_continuation_manifest(manifest)
    if isinstance(validated, PolicyManifest):
        config = validated.inference_config
        return {
            "schema": validated.schema,
            "runner_type": validated.runner_type,
            "inference_config": {
                "input_dim": config.input_dim,
                "layers": list(config.layers),
                "activation": config.activation,
                "output_dim": config.output_dim,
                "output_clip": list(config.output_clip),
            },
            "output_semantics": validated.output_semantics,
            "normalization": validated.normalization,
            "weight_pattern": validated.weight_pattern,
        }
    return _manifest_payload(validated)


def canonical_candidate_manifest_payload(
    manifest: (
        ContinuationManifest
        | PolicyManifest
        | PythonProgramManifest
        | Mapping[str, Any]
    ),
) -> dict[str, Any]:
    """Return the normalized JSON representation of every admitted manifest."""
    validated = validate_candidate_manifest(manifest)
    if isinstance(validated, PythonProgramManifest):
        return {
            "schema": validated.schema,
            "interface": validated.interface,
        }
    return canonical_manifest_payload(validated)


def _resolved_dimension(manifest: ContinuationManifest, input_dim: int | None) -> int:
    declared = manifest.inference_config.input_dim
    if input_dim is None:
        if isinstance(declared, int):
            return declared
        raise ValueError("input_dim is required when manifest input_dim is n_assets")
    resolved = _strict_int(input_dim, path="input_dim", minimum=1, maximum=MAX_INPUT_DIM)
    if isinstance(declared, int) and declared != resolved:
        raise ValueError("input_dim does not match policy manifest.inference_config.input_dim")
    return resolved


def _validate_artifact_root(root: Path, expected_names: set[str]) -> None:
    try:
        info = os.lstat(root)
    except FileNotFoundError as exc:
        raise ValueError("policy artifact directory not found") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("policy artifact root must be a real directory")
    actual = {entry.name for entry in os.scandir(root)}
    missing = sorted(expected_names - actual)
    unexpected = sorted(actual - expected_names)
    if missing:
        raise ValueError(f"policy artifact is missing file(s): {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"policy artifact has unexpected file(s): {', '.join(unexpected)}")


def _bundle_hash(files: Mapping[str, bytes]) -> str:
    # Reuse the same domain-separated hash framing as the MLP artifact so all
    # protocol digests can be compared by the surrounding provenance layer.
    return _mlp_bundle_hash(files)


def _immutable_vector(values: Any, *, expected: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite vector of length {expected}")
    data = array.tobytes(order="C")
    return np.frombuffer(data, dtype=np.float64).reshape((expected,))


def load_linear_artifact(
    manifest: ContinuationManifest | Mapping[str, Any] | str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    *,
    n_exercise_times: int,
    input_dim: int | None = None,
) -> LinearPolicyArtifact:
    """Load a ``continuation-linear.v1`` per-instance artifact.

    Each ``step_XXX.npy`` contains ``input_dim`` coefficients followed by one
    intercept.  The normalization file uses the same ``steps`` shape as the
    MLP protocol.  Coefficients are applied after normalization and the result
    is already discounted to time zero.
    """
    validated, manifest_bytes = _coerce_continuation_manifest(manifest)
    if not isinstance(validated, ContinuationManifest) or validated.runner_type != LINEAR_RUNNER_TYPE:
        raise ValueError("manifest is not continuation-linear.v1")
    times = _strict_int(n_exercise_times, path="n_exercise_times", minimum=2, maximum=1000)
    steps = times - 1
    dimension = _resolved_dimension(validated, input_dim)
    expected_names = {"normalization.json", *(LINEAR_WEIGHT_PATTERN.format(i) for i in range(steps))}
    root = Path(artifact_dir)
    _validate_artifact_root(root, expected_names)
    files: dict[str, bytes] = {"manifest.json": manifest_bytes}
    total = len(manifest_bytes)
    normalization_bytes = _read_regular_file(
        root / "normalization.json",
        min(MAX_NORMALIZATION_BYTES, max(0, MAX_ARTIFACT_BUNDLE_BYTES - total)),
        label="normalization.json",
    )
    files["normalization.json"] = normalization_bytes
    total += len(normalization_bytes)
    normalizations = _validate_normalizations(
        normalization_bytes, step_count=steps, input_dim=dimension
    )
    expected_count = dimension + 1
    if expected_count > MAX_PARAMETERS_PER_STEP:
        raise ValueError("linear policy exceeds the per-step parameter limit")
    linear_steps: list[LinearStep] = []
    for index in range(steps):
        name = LINEAR_WEIGHT_PATTERN.format(index)
        remaining = MAX_ARTIFACT_BUNDLE_BYTES - total
        data = _read_regular_file(
            root / name,
            min(MAX_STEP_FILE_BYTES, max(0, remaining)),
            label=name,
        )
        files[name] = data
        total += len(data)
        flat = _load_canonical_flat_weights(data, expected_count=expected_count, label=name)
        coefficients = _immutable_vector(flat[:-1], expected=dimension, label=f"{name} coefficients")
        bias = float(flat[-1])
        linear_steps.append(LinearStep(coefficients=coefficients, bias=bias))
    return LinearPolicyArtifact(
        manifest=validated,
        input_dim=dimension,
        normalizations=normalizations,
        steps=tuple(linear_steps),
        file_sha256=tuple((name, hashlib.sha256(files[name]).hexdigest()) for name in sorted(files)),
        bundle_sha256=_bundle_hash(files),
    )


def _validate_expression_node(node: Any, *, path: str, depth: int, counter: list[int]) -> dict[str, Any]:
    counter[0] += 1
    if counter[0] > EXPRESSION_MAX_NODES:
        raise ValueError(f"{path} exceeds {EXPRESSION_MAX_NODES} AST nodes")
    if depth > EXPRESSION_MAX_DEPTH:
        raise ValueError(f"{path} exceeds maximum AST depth {EXPRESSION_MAX_DEPTH}")
    if not isinstance(node, dict):
        raise ValueError(f"{path} must be an expression object")
    op = node.get("op")
    if not isinstance(op, str):
        raise ValueError(f"{path}.op must be a string")
    if op == "constant":
        _strict_keys(node, required={"op", "value"}, allowed={"op", "value"}, path=path)
        value = _strict_number(node["value"], path=f"{path}.value")
        if not -EXPRESSION_MAX_CONSTANT <= value <= EXPRESSION_MAX_CONSTANT:
            raise ValueError(
                f"{path}.value must be in [-{EXPRESSION_MAX_CONSTANT}, {EXPRESSION_MAX_CONSTANT}]"
            )
        return {
            "op": op,
            "value": value,
        }
    if op == "spot":
        _strict_keys(node, required={"op", "asset"}, allowed={"op", "asset"}, path=path)
        return {
            "op": op,
            "asset": _strict_int(node["asset"], path=f"{path}.asset", minimum=0, maximum=3),
        }
    if op in EXPRESSION_TERMINALS:
        _strict_keys(node, required={"op"}, allowed={"op"}, path=path)
        return {"op": op}
    if op in EXPRESSION_UNARY_OPS:
        _strict_keys(node, required={"op", "arg"}, allowed={"op", "arg"}, path=path)
        return {
            "op": op,
            "arg": _validate_expression_node(
                node["arg"], path=f"{path}.arg", depth=depth + 1, counter=counter
            ),
        }
    if op in EXPRESSION_BINARY_OPS:
        _strict_keys(node, required={"op", "left", "right"}, allowed={"op", "left", "right"}, path=path)
        return {
            "op": op,
            "left": _validate_expression_node(
                node["left"], path=f"{path}.left", depth=depth + 1, counter=counter
            ),
            "right": _validate_expression_node(
                node["right"], path=f"{path}.right", depth=depth + 1, counter=counter
            ),
        }
    raise ValueError(f"{path}.op is not supported: {op}")


def _expression_json_bytes(expression: Mapping[str, Any]) -> bytes:
    return json.dumps(
        expression,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _context_attr(instance: Any, name: str, default: Any = None) -> Any:
    if instance is None:
        return default
    if isinstance(instance, Mapping):
        return instance.get(name, default)
    return getattr(instance, name, default)


def _instance_dimension(instance: Any, fallback: int) -> int:
    dimension = _context_attr(instance, "dimension", None)
    if dimension is None:
        spots = _context_attr(instance, "spots", None)
        dimension = len(spots) if spots is not None else fallback
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
        raise ValueError("instance dimension must be an integer")
    return int(dimension)


def _evaluate_expression(
    expression: Mapping[str, Any],
    *,
    time_index: int,
    state: np.ndarray,
    instance: Any,
) -> np.ndarray:
    """Evaluate one validated scalar expression over ``state`` rows."""
    leading_shape = state.shape[:-1]
    strike = float(_context_attr(instance, "strike", 1.0))
    if not math.isfinite(strike) or strike <= 0.0:
        raise ValueError("instance strike must be positive and finite")
    maturity = float(_context_attr(instance, "maturity", 1.0))
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("instance maturity must be positive and finite")
    exercise_times = _context_attr(instance, "exercise_times", None)
    if exercise_times is None:
        # A caller that only uses spot terminals can omit contract timing; the
        # normalized time terminals still have a useful zero default.
        exercise_time = 0.0
    else:
        if time_index >= len(exercise_times):
            raise ValueError("time_index is outside instance exercise grid")
        exercise_time = float(exercise_times[time_index])
    time_fraction = exercise_time / maturity
    payoff_type = _context_attr(instance, "payoff_type", "put")
    weights = _context_attr(instance, "weights", None)
    if weights is None:
        weights_array = np.full(state.shape[-1], 1.0 / state.shape[-1], dtype=np.float64)
    else:
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.shape != (state.shape[-1],) or not np.all(np.isfinite(weights_array)):
            raise ValueError("instance weights must match state dimension")

    def underlying() -> np.ndarray:
        if payoff_type == "put":
            return state[..., 0] / strike
        if payoff_type == "max_call":
            return np.max(state, axis=-1) / strike
        if payoff_type == "basket_put":
            return np.sum(state * weights_array, axis=-1) / strike
        raise ValueError(f"unsupported payoff_type: {payoff_type}")

    def intrinsic() -> np.ndarray:
        if payoff_type == "put":
            value = np.maximum(strike - state[..., 0], 0.0)
        elif payoff_type == "max_call":
            value = np.maximum(np.max(state, axis=-1) - strike, 0.0)
        elif payoff_type == "basket_put":
            value = np.maximum(strike - np.sum(state * weights_array, axis=-1), 0.0)
        else:
            raise ValueError(f"unsupported payoff_type: {payoff_type}")
        return value / strike

    def compute(node: Mapping[str, Any]) -> np.ndarray:
        op = node["op"]
        if op == "constant":
            result = np.full(leading_shape, float(node["value"]), dtype=np.float64)
        elif op == "time":
            result = np.full(leading_shape, time_fraction, dtype=np.float64)
        elif op == "time_to_maturity":
            result = np.full(leading_shape, 1.0 - time_fraction, dtype=np.float64)
        elif op == "spot":
            asset = int(node["asset"])
            if asset >= state.shape[-1]:
                raise ValueError(f"expression references unavailable asset {asset}")
            result = state[..., asset] / strike
        elif op == "mean_spot":
            result = np.mean(state, axis=-1) / strike
        elif op == "max_spot":
            result = np.max(state, axis=-1) / strike
        elif op == "min_spot":
            result = np.min(state, axis=-1) / strike
        elif op == "basket_spot":
            result = np.sum(state * weights_array, axis=-1) / strike
        elif op == "underlying":
            result = underlying()
        elif op == "intrinsic":
            result = intrinsic()
        elif op in EXPRESSION_UNARY_OPS:
            arg = compute(node["arg"])
            if op == "abs":
                result = np.abs(arg)
            elif op == "square":
                result = np.square(np.clip(arg, -1_000.0, 1_000.0))
            elif op == "cube":
                result = np.power(np.clip(arg, -100.0, 100.0), 3)
            elif op == "sqrt_abs":
                result = np.sqrt(np.abs(arg))
            elif op == "log1p_abs":
                result = np.log1p(np.abs(arg))
            elif op == "exp_neg_abs":
                result = np.exp(-np.abs(arg))
            else:
                result = 1.0 / (1.0 + np.abs(arg))
        else:
            left = compute(node["left"])
            right = compute(node["right"])
            if op == "add":
                result = left + right
            elif op == "subtract":
                result = left - right
            elif op == "multiply":
                result = np.clip(left, -1_000.0, 1_000.0) * np.clip(right, -1_000.0, 1_000.0)
            elif op == "divide_safe":
                denominator = np.where(
                    np.abs(right) < 1e-8,
                    np.where(right < 0.0, -1e-8, 1e-8),
                    right,
                )
                result = left / denominator
            elif op == "minimum":
                result = np.minimum(left, right)
            elif op == "maximum":
                result = np.maximum(left, right)
            else:  # pragma: no cover - validation makes this unreachable
                raise ValueError(f"unsupported expression op: {op}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"expression op {op} produced NaN or infinity")
        return np.clip(result, -EXPRESSION_MAX_ABS, EXPRESSION_MAX_ABS)

    return compute(expression)


def load_expression_artifact(
    manifest: ContinuationManifest | Mapping[str, Any] | str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    *,
    n_exercise_times: int,
    input_dim: int | None = None,
) -> ExpressionPolicyArtifact:
    """Load per-step bounded AST expressions from JSON files."""
    validated, manifest_bytes = _coerce_continuation_manifest(manifest)
    if not isinstance(validated, ContinuationManifest) or validated.runner_type != EXPRESSION_RUNNER_TYPE:
        raise ValueError("manifest is not continuation-expression.v1")
    times = _strict_int(n_exercise_times, path="n_exercise_times", minimum=2, maximum=1000)
    steps = times - 1
    dimension = _resolved_dimension(validated, input_dim)
    step_names = [EXPRESSION_WEIGHT_PATTERN.format(i) for i in range(steps)]
    # Expression normalization is genuinely optional.  The manifest records
    # the preferred mode, but a hand-written symbolic candidate may omit the
    # file even when the mode is ``per_step`` (and a generic exporter may emit
    # it when the mode is ``none``).  If present, it is validated and applied;
    # no other extra file is accepted.
    root = Path(artifact_dir)
    normalization_present = (root / "normalization.json").exists()
    expected_names = set(step_names)
    if normalization_present:
        expected_names.add("normalization.json")
    _validate_artifact_root(root, expected_names)
    files: dict[str, bytes] = {"manifest.json": manifest_bytes}
    total = len(manifest_bytes)
    if normalization_present:
        normalization_bytes = _read_regular_file(
            root / "normalization.json",
            min(MAX_NORMALIZATION_BYTES, max(0, MAX_ARTIFACT_BUNDLE_BYTES - total)),
            label="normalization.json",
        )
        files["normalization.json"] = normalization_bytes
        total += len(normalization_bytes)
        normalizations = _validate_normalizations(
            normalization_bytes, step_count=steps, input_dim=dimension
        )
    else:
        normalizations = tuple()
    expressions: list[dict[str, Any]] = []
    for index, name in enumerate(step_names):
        remaining = MAX_ARTIFACT_BUNDLE_BYTES - total
        data = _read_regular_file(
            root / name,
            min(MAX_STEP_FILE_BYTES, max(0, remaining)),
            label=name,
        )
        files[name] = data
        total += len(data)
        raw = _load_json_bytes(data, label=name)
        # The canonical file is the expression object itself.  Accepting a
        # single ``{"expression": ...}`` wrapper keeps hand-written candidate
        # exporters readable without adding another protocol file.
        if isinstance(raw, dict) and set(raw) == {"expression"}:
            raw = raw["expression"]
        counter = [0]
        expressions.append(
            _validate_expression_node(raw, path=f"{name}", depth=1, counter=counter)
        )
    return ExpressionPolicyArtifact(
        manifest=validated,
        input_dim=dimension,
        expressions=tuple(expressions),
        normalizations=normalizations,
        file_sha256=tuple((name, hashlib.sha256(files[name]).hexdigest()) for name in sorted(files)),
        bundle_sha256=_bundle_hash(files),
    )


def _validate_runner_states(states: Any, input_dim: int) -> tuple[np.ndarray, tuple[int, ...]]:
    raw = np.asarray(states)
    if (
        not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError("states must contain real numeric values")
    state = np.asarray(raw, dtype=np.float64)
    if state.ndim < 1 or state.shape[-1] != input_dim:
        raise ValueError(f"states must have shape (..., {input_dim})")
    if not np.all(np.isfinite(state)):
        raise ValueError("states must contain only finite values")
    return state, state.shape[:-1]


def _normalize_time_stats(stats: Any, state: np.ndarray, *, label: str) -> np.ndarray:
    if stats is None:
        return state
    mean = np.asarray(stats.mean, dtype=np.float64)
    scale = np.asarray(stats.scale, dtype=np.float64)
    if mean.shape != (state.shape[-1],) or scale.shape != (state.shape[-1],):
        raise ValueError(f"{label} normalization dimension mismatch")
    normalized = (state - mean) / scale
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{label} normalization produced non-finite states")
    return normalized


@dataclass(frozen=True, slots=True)
class LinearContinuationRunner:
    artifact: LinearPolicyArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, LinearPolicyArtifact):
            raise TypeError("artifact must be a validated LinearPolicyArtifact")

    def continuation(
        self,
        time_index: int,
        states: np.ndarray,
        instance: Any | None = None,
    ) -> np.ndarray:
        if isinstance(time_index, bool) or not isinstance(time_index, int):
            raise ValueError("time_index must be an integer")
        if not 0 <= time_index < len(self.artifact.steps):
            raise ValueError("time_index is outside the non-terminal exercise grid")
        state, leading_shape = _validate_runner_states(states, self.artifact.input_dim)
        state = _normalize_time_stats(
            self.artifact.normalizations[time_index], state, label="linear"
        )
        flat = state.reshape(-1, self.artifact.input_dim)
        step = self.artifact.steps[time_index]
        output = np.empty(flat.shape[0], dtype=np.float64)
        lower, upper = self.artifact.manifest.inference_config.output_clip
        with np.errstate(over="ignore", invalid="ignore"):
            for row_index, row in enumerate(flat):
                accumulator = np.float64(step.bias)
                for column_index, coefficient in enumerate(step.coefficients):
                    accumulator = np.float64(
                        accumulator + np.float64(row[column_index] * coefficient)
                    )
                # Overflow is handled by the protocol clip (as in the MLP
                # runner); NaN is not orderable and remains a hard failure.
                if np.isnan(accumulator):
                    raise ValueError("linear continuation produced NaN")
                output[row_index] = np.clip(accumulator, lower, upper)
        return output.reshape(leading_shape)


@dataclass(frozen=True, slots=True)
class ExpressionContinuationRunner:
    artifact: ExpressionPolicyArtifact
    instance: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ExpressionPolicyArtifact):
            raise TypeError("artifact must be a validated ExpressionPolicyArtifact")

    def continuation(
        self,
        time_index: int,
        states: np.ndarray,
        instance: Any | None = None,
    ) -> np.ndarray:
        if isinstance(time_index, bool) or not isinstance(time_index, int):
            raise ValueError("time_index must be an integer")
        if not 0 <= time_index < len(self.artifact.expressions):
            raise ValueError("time_index is outside the non-terminal exercise grid")
        state, leading_shape = _validate_runner_states(states, self.artifact.input_dim)
        context = instance if instance is not None else self.instance
        # A context is only mandatory for finance-aware terminals.  Supplying
        # a small default keeps constant/spot expressions convenient in unit
        # tests and for generic affine rules.
        if context is None:
            context = {"strike": 1.0, "maturity": 1.0, "exercise_times": tuple(range(len(self.artifact.expressions) + 1)), "payoff_type": "put"}
        if _instance_dimension(context, self.artifact.input_dim) != self.artifact.input_dim:
            raise ValueError("instance dimension does not match expression input_dim")
        if self.artifact.normalizations:
            state_for_expression = _normalize_time_stats(
                self.artifact.normalizations[time_index], state, label="expression"
            )
        else:
            state_for_expression = state
        values = _evaluate_expression(
            self.artifact.expressions[time_index],
            time_index=time_index,
            state=state_for_expression,
            instance=context,
        )
        # Expression terminals intentionally mirror the legacy Feature IR:
        # spot/underlying/intrinsic are normalized by strike and are expressed
        # at the current exercise date.  The common evaluator interface,
        # however, compares every runner against time-zero discounted payoff
        # in currency units.  Perform that unit conversion at this trusted
        # boundary so a pure logic expression remains financially coherent for
        # non-unit strikes and non-zero exercise dates.  MLP/linear runners
        # already emit the common t0-currency units and are not affected.
        strike = float(_context_attr(context, "strike", 1.0))
        rate = float(_context_attr(context, "rate", 0.0))
        if not math.isfinite(strike) or strike <= 0.0:
            raise ValueError("instance strike must be positive and finite")
        if not math.isfinite(rate):
            raise ValueError("instance rate must be finite")
        exercise_times = _context_attr(context, "exercise_times", None)
        if exercise_times is None:
            exercise_time = float(time_index)
        else:
            if time_index >= len(exercise_times):
                raise ValueError("time_index is outside instance exercise grid")
            exercise_time = float(exercise_times[time_index])
        if not math.isfinite(exercise_time):
            raise ValueError("instance exercise time must be finite")
        values = values * np.float64(strike * math.exp(-rate * exercise_time))
        lower, upper = self.artifact.manifest.inference_config.output_clip
        values = np.clip(values, lower, upper)
        return np.asarray(values, dtype=np.float64).reshape(leading_shape)


def _adapt_mlp_runner(runner: MLPContinuationRunner) -> ContinuationRunner:
    """Return an MLP runner with the common optional-instance call surface."""
    return runner


def load_continuation_runner(
    manifest: Any,
    artifact_dir: str | os.PathLike[str],
    *,
    n_exercise_times: int,
    input_dim: int | None = None,
    instance: Any | None = None,
) -> ContinuationRunner:
    """Dispatch a frozen artifact to the registered trusted runner.

    The dispatch is deliberately explicit.  A candidate cannot select a new
    runner by adding a field; the manifest schema and runner type must match a
    protocol implemented here.
    """
    validated, _ = _coerce_continuation_manifest(manifest)
    if isinstance(validated, PolicyManifest):
        artifact = load_policy_artifact(
            validated,
            artifact_dir,
            n_exercise_times=n_exercise_times,
            input_dim=input_dim,
        )
        return _adapt_mlp_runner(MLPContinuationRunner(artifact))
    if validated.runner_type == LINEAR_RUNNER_TYPE:
        return LinearContinuationRunner(
            load_linear_artifact(
                validated,
                artifact_dir,
                n_exercise_times=n_exercise_times,
                input_dim=input_dim,
            )
        )
    if validated.runner_type == EXPRESSION_RUNNER_TYPE:
        return ExpressionContinuationRunner(
            load_expression_artifact(
                validated,
                artifact_dir,
                n_exercise_times=n_exercise_times,
                input_dim=input_dim,
            ),
            instance=instance,
        )
    raise ValueError(f"unsupported continuation runner type: {validated.runner_type}")


# Friendly aliases used by callers that treat the artifact as a policy rather
# than a protocol-specific object.
load_policy_runner = load_continuation_runner
dispatch_policy_runner = load_continuation_runner


def export_linear_artifact(
    output_dir: str | os.PathLike[str],
    coefficients_by_step: Sequence[tuple[Any, Any] | Any],
    normalizations: Sequence[tuple[Any, Any] | Mapping[str, Any]],
) -> None:
    """Small candidate-side helper for emitting the canonical linear files.

    ``coefficients_by_step`` accepts either a coefficient vector (with an
    implicit zero bias) or ``(coefficients, bias)`` pairs.  The helper is not
    used by the trusted evaluator; it simply makes a Python ``train.py`` easy
    to write and test.
    """
    root = Path(output_dir)
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError("output_dir must be an existing empty directory")
    if len(coefficients_by_step) != len(normalizations):
        raise ValueError("linear steps and normalizations must have equal length")
    for index, value in enumerate(coefficients_by_step):
        # A two-element numeric list is a perfectly valid coefficient vector;
        # only treat a value as ``(coefficients, bias)`` when its first item is
        # itself an array-like vector and the second item is scalar.
        is_pair = (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and np.asarray(value[0]).ndim >= 1
            and np.asarray(value[1]).ndim == 0
        )
        if is_pair:
            coefficients, bias = value  # type: ignore[misc]
        else:
            coefficients, bias = value, 0.0
        coefficient_array = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(coefficient_array)) or not math.isfinite(float(bias)):
            raise ValueError("linear coefficients and bias must be finite")
        flat = np.concatenate((coefficient_array, np.asarray([float(bias)], dtype=np.float64)))
        np.save(root / LINEAR_WEIGHT_PATTERN.format(index), np.ascontiguousarray(flat), allow_pickle=False)
    steps_payload = []
    for item in normalizations:
        if isinstance(item, Mapping):
            mean, scale = item["mean"], item["scale"]
        else:
            mean, scale = item
        mean_array = np.asarray(mean, dtype=np.float64).reshape(-1)
        scale_array = np.asarray(scale, dtype=np.float64).reshape(-1)
        if (
            not np.all(np.isfinite(mean_array))
            or not np.all(np.isfinite(scale_array))
            or np.any(scale_array <= NORMALIZATION_EPSILON)
        ):
            raise ValueError("normalization means/scales must be finite with positive scales")
        steps_payload.append({"mean": mean_array.tolist(), "scale": scale_array.tolist()})
    (root / "normalization.json").write_text(
        json.dumps({"steps": steps_payload}, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )


def export_expression_artifact(
    output_dir: str | os.PathLike[str],
    expressions: Sequence[Mapping[str, Any]],
    *,
    normalizations: Sequence[tuple[Any, Any] | Mapping[str, Any]] | None = None,
) -> None:
    """Emit validated per-step expression JSON files for a candidate train.py."""
    root = Path(output_dir)
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError("output_dir must be an existing empty directory")
    if normalizations is not None and len(normalizations) != len(expressions):
        raise ValueError("expression steps and normalizations must have equal length")
    for index, expression in enumerate(expressions):
        normalized = _validate_expression_node(expression, path=f"expression[{index}]", depth=1, counter=[0])
        (root / EXPRESSION_WEIGHT_PATTERN.format(index)).write_bytes(_expression_json_bytes(normalized))
    if normalizations is not None:
        steps_payload = []
        for item in normalizations:
            if isinstance(item, Mapping):
                mean, scale = item["mean"], item["scale"]
            else:
                mean, scale = item
            mean_array = np.asarray(mean, dtype=np.float64).reshape(-1)
            scale_array = np.asarray(scale, dtype=np.float64).reshape(-1)
            if (
                not np.all(np.isfinite(mean_array))
                or not np.all(np.isfinite(scale_array))
                or np.any(scale_array <= NORMALIZATION_EPSILON)
            ):
                raise ValueError("normalization means/scales must be finite with positive scales")
            steps_payload.append({"mean": mean_array.tolist(), "scale": scale_array.tolist()})
        (root / "normalization.json").write_text(
            json.dumps({"steps": steps_payload}, sort_keys=True, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )


__all__ = [
    "OUTPUT_SEMANTICS",
    "PROTOCOL_OUTPUT_CLIP",
    "POLICY_SCHEMA",
    "ContinuationRunner",
    "ContinuationInferenceConfig",
    "ContinuationManifest",
    "PythonProgramManifest",
    "LINEAR_SCHEMA",
    "EXPRESSION_SCHEMA",
    "PYTHON_PROGRAM_SCHEMA",
    "LINEAR_RUNNER_TYPE",
    "EXPRESSION_RUNNER_TYPE",
    "PYTHON_PROGRAM_RUNNER_TYPE",
    "PYTHON_PROGRAM_INTERFACES",
    "LINEAR_WEIGHT_PATTERN",
    "EXPRESSION_WEIGHT_PATTERN",
    "NORMALIZATION_NONE",
    "NORMALIZATION_PER_STEP",
    "LinearStep",
    "LinearPolicyArtifact",
    "ExpressionPolicyArtifact",
    "LinearContinuationRunner",
    "ExpressionContinuationRunner",
    "validate_continuation_manifest",
    "load_continuation_manifest",
    "canonical_manifest_payload",
    "validate_python_program_manifest",
    "validate_candidate_manifest",
    "load_candidate_manifest",
    "canonical_candidate_manifest_payload",
    "load_linear_artifact",
    "load_expression_artifact",
    "load_continuation_runner",
    "load_policy_runner",
    "dispatch_policy_runner",
    "export_linear_artifact",
    "export_expression_artifact",
]
