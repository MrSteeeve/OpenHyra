#!/usr/bin/env python3
"""Evaluator for Bermudan feature baselines and whole Python policy programs.

The evaluator owns the Black--Scholes problem, path samples, payoff, stopping
application, scores, and hidden audit.  The open track supplies an executable
fit/predict algorithm while the historical tracks keep their data-only
continuation representations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping

import numpy as np

# ``sandbox._trusted_score`` may launch this file with a candidate/sandbox cwd,
# so the repository root is not guaranteed to be on ``sys.path``.  Insert it
# before importing evaluator-owned helper modules (not after them), which also
# keeps direct ``python tasks/.../evaluator.py`` invocation working.
_EVALUATOR_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_EVALUATOR_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVALUATOR_REPO_ROOT))

from feedback import (
    NOT_OBSERVED,
    DirectionalFeedback,
    FeedbackPacket,
    not_observed,
)


TASK_NAME = "bermudan_optimal_stopping"
TASK_PROTOCOL = "bermudan-lsmc-feature-ir.v1"
# The Python-training protocol is additive.  Keep the historical protocol
# constant above so old requests and archived records remain readable, while
# accepting the explicit bundle protocol used by the open algorithm track.
ALGORITHM_TASK_PROTOCOL = "bermudan-lsmc-python.v1"
ALGORITHM_BUNDLE_PROTOCOL = "bermudan-lsmc-algorithm-bundle.v1"
PYTHON_PROGRAM_TASK_PROTOCOL = "bermudan-python-program-search.v1"
SUPPORTED_TASK_PROTOCOLS = frozenset({
    TASK_PROTOCOL, ALGORITHM_TASK_PROTOCOL, ALGORITHM_BUNDLE_PROTOCOL,
    PYTHON_PROGRAM_TASK_PROTOCOL,
})
# The Python track is an additive task surface.  Keeping the legacy name in
# this module preserves archived requests, while the explicit alias lets the
# harness give Python-bundle runs their own run directory and provenance.
SUPPORTED_TASK_NAMES = frozenset({TASK_NAME, "bermudan_python_search"})
ALGORITHM_SOURCE_FILES = ("train.py", "manifest.json")
PYTHON_PROGRAM_SOURCE_FILES = ("algorithm.py", "manifest.json")
PYTHON_PROGRAM_SCHEMA = "openhyra-python-program.v1"
PYTHON_PROGRAM_SOURCE_MAX_FILES = 64
PYTHON_PROGRAM_SOURCE_MAX_ENTRIES = 128
PYTHON_PROGRAM_SOURCE_EXTENSIONS = frozenset({".py", ".json", ".toml"})
ALGORITHM_BUNDLE_SCHEMAS = frozenset({
    "openhyra-algorithm-bundle.v1",
    "openhyra-candidate-algorithm-bundle.v1",
})
# These fields are the executable/source declarations in the current
# AlgorithmBundle envelope.  Lineage fields are intentionally not required by
# this low-level evaluator: archived ``candidate-algorithm`` records predate
# the current envelope and remain readable, while the current v1 spelling is
# required to make its execution boundary explicit.
ALGORITHM_BUNDLE_DECLARATION_FIELDS = (
    "entrypoint",
    "source_files",
    "artifact_protocol",
)
FEATURE_SCHEMA = "openhyra-feature-program.v1"
REQUEST_SCHEMA = "openhyra-evaluation-request.v1"
EVIDENCE_SCHEMA = "openhyra-bermudan-evidence.v1"

# Candidate training is deliberately bounded per instance.  These defaults
# mirror the standalone training bridge; task/request configuration can lower
# them for a cheap smoke run, but cannot silently make them unbounded.
DEFAULT_TRAINING_TIMEOUT_S = 60.0
DEFAULT_TRAINING_MEMORY_BYTES = 1024 * 1024 * 1024
DEFAULT_TRAINING_FILE_SIZE_BYTES = 64 * 1024 * 1024

MAX_FEATURES = 16
MAX_AST_NODES = 128
MAX_AST_DEPTH = 8
MAX_ASSETS = 4
MAX_ABS_FEATURE = 1_000_000.0
MAX_CONSTANT = 10.0
MAX_SUITE_ID_CHARS = 96
MAX_REQUEST_SEED = 2**63 - 1

UNARY_OPS = {
    "abs",
    "square",
    "cube",
    "sqrt_abs",
    "log1p_abs",
    "exp_neg_abs",
    "reciprocal_one_plus_abs",
}
BINARY_OPS = {"add", "subtract", "multiply", "divide_safe", "minimum", "maximum"}
TERMINAL_OPS = {
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
PAYOFF_TYPES = {"put", "max_call", "basket_put"}


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_array(value: Any) -> str:
    """Hash a numeric array with dtype and shape framing for provenance."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _strict_keys(payload: Any, *, required: set[str], allowed: set[str], path: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be an object")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path} is missing field(s): {', '.join(missing)}")


def _strict_int(value: Any, *, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{path} must be in [{minimum}, {maximum}]")
    return value


def _strict_float(value: Any, *, path: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{path} must be finite and in [{minimum}, {maximum}]")
    return result


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _derive_seed(master_seed: int, *labels: Any) -> int:
    material = ":".join([str(master_seed), *(str(label) for label in labels)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class BSInstance:
    """One risk-neutral correlated Black--Scholes Bermudan instance."""

    instance_id: str
    payoff_type: str
    spots: tuple[float, ...]
    strike: float
    rate: float
    dividends: tuple[float, ...]
    volatilities: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    maturity: float
    exercise_times: tuple[float, ...]
    weights: tuple[float, ...] | None = None

    @property
    def dimension(self) -> int:
        return len(self.spots)

    def __post_init__(self) -> None:
        dimension = self.dimension
        if not 1 <= dimension <= MAX_ASSETS:
            raise ValueError(f"instance dimension must be in [1, {MAX_ASSETS}]")
        if self.payoff_type not in PAYOFF_TYPES:
            raise ValueError(f"unsupported payoff_type: {self.payoff_type}")
        if self.payoff_type == "put" and dimension != 1:
            raise ValueError("put instances must be one-dimensional")
        for name, values in (
            ("spots", self.spots),
            ("dividends", self.dividends),
            ("volatilities", self.volatilities),
        ):
            if len(values) != dimension:
                raise ValueError(f"{name} length must equal dimension")
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain finite values")
        if any(value <= 0.0 for value in self.spots):
            raise ValueError("spots must be positive")
        if any(value <= 0.0 for value in self.volatilities):
            raise ValueError("volatilities must be positive")
        if not math.isfinite(self.strike) or self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if not math.isfinite(self.rate):
            raise ValueError("rate must be finite")
        if not math.isfinite(self.maturity) or self.maturity <= 0.0:
            raise ValueError("maturity must be positive")
        if len(self.exercise_times) < 2:
            raise ValueError("exercise_times must include t=0 and maturity")
        times = np.asarray(self.exercise_times, dtype=float)
        if not np.all(np.isfinite(times)) or abs(times[0]) > 1e-14:
            raise ValueError("exercise_times must start at zero")
        if abs(times[-1] - self.maturity) > 1e-12:
            raise ValueError("exercise_times must end at maturity")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("exercise_times must be strictly increasing")
        corr = np.asarray(self.correlation, dtype=float)
        if corr.shape != (dimension, dimension):
            raise ValueError("correlation shape must equal dimension by dimension")
        if not np.all(np.isfinite(corr)) or not np.allclose(corr, corr.T, atol=1e-12):
            raise ValueError("correlation must be finite and symmetric")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-12):
            raise ValueError("correlation diagonal must be one")
        if float(np.linalg.eigvalsh(corr).min()) < -1e-10:
            raise ValueError("correlation must be positive semidefinite")
        if self.weights is not None:
            if len(self.weights) != dimension:
                raise ValueError("weights length must equal dimension")
            if any(not math.isfinite(value) or value < 0.0 for value in self.weights):
                raise ValueError("weights must be finite and nonnegative")
            if abs(sum(self.weights) - 1.0) > 1e-12:
                raise ValueError("weights must sum to one")
        if self.payoff_type == "basket_put" and self.weights is None:
            raise ValueError("basket_put requires weights")


def _uniform_times(maturity: float, exercise_count: int) -> tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(0.0, maturity, exercise_count + 1))


def public_suite() -> tuple[BSInstance, ...]:
    """Frozen public development suite used with common random numbers."""
    return (
        BSInstance(
            "public-put-atm", "put", (1.0,), 1.0, 0.05, (0.0,), (0.20,),
            ((1.0,),), 1.0, _uniform_times(1.0, 5),
        ),
        BSInstance(
            "public-put-high-vol", "put", (1.08,), 1.0, 0.02, (0.01,), (0.35,),
            ((1.0,),), 1.5, _uniform_times(1.5, 8),
        ),
        BSInstance(
            "public-max-call-2d", "max_call", (0.95, 1.02), 1.0, 0.04,
            (0.08, 0.10), (0.22, 0.28), ((1.0, 0.30), (0.30, 1.0)),
            1.0, _uniform_times(1.0, 5),
        ),
        BSInstance(
            "public-basket-put-3d", "basket_put", (0.98, 1.03, 1.0), 1.0,
            0.03, (0.01, 0.02, 0.015), (0.18, 0.24, 0.21),
            ((1.0, 0.25, 0.15), (0.25, 1.0, 0.20), (0.15, 0.20, 1.0)),
            1.25, _uniform_times(1.25, 6), (0.3, 0.4, 0.3),
        ),
    )


def derive_hidden_suite(seed: int, count: int) -> tuple[BSInstance, ...]:
    """Derive a private multi-product suite solely from the sealed audit seed."""
    rng = np.random.default_rng(_derive_seed(seed, "hidden-suite-v1"))
    products = ("put", "max_call", "basket_put")
    suite: list[BSInstance] = []
    for index in range(count):
        product = products[index % len(products)]
        dimension = 1 if product == "put" else (2 if product == "max_call" else 3)
        spots = tuple(float(value) for value in rng.uniform(0.84, 1.16, size=dimension))
        vols = tuple(float(value) for value in rng.uniform(0.16, 0.42, size=dimension))
        dividends = tuple(float(value) for value in rng.uniform(0.0, 0.11, size=dimension))
        rho = float(rng.uniform(0.05, 0.48))
        corr = tuple(tuple(1.0 if i == j else rho for j in range(dimension)) for i in range(dimension))
        maturity = float(rng.uniform(0.65, 1.8))
        exercise_count = int(rng.integers(5, 10))
        weights = None
        if product == "basket_put":
            raw_weights = rng.uniform(0.5, 1.5, size=dimension)
            weights = tuple(float(value) for value in raw_weights / raw_weights.sum())
        suite.append(BSInstance(
            instance_id=f"hidden-{index:03d}",
            payoff_type=product,
            spots=spots,
            strike=1.0,
            rate=float(rng.uniform(0.005, 0.075)),
            dividends=dividends,
            volatilities=vols,
            correlation=corr,
            maturity=maturity,
            exercise_times=_uniform_times(maturity, exercise_count),
            weights=weights,
        ))
    return tuple(suite)


def _correlation_root(instance: BSInstance) -> np.ndarray:
    corr = np.asarray(instance.correlation, dtype=float)
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(corr)
        return vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))


def simulate_paths(
    instance: BSInstance,
    n_paths: int,
    rng: np.random.Generator | int,
) -> np.ndarray:
    """Simulate exact-grid correlated GBM paths under the risk-neutral measure."""
    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths <= 0:
        raise ValueError("n_paths must be a positive integer")
    times = np.asarray(instance.exercise_times, dtype=float)
    dt = np.diff(times)
    dimension = instance.dimension
    independent = rng.standard_normal((n_paths, len(dt), dimension))
    shocks = independent @ _correlation_root(instance).T
    vol = np.asarray(instance.volatilities, dtype=float)
    dividend = np.asarray(instance.dividends, dtype=float)
    drift = (instance.rate - dividend - 0.5 * vol**2)[None, None, :] * dt[None, :, None]
    diffusion = vol[None, None, :] * np.sqrt(dt)[None, :, None] * shocks
    log_steps = drift + diffusion
    paths = np.empty((n_paths, len(times), dimension), dtype=float)
    paths[:, 0, :] = np.asarray(instance.spots, dtype=float)
    paths[:, 1:, :] = paths[:, :1, :] * np.exp(np.cumsum(log_steps, axis=1))
    return paths


def simulate_conditional_next(
    previous_states: np.ndarray,
    dt: float,
    instance: BSInstance,
    n_inner: int,
    rng: np.random.Generator | int,
) -> np.ndarray:
    """Draw iid one-step successors conditional on each supplied state."""
    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    previous = np.asarray(previous_states, dtype=float)
    if previous.ndim != 2 or previous.shape[1] != instance.dimension:
        raise ValueError("previous_states has invalid shape")
    if dt <= 0.0 or n_inner <= 0:
        raise ValueError("dt and n_inner must be positive")
    normal = rng.standard_normal((previous.shape[0], n_inner, instance.dimension))
    shocks = normal @ _correlation_root(instance).T
    vol = np.asarray(instance.volatilities, dtype=float)
    dividend = np.asarray(instance.dividends, dtype=float)
    drift = (instance.rate - dividend - 0.5 * vol**2) * dt
    diffusion = vol * math.sqrt(dt) * shocks
    return previous[:, None, :] * np.exp(drift[None, None, :] + diffusion)


def payoff(states: np.ndarray, instance: BSInstance) -> np.ndarray:
    """Evaluator-owned non-discounted payoff on arbitrary leading dimensions."""
    state = np.asarray(states, dtype=float)
    if state.shape[-1] != instance.dimension:
        raise ValueError("state dimension does not match instance")
    if instance.payoff_type == "put":
        return np.maximum(instance.strike - state[..., 0], 0.0)
    if instance.payoff_type == "max_call":
        return np.maximum(np.max(state, axis=-1) - instance.strike, 0.0)
    weights = np.asarray(instance.weights, dtype=float)
    return np.maximum(instance.strike - np.sum(state * weights, axis=-1), 0.0)


def discounted_rewards(paths: np.ndarray, instance: BSInstance) -> np.ndarray:
    state_paths = np.asarray(paths, dtype=float)
    if state_paths.ndim != 3 or state_paths.shape[1] != len(instance.exercise_times):
        raise ValueError("paths have invalid shape")
    discounts = np.exp(-instance.rate * np.asarray(instance.exercise_times, dtype=float))
    return payoff(state_paths, instance) * discounts[None, :]


def black_scholes_european_price(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    option_type: str = "put",
) -> float:
    """Analytic Black--Scholes European call or put price."""
    if min(spot, strike, volatility, maturity) <= 0.0:
        raise ValueError("spot, strike, volatility, and maturity must be positive")
    if option_type not in {"put", "call"}:
        raise ValueError("option_type must be put or call")
    normal = NormalDist()
    root_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * maturity) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    call = spot * math.exp(-dividend * maturity) * normal.cdf(d1) - strike * math.exp(-rate * maturity) * normal.cdf(d2)
    if option_type == "call":
        return call
    return call - spot * math.exp(-dividend * maturity) + strike * math.exp(-rate * maturity)


def crr_price(
    instance: BSInstance,
    steps: int = 800,
    *,
    exercise: str = "bermudan",
) -> float:
    """CRR value for a one-dimensional put; used as an independent oracle."""
    if instance.payoff_type != "put" or instance.dimension != 1:
        raise ValueError("CRR reference currently supports one-dimensional puts")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer >= 2")
    if exercise not in {"european", "bermudan", "american"}:
        raise ValueError("exercise must be european, bermudan, or american")
    dt = instance.maturity / steps
    sigma = instance.volatilities[0]
    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((instance.rate - instance.dividends[0]) * dt)
    probability = (growth - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("CRR risk-neutral probability is outside [0, 1]")
    indices = np.arange(steps + 1)
    terminal_spots = instance.spots[0] * up**indices * down ** (steps - indices)
    values = np.maximum(instance.strike - terminal_spots, 0.0)
    exercise_steps = {
        int(round(time / instance.maturity * steps)) for time in instance.exercise_times
    }
    discount = math.exp(-instance.rate * dt)
    for step in range(steps - 1, -1, -1):
        values = discount * (probability * values[1:] + (1.0 - probability) * values[:-1])
        may_exercise = exercise == "american" or (exercise == "bermudan" and step in exercise_steps)
        if may_exercise:
            node_indices = np.arange(step + 1)
            node_spots = instance.spots[0] * up**node_indices * down ** (step - node_indices)
            values = np.maximum(values, np.maximum(instance.strike - node_spots, 0.0))
    return float(values[0])


def validate_feature_program(raw: Any) -> dict[str, Any]:
    """Validate and normalize the bounded typed feature-expression IR."""
    _strict_keys(raw, required={"schema", "features"}, allowed={"schema", "features"}, path="feature program")
    if raw["schema"] != FEATURE_SCHEMA:
        raise ValueError(f"feature program schema must be {FEATURE_SCHEMA}")
    features = raw["features"]
    if not isinstance(features, list) or not 1 <= len(features) <= MAX_FEATURES:
        raise ValueError(f"features must contain between 1 and {MAX_FEATURES} expressions")
    node_count = 0

    def visit(node: Any, path: str, depth: int) -> dict[str, Any]:
        nonlocal node_count
        node_count += 1
        if node_count > MAX_AST_NODES:
            raise ValueError(f"feature program exceeds {MAX_AST_NODES} AST nodes")
        if depth > MAX_AST_DEPTH:
            raise ValueError(f"{path} exceeds maximum AST depth {MAX_AST_DEPTH}")
        if not isinstance(node, dict):
            raise ValueError(f"{path} must be an expression object")
        op = node.get("op")
        if not isinstance(op, str):
            raise ValueError(f"{path}.op must be a string")
        if op == "constant":
            _strict_keys(node, required={"op", "value"}, allowed={"op", "value"}, path=path)
            return {"op": op, "value": _strict_float(node["value"], path=f"{path}.value", minimum=-MAX_CONSTANT, maximum=MAX_CONSTANT)}
        if op == "spot":
            _strict_keys(node, required={"op", "asset"}, allowed={"op", "asset"}, path=path)
            return {"op": op, "asset": _strict_int(node["asset"], path=f"{path}.asset", minimum=0, maximum=MAX_ASSETS - 1)}
        if op in TERMINAL_OPS:
            _strict_keys(node, required={"op"}, allowed={"op"}, path=path)
            return {"op": op}
        if op in UNARY_OPS:
            _strict_keys(node, required={"op", "arg"}, allowed={"op", "arg"}, path=path)
            return {"op": op, "arg": visit(node["arg"], f"{path}.arg", depth + 1)}
        if op in BINARY_OPS:
            _strict_keys(node, required={"op", "left", "right"}, allowed={"op", "left", "right"}, path=path)
            return {
                "op": op,
                "left": visit(node["left"], f"{path}.left", depth + 1),
                "right": visit(node["right"], f"{path}.right", depth + 1),
            }
        raise ValueError(f"{path}.op is not supported: {op}")

    normalized = [visit(node, f"features[{index}]", 1) for index, node in enumerate(features)]
    return {"schema": FEATURE_SCHEMA, "features": normalized}


def _underlying(states: np.ndarray, instance: BSInstance) -> np.ndarray:
    if instance.payoff_type == "put":
        return states[..., 0]
    if instance.payoff_type == "max_call":
        return np.max(states, axis=-1)
    return np.sum(states * np.asarray(instance.weights, dtype=float), axis=-1)


def evaluate_features(
    program: dict[str, Any],
    time_index: int,
    states: np.ndarray,
    instance: BSInstance,
) -> np.ndarray:
    """Vectorized, total evaluator for an already validated feature program."""
    state = np.asarray(states, dtype=float)
    if state.ndim < 2 or state.shape[-1] != instance.dimension:
        raise ValueError("states must have shape (..., dimension)")
    if not 0 <= time_index < len(instance.exercise_times):
        raise ValueError("time_index is outside exercise grid")
    leading_shape = state.shape[:-1]
    strike = instance.strike
    time_fraction = instance.exercise_times[time_index] / instance.maturity

    def compute(node: dict[str, Any]) -> np.ndarray:
        op = node["op"]
        if op == "constant":
            result = np.full(leading_shape, node["value"], dtype=float)
        elif op == "time":
            result = np.full(leading_shape, time_fraction, dtype=float)
        elif op == "time_to_maturity":
            result = np.full(leading_shape, 1.0 - time_fraction, dtype=float)
        elif op == "spot":
            asset = node["asset"]
            if asset >= instance.dimension:
                raise ValueError(f"feature references unavailable asset {asset} for dimension {instance.dimension}")
            result = state[..., asset] / strike
        elif op == "mean_spot":
            result = np.mean(state, axis=-1) / strike
        elif op == "max_spot":
            result = np.max(state, axis=-1) / strike
        elif op == "min_spot":
            result = np.min(state, axis=-1) / strike
        elif op == "basket_spot":
            weights = np.asarray(instance.weights if instance.weights is not None else tuple([1.0 / instance.dimension] * instance.dimension))
            result = np.sum(state * weights, axis=-1) / strike
        elif op == "underlying":
            result = _underlying(state, instance) / strike
        elif op == "intrinsic":
            result = payoff(state, instance) / strike
        elif op in UNARY_OPS:
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
        elif op in BINARY_OPS:
            left, right = compute(node["left"]), compute(node["right"])
            if op == "add":
                result = left + right
            elif op == "subtract":
                result = left - right
            elif op == "multiply":
                result = np.clip(left, -1_000.0, 1_000.0) * np.clip(right, -1_000.0, 1_000.0)
            elif op == "divide_safe":
                denominator = np.where(np.abs(right) < 1e-8, np.where(right < 0.0, -1e-8, 1e-8), right)
                result = left / denominator
            elif op == "minimum":
                result = np.minimum(left, right)
            else:
                result = np.maximum(left, right)
        else:  # pragma: no cover - validator makes the interpreter total
            raise ValueError(f"unsupported feature op: {op}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"feature op {op} produced NaN or infinity")
        return np.clip(result, -MAX_ABS_FEATURE, MAX_ABS_FEATURE)

    columns = [compute(node).reshape(-1) for node in program["features"]]
    return np.column_stack(columns).reshape(*leading_shape, len(columns))


BASELINE_PROGRAM = validate_feature_program({
    "schema": FEATURE_SCHEMA,
    "features": [
        {"op": "underlying"},
        {"op": "square", "arg": {"op": "underlying"}},
        {"op": "cube", "arg": {"op": "underlying"}},
        {"op": "intrinsic"},
    ],
})


@dataclass(frozen=True)
class RidgeStep:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class FrozenPolicy:
    program: dict[str, Any]
    instance: BSInstance
    steps: tuple[RidgeStep, ...]
    ridge_alpha: float

    def continuation(self, time_index: int, states: np.ndarray) -> np.ndarray:
        features = evaluate_features(self.program, time_index, states, self.instance)
        model = self.steps[time_index]
        flat = features.reshape(-1, features.shape[-1])
        standardized = (flat - model.mean) / model.scale
        design = np.column_stack((np.ones(flat.shape[0]), standardized))
        prediction = design @ model.coefficients
        return prediction.reshape(features.shape[:-1])

    def approximate_value(self, time_index: int, states: np.ndarray) -> np.ndarray:
        immediate = payoff(states, self.instance) * math.exp(-self.instance.rate * self.instance.exercise_times[time_index])
        if time_index == len(self.instance.exercise_times) - 1:
            return immediate
        return np.maximum(immediate, self.continuation(time_index, states))


@dataclass(frozen=True)
class TrustedRunnerPolicy:
    """Bind a protocol runner to one evaluator-owned contract instance.

    ``policy_artifact.MLPContinuationRunner`` intentionally knows nothing
    about payoffs or exercise dates.  The stopping evaluator does.  This
    small adapter gives the existing ``apply_policy`` and dual code the same
    interface as :class:`FrozenPolicy`, while keeping all decisions in this
    module.  Candidate code is never imported here; only the data-only runner
    object returned by the training bridge is retained.
    """

    runner: Any
    instance: BSInstance
    runner_type: str = "mlp"
    policy_interface: str = "continuation"
    policy_artifact_sha256: str | None = None

    def continuation(
        self,
        time_index: int,
        states: np.ndarray,
        *,
        history: np.ndarray | None = None,
        immediate_payoffs: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.policy_interface != "continuation":
            raise ValueError("policy exposes direct decisions, not continuation")
        if self.runner_type == "python_program":
            values = self.runner.continuation(
                time_index,
                states,
                self.instance,
                history=history,
                immediate_payoffs=immediate_payoffs,
            )
        else:
            values = self.runner.continuation(time_index, states, self.instance)
        result = np.asarray(values, dtype=np.float64)
        expected_shape = np.asarray(states).shape[:-1]
        if result.shape != expected_shape:
            raise ValueError(
                "trusted runner returned shape "
                f"{result.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("trusted runner returned NaN or infinity")
        # The protocol runner normally performs this clip itself.  Retaining
        # the bound at the adapter boundary protects future registered
        # runners and keeps the stopping/dual arithmetic total.
        return np.clip(result, -1_000_000.0, 1_000_000.0)

    def decision(
        self,
        time_index: int,
        states: np.ndarray,
        *,
        history: np.ndarray,
        immediate_payoffs: np.ndarray,
    ) -> np.ndarray:
        if self.policy_interface != "decision":
            raise ValueError("policy exposes continuation, not direct decisions")
        values = self.runner.decision(
            time_index,
            states,
            self.instance,
            history=history,
            immediate_payoffs=immediate_payoffs,
        )
        result = np.asarray(values)
        expected_shape = np.asarray(states).shape[:-1]
        if result.shape != expected_shape:
            raise ValueError(
                "trusted runner returned decision shape "
                f"{result.shape}, expected {expected_shape}"
            )
        if result.dtype != np.dtype(np.bool_):
            raise ValueError("trusted runner decisions must have boolean dtype")
        return result

    def approximate_value(self, time_index: int, states: np.ndarray) -> np.ndarray:
        if self.policy_interface != "continuation":
            raise ValueError("direct-decision policy has no continuation value")
        immediate = payoff(states, self.instance) * math.exp(
            -self.instance.rate * self.instance.exercise_times[time_index]
        )
        if time_index == len(self.instance.exercise_times) - 1:
            return immediate
        return np.maximum(immediate, self.continuation(time_index, states))


def _fit_ridge(features: np.ndarray, target: np.ndarray, ridge_alpha: float) -> RidgeStep:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] == 0:
        raise ValueError("ridge inputs have invalid shape")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(x.shape[0]), standardized))
    gram = design.T @ design / x.shape[0]
    penalty = np.eye(design.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    rhs = design.T @ y / x.shape[0]
    try:
        coefficients = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("ridge fit produced non-finite coefficients")
    return RidgeStep(mean=mean, scale=scale, coefficients=coefficients)


def fit_lsmc(
    program: dict[str, Any],
    instance: BSInstance,
    training_paths: np.ndarray,
    *,
    ridge_alpha: float = 1e-6,
) -> FrozenPolicy:
    """Fit the fixed evaluator-owned Ridge LSMC and freeze its policy."""
    if not 0.0 < ridge_alpha <= 1.0:
        raise ValueError("ridge_alpha must be in (0, 1]")
    paths = np.asarray(training_paths, dtype=float)
    rewards = discounted_rewards(paths, instance)
    cashflow = rewards[:, -1].copy()
    step_models: list[RidgeStep | None] = [None] * (rewards.shape[1] - 1)
    for time_index in range(rewards.shape[1] - 2, -1, -1):
        current_payoff = rewards[:, time_index]
        eligible = current_payoff > 0.0
        # If no path is in the money, a constant continuation estimate remains
        # well-defined and the exercise guard below forbids zero-payoff exercise.
        fit_mask = eligible if int(eligible.sum()) >= 2 else np.ones(len(paths), dtype=bool)
        x = evaluate_features(program, time_index, paths[fit_mask, time_index, :], instance)
        model = _fit_ridge(x, cashflow[fit_mask], ridge_alpha)
        step_models[time_index] = model
        if np.any(eligible):
            eligible_x = evaluate_features(program, time_index, paths[eligible, time_index, :], instance)
            flat = (eligible_x - model.mean) / model.scale
            continuation = np.column_stack((np.ones(flat.shape[0]), flat)) @ model.coefficients
            exercise_indices = np.flatnonzero(eligible)[current_payoff[eligible] >= continuation]
            cashflow[exercise_indices] = current_payoff[exercise_indices]
    return FrozenPolicy(program=program, instance=instance, steps=tuple(step_models), ridge_alpha=ridge_alpha)  # type: ignore[arg-type]


def apply_policy(policy: Any, paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Execute a frozen causal policy and return discounted payoffs and stops."""
    state_paths = np.asarray(paths, dtype=float)
    rewards = discounted_rewards(state_paths, policy.instance)
    n_paths, n_times = rewards.shape
    realized = np.empty(n_paths, dtype=float)
    stopping_times = np.full(n_paths, n_times - 1, dtype=int)
    alive = np.ones(n_paths, dtype=bool)
    for time_index in range(n_times - 1):
        indices = np.flatnonzero(alive)
        if not len(indices):
            break
        immediate = rewards[indices, time_index]
        current_states = state_paths[indices, time_index, :]
        history = state_paths[indices, : time_index + 1, :]
        if getattr(policy, "policy_interface", "continuation") == "decision":
            requested = policy.decision(
                time_index,
                current_states,
                history=history,
                immediate_payoffs=immediate,
            )
            # Zero-payoff exercise is evaluator-owned and always dominated by
            # continuing to the next date, even for a direct decision program.
            exercise = (immediate > 0.0) & requested
        elif getattr(policy, "runner_type", "") == "python_program":
            continuation = policy.continuation(
                time_index,
                current_states,
                history=history,
                immediate_payoffs=immediate,
            )
            exercise = (immediate > 0.0) & (immediate >= continuation)
        else:
            continuation = policy.continuation(time_index, current_states)
            exercise = (immediate > 0.0) & (immediate >= continuation)
        chosen = indices[exercise]
        realized[chosen] = immediate[exercise]
        stopping_times[chosen] = time_index
        alive[chosen] = False
    remaining = np.flatnonzero(alive)
    realized[remaining] = rewards[remaining, -1]
    return realized, stopping_times


def dual_upper_bound_samples(
    policy: FrozenPolicy,
    outer_paths: np.ndarray,
    *,
    inner_paths: int,
    inner_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid nested dual samples and the martingale terminal values.

    For m >= 1, the increment is
      f_m(S_m^outer) - mean_b f_m(S_m^(inner,b)),
    where outer and all inner successors are iid conditional on S_{m-1}.
    Therefore each increment has conditional mean zero (including finite B),
    M_0=0, and E[max_m(Z_m-M_m)] is a valid Rogers/Haugh-Kogan upper bound.
    """
    paths = np.asarray(outer_paths, dtype=float)
    rewards = discounted_rewards(paths, policy.instance)
    n_outer, n_times = rewards.shape
    martingale = np.zeros(n_outer, dtype=float)
    adjusted = np.empty_like(rewards)
    adjusted[:, 0] = rewards[:, 0]  # M_0 = 0 exactly.
    rng = np.random.default_rng(inner_seed)
    times = np.asarray(policy.instance.exercise_times, dtype=float)
    for time_index in range(1, n_times):
        outer_value = policy.approximate_value(time_index, paths[:, time_index, :])
        inner_states = simulate_conditional_next(
            paths[:, time_index - 1, :],
            float(times[time_index] - times[time_index - 1]),
            policy.instance,
            inner_paths,
            rng,
        )
        inner_value = policy.approximate_value(time_index, inner_states)
        increment = outer_value - inner_value.mean(axis=1)
        martingale += increment
        adjusted[:, time_index] = rewards[:, time_index] - martingale
    return np.max(adjusted, axis=1), martingale


def _mean_se(samples: np.ndarray) -> tuple[float, float]:
    values = np.asarray(samples, dtype=float).reshape(-1)
    if not len(values):
        raise ValueError("cannot summarize empty samples")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, se


def _policy_behavior_metrics(
    values: np.ndarray,
    stopping_times: np.ndarray,
    n_times: int,
) -> dict[str, Any]:
    """Return a small, evaluator-owned descriptor of a frozen policy.

    The descriptor is deliberately about the *observed policy outcome* on the
    independent pricing paths, not about candidate internals.  ``exercise_rate``
    means the fraction of paths exercised before maturity (maturity settlement
    is excluded), which gives BehaviorProfile a useful scalar instead of the
    trivial fraction that would result from counting every terminal stop.
    ``exercise_rate_by_time`` and the normalized stopping-time moments retain a
    little more geometry for mechanism comparisons while remaining compact.
    """
    rewards = np.asarray(values, dtype=float).reshape(-1)
    stops = np.asarray(stopping_times).reshape(-1)
    if rewards.size == 0 or stops.size == 0 or rewards.size != stops.size:
        raise ValueError("policy behavior samples must be non-empty and aligned")
    if n_times < 2:
        raise ValueError("n_times must be at least two")

    # ``apply_policy`` returns integer, in-range stopping indices.  Keep the
    # descriptor total for direct callers as well: invalid indices are marked
    # in the finite/valid flag and omitted from the histogram rather than
    # causing a secondary metrics failure.
    finite_values = bool(np.all(np.isfinite(rewards)))
    integer_stops = np.issubdtype(stops.dtype, np.integer)
    finite_stops = bool(np.all(np.isfinite(stops))) if np.issubdtype(stops.dtype, np.number) else False
    valid_stops = (
        integer_stops
        and finite_stops
        and bool(np.all((stops >= 0) & (stops < n_times)))
    )
    if valid_stops:
        integer_indices = stops.astype(np.int64, copy=False)
        counts = np.bincount(integer_indices, minlength=n_times)[:n_times]
        rates = counts.astype(float) / float(stops.size)
        normalized_stops = integer_indices.astype(float) / float(n_times - 1)
        exercise_rate = float(np.mean(integer_indices < n_times - 1))
        stop_time_mean = float(np.mean(normalized_stops))
        stop_time_std = float(np.std(normalized_stops))
    else:
        # This branch is only defensive telemetry; the trusted policy path
        # should never produce invalid indices.  Returning finite placeholders
        # keeps diagnostics serializable if a future runner violates the
        # interface and lets its validity flag explain the issue.
        rates = np.zeros(n_times, dtype=float)
        exercise_rate = 0.0
        stop_time_mean = 0.0
        stop_time_std = 0.0

    return {
        "exercise_rate": exercise_rate,
        "exercise_rate_by_time": rates.tolist(),
        "stop_time_mean": stop_time_mean,
        "stop_time_std": stop_time_std,
        "finite": bool(finite_values and valid_stops),
        "valid_stop_rate": 1.0 if valid_stops else 0.0,
    }


def _aggregate_behavior_metrics(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate compact per-cell behavior descriptors by instance.

    Repeated cells are averaged per instance.  The standard deviation of the
    exercise rate across repeats is retained as a lightweight stability signal
    for the V5 behavior profile; with one repeat it is exactly zero.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        instance_id = summary.get("instance_id")
        if isinstance(instance_id, str) and instance_id:
            grouped.setdefault(instance_id, []).append(summary)

    def numeric_by_instance(field: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for instance_id in sorted(grouped):
            values = [float(row[field]) for row in grouped[instance_id] if field in row]
            if values:
                result[instance_id] = float(np.mean(values))
        return result

    def std_by_instance(field: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for instance_id in sorted(grouped):
            values = [float(row[field]) for row in grouped[instance_id] if field in row]
            if values:
                result[instance_id] = float(np.std(values))
        return result

    finite_rates: dict[str, float] = {}
    for instance_id in sorted(grouped):
        flags = [bool(row.get("candidate_finite", False)) for row in grouped[instance_id]]
        finite_rates[instance_id] = float(np.mean(flags)) if flags else 0.0

    exercise_rates = numeric_by_instance("candidate_exercise_rate")
    result: dict[str, Any] = {
        # These score projections are useful to callers that do not want to
        # re-parse summaries (including the V5 adapter).  They preserve the
        # historical, unnormalized lower-bound units; normalized improvements
        # remain available in each summary and below.
        "per_instance_scores": numeric_by_instance("candidate_lower_bound"),
        "baseline_scores": numeric_by_instance("baseline_lower_bound"),
        "per_instance_normalized_improvements": numeric_by_instance(
            "paired_normalized_improvement"
        ),
        "per_instance_exercise_rates": exercise_rates,
        "baseline_exercise_rates": numeric_by_instance("baseline_exercise_rate"),
        "per_instance_exercise_rate_std": std_by_instance(
            "candidate_exercise_rate"
        ),
        "baseline_exercise_rate_std": std_by_instance("baseline_exercise_rate"),
        "per_instance_finite_rates": finite_rates,
        "behavior_finite": bool(
            summaries
            and all(bool(row.get("candidate_finite", False)) for row in summaries)
        ),
        "behavior_exercise_rate_mean": (
            float(np.mean(list(exercise_rates.values()))) if exercise_rates else 0.0
        ),
        "behavior_exercise_rate_std": (
            float(np.std(list(exercise_rates.values()))) if exercise_rates else 0.0
        ),
        "behavior_descriptor": {
            "exercise_rate": "fraction_of_pricing_paths_stopped_before_maturity",
            "stability": "standard_deviation_of_exercise_rate_across_repeats",
            "finite": "all_candidate_values_and_stopping_indices_finite_and_valid",
        },
    }
    return result


def _feedback_direction(value: Any, *, lower_is_better: bool = False) -> str:
    """Map an observed effect to a deliberately coarse direction label."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "uncertain"
    value = float(value)
    if not math.isfinite(value) or abs(value) <= 1e-15:
        return "uncertain"
    if lower_is_better:
        value = -value
    return "positive" if value > 0.0 else "negative"


def _feedback_marker(reason: str) -> dict[str, str]:
    """Short local spelling keeps the projection readable at call sites."""

    return not_observed(reason)


def _build_domain_feedback_packet(
    *,
    stage: str,
    task: str,
    suite_id: str,
    request: Mapping[str, Any],
    score: float,
    summaries: list[dict[str, Any]],
    confidence_level: float,
    aggregate_effect: float | None = None,
    aggregate_standard_error: float | None = None,
    runtime_seconds: float | None = None,
    candidate_id: str = "",
) -> FeedbackPacket:
    """Project trusted Bermudan outcomes into a reusable feedback packet.

    This is a sidecar projection: it reads values already produced by the
    evaluator and never participates in score calculation or candidate
    validation.  Missing probes are explicit ``not_observed`` markers rather
    than zero-filled pseudo-measurements.
    """

    if stage not in {"search", "audit"}:
        raise ValueError("feedback stage must be search or audit")
    if aggregate_effect is None:
        if stage == "search":
            aggregate_effect = float(np.mean([
                float(row["paired_normalized_improvement"])
                for row in summaries
                if isinstance(row.get("paired_normalized_improvement"), (int, float))
            ])) if summaries else None
        else:
            aggregate_effect = float(-np.mean([
                float(row["normalized_primal_dual_confidence_gap"])
                for row in summaries
                if isinstance(row.get("normalized_primal_dual_confidence_gap"), (int, float))
            ])) if summaries else None

    finite_flags = [
        bool(row.get("candidate_finite"))
        for row in summaries
        if "candidate_finite" in row
    ]
    failure_rate: Any = (
        float(1.0 - np.mean(finite_flags)) if finite_flags else
        _feedback_marker("candidate validity was not emitted for this stage")
    )
    runtime_value: Any = (
        float(runtime_seconds) if runtime_seconds is not None and math.isfinite(float(runtime_seconds))
        else _feedback_marker("stage runtime is filled by the outer evaluator")
    )
    training_seconds = [
        float(row["candidate_training_seconds"])
        for row in summaries
        if isinstance(row.get("candidate_training_seconds"), (int, float))
        and math.isfinite(float(row["candidate_training_seconds"]))
    ]

    if stage == "search":
        primary_metric = "paired_normalized_improvement"
        objective_direction = "max"
        tail_value = {
            "loss_definition": "negative_paired_normalized_improvement",
            "var95_by_cell": [
                row.get("paired_loss_var95", _feedback_marker("missing"))
                for row in summaries
            ],
            "cvar95_by_cell": [
                row.get("paired_loss_cvar95", _feedback_marker("missing"))
                for row in summaries
            ],
        }
        bound_value: Any = _feedback_marker(
            "boundary agreement and monotonicity probes are not run"
        )
    else:
        primary_metric = "normalized_primal_dual_confidence_gap"
        objective_direction = "min"
        tail_value = {
            "normalized_confidence_gap": [
                row.get("normalized_primal_dual_confidence_gap", _feedback_marker("missing"))
                for row in summaries
            ],
            "raw_gap": [
                row.get("raw_primal_dual_gap", _feedback_marker("missing"))
                for row in summaries
            ],
        }
        bound_value = {
            "raw_bound_order_ok": [
                row.get("raw_bound_order_ok", _feedback_marker("missing"))
                for row in summaries
            ],
            "all_ok": all(bool(row.get("raw_bound_order_ok", False)) for row in summaries)
            if summaries else _feedback_marker("no audit cells")
        }

    observed: dict[str, Any] = {
        "primary_score": float(score),
        "primary_metric": primary_metric,
        "objective_direction": objective_direction,
        "aggregate_effect": (
            float(aggregate_effect)
            if aggregate_effect is not None and math.isfinite(float(aggregate_effect))
            else _feedback_marker("no finite aggregate effect")
        ),
        "aggregate_standard_error": (
            float(aggregate_standard_error)
            if aggregate_standard_error is not None and math.isfinite(float(aggregate_standard_error))
            else _feedback_marker("aggregate standard error unavailable")
        ),
        "cell_count": len(summaries),
        "failure_rate": failure_rate,
        "runtime_seconds": runtime_value,
        "training_seconds_by_cell": training_seconds or _feedback_marker(
            "candidate training runtime is unavailable for this runner"
        ),
        "tail_risk": tail_value,
        "boundary_diagnostics": bound_value,
        # This field used to be a hard-coded true flag in the legacy metrics.
        # A sidecar must state that no independent replay was actually run.
        "independent_reproduction": _feedback_marker(
            "independent replay is not part of this evaluation stage"
        ),
    }

    directional: list[DirectionalFeedback] = []
    for index, row in enumerate(summaries):
        instance_id = str(row.get("instance_id", f"cell-{index}"))
        repeat = row.get("repeat", index)
        if stage == "search":
            raw_effect = row.get("paired_normalized_improvement")
            raw_se = row.get("paired_normalized_standard_error")
            metric = "paired_normalized_improvement"
            lower_is_better = False
        else:
            gap = row.get("normalized_primal_dual_confidence_gap")
            raw_effect = (-float(gap)) if isinstance(gap, (int, float)) else None
            raw_se = row.get("upper_bound_standard_error")
            metric = "negative_normalized_primal_dual_confidence_gap"
            lower_is_better = False
        direction = _feedback_direction(raw_effect, lower_is_better=lower_is_better)
        if direction == "positive":
            recommendation = {
                "action": "exploit",
                "scope": "mechanism_or_representation",
                "target_slice": f"instance:{instance_id}",
                "next_probe": "held_out_regime",
            }
        elif direction == "negative":
            recommendation = {
                "action": "falsify_or_switch",
                "scope": "mechanism_family",
                "target_slice": f"instance:{instance_id}",
                "next_probe": "matched_control_or_counterexample",
            }
        else:
            recommendation = {
                "action": "probe",
                "scope": "slice",
                "target_slice": f"instance:{instance_id}",
                "next_probe": "repeat_or_held_out_regime",
            }
        observed_row: dict[str, Any] = {
            "metric": metric,
            "effect": (
                float(raw_effect)
                if isinstance(raw_effect, (int, float)) and math.isfinite(float(raw_effect))
                else _feedback_marker("effect was not emitted")
            ),
            "standard_error": (
                float(raw_se)
                if isinstance(raw_se, (int, float)) and math.isfinite(float(raw_se))
                else _feedback_marker("standard error was not emitted")
            ),
        }
        # These fields are evaluator-owned and already present in search
        # summaries.  Audit rows legitimately lack exercise geometry.
        for field_name in (
            "candidate_exercise_rate", "baseline_exercise_rate",
            "candidate_stop_time_mean", "baseline_stop_time_mean",
            "candidate_stop_time_std", "baseline_stop_time_std",
            "candidate_finite", "baseline_finite",
            "candidate_training_seconds",
            "raw_primal_dual_gap", "raw_bound_order_ok",
        ):
            observed_field = row.get(field_name)
            observed_row[field_name] = (
                observed_field if observed_field is not None else
                _feedback_marker(f"{field_name} is not observed in {stage} stage")
            )
        # Emit one instance-level item plus a handful of stable domain strata.
        # All share the same measured cell effect; the strata are for routing
        # and coverage, never extra contributions to the primary score.
        slice_keys = [f"instance:{instance_id}"]
        raw_labels = row.get("slice_labels", [])
        if isinstance(raw_labels, (list, tuple)):
            for label in raw_labels:
                if isinstance(label, str) and label and label not in slice_keys:
                    slice_keys.append(label)
        for slice_index, slice_key in enumerate(slice_keys):
            directional.append(DirectionalFeedback(
                id=f"{suite_id}:{stage}:{instance_id}:{repeat}:{slice_index}",
                candidate_id=candidate_id,
                mechanism_id=(
                    "candidate_vs_baseline" if stage == "search"
                    else "candidate_vs_dual"
                ),
                slice_key=slice_key,
                direction=direction,
                prediction=NOT_OBSERVED,
                confidence=NOT_OBSERVED,
                observed=observed_row,
                recommendation={**recommendation, "target_slice": slice_key},
                evidence={
                    "source": "trusted_evaluator",
                    "request_sha256": _sha256_json(request),
                    "record_ids": [],
                },
                probe={
                    "probe_version": "bermudan-domain-probe.v1",
                    "stage": stage,
                    "metric": metric,
                    "slice_source": "instance_contract_descriptor",
                },
                data={
                    "split": "private" if stage == "audit" else "public",
                    "task": task,
                    "suite_id": suite_id,
                    "instance_id": instance_id,
                    "repeat": repeat,
                    "slice_label": slice_key,
                },
                falsifier=(
                    "paired_effect_ci_upper_le_0" if stage == "search"
                    else "negative_primal_dual_gap_or_bound_order_failure"
                ),
            ))

    packet_payload = {
        "stage": stage,
        "task": task,
        "suite_id": suite_id,
        "request": request,
        "summary_ids": [item.id for item in directional],
    }
    packet = FeedbackPacket(
        packet_id=_canonical_feedback_id(packet_payload),
        candidate_id=candidate_id,
        mechanism_id="candidate_vs_baseline" if stage == "search" else "candidate_vs_dual",
        directional=directional,
        observed=observed,
        recommendation={
            "policy": "deterministic_projection_only",
            "action": "use_directional_items_with_router",
            "primary_metric": primary_metric,
        },
        evidence={
            "source": "trusted_evaluator",
            "evidence_version": "bermudan-evidence.v1",
            "request_sha256": _sha256_json(request),
            "candidate_supplied_telemetry_ignored": True,
        },
        probe={
            "probe_version": "bermudan-domain-probe.v1",
            "stage": stage,
            "confidence_level": confidence_level,
            "observed_fields": sorted(observed),
        },
        data={
            "split": "private" if stage == "audit" else "public",
            "task": task,
            "suite_id": suite_id,
            "candidate_boundary": "evaluator_owned",
        },
    )
    packet.validate()
    return packet


def _canonical_feedback_id(payload: Mapping[str, Any]) -> str:
    """Stable packet id without exposing candidate source contents."""

    return f"feedback_{_sha256_json(payload)[:20]}"


def _fixed_suite_lcb(
    cell_means: list[float],
    cell_standard_errors: list[float],
    confidence_level: float,
) -> tuple[float, float, float]:
    """LCB for the equally weighted mean of fixed, independent suite cells."""
    means = np.asarray(cell_means, dtype=float)
    standard_errors = np.asarray(cell_standard_errors, dtype=float)
    if not len(means) or means.shape != standard_errors.shape:
        raise ValueError("cell means and standard errors must be non-empty and aligned")
    if np.any(standard_errors < 0.0) or not np.all(np.isfinite(standard_errors)):
        raise ValueError("cell standard errors must be finite and nonnegative")
    mean = float(means.mean())
    se = float(np.sqrt(np.sum(np.square(standard_errors))) / len(means))
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return mean - z * se, mean, se


def _bonferroni_gap_z(confidence_level: float) -> float:
    """One-sided component quantile giving joint coverage >= confidence."""
    alpha = 1.0 - confidence_level
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def _instance_digest(instance: BSInstance) -> str:
    payload = {
        "payoff_type": instance.payoff_type,
        "spots": instance.spots,
        "strike": instance.strike,
        "rate": instance.rate,
        "dividends": instance.dividends,
        "volatilities": instance.volatilities,
        "correlation": instance.correlation,
        "maturity": instance.maturity,
        "exercise_times": instance.exercise_times,
        "weights": instance.weights,
    }
    return _sha256_json(payload)


def _instance_slice_labels(instance: BSInstance) -> list[str]:
    """Return stable, evaluator-owned domain slices for directional feedback.

    These labels are descriptive strata, not additional score terms.  They
    let the Context distinguish a mechanism that helps (say) high-volatility
    baskets from one that only improves a particular instance, without
    exposing paths or candidate internals.
    """
    mean_moneyness = float(np.mean(np.asarray(instance.spots, dtype=float))) / float(instance.strike)
    mean_vol = float(np.mean(np.asarray(instance.volatilities, dtype=float)))
    corr = np.asarray(instance.correlation, dtype=float)
    off_diag = corr[~np.eye(instance.dimension, dtype=bool)]
    mean_corr = float(np.mean(off_diag)) if off_diag.size else 0.0
    if mean_moneyness < 0.95:
        moneyness = "deep_itm_or_low_spot"
    elif mean_moneyness > 1.05:
        moneyness = "deep_otm_or_high_spot"
    else:
        moneyness = "near_atm"
    volatility = "high_volatility" if mean_vol >= 0.30 else (
        "low_volatility" if mean_vol <= 0.20 else "medium_volatility"
    )
    correlation = "high_correlation" if mean_corr >= 0.35 else (
        "low_correlation" if mean_corr <= 0.15 else "medium_correlation"
    )
    times = np.asarray(instance.exercise_times, dtype=float)
    gaps = np.diff(times)
    grid = "irregular_grid" if gaps.size and not np.allclose(gaps, gaps[0]) else "regular_grid"
    return [
        f"payoff:{instance.payoff_type}",
        f"dimension:{instance.dimension}d",
        f"moneyness:{moneyness}",
        f"volatility:{volatility}",
        f"correlation:{correlation}",
        f"exercise_grid:{grid}",
    ]


def _regular_source_file(path: Path, label: str) -> None:
    """Require a candidate bundle entry to be a stable ordinary file.

    The parent harness normally supplies a sealed source tree.  The evaluator
    still performs this small check because it is also callable directly (for
    example from a task runner or a unit test), and passing a symlink here
    would make the training bridge's provenance ambiguous.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"candidate {label} not found") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"candidate {label} must be a regular file")


def _is_python_program_manifest(manifest: Any) -> bool:
    return getattr(manifest, "schema", None) == PYTHON_PROGRAM_SCHEMA


def _candidate_source_files(manifest: Any) -> tuple[str, ...]:
    return (
        PYTHON_PROGRAM_SOURCE_FILES
        if _is_python_program_manifest(manifest)
        else ALGORITHM_SOURCE_FILES
    )


def _candidate_entrypoint(manifest: Any) -> str:
    return "algorithm.py" if _is_python_program_manifest(manifest) else "train.py"


def _candidate_kind(manifest: Any) -> str:
    return "python_program" if _is_python_program_manifest(manifest) else "algorithm_bundle"


def _candidate_runner_type(manifest: Any) -> str:
    return "python_program" if _is_python_program_manifest(manifest) else getattr(
        manifest, "runner_type", "mlp"
    )


def _validate_algorithm_source_tree(
    source: Path,
    allowed_files: Iterable[str] = ALGORITHM_SOURCE_FILES,
) -> None:
    """Require the standalone evaluator source root to match the v1 allowlist.

    The harness keeps solver plumbing (for example ``solve.sh``) beside the
    candidate files, then passes a filtered root here.  Enforcing the same
    two-file contract at this lower-level API prevents an untracked helper
    module from changing training behavior without changing the bundle digest.
    """
    # Whole-program candidates use a task-owned recursive policy.  The two
    # anchor files are mandatory, while helper modules/configuration may live
    # below nested directories.  Legacy AlgorithmBundle callers retain the
    # exact allowlist behavior below.
    if tuple(allowed_files) == PYTHON_PROGRAM_SOURCE_FILES:
        try:
            files = []
            entries_seen = 0
            for current, directories, filenames in os.walk(source, topdown=True, followlinks=False):
                kept_dirs = []
                for dirname in sorted(directories):
                    if dirname in {".git", "__pycache__"}:
                        continue
                    entries_seen += 1
                    if entries_seen > PYTHON_PROGRAM_SOURCE_MAX_ENTRIES:
                        raise ValueError(
                            f"python program source tree exceeds {PYTHON_PROGRAM_SOURCE_MAX_ENTRIES} entries"
                        )
                    dinfo = os.lstat(Path(current) / dirname)
                    if stat.S_ISLNK(dinfo.st_mode) or not stat.S_ISDIR(dinfo.st_mode):
                        raise ValueError(f"python program source directory {Path(current, dirname).relative_to(source)} must be a real directory")
                    kept_dirs.append(dirname)
                directories[:] = kept_dirs
                for name in sorted(filenames):
                    path = Path(current) / name
                    # Task transport files are present in some direct
                    # evaluator invocations but are not candidate code.
                    if path.relative_to(source).as_posix() == "solve.sh":
                        continue
                    entries_seen += 1
                    if entries_seen > PYTHON_PROGRAM_SOURCE_MAX_ENTRIES:
                        raise ValueError(
                            f"python program source tree exceeds {PYTHON_PROGRAM_SOURCE_MAX_ENTRIES} entries"
                        )
                    finfo = os.lstat(path)
                    if stat.S_ISLNK(finfo.st_mode) or not stat.S_ISREG(finfo.st_mode):
                        raise ValueError(f"python program source file {path.relative_to(source)} must be a regular file")
                    relative = path.relative_to(source).as_posix()
                    if name in {"manifest.json", "algorithm.py"} or Path(name).suffix in PYTHON_PROGRAM_SOURCE_EXTENSIONS:
                        files.append(relative)
                    else:
                        raise ValueError(
                            f"python program source file {relative} has an unsupported extension"
                        )
            if len(files) > PYTHON_PROGRAM_SOURCE_MAX_FILES:
                raise ValueError(
                    f"python program source tree exceeds {PYTHON_PROGRAM_SOURCE_MAX_FILES} files"
                )
            required = set(PYTHON_PROGRAM_SOURCE_FILES)
            missing = sorted(required - set(files))
            if missing:
                raise ValueError("candidate source is missing file(s): " + ", ".join(missing))
            return
        except OSError as exc:
            raise ValueError(f"could not inspect candidate source: {exc}") from exc

    allowed_names = tuple(allowed_files)
    allowed = set(allowed_names)
    try:
        entries = list(os.scandir(source))
    except OSError as exc:
        raise ValueError(f"could not inspect candidate source: {exc}") from exc
    actual = set()
    for entry in entries:
        relative = entry.name
        if entry.is_symlink():
            raise ValueError(
                f"candidate source entry {relative} must not be a symbolic link"
            )
        if not entry.is_file(follow_symlinks=False):
            raise ValueError(
                f"candidate source entry {relative} must be one of: "
                + ", ".join(allowed_names)
            )
        actual.add(relative)
    unexpected = sorted(actual - allowed)
    if unexpected:
        raise ValueError(
            "candidate source contains undeclared file(s): "
            + ", ".join(unexpected)
        )
    missing = sorted(allowed - actual)
    if missing:
        raise ValueError(
            "candidate source is missing file(s): " + ", ".join(missing)
        )


def _candidate_source_manifest(
    source_dir: str | os.PathLike[str],
    submission: Any = None,
) -> tuple[Path, Any]:
    """Validate a Python algorithm source directory and its manifest.

    ``solution.json`` is intentionally only a transport envelope.  The
    authoritative manifest is the sealed ``manifest.json`` beside
    ``train.py``; if the envelope itself contains a manifest, the two must
    describe the same protocol.  We accept both schema spellings used by the
    V5 records, as well as a bare policy manifest copied by ``solve.sh``.
    """
    from tasks.bermudan_optimal_stopping.policy_protocols import (
        canonical_candidate_manifest_payload,
        load_candidate_manifest,
        validate_candidate_manifest,
    )

    raw = Path(source_dir)
    try:
        info = os.lstat(raw)
    except FileNotFoundError as exc:
        raise ValueError("candidate source directory not found") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("candidate source must be a directory")
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError("candidate source must be a real directory")
    source = raw.resolve(strict=True)
    _regular_source_file(source / "manifest.json", "manifest.json")
    manifest = load_candidate_manifest(source / "manifest.json")
    expected_source_files = _candidate_source_files(manifest)
    expected_entrypoint = _candidate_entrypoint(manifest)
    _validate_algorithm_source_tree(source, expected_source_files)
    _regular_source_file(source / expected_entrypoint, expected_entrypoint)

    if isinstance(submission, Mapping):
        schema = submission.get("schema")
        if schema in {
            "openhyra-policy-spec.v1",
            "continuation-linear.v1",
            "continuation-expression.v1",
            PYTHON_PROGRAM_SCHEMA,
        }:
            supplied = validate_candidate_manifest(dict(submission))
            if canonical_candidate_manifest_payload(supplied) != canonical_candidate_manifest_payload(manifest):
                raise ValueError(
                    "solution manifest does not match candidate manifest.json"
                )
        elif schema in ALGORITHM_BUNDLE_SCHEMAS:
            # A bundle envelope is metadata, not an alternate source of
            # executable bytes.  For the current v1 schema require its three
            # execution-boundary declarations and bind each one to what the
            # evaluator actually runs.  The older candidate-algorithm spelling
            # is kept readable for archived callers; if it carries any of
            # these declarations, they are still checked rather than trusted.
            if schema == "openhyra-algorithm-bundle.v1":
                missing = [
                    field for field in ALGORITHM_BUNDLE_DECLARATION_FIELDS
                    if field not in submission
                ]
                if missing:
                    raise ValueError(
                        "algorithm bundle envelope is missing required field(s): "
                        + ", ".join(missing)
                    )

            entrypoint = submission.get("entrypoint")
            if "entrypoint" in submission and entrypoint != expected_entrypoint:
                raise ValueError(
                    f"algorithm bundle entrypoint must be {expected_entrypoint}"
                )

            source_files = submission.get("source_files")
            if "source_files" in submission:
                if (
                    not isinstance(source_files, (list, tuple))
                    or any(not isinstance(name, str) for name in source_files)
                    or len(source_files) != len(expected_source_files)
                    or set(source_files) != set(expected_source_files)
                ):
                    raise ValueError(
                        "algorithm bundle source_files must contain exactly "
                        + " and ".join(expected_source_files)
                    )

            artifact_protocol = submission.get("artifact_protocol")
            if (
                "artifact_protocol" in submission
                and artifact_protocol != manifest.schema
            ):
                raise ValueError(
                    "algorithm bundle artifact_protocol does not match manifest"
                )
            supplied_manifest = submission.get("manifest")
            if supplied_manifest is not None:
                supplied = validate_candidate_manifest(supplied_manifest)
                if canonical_candidate_manifest_payload(supplied) != canonical_candidate_manifest_payload(manifest):
                    raise ValueError(
                        "algorithm bundle manifest does not match manifest.json"
                    )
        elif schema is not None:
            raise ValueError(
                "algorithm candidate solution must be a policy manifest or "
                "algorithm bundle envelope"
            )
    return source, manifest


def _manifest_payload_for_metrics(value: Any) -> dict[str, Any]:
    """Canonical JSON-like representation for every registered manifest."""
    from tasks.bermudan_optimal_stopping.policy_protocols import (
        canonical_candidate_manifest_payload,
    )

    return canonical_candidate_manifest_payload(value)


def _source_bundle_digest(
    source_dir: Path,
    source_files: Iterable[str] | None = ALGORITHM_SOURCE_FILES,
) -> str:
    """Compute a deterministic digest for source provenance/metrics.

    This is not used as an authorization token (the harness seals the source
    separately); it simply makes algorithm results distinguishable from the
    legacy feature-program hash and gives EB a compact reproducibility key.
    """
    from sandbox import read_regular_file

    files = []
    if source_files is None:
        _validate_algorithm_source_tree(source_dir, PYTHON_PROGRAM_SOURCE_FILES)
        from sandbox import read_source_tree
        _tree_hash, hashes, _payloads = read_source_tree(source_dir, 64 * 1024 * 1024)
        for name in sorted(hashes):
            if name == "solve.sh":
                continue
            data = read_regular_file(
                source_dir / name, 64 * 1024 * 1024,
                label=f"candidate algorithm file {name}",
            )
            files.append({"path": name, "size_bytes": len(data), "sha256": hashes[name]})
    else:
        for name in sorted(set(source_files)):
            if (
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or "\\" in name
            ):
                raise ValueError("algorithm source file path is unsafe")
            data = read_regular_file(
                source_dir / name,
                64 * 1024 * 1024,
                label=f"candidate algorithm file {name}",
            )
            files.append({
                "path": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
    return _sha256_json({
        "schema": "openhyra-algorithm-bundle.v1",
        "files": files,
    })


def _training_runtime_roots() -> tuple[Path, ...]:
    """Return narrow, non-overlapping roots needed by the candidate runtime."""
    import sysconfig

    prefix = Path(sys.prefix)
    exec_prefix = Path(getattr(sys, "exec_prefix", sys.prefix))
    candidates = [
        # ``sys.prefix`` itself can be a protected broad root (notably
        # ``/usr/local`` on Linux).  Expose the versioned stdlib/site-packages
        # directories and the binary/lib directories instead.
        prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}",
        exec_prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}",
        prefix / "lib",
        exec_prefix / "lib",
        Path(sys.executable).resolve().parent,
    ]
    # The macOS framework build links the interpreter against a sibling
    # ``Python`` dynamic library (``$prefix/Python``), not merely the files
    # under ``lib/pythonX.Y`` and ``bin``.  Allow that *versioned* framework
    # directory as one narrow root when the library is present.  Do not add
    # ``sys.prefix`` on Linux/Unix installations where it is commonly the
    # broad ``/usr/local`` root rejected by the training sandbox.
    if (
        sys.platform == "darwin"
        and (prefix / "Python").is_file()
    ):
        candidates.insert(0, prefix)
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        configured = sysconfig.get_path(key)
        if configured:
            candidates.append(Path(configured))
    roots: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_dir():
            continue
        if any(
            resolved == existing
            or resolved in existing.parents
            or existing in resolved.parents
            for existing in roots
        ):
            continue
        roots.append(resolved)
    if not roots:
        raise ValueError("no usable Python runtime root for candidate training")
    return tuple(roots)


def _training_result_metrics(result: Any) -> dict[str, Any]:
    """Convert a TrainingCellResult into bounded JSON-friendly diagnostics."""
    def hash_records(value: Any) -> list[dict[str, str]] | None:
        if value is None:
            return None
        return [
            {"path": str(name), "sha256": str(digest)}
            for name, digest in value
        ]

    return {
        "status": result.status,
        "returncode": result.returncode,
        "isolation": result.isolation,
        "research_fallback": bool(getattr(result, "research_fallback", False)),
        "wall_seconds": float(result.wall_seconds),
        "peak_memory_bytes": int(result.peak_memory_bytes),
        "output_entries": int(result.output_entries),
        "output_bytes": int(result.output_bytes),
        "train_seed": int(result.train_seed),
        # Keep per-file identities alongside the compact bundle digests.  They
        # are supervisor-produced telemetry and never become candidate input.
        "input_file_sha256": hash_records(result.input_file_sha256),
        "input_bundle_sha256": result.input_bundle_sha256,
        "policy_file_sha256": hash_records(result.policy_file_sha256),
        "policy_artifact_sha256": result.policy_artifact_sha256,
        "log_tail": str(result.log_tail)[-2000:],
    }


def _training_provenance(
    instance: BSInstance,
    training_paths: np.ndarray,
    train_seed: int,
    train_details: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach evaluator-owned path/payoff/target provenance to a fit cell.

    A Python candidate owns its continuation-target construction, so that
    internal target array is intentionally not claimed as observed.  The
    evaluator nevertheless records the exact path and discounted-payoff
    streams that were supplied to the candidate, plus a deterministic target
    reference for the supplied stream.  This makes a later replay auditable
    without confusing evaluator inputs with candidate internals.
    """
    paths = np.asarray(training_paths)
    rewards = discounted_rewards(paths, instance)
    return {
        **dict(train_details),
        "instance_id": instance.instance_id,
        "path_seed": int(train_seed),
        "train_seed": int(train_details.get("train_seed", train_seed)),
        "training_path_count": int(paths.shape[0]),
        "training_paths_shape": list(paths.shape),
        "training_paths_sha256": _sha256_array(paths),
        "payoffs_sha256": _sha256_array(rewards),
        "target_sha256": _sha256_array(rewards),
        "target_kind": "evaluator_discounted_payoff_stream",
        "candidate_target_observed": False,
        "model_file_sha256": (
            train_details.get("policy_artifact_sha256")
            or train_details.get("policy_file_sha256")
        ),
        "fit_wall_seconds": float(train_details.get("wall_seconds", 0.0)),
    }


def _fit_algorithm_policy(
    source_dir: Path,
    manifest: Any,
    instance: BSInstance,
    training_paths: np.ndarray,
    *,
    train_seed: int,
    cell_dir: Path,
    config: Mapping[str, Any],
) -> tuple[TrustedRunnerPolicy, dict[str, Any]]:
    """Run one candidate ``train.py`` in a fresh sandbox and load its runner."""
    # Import lazily: training_pipeline imports this evaluator's BSInstance for
    # its input serializer, and eager import would create a script/module
    # cycle when evaluator.py is launched as ``__main__``.
    from tasks.bermudan_optimal_stopping.training_pipeline import (
        run_per_instance_training,
    )

    timeout_s = float(config.get("training_timeout_s", DEFAULT_TRAINING_TIMEOUT_S))
    memory_bytes = int(config.get("training_memory_bytes", DEFAULT_TRAINING_MEMORY_BYTES))
    file_size_bytes = int(config.get("training_file_size_bytes", DEFAULT_TRAINING_FILE_SIZE_BYTES))
    result = run_per_instance_training(
        instance=instance,
        training_paths=training_paths,
        candidate_source_dir=source_dir,
        cell_dir=cell_dir,
        # ``_derive_seed`` intentionally uses the full 64-bit digest for the
        # simulator.  The training bridge's public contract is a non-negative
        # 63-bit seed, so fold the derived value without changing the legacy
        # simulator streams.
        train_seed=train_seed & ((1 << 63) - 1),
        runtime_roots=_training_runtime_roots(),
        timeout_s=timeout_s,
        cpu_seconds=config.get("training_cpu_seconds"),
        memory_bytes=memory_bytes,
        file_size_bytes=file_size_bytes,
        prediction_timeout_s=float(config.get("prediction_timeout_s", 5.0)),
        prediction_cpu_seconds=config.get("prediction_cpu_seconds"),
        prediction_memory_bytes=int(config.get(
            "prediction_memory_bytes", DEFAULT_TRAINING_MEMORY_BYTES,
        )),
        prediction_file_size_bytes=int(config.get(
            "prediction_file_size_bytes", 16 * 1024 * 1024,
        )),
        externally_isolated=sys.platform != "darwin",
    )
    details = _training_result_metrics(result)
    if result.status != "ok" or result.runner is None:
        note = result.log_tail.strip() or "candidate training failed"
        raise ValueError(
            f"candidate training failed for {instance.instance_id}: "
            f"{result.status}: {note[-1000:]}"
        )
    policy = TrustedRunnerPolicy(
        runner=result.runner,
        instance=instance,
        runner_type=_candidate_runner_type(manifest),
        policy_interface=getattr(manifest, "interface", "continuation"),
        policy_artifact_sha256=result.policy_artifact_sha256,
    )
    return policy, details


def _fit_candidate_policy(
    program: dict[str, Any] | None,
    source_dir: Path | None,
    manifest: Any,
    instance: BSInstance,
    training_paths: np.ndarray,
    *,
    train_seed: int,
    cell_dir: Path | None,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Fit either the legacy in-process Ridge policy or a bundle runner."""
    if source_dir is None:
        if program is None:
            raise ValueError("legacy candidate is missing feature program")
        return (
            fit_lsmc(
                program, instance, training_paths,
                ridge_alpha=config["ridge_alpha"],
            ),
            {"status": "builtin", "runner_type": "ridge_lsmc"},
        )
    if manifest is None or cell_dir is None:
        raise ValueError("algorithm candidate is missing manifest or training cell")
    policy, details = _fit_algorithm_policy(
        source_dir,
        manifest,
        instance,
        training_paths,
        train_seed=train_seed,
        cell_dir=cell_dir,
        config=config,
    )
    details["runner_type"] = _candidate_runner_type(manifest)
    details["policy_interface"] = getattr(manifest, "interface", "continuation")
    return policy, details


def _policy_probe_output(policy: Any, paths: np.ndarray, time_index: int) -> np.ndarray:
    """Run one bounded prediction query through the evaluator-owned adapter."""
    rewards = discounted_rewards(paths, policy.instance)
    states = np.asarray(paths[:, time_index, :], dtype=float)
    history = np.asarray(paths[:, : time_index + 1, :], dtype=float)
    immediate = np.asarray(rewards[:, time_index], dtype=float)
    if getattr(policy, "policy_interface", "continuation") == "decision":
        return np.asarray(
            policy.decision(
                time_index, states, history=history,
                immediate_payoffs=immediate,
            )
        )
    if getattr(policy, "runner_type", "") == "python_program":
        return np.asarray(
            policy.continuation(
                time_index, states, history=history,
                immediate_payoffs=immediate,
            )
        )
    return np.asarray(policy.continuation(time_index, states))


def _lookahead_probe(policy: Any, instance: BSInstance, seed: int) -> dict[str, Any]:
    """Check that changing an unobserved future suffix cannot change output.

    The evaluator deliberately supplies only the current prefix to both
    queries.  The two full paths differ after the prefix and their suffix
    digests are recorded so the resulting equality is an auditable causal
    probe rather than an assertion inferred from source inspection.
    """
    try:
        probe_paths = simulate_paths(instance, 8, int(seed))
        time_index = 0 if len(instance.exercise_times) <= 2 else 1
        prefix = probe_paths[:, : time_index + 1, :].copy()
        changed = simulate_paths(instance, 8, int(seed) + 1)
        changed[:, : time_index + 1, :] = prefix
        if np.array_equal(
            probe_paths[:, time_index + 1 :, :],
            changed[:, time_index + 1 :, :],
        ):
            changed[:, time_index + 1 :, :] *= 1.0001
        first = _policy_probe_output(policy, probe_paths, time_index)
        second = _policy_probe_output(policy, changed, time_index)
        equal = bool(np.array_equal(first, second))
        return {
            "status": "passed" if equal else "failed",
            "observed": True,
            "time_index": time_index,
            "prefix_equal": bool(np.array_equal(
                probe_paths[:, : time_index + 1, :],
                changed[:, : time_index + 1, :],
            )),
            "future_changed": bool(not np.array_equal(
                probe_paths[:, time_index + 1 :, :],
                changed[:, time_index + 1 :, :],
            )),
            "prediction_equal": equal,
            "future_a_sha256": _sha256_json(
                probe_paths[:, time_index + 1 :, :].tolist()
            ),
            "future_b_sha256": _sha256_json(
                changed[:, time_index + 1 :, :].tolist()
            ),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "observed": True,
            "prediction_equal": False,
            "reason": f"{type(exc).__name__}: {exc}"[:1000],
        }


def _independent_validation(
    *,
    policy: Any,
    source_dir: Path | None,
    manifest: Any,
    instance: BSInstance,
    training_paths: np.ndarray,
    train_seed: int,
    training_root: Path | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one fresh replay and a causal lookahead probe for a top candidate."""
    probe_started = time.perf_counter()
    lookahead = _lookahead_probe(policy, instance, train_seed ^ 0x5EED)
    probe_seconds = time.perf_counter() - probe_started
    result: dict[str, Any] = {
        "schema": "openhyra-bermudan-independent-validation.v1",
        "observed": True,
        "lookahead_probe": lookahead,
        "deterministic_replay": {
            "status": "not_applicable",
            "observed": False,
            "reason": "legacy in-process policy has no sandbox model artifact",
        },
        "cost": {"replay_wall_seconds": 0.0, "probe_wall_seconds": 0.0},
    }
    started = time.perf_counter()
    replay_details: dict[str, Any] | None = None
    replay_policy: Any = None
    if source_dir is not None and training_root is not None:
        try:
            replay_cell = training_root / "independent-replay"
            replay_policy, replay_details = _fit_candidate_policy(
                None,
                source_dir,
                manifest,
                instance,
                np.array(training_paths, copy=True),
                train_seed=train_seed,
                cell_dir=replay_cell,
                config=config,
            )
            model_a = getattr(policy, "policy_artifact_sha256", None)
            model_b = getattr(replay_policy, "policy_artifact_sha256", None)
            prediction_equal = False
            try:
                first_query = simulate_paths(instance, 8, train_seed ^ 0xC0DE)
                left = _policy_probe_output(policy, first_query, 0)
                right = _policy_probe_output(replay_policy, first_query, 0)
                prediction_equal = bool(np.array_equal(left, right))
            except Exception:
                prediction_equal = False
            model_equal = (
                isinstance(model_a, str)
                and isinstance(model_b, str)
                and model_a == model_b
            )
            result["deterministic_replay"] = {
                "status": "passed" if model_equal and prediction_equal else "failed",
                "observed": True,
                "model_equal": model_equal,
                "prediction_equal": prediction_equal,
                "model_sha256": model_a,
                "replay_model_sha256": model_b,
                "replay_training": replay_details,
            }
        except Exception as exc:
            result["deterministic_replay"] = {
                "status": "failed",
                "observed": True,
                "model_equal": False,
                "prediction_equal": False,
                "reason": f"{type(exc).__name__}: {exc}"[:1000],
            }
    # Timing is an observed cost, not part of numerical replay identity.
    result["cost"]["replay_wall_seconds"] = time.perf_counter() - started
    result["cost"]["probe_wall_seconds"] = probe_seconds
    return result


def _evaluate_search(
    program: dict[str, Any] | None,
    request: dict[str, Any],
    config: dict[str, Any],
    *,
    candidate_source_dir: Path | None = None,
    candidate_manifest: Any = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    if candidate_source_dir is not None:
        candidate_source_dir, loaded_manifest = _candidate_source_manifest(
            candidate_source_dir,
        )
        if candidate_manifest is None:
            candidate_manifest = loaded_manifest
    suite = public_suite()[: config["instance_count"]]
    summaries: list[dict[str, Any]] = []
    cluster_improvements: list[float] = []
    cluster_standard_errors: list[float] = []
    training_cells: list[dict[str, Any]] = []
    independent_validation: dict[str, Any] | None = None
    training_root: Path | None = None
    if candidate_source_dir is not None:
        # A single root is used only as a parent; every cell remains a fresh
        # directory and is removed after this stage.  The candidate source is
        # never copied into it or imported by this process.
        training_root = Path(tempfile.mkdtemp(prefix="openhyra-bermudan-train-"))
    try:
        for instance_index, instance in enumerate(suite):
            for repeat in range(config["repeats"]):
                train_seed = _derive_seed(
                    request["seed"], request["suite_id"], instance_index,
                    repeat, "train",
                )
                pricing_seed = _derive_seed(
                    request["seed"], request["suite_id"], instance_index,
                    repeat, "pricing",
                )
                training = simulate_paths(instance, config["training_paths"], train_seed)
                pricing = simulate_paths(instance, config["pricing_paths"], pricing_seed)
                cell_dir = (
                    training_root / f"search-{instance_index:03d}-{repeat:03d}"
                    if training_root is not None else None
                )
                candidate_policy, train_details = _fit_candidate_policy(
                    program,
                    candidate_source_dir,
                    candidate_manifest,
                    instance,
                    training,
                    train_seed=train_seed,
                    cell_dir=cell_dir,
                    config=config,
                )
                if (
                    candidate_source_dir is not None
                    and independent_validation is None
                    and config.get("independent_validation", True)
                ):
                    independent_validation = _independent_validation(
                        policy=candidate_policy,
                        source_dir=candidate_source_dir,
                        manifest=candidate_manifest,
                        instance=instance,
                        training_paths=training,
                        train_seed=train_seed,
                        training_root=training_root,
                        config=config,
                    )
                if candidate_source_dir is not None:
                    training_cells.append({
                        "repeat": repeat,
                        **_training_provenance(
                            instance, training, train_seed, train_details
                        ),
                    })
                baseline_policy = fit_lsmc(
                    BASELINE_PROGRAM, instance, training,
                    ridge_alpha=config["ridge_alpha"],
                )
                candidate_values, candidate_stops = apply_policy(candidate_policy, pricing)
                baseline_values, baseline_stops = apply_policy(baseline_policy, pricing)
                candidate_behavior = _policy_behavior_metrics(
                    candidate_values,
                    candidate_stops,
                    len(instance.exercise_times),
                )
                baseline_behavior = _policy_behavior_metrics(
                    baseline_values,
                    baseline_stops,
                    len(instance.exercise_times),
                )
                paired = (candidate_values - baseline_values) / instance.strike
                improvement, paired_se = _mean_se(paired)
                # The tail probe is descriptive only; it is not part of the
                # fixed-suite primary LCB.  ``loss`` is the negative paired
                # improvement, so larger values mean a worse tail outcome.
                paired_losses = -np.asarray(paired, dtype=float).reshape(-1)
                paired_loss_q95 = float(np.quantile(paired_losses, 0.95))
                paired_loss_tail = paired_losses[paired_losses >= paired_loss_q95]
                paired_loss_cvar95 = float(np.mean(paired_loss_tail))
                cluster_improvements.append(improvement)
                cluster_standard_errors.append(paired_se)
                candidate_mean, candidate_se = _mean_se(candidate_values)
                baseline_mean, baseline_se = _mean_se(baseline_values)
                summaries.append({
                    "instance_id": instance.instance_id,
                    "instance_sha256": _instance_digest(instance),
                    "payoff_type": instance.payoff_type,
                    "dimension": instance.dimension,
                    "moneyness": float(np.mean(np.asarray(instance.spots, dtype=float)) / instance.strike),
                    "mean_volatility": float(np.mean(np.asarray(instance.volatilities, dtype=float))),
                    "mean_correlation": float(
                        np.mean(np.asarray(instance.correlation, dtype=float)[
                            ~np.eye(instance.dimension, dtype=bool)
                        ])
                    ) if instance.dimension > 1 else 0.0,
                    "slice_labels": _instance_slice_labels(instance),
                    "repeat": repeat,
                    "candidate_lower_bound": candidate_mean,
                    "candidate_lower_bound_standard_error": candidate_se,
                    "baseline_lower_bound": baseline_mean,
                    "baseline_lower_bound_standard_error": baseline_se,
                    "paired_normalized_improvement": improvement,
                    "paired_normalized_standard_error": paired_se,
                    "paired_loss_var95": paired_loss_q95,
                    "paired_loss_cvar95": paired_loss_cvar95,
                    **({
                        "pricing_paths_sha256": _sha256_array(pricing),
                        "paired_pathwise_improvements": paired.tolist(),
                    } if config.get("public_pathwise_samples", False) else {}),
                    # Compact policy-geometry descriptors used by V5's
                    # BehaviorProfile and by the mechanism critic.  These are
                    # computed from evaluator-owned pricing outcomes, never
                    # from candidate-reported telemetry.
                    "candidate_exercise_rate": candidate_behavior["exercise_rate"],
                    "candidate_exercise_rate_by_time": candidate_behavior[
                        "exercise_rate_by_time"
                    ],
                    "candidate_stop_time_mean": candidate_behavior[
                        "stop_time_mean"
                    ],
                    "candidate_stop_time_std": candidate_behavior[
                        "stop_time_std"
                    ],
                    "candidate_finite": candidate_behavior["finite"],
                    "candidate_valid_stop_rate": candidate_behavior[
                        "valid_stop_rate"
                    ],
                    "baseline_exercise_rate": baseline_behavior["exercise_rate"],
                    "baseline_exercise_rate_by_time": baseline_behavior[
                        "exercise_rate_by_time"
                    ],
                    "baseline_stop_time_mean": baseline_behavior[
                        "stop_time_mean"
                    ],
                    "baseline_stop_time_std": baseline_behavior[
                        "stop_time_std"
                    ],
                    "baseline_finite": baseline_behavior["finite"],
                    "candidate_training_seconds": train_details.get(
                        "wall_seconds"
                    ),
                })
    finally:
        if training_root is not None:
            shutil.rmtree(training_root, ignore_errors=True)
    score, mean_improvement, aggregate_se = _fixed_suite_lcb(
        cluster_improvements,
        cluster_standard_errors,
        config["confidence_level"],
    )
    cell_count = len(cluster_improvements)
    metrics = {
        "metric": "paired_lower_bound_lcb",
        "research_mode": bool(config.get("research_mode", False)),
        "independent_validation_enabled": bool(
            config.get("independent_validation", True)
        ),
        "search_score": score,
        "mean_paired_normalized_improvement": mean_improvement,
        "paired_aggregate_standard_error": aggregate_se,
        "estimator_scope": "fixed_public_suite_mean",
        "instance_count": len(suite),
        "repeat_count": config["repeats"],
        "evaluation_cell_count": cell_count,
        "training_path_count": config["training_paths"],
        "test_path_count": config["pricing_paths"],
        "total_training_path_budget": config["training_paths"] * cell_count,
        "total_test_path_budget": config["pricing_paths"] * cell_count,
        "failure_rate": (
            float(1.0 - np.mean([
                bool(row.get("candidate_finite", False)) for row in summaries
            ])) if summaries else None
        ),
        "failure_rate_observed": bool(summaries),
        "lookahead_violation": (
            None if independent_validation is None else
            independent_validation.get("lookahead_probe", {}).get("status") != "passed"
        ),
        "lookahead_violation_observed": bool(
            independent_validation is not None
            and independent_validation.get("lookahead_probe", {}).get("observed")
        ),
        "deterministic_reproduction_passed": (
            None if independent_validation is None else
            independent_validation.get("deterministic_replay", {}).get("status") == "passed"
        ),
        "deterministic_reproduction_observed": bool(
            independent_validation is not None
            and independent_validation.get("deterministic_replay", {}).get("observed")
        ),
        "independent_validation": independent_validation or {
            "schema": "openhyra-bermudan-independent-validation.v1",
            "observed": False,
            "reason": "independent validation disabled",
        },
        "summaries": summaries,
    }
    # Emit direct projections so downstream V5 code can build a profile
    # without reimplementing repeat aggregation.  The legacy summary fields
    # above remain unchanged and are still the source of truth for old
    # consumers.
    metrics.update(_aggregate_behavior_metrics(summaries))
    if candidate_source_dir is not None:
        metrics.update({
            "candidate_kind": _candidate_kind(candidate_manifest),
            "runner_type": _candidate_runner_type(candidate_manifest),
            "policy_interface": getattr(
                candidate_manifest, "interface", "continuation",
            ),
            "training_cell_count": len(training_cells),
            "training_cells": training_cells,
            "mean_training_seconds_per_instance": (
                float(np.mean([item["wall_seconds"] for item in training_cells]))
                if training_cells else 0.0
            ),
            "max_training_seconds_per_instance": (
                float(max(item["wall_seconds"] for item in training_cells))
                if training_cells else 0.0
            ),
        })
    # Sidecar only: the primary score above remains exactly the historical
    # fixed-suite LCB.  The packet gives Context a typed projection of the
    # evaluator-owned outcome geometry and explicit not-observed markers.
    feedback_packet = _build_domain_feedback_packet(
        stage="search",
        task=request["task"],
        suite_id=request["suite_id"],
        request=request,
        score=score,
        summaries=summaries,
        confidence_level=config["confidence_level"],
        aggregate_effect=mean_improvement,
        aggregate_standard_error=aggregate_se,
    )
    feedback_payload = feedback_packet.to_dict()
    if independent_validation is not None:
        feedback_payload.setdefault("observed", {})[
            "independent_reproduction"
        ] = {
            "status": independent_validation.get(
                "deterministic_replay", {}
            ).get("status", "not_observed"),
            "lookahead_probe": independent_validation.get(
                "lookahead_probe", {}
            ),
        }
    metrics["feedback_packet"] = feedback_payload
    evidence = {
        "search": {
            "status": "paired_oos_checked",
            "baseline_feature_sha256": _sha256_json(BASELINE_PROGRAM),
            "common_random_numbers": True,
            "training_pricing_independent": True,
            "cluster_unit": "instance_repeat",
            "candidate_kind": (
                _candidate_kind(candidate_manifest) if candidate_source_dir is not None
                else "feature_ir"
            ),
            "feedback_packet_id": feedback_packet.packet_id,
            "feedback_packet_schema": feedback_packet.schema,
            "independent_reproduction": {
                "status": (
                    (independent_validation or {}).get(
                        "deterministic_replay", {}
                    ).get("status", "not_observed")
                ),
                "lookahead_probe": (independent_validation or {}).get(
                    "lookahead_probe", {"status": "not_observed"}
                ),
                "reason": "one fresh top-candidate replay and causal probe",
            },
        }
    }
    return score, metrics, evidence


def _evaluate_audit(
    program: dict[str, Any] | None,
    request: dict[str, Any],
    config: dict[str, Any],
    *,
    candidate_source_dir: Path | None = None,
    candidate_manifest: Any = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    if candidate_source_dir is not None:
        candidate_source_dir, loaded_manifest = _candidate_source_manifest(
            candidate_source_dir,
        )
        if candidate_manifest is None:
            candidate_manifest = loaded_manifest
    suite = derive_hidden_suite(request["seed"], config["instance_count"])
    summaries: list[dict[str, Any]] = []
    confidence_gaps: list[float] = []
    training_cells: list[dict[str, Any]] = []
    independent_validation: dict[str, Any] | None = None
    dual_verifiers: set[str] = set()
    training_root: Path | None = None
    if candidate_source_dir is not None:
        training_root = Path(tempfile.mkdtemp(prefix="openhyra-bermudan-audit-"))
    # Allocate alpha/2 to each of the upper and lower one-sided endpoint
    # failures.  The union bound then gives at least (1-alpha) simultaneous
    # coverage without assuming independence between the estimators.
    z = _bonferroni_gap_z(config["confidence_level"])
    all_bound_order_ok = True
    try:
        for instance_index, instance in enumerate(suite):
            for repeat in range(config["repeats"]):
                train_seed = _derive_seed(
                    request["seed"], request["suite_id"], instance_index,
                    repeat, "audit-train",
                )
                pricing_seed = _derive_seed(
                    request["seed"], request["suite_id"], instance_index,
                    repeat, "audit-pricing",
                )
                outer_seed = _derive_seed(
                    request["seed"], request["suite_id"], instance_index,
                    repeat, "audit-outer",
                )
                inner_seed = _derive_seed(
                    request["seed"], request["suite_id"], instance_index,
                    repeat, "audit-inner",
                )
                training = simulate_paths(instance, config["training_paths"], train_seed)
                pricing = simulate_paths(instance, config["pricing_paths"], pricing_seed)
                outer = simulate_paths(instance, config["outer_paths"], outer_seed)
                cell_dir = (
                    training_root / f"audit-{instance_index:03d}-{repeat:03d}"
                    if training_root is not None else None
                )
                policy, train_details = _fit_candidate_policy(
                    program,
                    candidate_source_dir,
                    candidate_manifest,
                    instance,
                    training,
                    train_seed=train_seed,
                    cell_dir=cell_dir,
                    config=config,
                )
                if (
                    candidate_source_dir is not None
                    and independent_validation is None
                    and config.get("independent_validation", True)
                ):
                    independent_validation = _independent_validation(
                        policy=policy,
                        source_dir=candidate_source_dir,
                        manifest=candidate_manifest,
                        instance=instance,
                        training_paths=training,
                        train_seed=train_seed,
                        training_root=training_root,
                        config=config,
                    )
                if candidate_source_dir is not None:
                    training_cells.append({
                        "repeat": repeat,
                        **_training_provenance(
                            instance, training, train_seed, train_details
                        ),
                    })
                lower_samples, _ = apply_policy(policy, pricing)
                # Open Python programs may return direct decisions and may use
                # full causal history.  Neither output is a value function for
                # the nested martingale.  Use one deterministic,
                # evaluator-owned continuation approximation for every Python
                # program so decision and continuation interfaces have the same
                # independent upper-bound verifier.  Legacy frozen runners keep
                # their historical candidate-specific dual unchanged.
                if getattr(policy, "runner_type", "") == "python_program":
                    dual_policy = fit_lsmc(
                        BASELINE_PROGRAM,
                        instance,
                        training,
                        ridge_alpha=config["ridge_alpha"],
                    )
                    dual_verifier = "evaluator_owned_ridge_lsmc"
                else:
                    dual_policy = policy
                    dual_verifier = "candidate_continuation"
                dual_verifiers.add(dual_verifier)
                upper_samples, martingale_terminal = dual_upper_bound_samples(
                    dual_policy,
                    outer,
                    inner_paths=config["inner_paths"],
                    inner_seed=inner_seed,
                )
                lower_mean, lower_se = _mean_se(lower_samples)
                upper_mean, upper_se = _mean_se(upper_samples)
                raw_gap = upper_mean - lower_mean
                confidence_gap = raw_gap + z * (upper_se + lower_se)
                normalized_gap = confidence_gap / instance.strike
                confidence_gaps.append(normalized_gap)
                bound_order_ok = upper_mean >= lower_mean
                all_bound_order_ok = all_bound_order_ok and bound_order_ok
                martingale_mean, martingale_se = _mean_se(martingale_terminal)
                summaries.append({
                    "instance_id": instance.instance_id,
                    "instance_sha256": _instance_digest(instance),
                    "payoff_type": instance.payoff_type,
                    "dimension": instance.dimension,
                    "moneyness": float(np.mean(np.asarray(instance.spots, dtype=float)) / instance.strike),
                    "mean_volatility": float(np.mean(np.asarray(instance.volatilities, dtype=float))),
                    "mean_correlation": float(
                        np.mean(np.asarray(instance.correlation, dtype=float)[
                            ~np.eye(instance.dimension, dtype=bool)
                        ])
                    ) if instance.dimension > 1 else 0.0,
                    "slice_labels": _instance_slice_labels(instance),
                    "repeat": repeat,
                    "lower_bound": lower_mean,
                    "lower_bound_standard_error": lower_se,
                    "upper_bound": upper_mean,
                    "upper_bound_standard_error": upper_se,
                    "raw_primal_dual_gap": raw_gap,
                    "confidence_gap": confidence_gap,
                    "normalized_primal_dual_confidence_gap": normalized_gap,
                    "raw_bound_order_ok": bound_order_ok,
                    "martingale_terminal_mean": martingale_mean,
                    "martingale_terminal_standard_error": martingale_se,
                    "dual_verifier": dual_verifier,
                    "candidate_training_seconds": train_details.get(
                        "wall_seconds"
                    ),
                })
    finally:
        if training_root is not None:
            shutil.rmtree(training_root, ignore_errors=True)
    mean_gap = float(np.mean(confidence_gaps))
    q90_gap = float(np.quantile(confidence_gaps, 0.9))
    score = mean_gap + 0.25 * q90_gap
    cell_count = len(confidence_gaps)
    metrics = {
        "metric": "normalized_primal_dual_confidence_gap",
        "research_mode": bool(config.get("research_mode", False)),
        "independent_validation_enabled": bool(
            config.get("independent_validation", True)
        ),
        "normalized_primal_dual_confidence_gap": score,
        "confidence_level": config["confidence_level"],
        "confidence_construction": "bonferroni_one_sided_components",
        "confidence_component_z": z,
        "mean_normalized_confidence_gap": mean_gap,
        "q90_normalized_confidence_gap": q90_gap,
        "raw_bound_order_all_ok": all_bound_order_ok,
        "estimator_scope": "fixed_hidden_suite_and_repeat_mean",
        "instance_count": len(suite),
        "repeat_count": config["repeats"],
        "evaluation_cell_count": cell_count,
        "training_path_count": config["training_paths"],
        "test_path_count": config["pricing_paths"],
        "dual_outer_path_count": config["outer_paths"],
        "dual_inner_path_count": config["inner_paths"],
        "total_training_path_budget": config["training_paths"] * cell_count,
        "total_test_path_budget": config["pricing_paths"] * cell_count,
        "total_dual_outer_path_budget": config["outer_paths"] * cell_count,
        "total_dual_inner_draw_budget": sum(
            config["outer_paths"] * config["inner_paths"]
            * (len(instance.exercise_times) - 1) * config["repeats"]
            for instance in suite
        ),
        # A failed audit cell aborts the trusted evaluation before a complete
        # summary is emitted, so this stage cannot estimate a failure rate from
        # its current contract.  ``None`` is intentional and is mirrored by
        # the packet's not_observed marker.
        "failure_rate": None,
        "failure_rate_observed": False,
        "lookahead_violation": (
            None if independent_validation is None else
            independent_validation.get("lookahead_probe", {}).get("status") != "passed"
        ),
        "lookahead_violation_observed": bool(
            independent_validation is not None
            and independent_validation.get("lookahead_probe", {}).get("observed")
        ),
        "deterministic_reproduction_passed": (
            None if independent_validation is None else
            independent_validation.get("deterministic_replay", {}).get("status") == "passed"
        ),
        "deterministic_reproduction_observed": bool(
            independent_validation is not None
            and independent_validation.get("deterministic_replay", {}).get("observed")
        ),
        "independent_validation": independent_validation or {
            "schema": "openhyra-bermudan-independent-validation.v1",
            "observed": False,
            "reason": "independent validation disabled",
        },
        "summaries": summaries,
        "dual_verifier": (
            next(iter(dual_verifiers)) if len(dual_verifiers) == 1 else "mixed"
        ),
    }
    if candidate_source_dir is not None:
        metrics.update({
            "candidate_kind": _candidate_kind(candidate_manifest),
            "runner_type": _candidate_runner_type(candidate_manifest),
            "policy_interface": getattr(
                candidate_manifest, "interface", "continuation",
            ),
            "training_cell_count": len(training_cells),
            "training_cells": training_cells,
            "mean_training_seconds_per_instance": (
                float(np.mean([item["wall_seconds"] for item in training_cells]))
                if training_cells else 0.0
            ),
            "max_training_seconds_per_instance": (
                float(max(item["wall_seconds"] for item in training_cells))
                if training_cells else 0.0
            ),
        })
    feedback_packet = _build_domain_feedback_packet(
        stage="audit",
        task=request["task"],
        suite_id=request["suite_id"],
        request=request,
        score=score,
        summaries=summaries,
        confidence_level=config["confidence_level"],
        aggregate_effect=-mean_gap,
    )
    feedback_payload = feedback_packet.to_dict()
    if independent_validation is not None:
        feedback_payload.setdefault("observed", {})[
            "independent_reproduction"
        ] = {
            "status": independent_validation.get(
                "deterministic_replay", {}
            ).get("status", "not_observed"),
            "lookahead_probe": independent_validation.get(
                "lookahead_probe", {}
            ),
        }
    metrics["feedback_packet"] = feedback_payload
    evidence = {
        "audit": {
            "status": "private_primal_dual_checked",
            "hidden_suite_derived_from_request_seed": True,
            "training_pricing_outer_inner_streams_independent": True,
            "discounted_reward": True,
            "martingale": {
                "m0": 0.0,
                "increment": "f_m(S_outer)-mean_b(f_m(S_inner_b))",
                "inner_conditioning_state": "S_outer[m-1]",
                "finite_inner_estimator_conditionally_unbiased": True,
            },
            "negative_raw_gaps_clipped": False,
            "candidate_kind": (
                _candidate_kind(candidate_manifest) if candidate_source_dir is not None
                else "feature_ir"
            ),
            "dual_verifier": (
                next(iter(dual_verifiers)) if len(dual_verifiers) == 1 else "mixed"
            ),
            "feedback_packet_id": feedback_packet.packet_id,
            "feedback_packet_schema": feedback_packet.schema,
            "independent_reproduction": {
                "status": (
                    (independent_validation or {}).get(
                        "deterministic_replay", {}
                    ).get("status", "not_observed")
                ),
                "lookahead_probe": (independent_validation or {}).get(
                    "lookahead_probe", {"status": "not_observed"}
                ),
                "reason": "one fresh top-candidate replay and causal probe",
            },
        }
    }
    return score, metrics, evidence


def _validate_config(stage: str, raw: Any) -> dict[str, Any]:
    common_allowed = {
        "suite_id", "direction", "metric", "confidence_level", "instance_count",
        "repeats", "training_paths", "pricing_paths", "ridge_alpha",
        # Open algorithm-track limits.  They are optional and intentionally
        # omitted from the legacy task's canonical request unless supplied.
        "training_timeout_s", "training_cpu_seconds",
        "training_memory_bytes", "training_file_size_bytes",
        "prediction_timeout_s", "prediction_cpu_seconds",
        "prediction_memory_bytes", "prediction_file_size_bytes",
        # Research-only diagnostics are additive and never alter the primary
        # score, suite, seed, causal inputs, or evaluator-owned arithmetic.
        "research_mode", "independent_validation",
    }
    allowed = common_allowed | ({"outer_paths", "inner_paths"} if stage == "audit" else {"public_pathwise_samples"})
    _strict_keys(raw, required=set(), allowed=allowed, path="evaluation request.config")
    expected_direction = "max" if stage == "search" else "min"
    direction = raw.get("direction", expected_direction)
    if direction != expected_direction:
        raise ValueError(f"{stage} config.direction must be {expected_direction}")
    expected_metric = "paired_lower_bound_lcb" if stage == "search" else "normalized_primal_dual_confidence_gap"
    metric = raw.get("metric", expected_metric)
    if metric != expected_metric:
        raise ValueError(f"{stage} config.metric must be {expected_metric}")
    confidence = _strict_float(raw.get("confidence_level", 0.95), path="config.confidence_level", minimum=0.80, maximum=0.999)
    defaults = ({
        "instance_count": 4, "repeats": 2, "training_paths": 1024,
        "pricing_paths": 2048, "ridge_alpha": 1e-6,
    } if stage == "search" else {
        "instance_count": 3, "repeats": 1, "training_paths": 1536,
        "pricing_paths": 3072, "outer_paths": 768, "inner_paths": 16,
        "ridge_alpha": 1e-6,
    })
    result: dict[str, Any] = {
        "direction": direction,
        "metric": metric,
        "confidence_level": confidence,
        # Research mode labels the experiment; it never bypasses isolation or
        # changes the evaluator-owned score or causal input boundary.
        "research_mode": bool(raw.get("research_mode", False)),
        "independent_validation": bool(raw.get("independent_validation", True)),
        "instance_count": _strict_int(raw.get("instance_count", defaults["instance_count"]), path="config.instance_count", minimum=1, maximum=4 if stage == "search" else 8),
        "repeats": _strict_int(raw.get("repeats", defaults["repeats"]), path="config.repeats", minimum=1, maximum=5),
        "training_paths": _strict_int(raw.get("training_paths", defaults["training_paths"]), path="config.training_paths", minimum=64, maximum=20_000),
        "pricing_paths": _strict_int(raw.get("pricing_paths", defaults["pricing_paths"]), path="config.pricing_paths", minimum=64, maximum=50_000),
        "ridge_alpha": _strict_float(raw.get("ridge_alpha", defaults["ridge_alpha"]), path="config.ridge_alpha", minimum=1e-12, maximum=1.0),
    }
    for key in ("research_mode", "independent_validation", "public_pathwise_samples"):
        if key in raw:
            if not isinstance(raw[key], bool):
                raise ValueError(f"config.{key} must be boolean")
            result[key] = raw[key]
    if "training_timeout_s" in raw:
        result["training_timeout_s"] = _strict_float(
            raw["training_timeout_s"], path="config.training_timeout_s",
            minimum=0.1, maximum=600.0,
        )
    if "training_cpu_seconds" in raw:
        result["training_cpu_seconds"] = _strict_float(
            raw["training_cpu_seconds"], path="config.training_cpu_seconds",
            minimum=0.1, maximum=600.0,
        )
    if "training_memory_bytes" in raw:
        result["training_memory_bytes"] = _strict_int(
            raw["training_memory_bytes"], path="config.training_memory_bytes",
            minimum=16 * 1024 * 1024, maximum=8 * 1024 * 1024 * 1024,
        )
    if "training_file_size_bytes" in raw:
        result["training_file_size_bytes"] = _strict_int(
            raw["training_file_size_bytes"], path="config.training_file_size_bytes",
            minimum=1024, maximum=256 * 1024 * 1024,
        )
    if "prediction_timeout_s" in raw:
        result["prediction_timeout_s"] = _strict_float(
            raw["prediction_timeout_s"], path="config.prediction_timeout_s",
            minimum=0.05, maximum=60.0,
        )
    if "prediction_cpu_seconds" in raw:
        result["prediction_cpu_seconds"] = _strict_float(
            raw["prediction_cpu_seconds"], path="config.prediction_cpu_seconds",
            minimum=0.05, maximum=60.0,
        )
    if "prediction_memory_bytes" in raw:
        result["prediction_memory_bytes"] = _strict_int(
            raw["prediction_memory_bytes"], path="config.prediction_memory_bytes",
            minimum=16 * 1024 * 1024, maximum=8 * 1024 * 1024 * 1024,
        )
    if "prediction_file_size_bytes" in raw:
        result["prediction_file_size_bytes"] = _strict_int(
            raw["prediction_file_size_bytes"], path="config.prediction_file_size_bytes",
            minimum=1024, maximum=256 * 1024 * 1024,
        )
    if "suite_id" in raw:
        suite_id = raw["suite_id"]
        if not isinstance(suite_id, str) or not suite_id or len(suite_id) > MAX_SUITE_ID_CHARS:
            raise ValueError("config.suite_id must be a bounded non-empty string")
        result["suite_id"] = suite_id
    if stage == "audit":
        result["outer_paths"] = _strict_int(raw.get("outer_paths", defaults["outer_paths"]), path="config.outer_paths", minimum=64, maximum=10_000)
        result["inner_paths"] = _strict_int(raw.get("inner_paths", defaults["inner_paths"]), path="config.inner_paths", minimum=2, maximum=128)
    return result


def validate_evaluation_request(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {"schema", "stage", "task", "protocol", "seed", "suite_id", "config"}
    _strict_keys(raw, required=required, allowed=required, path="evaluation request")
    if raw["schema"] != REQUEST_SCHEMA:
        raise ValueError(f"evaluation request.schema must be {REQUEST_SCHEMA}")
    if raw["stage"] not in {"search", "audit"}:
        raise ValueError("evaluation request.stage must be search or audit")
    if raw["task"] not in SUPPORTED_TASK_NAMES:
        raise ValueError(
            "evaluation request.task must be one of: "
            + ", ".join(sorted(SUPPORTED_TASK_NAMES))
        )
    if raw["protocol"] not in SUPPORTED_TASK_PROTOCOLS:
        raise ValueError(
            "evaluation request.protocol must be one of: "
            + ", ".join(sorted(SUPPORTED_TASK_PROTOCOLS))
        )
    seed = _strict_int(raw["seed"], path="evaluation request.seed", minimum=0, maximum=MAX_REQUEST_SEED)
    suite_id = raw["suite_id"]
    if not isinstance(suite_id, str) or not suite_id or len(suite_id) > MAX_SUITE_ID_CHARS or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", suite_id):
        raise ValueError("evaluation request.suite_id has invalid syntax")
    config = _validate_config(raw["stage"], raw["config"])
    if config.get("suite_id", suite_id) != suite_id:
        raise ValueError("config.suite_id must match evaluation request.suite_id")
    normalized = {
        "schema": REQUEST_SCHEMA,
        "stage": raw["stage"],
        "task": raw["task"],
        "protocol": raw["protocol"],
        "seed": seed,
        "suite_id": suite_id,
        "config": config,
    }
    return normalized, config


def default_search_request() -> dict[str, Any]:
    """Manual smoke-test request; production harnesses pass a sealed argv[2]."""
    suite_id = (
        "bermudan-python-public-v2"
        if TASK_NAME == "bermudan_python_search"
        else "bermudan-public-v1"
    )
    raw = {
        "schema": REQUEST_SCHEMA,
        "stage": "search",
        "task": TASK_NAME,
        "protocol": TASK_PROTOCOL,
        "seed": 1729,
        "suite_id": suite_id,
        "config": {},
    }
    return validate_evaluation_request(raw)[0]


def _validate_candidate_protocol_binding(
    request: Mapping[str, Any],
    candidate_source: Path | None,
    candidate_manifest: Any,
) -> None:
    """Bind the whole-program task protocol to its executable program contract."""
    is_python_program = (
        candidate_source is not None
        and _is_python_program_manifest(candidate_manifest)
    )
    if request["protocol"] != PYTHON_PROGRAM_TASK_PROTOCOL:
        if is_python_program:
            raise ValueError(
                f"{PYTHON_PROGRAM_SCHEMA} requires protocol "
                f"{PYTHON_PROGRAM_TASK_PROTOCOL}"
            )
        return
    if request["task"] != "bermudan_python_search":
        raise ValueError(
            f"{PYTHON_PROGRAM_TASK_PROTOCOL} requires task bermudan_python_search"
        )
    if not is_python_program:
        raise ValueError(
            f"{PYTHON_PROGRAM_TASK_PROTOCOL} requires "
            f"{PYTHON_PROGRAM_SCHEMA} with entrypoint algorithm.py"
        )


def evaluate_submission(
    submission: Any,
    request: dict[str, Any] | None = None,
    *,
    candidate_source_dir: str | os.PathLike[str] | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]:
    algorithm_source: Path | None = None
    algorithm_manifest: Any = None
    candidate_source_names: list[str] | None = None
    if candidate_source_dir is not None:
        algorithm_source, algorithm_manifest = _candidate_source_manifest(
            candidate_source_dir, submission,
        )
        program = None
        normalized_candidate = _manifest_payload_for_metrics(algorithm_manifest)
        candidate_hash = _source_bundle_digest(
            algorithm_source,
            None if _is_python_program_manifest(algorithm_manifest)
            else _candidate_source_files(algorithm_manifest),
        )
        if _is_python_program_manifest(algorithm_manifest):
            from sandbox import read_source_tree
            _tree_hash, _hashes, _files = read_source_tree(
                algorithm_source, 64 * 1024 * 1024,
            )
            candidate_source_names = sorted(
                name for name in _files if name != "solve.sh"
            )
        else:
            candidate_source_names = list(
                _candidate_source_files(algorithm_manifest)
            )
        feature_hash = None
    else:
        program = validate_feature_program(submission)
        normalized_candidate = program
        feature_hash = _sha256_json(program)
        candidate_hash = feature_hash
    if request is None:
        request = default_search_request()
        config = _validate_config("search", request["config"])
    else:
        request, config = validate_evaluation_request(request)
    _validate_candidate_protocol_binding(
        request,
        algorithm_source,
        algorithm_manifest,
    )
    started = time.perf_counter()
    if request["stage"] == "search":
        score, stage_metrics, stage_evidence = _evaluate_search(
            program,
            request,
            config,
            candidate_source_dir=algorithm_source,
            candidate_manifest=algorithm_manifest,
        )
    else:
        score, stage_metrics, stage_evidence = _evaluate_audit(
            program,
            request,
            config,
            candidate_source_dir=algorithm_source,
            candidate_manifest=algorithm_manifest,
        )
    request_hash = _sha256_json(request)
    metrics = {
        "stage": request["stage"],
        "suite_id": request["suite_id"],
        "evaluation_request_sha256": request_hash,
        "feature_program_sha256": feature_hash,
        "candidate_hash": candidate_hash,
        "source_files": candidate_source_names,
        "candidate_kind": (
            _candidate_kind(algorithm_manifest)
            if algorithm_source is not None else "feature_ir"
        ),
        "feature_count": len(program["features"]) if program is not None else None,
        "algorithm_bundle_sha256": (
            candidate_hash
            if algorithm_source is not None
            and not _is_python_program_manifest(algorithm_manifest)
            else None
        ),
        "runner_type": (
            getattr(algorithm_manifest, "runner_type", None)
            if algorithm_source is not None else "ridge_lsmc"
        ),
        "runtime_seconds": time.perf_counter() - started,
        **stage_metrics,
    }
    if algorithm_source is not None:
        # ``protocol`` identifies the task/evaluation envelope.  The V5
        # mechanism card needs the separate wire protocol of the frozen policy
        # artifact so MLP, linear, and expression candidates remain
        # distinguishable.  The training entrypoint is part of the trusted
        # AlgorithmBundle contract and is not candidate-reported telemetry.
        metrics.update({
            "artifact_protocol": algorithm_manifest.schema,
            "entrypoint": _candidate_entrypoint(algorithm_manifest),
        })
    # Complete the packet's evaluator-side identity/runtime fields only after
    # the stage has returned.  This keeps the stage helpers reusable while
    # ensuring Context sees measured runtime rather than a zero placeholder.
    feedback_payload = metrics.get("feedback_packet")
    if isinstance(feedback_payload, dict):
        feedback_payload["candidate_id"] = candidate_hash
        # The stage helper cannot know the submitted candidate digest.  Make
        # packet and directional identities candidate-specific here so the
        # append-only ProblemState does not deduplicate observations from
        # different algorithms evaluated on the same suite/request.
        feedback_payload["packet_id"] = _canonical_feedback_id({
            "stage": request["stage"],
            "suite_id": request["suite_id"],
            "request_sha256": request_hash,
            "candidate_hash": candidate_hash,
        })
        observed_payload = feedback_payload.get("observed")
        if isinstance(observed_payload, dict):
            # Wall-clock timing is intentionally kept in the outer metrics and
            # runtime object.  It is host-load dependent, so embedding it in
            # the content-addressed feedback packet would make fixed-seed
            # replay differ for a reason unrelated to the algorithm.
            observed_payload["runtime_seconds"] = _feedback_marker(
                "wall-clock runtime is recorded outside the deterministic packet"
            )
        for directional_payload in feedback_payload.get("directional", []):
            if isinstance(directional_payload, dict):
                directional_payload["candidate_id"] = candidate_hash
                directional_payload["id"] = (
                    f"{candidate_hash}:" + str(
                        directional_payload.get("id", "observation")
                    )
                )
        evidence_payload = feedback_payload.get("evidence")
        if isinstance(evidence_payload, dict):
            evidence_payload["candidate_hash"] = candidate_hash
        stage_evidence_payload = stage_evidence.get(request["stage"])
        if isinstance(stage_evidence_payload, dict):
            stage_evidence_payload["feedback_packet_id"] = feedback_payload[
                "packet_id"
            ]
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "stage": request["stage"],
        "suite_id": request["suite_id"],
        "evaluation_request_sha256": request_hash,
        "feature_program_sha256": feature_hash,
        "candidate_hash": candidate_hash,
        "candidate_kind": metrics["candidate_kind"],
        "source_files": candidate_source_names,
        "candidate_supplied_prices_ignored": True,
        **stage_evidence,
    }
    return score, metrics, normalized_candidate, evidence


def fail(message: str) -> None:
    print(json.dumps({"error": message}))
    raise SystemExit(0)


def main() -> None:
    if len(sys.argv) < 2:
        fail(
            "usage: evaluator.py ARTIFACT_JSON [EVALUATION_REQUEST_JSON] "
            "[--candidate-source SOURCE_DIR]"
        )
    try:
        artifact_path = Path(sys.argv[1])
        candidate_source_dir = None
        request_path = None
        positional: list[str] = []
        index = 2
        while index < len(sys.argv):
            argument = sys.argv[index]
            if argument == "--candidate-source":
                if index + 1 >= len(sys.argv) or candidate_source_dir is not None:
                    raise ValueError("--candidate-source requires exactly one path")
                candidate_source_dir = sys.argv[index + 1]
                index += 2
                continue
            if argument.startswith("--candidate-source="):
                if candidate_source_dir is not None:
                    raise ValueError("--candidate-source may only be supplied once")
                candidate_source_dir = argument.split("=", 1)[1]
                if not candidate_source_dir:
                    raise ValueError("--candidate-source requires a path")
                index += 1
                continue
            positional.append(argument)
            index += 1
        if len(positional) > 1:
            raise ValueError("at most one evaluation request path is allowed")
        if artifact_path.is_dir():
            if candidate_source_dir is None and (
                (artifact_path / "manifest.json").is_file()
                and (
                    (artifact_path / "train.py").is_file()
                    or (artifact_path / "algorithm.py").is_file()
                )
            ):
                candidate_source_dir = str(artifact_path)
            artifact_path = artifact_path / "solution.json"
            if not artifact_path.is_file() and candidate_source_dir is not None:
                artifact_path = Path(candidate_source_dir) / "manifest.json"
        if not artifact_path.is_file():
            fail("artifact JSON not found")
        submission = json.loads(
            artifact_path.read_text(), object_pairs_hook=_unique_object,
        )
        request = None
        if positional:
            request_path = Path(positional[0])
            if not request_path.is_file():
                raise ValueError("evaluation request JSON not found")
            request = json.loads(
                request_path.read_text(), object_pairs_hook=_unique_object,
            )
        score, metrics, normalized, evidence = evaluate_submission(
            submission,
            request,
            candidate_source_dir=candidate_source_dir,
        )
    except (OSError, RecursionError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        fail(str(exc))
    print(json.dumps({
        "score": score,
        "metrics": metrics,
        "normalized_solution": normalized,
        "evidence": evidence,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
