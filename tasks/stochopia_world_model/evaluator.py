"""Trusted evaluator for the Stochopia world-model task.

A candidate is the full text of ``world_model.py``.  For every walk-forward
origin the evaluator writes that origin's public history prefix and query to
a fresh input directory and runs the program once in the training sandbox
(macOS Seatbelt: no network, writes only to its own output/tmp directories,
no access to the panel or other origins).  Because each origin is a separate
process that only ever receives data up to its own origin, look-ahead is
impossible by construction.  Scoring is done in this trusted process by
Stochopia's ``forecast_protocol``; candidate-reported metrics are never used.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sandbox import run_training_sandbox  # noqa: E402
from stochopia.research import forecast_protocol as fp  # noqa: E402
from stochopia.research.market_panel import MarketPanel  # noqa: E402

TASK_PROTOCOL = "openhyra-stochopia-world-model.v1"
PROGRAM_FILE = "world_model.py"
SEED_PROGRAM = Path(__file__).with_name("seed_solution") / PROGRAM_FILE
MAX_PROGRAM_BYTES = 64 * 1024
DEFAULT_ORIGIN_TIMEOUT_S = 20.0
DEFAULT_MEMORY_BYTES = 1024 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024  # uncompressed; a valid forecast is ~100 KB
EXPECTED_MEMBERS = {
    "returns.npy": (fp.N_PATHS, fp.HORIZON, 2),
    "q.npy": (fp.N_PATHS, len(fp.Q_NAMES)),
}


def _read_forecast(target: Path) -> tuple[np.ndarray, np.ndarray]:
    """Check every npz member header before any array is materialised."""

    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        if names != set(EXPECTED_MEMBERS):
            raise fp.ForecastError("forecast.npz must contain exactly 'returns' and 'q'")
        for name, shape in EXPECTED_MEMBERS.items():
            info = archive.getinfo(name)
            if info.file_size > MAX_MEMBER_BYTES:
                raise fp.ForecastError(f"{name} is larger than {MAX_MEMBER_BYTES} bytes")
            with archive.open(name) as handle:
                version = np.lib.format.read_magic(handle)
                header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                found_shape, _, dtype = header(handle)
            if tuple(found_shape) != shape or dtype.kind not in "fiu":
                raise fp.ForecastError(f"{name} must be numeric with shape {shape}, got {found_shape} {dtype}")
    with np.load(target, allow_pickle=False) as data:
        return np.array(data["returns"], dtype=float), np.array(data["q"], dtype=float)


def runtime_roots() -> tuple[Path, ...]:
    """The running interpreter's environment plus its base installation."""

    roots: list[Path] = []
    for candidate in (Path(sys.prefix), Path(sys.base_prefix)):
        resolved = candidate.resolve()
        if not any(resolved == item or item in resolved.parents for item in roots):
            roots.append(resolved)
    return tuple(roots)


class SandboxedProgram:
    """Callable forecaster that runs one candidate program per origin."""

    def __init__(self, program: str, *, workdir: Path, timeout_s: float = DEFAULT_ORIGIN_TIMEOUT_S,
                 memory_bytes: int = DEFAULT_MEMORY_BYTES):
        encoded = program.encode("utf-8")
        if not program.strip():
            raise ValueError("program is empty")
        if len(encoded) > MAX_PROGRAM_BYTES:
            raise ValueError(f"program exceeds {MAX_PROGRAM_BYTES} bytes")
        compile(program, PROGRAM_FILE, "exec")  # syntax check only; never executed here
        Path(workdir).mkdir(parents=True, exist_ok=False)
        # Seatbelt matches resolved paths (macOS /var -> /private/var).
        self.workdir = Path(workdir).resolve(strict=True)
        self.source_dir = self.workdir / "source"
        self.source_dir.mkdir()
        (self.source_dir / PROGRAM_FILE).write_text(program, encoding="utf-8")
        self.timeout_s = float(timeout_s)
        self.memory_bytes = int(memory_bytes)
        self.calls = 0
        self.wall_seconds = 0.0
        self.roots = runtime_roots()

    def __call__(self, history: Mapping[str, np.ndarray], query: Mapping[str, Any]):
        cell = self.workdir / f"origin_{self.calls:04d}"
        self.calls += 1
        input_dir, output_dir, tmp_dir = cell / "input", cell / "output", cell / "tmp"
        for path in (input_dir, output_dir, tmp_dir):
            path.mkdir(parents=True)
        np.savez(input_dir / "history.npz", **{k: np.asarray(v) for k, v in history.items()})
        (input_dir / "query.json").write_text(json.dumps(dict(query)), encoding="utf-8")
        result = run_training_sandbox(
            [sys.executable, PROGRAM_FILE, "--input", str(input_dir), "--output", str(output_dir)],
            source_dir=self.source_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            runtime_roots=self.roots,
            timeout_s=self.timeout_s,
            memory_bytes=self.memory_bytes,
            externally_isolated=False,  # fail closed without Seatbelt
        )
        self.wall_seconds += float(result.get("wall_seconds") or 0.0)
        try:
            if result["status"] != "ok":
                tail = self._sanitize((result.get("log_tail") or "").strip()[-800:])
                raise fp.ForecastError(f"sandbox {result['status']}: {tail}")
            target = output_dir / "forecast.npz"
            if not target.is_file() or target.is_symlink():
                raise fp.ForecastError("program did not write forecast.npz")
            try:
                return _read_forecast(target)
            except fp.ForecastError:
                raise
            except Exception as exc:  # malformed archives must not crash the trusted side
                raise fp.ForecastError(f"unreadable forecast.npz: {type(exc).__name__}: {exc}") from exc
        finally:
            shutil.rmtree(cell, ignore_errors=True)


    def _sanitize(self, text: str) -> str:
        """Drop local absolute paths before text can reach a model provider."""

        for path, label in ((str(self.workdir), "<sandbox>"), (str(Path.home()), "~")):
            text = text.replace(path, label)
        return text


def public_feedback(report: Mapping[str, Any]) -> dict[str, Any]:
    """Development feedback shown to the proposer: aggregates only, no dates."""

    def rounded(value):
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        return value

    keys = ("origins", "score", "score_lcb_95", "family_skill", "target_skill", "slices", "calibration")
    return {key: rounded(report[key]) for key in keys if key in report}


def evaluate_program(
    program: str,
    *,
    panel: MarketPanel,
    split: str,
    workdir: str | Path | None = None,
    timeout_s: float = DEFAULT_ORIGIN_TIMEOUT_S,
) -> dict[str, Any]:
    """Return ``{"status", "score", "report"|"error", "cost"}`` for one split."""

    started = time.monotonic()
    temporary = None
    if workdir is None:
        temporary = tempfile.TemporaryDirectory(prefix="openhyra-wm-")
        workdir = temporary.name
    run_dir = Path(workdir) / f"eval_{split}_{time.monotonic_ns()}"
    try:
        try:
            forecaster = SandboxedProgram(program, workdir=run_dir, timeout_s=timeout_s)
        except (SyntaxError, ValueError) as exc:
            return {"status": "invalid_program", "score": None, "error": f"{type(exc).__name__}: {exc}",
                    "cost": {"origins_run": 0, "seconds": time.monotonic() - started}}
        try:
            candidate = fp.collect_forecasts(forecaster, panel, split)
        except fp.ForecastError as exc:
            return {"status": "forecast_error", "score": None, "error": str(exc)[-1500:],
                    "cost": {"origins_run": forecaster.calls, "seconds": time.monotonic() - started}}
        report = fp.evaluate_forecasts(candidate, panel, reference=fp.reference_forecasts(panel, split))
        return {
            "status": "ok",
            "score": report["score"],
            "report": report,
            "cost": {"origins_run": forecaster.calls, "sandbox_seconds": round(forecaster.wall_seconds, 3),
                     "seconds": round(time.monotonic() - started, 3)},
        }
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)  # programs are archived by the caller
        if temporary is not None:
            temporary.cleanup()
