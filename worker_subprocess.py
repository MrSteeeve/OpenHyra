"""Lifecycle management for trusted research workers on POSIX systems.

This module does not enforce host filesystem or network permissions. Those
belong to the deployment environment. Environment filtering prevents implicit
credential inheritance, not access to files readable by the worker's account.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Event
from typing import TypedDict
import math
import os
import signal
import subprocess
import tempfile
import time

LOG_BYTES = 64 * 1024
HOST_ENV_KEYS = (
    "PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX",
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
    "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "SYSTEMROOT", "LANG", "LC_ALL",
)
NUMERIC_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


class WorkerResult(TypedDict):
    returncode: int | None
    timed_out: bool
    cancelled: bool
    elapsed_s: float
    stdout: str
    stderr: str


def worker_environment(tmp_dir: Path, extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy only named host settings; put default home and temporary files in tmp_dir."""
    env = {key: os.environ[key] for key in HOST_ENV_KEYS if key in os.environ}
    env.update(NUMERIC_THREAD_ENV)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    if extra_env:
        for key, value in extra_env.items():
            if key.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")):
                raise ValueError("Credentials are not worker environment settings")
            env[key] = value
    env.update(HOME=str(tmp_dir), TMPDIR=str(tmp_dir), TEMP=str(tmp_dir), TMP=str(tmp_dir))
    return env


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    """Reclaim processes sharing this worker's POSIX process group."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _log_tail(path: Path, limit: int) -> str:
    """Read at most limit trailing bytes, replacing incomplete UTF-8 sequences."""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read(limit).decode("utf-8", errors="replace")


def _wait(
    process: subprocess.Popen[bytes], deadline: float, cancel_event: Event | None,
) -> tuple[bool, bool]:
    """Wait until exit, the wall-clock deadline, or a caller cancellation."""
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return False, True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True, False
        interval = remaining if cancel_event is None else min(remaining, 0.1)
        try:
            process.communicate(timeout=interval)
            return False, False
        except subprocess.TimeoutExpired:
            if cancel_event is None:
                return True, False


def run_worker_subprocess(
    command: Sequence[str | os.PathLike[str]], *, timeout_s: float,
    work_dir: Path | None = None, extra_env: Mapping[str, str] | None = None,
    cancel_event: Event | None = None, log_bytes: int = LOG_BYTES,
    scratch_dir: Path | None = None,
) -> WorkerResult:
    """Run one trusted worker with dedicated scratch, deadline and bounded log readback.

    The command is an argv sequence (never a shell). If work_dir is absent,
    cwd is automatically removed on return; otherwise the caller owns cwd.
    Scratch lives in the system temporary area, or the caller's scratch_dir.
    Stdout and stderr are spooled rather than buffered unboundedly in memory.
    Log readback retains their respective last log_bytes (64 KiB by default).
    Deadline uses monotonic wall-clock seconds, and cancellation or expiry
    kills the POSIX process group before the parent process is reaped.
    """
    if not command or isinstance(command, (str, bytes)):
        raise ValueError("command must be a nonempty argv sequence")
    if not math.isfinite(timeout_s) or timeout_s <= 0 or log_bytes <= 0:
        raise ValueError("timeout and log_bytes must be positive and finite")
    parent = None if work_dir is None else Path(work_dir).resolve(strict=True)
    scratch_parent = None if scratch_dir is None else Path(scratch_dir).resolve(strict=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="openhyra-worker-", dir=scratch_parent) as directory:
        scratch = Path(directory).resolve()
        tmp_dir = scratch / "tmp"
        tmp_dir.mkdir()
        cwd = scratch if parent is None else parent
        stdout_path = scratch / "stdout"
        stderr_path = scratch / "stderr"
        timed_out = False
        cancelled = False
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                process = subprocess.Popen(
                    list(command), cwd=cwd, env=worker_environment(tmp_dir, extra_env),
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    start_new_session=True,
                )
            except OSError as exc:
                return {
                    "returncode": None, "timed_out": False, "cancelled": False,
                    "elapsed_s": time.monotonic() - started, "stdout": "", "stderr": str(exc),
                }
            try:
                timed_out, cancelled = _wait(process, started + timeout_s, cancel_event)
            finally:
                _kill_group(process)
                process.communicate()
        return {
            "returncode": process.returncode, "timed_out": timed_out, "cancelled": cancelled,
            "elapsed_s": time.monotonic() - started, "stdout": _log_tail(stdout_path, log_bytes),
            "stderr": _log_tail(stderr_path, log_bytes),
        }
