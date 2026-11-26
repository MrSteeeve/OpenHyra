"""Trusted research worker lifecycle and legacy scientific artifact APIs.

Host permissions are a deployment concern. The subprocess runner only
controls execution, temporary files, implicit environment and wall time.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Event
import os

from worker_subprocess import (
    LOG_BYTES, NUMERIC_THREAD_ENV, WorkerResult, run_worker_subprocess, worker_environment,
)
from sandbox_artifacts import (
    ALGORITHM_SOURCE_FILES, DEFAULT_MAX_ARTIFACT_BYTES, EVALUATION_REQUEST_SCHEMA,
    PYTHON_PROGRAM_SOURCE_TREE_SENTINEL, SOURCE_TREE_IGNORES,
    _apply_formalization_verdict, _snapshot_artifact, evaluate_trusted_artifact,
    read_regular_file, read_source_tree, remove_sealed_tree, run_solution,
    snapshot_algorithm_source, snapshot_source_tree, source_tree_hash,
    trusted_artifact_dir, validate_evaluation_request, write_evaluation_request,
)


def run_training_sandbox(
    command: Sequence[str | os.PathLike[str]], *, source_dir: Path,
    input_dir: Path, output_dir: Path, tmp_dir: Path,
    runtime_roots: Sequence[Path], timeout_s: float = 60,
    cancel_event: Event | None = None,
) -> dict[str, object]:
    """Compatibility adapter for legacy train/predict cells.

    Input and runtime roots are retained as historical interface parameters;
    they no longer define host access permissions. No memory or CPU limit is
    measured or enforced. Output counts are descriptive after-run statistics.
    """
    result = run_worker_subprocess(
        command, timeout_s=timeout_s, work_dir=source_dir, scratch_dir=tmp_dir,
        cancel_event=cancel_event,
    )
    if result["cancelled"]:
        status = "cancelled"
    elif result["timed_out"]:
        status = "timeout"
    elif result["returncode"] == 0:
        status = "ok"
    else:
        status = "crash"
    entries = 0
    output_bytes = 0
    for path in Path(output_dir).rglob("*"):
        entries += 1
        if path.is_file() and not path.is_symlink():
            output_bytes += path.stat().st_size
    tail = (result["stdout"] + "\n" + result["stderr"]).encode("utf-8")
    return {
        "status": status,
        "returncode": result["returncode"],
        "log_tail": tail[-LOG_BYTES:].decode("utf-8", errors="replace").strip(),
        "wall_seconds": result["elapsed_s"],
        "isolation": "process_group",
        "research_fallback": False,
        "memory_limit_enforced": False,
        "peak_memory_bytes": None,
        "output_entries": entries,
        "output_bytes": output_bytes,
    }
