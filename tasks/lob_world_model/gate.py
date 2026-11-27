"""Run zero-data scientific checks in a dedicated worker process."""

from __future__ import annotations

import os
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path

from sandbox import run_worker_subprocess


def _runtime_pythonpath() -> str:
    """Resolve the two local package roots without inheriting the host PYTHONPATH."""
    location = find_spec("stochopia")
    if location is None or location.origin is None:
        raise RuntimeError("World-model gate requires the Stochopia neural runtime")
    stochopia = Path(location.origin).resolve().parents[1]
    openhyra = Path(__file__).resolve().parents[2]
    return os.pathsep.join((str(openhyra), str(stochopia)))


class CandidateGate:
    """Check one trusted research candidate in a separate, empty work directory.

    This verifies the tensor contract, not host permissions. Deployment-level
    isolation is the responsibility of the surrounding execution environment.
    """

    def __init__(self, *, timeout_s: float = 20.0, max_state_dim: int = 128) -> None:
        if type(max_state_dim) is not int or max_state_dim < 1:
            raise ValueError("max_state_dim must be a positive integer")
        self.timeout_s = timeout_s
        self.max_state_dim = max_state_dim

    def __call__(self, source: str) -> str | None:
        with tempfile.TemporaryDirectory(prefix="lob-world-model-gate-") as directory:
            root = Path(directory).resolve()
            worker = Path(__file__).resolve().with_name("gate_worker.py")
            candidate = root / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            completion = root / "checked"
            result = run_worker_subprocess(
                [sys.executable, "-s", "-B", str(worker), str(candidate), str(completion),
                 str(self.max_state_dim)],
                work_dir=root,
                extra_env={"PYTHONPATH": _runtime_pythonpath()},
                timeout_s=self.timeout_s,
            )
            if result["returncode"] == 0 and not result["timed_out"] and not result["cancelled"]:
                try:
                    with completion.open("rb") as marker:
                        value = marker.read(4)
                except OSError:
                    return "Candidate exited before completing its tensor checks"
                return None if value == b"ok\n" else "Candidate did not complete its tensor checks"
            if result["timed_out"]:
                status = "timeout"
            elif result["cancelled"]:
                status = "cancelled"
            else:
                status = "crash"
            trace = "\n".join((result["stdout"], result["stderr"])).strip()
            trace = trace.replace(str(root), "<worker>").replace(str(Path.home()), "~")
            return f"Worker {status} (returncode {result['returncode']})\n{trace[-3500:]}"
