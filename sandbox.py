"""Sandbox runner: executes a solution's solve.sh in an isolated copy and parses the score.

One GPU -> one sandbox at a time (the semaphore of the Hyra pipeline degenerates to 1).
"""

import re
import shutil
import subprocess
from pathlib import Path

SCORE_RE = re.compile(r"^val_bpb:\s*([0-9.]+)", re.MULTILINE)


def run_solution(solution_dir: Path, sandbox_dir: Path, python_bin: str, timeout_s: int = 900):
    """Copy solution to sandbox, run solve.sh, return (score, status, log_tail).

    score is val_bpb (lower is better) or None on crash/timeout.
    """
    sandbox_dir = Path(sandbox_dir)
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    shutil.copytree(solution_dir, sandbox_dir, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".git"))

    log_path = sandbox_dir / "run.log"
    try:
        with open(log_path, "w") as log_f:
            subprocess.run(
                ["bash", "solve.sh"],
                cwd=sandbox_dir,
                env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home()),
                     "OPENHYRA_PYTHON": python_bin},
                stdout=log_f, stderr=subprocess.STDOUT,
                timeout=timeout_s, check=False,
            )
        log = log_path.read_text(errors="replace")
        m = SCORE_RE.search(log)
        tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
        if m:
            return float(m.group(1)), "ok", tail
        return None, "crash", tail
    except subprocess.TimeoutExpired:
        return None, "timeout", f"killed after {timeout_s}s"
