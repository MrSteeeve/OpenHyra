"""Sandbox runner: executes a solution's solve.sh in an isolated copy, then
scores it with the task's TRUSTED evaluator (which lives outside the sandbox
and recomputes the score from the emitted artifact — candidate-reported
numbers are never trusted).

Tasks without an evaluator (legacy nanochat) fall back to parsing the
candidate's own output; that weaker trust model is documented in their TASK.md.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

# macOS seatbelt profile: candidate code gets no network and may write only
# inside its own sandbox directory (plus the system temp dirs Python needs).
SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{sandbox}"))
(allow file-write* (subpath "/private/var/folders"))
(allow file-write* (subpath "/private/tmp"))
(allow file-write* (subpath "/dev"))
"""


def _sandboxed_cmd(sandbox_dir, cmd):
    if sys.platform == "darwin":
        profile = SANDBOX_PROFILE.format(sandbox=sandbox_dir.resolve())
        return ["sandbox-exec", "-p", profile] + cmd
    return cmd  # non-macOS: no seatbelt available; run unconfined


def _trusted_score(evaluator, sandbox_dir, timeout_s=300):
    """Run the task's trusted evaluator on the sandbox output."""
    try:
        res = subprocess.run([sys.executable, str(evaluator), str(sandbox_dir)],
                             capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return None, "crash", {}, "evaluator timed out"
    line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    try:
        result = json.loads(line)
    except ValueError:
        return None, "crash", {}, f"evaluator produced no verdict: {res.stderr.strip()[:300]}"
    if "error" in result:
        return None, "crash", {}, f"evaluator rejected solution: {result['error']}"
    return float(result["score"]), "ok", result.get("metrics", {}), ""


def run_solution(solution_dir: Path, sandbox_dir: Path, task):
    """Copy solution to sandbox, run solve.sh, score via the trusted evaluator.

    Returns (score, status, log_tail, metrics).
    """
    sandbox_dir = Path(sandbox_dir)
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    shutil.copytree(solution_dir, sandbox_dir, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".git"))

    log_path = sandbox_dir / "run.log"
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.run(
                _sandboxed_cmd(sandbox_dir, ["bash", "solve.sh"]),
                cwd=sandbox_dir,
                env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(sandbox_dir),
                     "OPENHYRA_PYTHON": task.python_bin},
                stdout=log_f, stderr=subprocess.STDOUT,
                timeout=task.timeout_s, check=False,
            )
    except subprocess.TimeoutExpired:
        tail = f"killed after {task.timeout_s}s"
        if log_path.exists():
            partial = log_path.read_text(errors="replace").replace("\r", "\n")
            tail += "\n" + "\n".join(partial.splitlines()[-10:])
        return None, "timeout", tail, {}

    log = log_path.read_text(errors="replace")
    tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
    # A non-zero exit is a crash even if output was produced.
    if proc.returncode != 0:
        return None, "crash", tail, {}

    score, status, metrics, note = _trusted_score(task.evaluator, sandbox_dir)
    if note:
        tail = (tail + "\n[evaluator] " + note).strip()
    return score, status, tail, metrics
