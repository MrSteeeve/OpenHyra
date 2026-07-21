"""Sandbox runner: executes a solution's solve.sh in an isolated copy, then
scores it with the task's TRUSTED evaluator (which lives outside the sandbox
and recomputes the score from the emitted artifact — candidate-reported
numbers are never trusted).

Tasks without an evaluator (legacy nanochat) fall back to parsing the
candidate's own output; that weaker trust model is documented in their TASK.md.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCORE_RE = re.compile(r"^val_bpb:\s*([0-9.]+)", re.MULTILINE)

METRIC_KEYS = ["val_bpb", "training_seconds", "total_seconds", "peak_vram_mb",
               "mfu_percent", "total_tokens_M", "num_steps", "num_params_M", "depth"]


def parse_metrics(log_text):
    metrics = {}
    for key in METRIC_KEYS:
        m = re.search(rf"^{key}:\s*([0-9.]+)", log_text, re.MULTILINE)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def _legacy_score(sandbox_dir, log):
    """Candidate-reported score (no trusted evaluator): solution.json then log grep."""
    sol_json = sandbox_dir / "solution.json"
    if sol_json.exists():
        try:
            result = json.loads(sol_json.read_text())
            if "val_bpb" in result:
                return float(result["val_bpb"]), "ok", parse_metrics(log)
        except (ValueError, TypeError):
            pass
    m = SCORE_RE.search(log)
    if m:
        return float(m.group(1)), "ok", parse_metrics(log)
    return None, "crash", {}


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
                ["bash", "solve.sh"],
                cwd=sandbox_dir,
                env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home()),
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
        return None, "crash", tail, parse_metrics(log)

    if task.evaluator is not None:
        score, status, metrics, note = _trusted_score(task.evaluator, sandbox_dir)
        if note:
            tail = (tail + "\n[evaluator] " + note).strip()
        return score, status, tail, metrics

    score, status, metrics = _legacy_score(sandbox_dir, log)
    return score, status, tail, metrics
