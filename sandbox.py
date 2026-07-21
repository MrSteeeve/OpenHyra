"""Sandbox runner: executes a solution's solve.sh in an isolated copy and parses the score.

One GPU -> one sandbox at a time (the semaphore of the Hyra pipeline degenerates to 1).
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

SCORE_RE = re.compile(r"^val_bpb:\s*([0-9.]+)", re.MULTILINE)

# Full diagnostic block printed by train.py — stored per-run so the Context Agent
# can expose throughput/size tradeoffs (tokens seen, steps, MFU) to Proposal Agents.
METRIC_KEYS = ["val_bpb", "training_seconds", "total_seconds", "peak_vram_mb",
               "mfu_percent", "total_tokens_M", "num_steps", "num_params_M", "depth"]


def parse_metrics(log_text):
    metrics = {}
    for key in METRIC_KEYS:
        m = re.search(rf"^{key}:\s*([0-9.]+)", log_text, re.MULTILINE)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def run_solution(solution_dir: Path, sandbox_dir: Path, python_bin: str, timeout_s: int = 660):
    """Copy solution to sandbox, run solve.sh, return (score, status, log_tail, metrics).

    score is val_bpb (lower is better) or None on crash/timeout.
    metrics is the parsed summary block from train.py (empty dict on crash).
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
        tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
        # Prefer the machine-readable result (solution.json, as in Hyra-results);
        # fall back to grepping the log for older solution formats.
        sol_json = sandbox_dir / "solution.json"
        if sol_json.exists():
            try:
                result = json.loads(sol_json.read_text())
                if "val_bpb" in result:
                    return float(result["val_bpb"]), "ok", tail, parse_metrics(log)
            except (ValueError, TypeError):
                pass
        m = SCORE_RE.search(log)
        if m:
            return float(m.group(1)), "ok", tail, parse_metrics(log)
        return None, "crash", tail, {}
    except subprocess.TimeoutExpired:
        return None, "timeout", f"killed after {timeout_s}s", {}
