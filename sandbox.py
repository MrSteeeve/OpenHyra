"""Isolated candidate execution followed by snapshot-based trusted scoring."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{sandbox}"))
(allow file-write* (literal "/dev/null"))
(deny file-read* (literal "{evaluator}"))
"""


def _seatbelt_escape(path):
    return str(Path(path).resolve()).replace("\\", "\\\\").replace('"', '\\"')


def _sandboxed_cmd(sandbox_dir, evaluator, cmd):
    if sys.platform == "darwin":
        profile = SANDBOX_PROFILE.format(
            sandbox=_seatbelt_escape(sandbox_dir),
            evaluator=_seatbelt_escape(evaluator),
        )
        return ["sandbox-exec", "-p", profile] + cmd
    if os.environ.get("OPENHYRA_ALLOW_UNSANDBOXED") == "1":
        return cmd
    raise RuntimeError(
        "OpenHyra fails closed without macOS Seatbelt; set "
        "OPENHYRA_ALLOW_UNSANDBOXED=1 only inside an external container/VM"
    )


LIMIT_WRAPPER = r"""
import os, resource, sys
limits = (
    (resource.RLIMIT_AS, int(sys.argv[1])),
    (resource.RLIMIT_FSIZE, int(sys.argv[2])),
    (resource.RLIMIT_CPU, int(sys.argv[3])),
)
for key, value in limits:
    try:
        _soft, hard = resource.getrlimit(key)
        target = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(key, (target, target))
    except (OSError, ValueError):
        pass
os.execvp(sys.argv[4], sys.argv[4:])
"""


def _limited_cmd(task, command):
    memory = int(getattr(task, "max_memory_mb", 1024)) * 1024 * 1024
    output = int(getattr(task, "max_output_mb", 64)) * 1024 * 1024
    return [
        sys.executable, "-c", LIMIT_WRAPPER,
        str(memory), str(output), str(int(task.timeout_s) + 5),
        *command,
    ]


def _kill_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _trusted_score(evaluator, snapshot_path, timeout_s=300):
    started = time.perf_counter()
    try:
        res = subprocess.run(
            [sys.executable, str(evaluator), str(snapshot_path)],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "crash", {}, "evaluator timed out", time.perf_counter() - started, None
    elapsed = time.perf_counter() - started
    line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    try:
        result = json.loads(line)
    except ValueError:
        note = f"evaluator produced no verdict: {res.stderr.strip()[:300]}"
        return None, "crash", {}, note, elapsed, None
    if "error" in result:
        return None, "crash", {}, f"evaluator rejected solution: {result['error']}", elapsed, None
    return (
        float(result["score"]), "ok", result.get("metrics", {}), "", elapsed,
        result.get("normalized_A"),
    )


def run_solution(solution_dir: Path, sandbox_dir: Path, task):
    """Run a candidate, kill its process group, snapshot output, then score."""
    total_started = time.perf_counter()
    sandbox_dir = Path(sandbox_dir)
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    shutil.copytree(
        solution_dir, sandbox_dir,
        ignore=shutil.ignore_patterns(
            ".venv", "__pycache__", ".git", ".tmp",
            "run.log", "train.log", "solution.json", "solution.snapshot.json",
        ),
    )
    tmp_dir = sandbox_dir / ".tmp"
    tmp_dir.mkdir()
    log_path = sandbox_dir / "run.log"

    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(sandbox_dir),
        "TMPDIR": str(tmp_dir),
        "OPENHYRA_PYTHON": task.python_bin,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        command = _limited_cmd(task, _sandboxed_cmd(
            sandbox_dir, task.evaluator, ["bash", "solve.sh"],
        ))
    except RuntimeError as exc:
        return None, "crash", str(exc), {}

    solver_started = time.perf_counter()
    timed_out = False
    with open(log_path, "w") as log_stream:
        proc = subprocess.Popen(
            command, cwd=sandbox_dir, env=env,
            stdout=log_stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=task.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also removes descendants deliberately left behind after a normal
            # parent exit, closing the artifact mutation race before snapshot.
            _kill_process_group(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    solver_seconds = time.perf_counter() - solver_started

    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
    base_metrics = {"solver_seconds": solver_seconds}
    if timed_out:
        return None, "timeout", (
            f"killed process group after {task.timeout_s}s\n{log_tail}"
        ).strip(), base_metrics
    if proc.returncode != 0:
        return None, "crash", log_tail, base_metrics

    artifact = sandbox_dir / "solution.json"
    if not artifact.exists():
        return None, "crash", (log_tail + "\nsolution.json not found").strip(), base_metrics
    snapshot = sandbox_dir / "solution.snapshot.json"
    snapshot_bytes = artifact.read_bytes()
    snapshot.write_bytes(snapshot_bytes)
    snapshot.chmod(0o444)
    candidate_artifact_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()

    score, status, metrics, note, evaluator_seconds, normalized = _trusted_score(
        task.evaluator, snapshot,
    )
    evaluated_artifact_sha256 = candidate_artifact_sha256
    if normalized is not None:
        evaluated_bytes = json.dumps(
            {"A": normalized}, separators=(",", ":"),
        ).encode()
        evaluated = sandbox_dir / "evaluated_solution.json"
        evaluated.write_bytes(evaluated_bytes)
        evaluated.chmod(0o444)
        evaluated_artifact_sha256 = hashlib.sha256(evaluated_bytes).hexdigest()
    metrics.update(base_metrics)
    metrics.update({
        "evaluator_seconds": evaluator_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "artifact_sha256": evaluated_artifact_sha256,
    })
    if note:
        log_tail = (log_tail + "\n[evaluator] " + note).strip()
    return score, status, log_tail, metrics
