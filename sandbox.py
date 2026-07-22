"""Isolated candidate execution followed by snapshot-based trusted scoring."""

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MAX_ARTIFACT_BYTES = 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024

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


def trusted_artifact_dir(sandbox_dir):
    """Return a parent-controlled directory outside the candidate write root."""
    sandbox_dir = Path(sandbox_dir)
    return sandbox_dir.parent / ".trusted_artifacts" / sandbox_dir.name


def _read_regular_file(path, max_bytes):
    """Read one untrusted artifact without following links or blocking on FIFOs."""
    path = Path(path)
    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError("solution.json not found") from exc
    if stat.S_ISLNK(before.st_mode):
        raise ValueError("solution.json must not be a symbolic link")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"could not safely open solution.json: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("solution.json must be a regular file")
        if info.st_nlink != 1:
            raise ValueError("solution.json must have exactly one hard link")
        if info.st_size > max_bytes:
            raise ValueError(
                f"solution.json exceeds the {max_bytes}-byte artifact limit"
            )
        chunks = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(fd, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise ValueError(
                f"solution.json exceeds the {max_bytes}-byte artifact limit"
            )
        return data
    finally:
        os.close(fd)


def _snapshot_artifact(artifact, trusted_dir, max_bytes):
    """Copy a validated candidate artifact into a fresh trusted directory."""
    data = _read_regular_file(artifact, max_bytes)
    trusted_dir = Path(trusted_dir)
    if trusted_dir.exists():
        shutil.rmtree(trusted_dir)
    trusted_dir.mkdir(parents=True)
    snapshot = trusted_dir / "solution.snapshot.json"
    snapshot.write_bytes(data)
    snapshot.chmod(0o444)
    return snapshot, data


def _kill_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _trusted_score(task, snapshot_path):
    started = time.perf_counter()
    timeout_s = int(getattr(task, "evaluator_timeout_s", 300))
    memory_mb = int(getattr(task, "evaluator_max_memory_mb", 512))
    output_mb = int(getattr(task, "max_output_mb", 64))
    command = [sys.executable, str(task.evaluator), str(snapshot_path)]
    limited = [
        sys.executable, "-c", LIMIT_WRAPPER,
        str(memory_mb * 1024 * 1024),
        str(output_mb * 1024 * 1024),
        str(timeout_s + 5),
        *command,
    ]
    try:
        res = subprocess.run(
            limited,
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
    trusted_dir = trusted_artifact_dir(sandbox_dir)
    max_artifact_bytes = int(getattr(
        task, "max_artifact_bytes", DEFAULT_MAX_ARTIFACT_BYTES,
    ))
    try:
        snapshot, snapshot_bytes = _snapshot_artifact(
            artifact, trusted_dir, max_artifact_bytes,
        )
    except (OSError, ValueError) as exc:
        return None, "crash", (log_tail + f"\n{exc}").strip(), base_metrics
    candidate_artifact_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()

    score, status, metrics, note, evaluator_seconds, normalized = _trusted_score(
        task, snapshot,
    )
    evaluated_artifact_sha256 = candidate_artifact_sha256
    if normalized is not None:
        evaluated_bytes = json.dumps(
            {"A": normalized}, separators=(",", ":"),
        ).encode()
        evaluated = trusted_dir / "evaluated_solution.json"
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
