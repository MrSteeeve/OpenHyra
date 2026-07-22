"""Immutable run provenance and single-writer locking."""

import fcntl
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

RUN_MANIFEST_SCHEMA = 1
SOURCE_FILES = (
    "context_agent.py",
    "eb.py",
    "harness.py",
    "llm_backend.py",
    "proposal_agent.py",
    "provenance.py",
    "reporting.py",
    "sandbox.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload):
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _command_output(command, cwd=None):
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True,
            timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or result.stderr).strip()


def command_version(command):
    output = _command_output(command)
    return output.splitlines()[0] if output else None


def git_metadata(root):
    root = Path(root)
    commit = _command_output(["git", "rev-parse", "HEAD"], cwd=root)
    status = _command_output(["git", "status", "--porcelain"], cwd=root)
    diff = _command_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    dirty_material = ((status or "") + "\n" + (diff or "")).encode()
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_state_sha256": hashlib.sha256(dirty_material).hexdigest(),
    }


def build_run_manifest(task, root, *, backend, model, workers,
                       candidates_per_context, trial_seed):
    root = Path(root)
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": task.run_id,
        "task": {
            "name": task.name,
            "protocol": task.protocol,
            "config_sha256": sha256_file(task.dir / "task.json"),
            "description_sha256": sha256_file(task.dir / "TASK.md"),
            "evaluator_sha256": sha256_file(task.evaluator),
        },
        "source_sha256": {
            name: sha256_file(root / name)
            for name in SOURCE_FILES
        },
        "search": {
            "backend": backend,
            "model": model,
            "workers": workers,
            "eval_concurrency": task.eval_concurrency,
            "candidates_per_context": candidates_per_context,
            "candidate_repair_attempts": task.candidate_repair_attempts,
            "trial_seed": trial_seed,
        },
        "limits": {
            "candidate_timeout_s": task.timeout_s,
            "max_memory_mb": task.max_memory_mb,
            "max_output_mb": task.max_output_mb,
            "max_artifact_bytes": task.max_artifact_bytes,
            "evaluator_timeout_s": task.evaluator_timeout_s,
            "evaluator_max_memory_mb": task.evaluator_max_memory_mb,
        },
        "git": git_metadata(root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "backend_cli": command_version([backend, "--version"]),
        },
        "initial_invocation": [sys.executable, *sys.argv],
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def write_run_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_run_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(
            f"run provenance is missing: {path}; legacy runs cannot be resumed"
        )
    manifest = json.loads(path.read_text())
    expected_hash = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items()
                if key != "manifest_sha256"}
    if expected_hash != sha256_json(unsigned):
        raise RuntimeError(f"run provenance checksum mismatch: {path}")
    return manifest


def validate_run_manifest(recorded, current):
    """Reject resume when any result-affecting source or setting drifted."""
    mismatches = []
    for field in ("task", "source_sha256", "search", "limits", "environment"):
        if recorded.get(field) != current.get(field):
            mismatches.append(field)
    if mismatches:
        raise RuntimeError(
            "run provenance drift in " + ", ".join(mismatches) +
            "; start a new --run-id instead of mixing experiments"
        )
    return recorded


class RunLock:
    """Non-blocking, process-wide single-writer lock for one run directory."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(self.path, "a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError(
                f"run {self.path.parent.name!r} is already owned by another harness process"
            ) from exc

    def release(self):
        if self.stream is None:
            return
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None
