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
EVALUATION_REQUEST_SCHEMA = "openhyra-evaluation-request.v1"
SOURCE_FILES = (
    "auditing.py",
    "context_agent.py",
    "eb.py",
    "external_formal_runner.py",
    "harness.py",
    "llm_backend.py",
    "proposal_agent.py",
    "provenance.py",
    "reporting.py",
    "sandbox.py",
    "stopping.py",
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


def _stage_config(task, stage):
    evaluation = getattr(task, "evaluation", {}) or {}
    config = evaluation.get(f"{stage}_stage")
    return dict(config) if isinstance(config, dict) else None


def evaluation_suite_id(task, stage, config=None):
    config = _stage_config(task, stage) if config is None else config
    configured = config.get("suite_id") if isinstance(config, dict) else None
    if configured is not None:
        if (
            not isinstance(configured, str)
            or not configured.strip()
            or len(configured) > 256
        ):
            raise ValueError(
                f"evaluation.{stage}_stage.suite_id must be bounded text"
            )
        return configured
    return f"{task.name}.{stage}.v1"


def derive_search_seed(task, trial_seed):
    """Derive one run-level CRN seed shared by every search candidate."""
    material = {
        "domain": EVALUATION_REQUEST_SCHEMA,
        "stage": "search",
        "task": task.name,
        "protocol": task.protocol,
        "suite_id": evaluation_suite_id(task, "search"),
        "trial_seed": trial_seed,
    }
    return int(sha256_json(material)[:16], 16) & ((1 << 63) - 1)


def build_evaluation_request(task, stage, seed):
    if stage not in {"search", "audit"}:
        raise ValueError("evaluation stage must be search or audit")
    config = _stage_config(task, stage)
    if config is None:
        return None
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= (1 << 63) - 1
    ):
        raise ValueError("evaluation seed must be a 63-bit non-negative integer")
    return {
        "schema": EVALUATION_REQUEST_SCHEMA,
        "stage": stage,
        "task": task.name,
        "protocol": task.protocol,
        "seed": seed,
        "suite_id": evaluation_suite_id(task, stage, config),
        # suite_id is promoted to the envelope; top_k is a harness-only
        # selection control and must never become trusted evaluator input.
        "config": {
            key: value for key, value in config.items()
            if key not in {"suite_id", "top_k"}
        },
    }


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
                       candidates_per_context, trial_seed,
                       stopping_policy=None):
    root = Path(root)
    task_support_sha256 = {}
    for path in sorted(task.dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(task.dir)
        if (
            "seed_solution" in relative.parts
            or "__pycache__" in relative.parts
            or relative.as_posix() in {
                "task.json", "TASK.md", "evaluator.py",
            }
        ):
            continue
        task_support_sha256[relative.as_posix()] = sha256_file(path)
    seed_solution_sha256 = {
        path.relative_to(task.seed_dir).as_posix(): sha256_file(path)
        for path in sorted(task.seed_dir.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    search_config = _stage_config(task, "search")
    search_request = (
        build_evaluation_request(
            task, "search", derive_search_seed(task, trial_seed),
        )
        if search_config is not None else None
    )
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
            "support_sha256": task_support_sha256,
            "seed_solution_sha256": seed_solution_sha256,
            "formalization": {
                "config": getattr(task, "formalization", {}),
                "runner": getattr(task, "formal_runner_identity", None),
            },
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
            "research_revision_attempts": getattr(
                task, "research_revision_attempts", 0,
            ),
            "trial_seed": trial_seed,
            "evaluation_request": search_request,
        },
        "evaluation": {
            "search_request_sha256": (
                sha256_json(search_request) if search_request is not None else None
            ),
            "audit_stage": _stage_config(task, "audit"),
            "audit_seed_strategy": (
                "secrets.randbits(63)-after-top-k-freeze"
                if _stage_config(task, "audit") is not None else None
            ),
        },
        "limits": {
            "candidate_timeout_s": task.timeout_s,
            "max_memory_mb": task.max_memory_mb,
            "max_output_mb": task.max_output_mb,
            "max_artifact_bytes": task.max_artifact_bytes,
            "evaluator_timeout_s": task.evaluator_timeout_s,
            "evaluator_max_memory_mb": task.evaluator_max_memory_mb,
        },
        "stopping_policy": stopping_policy or {},
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
    for field in (
            "task", "source_sha256", "search", "limits",
            "evaluation", "stopping_policy", "environment"):
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
