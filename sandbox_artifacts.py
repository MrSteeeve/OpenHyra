"""Snapshot-based trusted scoring and provenance for historical research tasks.

Process lifecycle management lives in worker_subprocess. Deployment owns host
permission boundaries; this module preserves artifact, source, and request
identity and trusted evaluator verdicts.
"""

import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

from worker_subprocess import run_worker_subprocess

DEFAULT_MAX_ARTIFACT_BYTES = 1024 * 1024
SOURCE_TREE_IGNORES = {
    ".venv", "__pycache__", ".git", ".tmp",
    "run.log", "train.log", "solution.json", "solution.snapshot.json",
    "evidence.json", "PROPOSAL.md",
}
ALGORITHM_SOURCE_FILES = ("train.py", "manifest.json")
PYTHON_PROGRAM_SOURCE_TREE_SENTINEL = "__openhyra_python_source_tree__"
HEX_DIGITS = frozenset("0123456789abcdef")
EVALUATION_REQUEST_SCHEMA = "openhyra-evaluation-request.v1"


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_DIGITS)
    )


def _canonical_json_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()


def validate_evaluation_request(request):
    """Validate the exact trusted evaluator request envelope."""
    if not isinstance(request, dict):
        raise ValueError("evaluation request must be an object")
    expected = {
        "schema", "stage", "task", "protocol", "seed", "suite_id", "config",
    }
    if set(request) != expected:
        raise ValueError(
            "evaluation request fields must be exactly: "
            + ", ".join(sorted(expected))
        )
    if request.get("schema") != EVALUATION_REQUEST_SCHEMA:
        raise ValueError("unsupported evaluation request schema")
    if request.get("stage") not in {"search", "audit"}:
        raise ValueError("evaluation request stage must be search or audit")
    for field in ("task", "protocol", "suite_id"):
        value = request.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise ValueError(
                f"evaluation request {field} must be bounded non-empty text"
            )
    seed = request.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= (1 << 63) - 1
    ):
        raise ValueError("evaluation request seed must be a 63-bit non-negative integer")
    if not isinstance(request.get("config"), dict):
        raise ValueError("evaluation request config must be an object")
    try:
        encoded = _canonical_json_bytes(request)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation request must be canonical-JSON serializable") from exc
    if len(encoded) > 1024 * 1024:
        raise ValueError("evaluation request exceeds the 1 MiB limit")
    return encoded


def write_evaluation_request(path, request):
    """Write one immutable, parent-controlled request and return its digest."""
    encoded = validate_evaluation_request(request)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    path.write_bytes(encoded)
    path.chmod(0o400)
    return hashlib.sha256(encoded).hexdigest()


def trusted_artifact_dir(sandbox_dir):
    """Return a parent-controlled directory outside the candidate write root."""
    sandbox_dir = Path(sandbox_dir)
    return sandbox_dir.parent / ".trusted_artifacts" / sandbox_dir.name


def read_regular_file(path: str | Path, max_bytes: int, *, label: str | None = None) -> bytes:
    """Read a bounded ordinary artifact for snapshot identity and validation.

    Source snapshots must be self-contained regular files, not symbolic or hard links to
    other inputs.  Host-level file permissions are the deployment's concern.
    """
    path = Path(path)
    label = label or path.name
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    if not path.exists():
        raise ValueError(f"{label} not found")
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    if path.stat().st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")
    if max_bytes < 0:
        raise ValueError(f"{label} byte limit must be non-negative")
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
    return data


def _configured_algorithm_source_files(task):
    """Return the task-owned source allowlist for an AlgorithmBundle.

    The harness validates real task specifications before execution, but this
    helper also protects the lower-level sandbox API when it is called with a
    lightweight task object or from a standalone test.
    """
    if (
        getattr(task, "candidate_mode", "legacy") == "python_program"
        and getattr(task, "candidate_source_tree", False)
    ):
        return (PYTHON_PROGRAM_SOURCE_TREE_SENTINEL,)
    configured = getattr(task, "candidate_source_files", None)
    if configured is None:
        configured = ALGORITHM_SOURCE_FILES
    if not isinstance(configured, (list, tuple)) or not configured:
        raise ValueError("algorithm bundle source_files must be a non-empty list")
    result = []
    for name in configured:
        if (
            not isinstance(name, str)
            or not name
            or "\x00" in name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or "\\" in name
            or name in result
        ):
            raise ValueError("algorithm bundle source_files contains an unsafe path")
        result.append(name)
    return tuple(result)


def remove_sealed_tree(root):
    """Delete a trusted tree whose directories were sealed read-only (0500).

    Plain ``rmtree`` cannot unlink entries of a read-only directory, so every
    directory is made owner-writable (top-down, before descending) first; a
    leftover tree may keep salted pricing paths, hidden-suite inputs or frozen
    sources readable by later processes.  A cleanup failure is raised only
    when no exception is already propagating (this is meant for ``finally`` /
    ``except`` blocks); otherwise the primary exception keeps priority and the
    cleanup failure is attached to it as a note (logged on Python < 3.11).
    """
    primary = sys.exc_info()[1]
    root = Path(root)
    nofollow = os.chmod in os.supports_follow_symlinks

    def unseal(path):
        if nofollow:
            os.chmod(path, 0o700, follow_symlinks=False)
        elif not os.path.islink(path):
            os.chmod(path, 0o700)

    try:
        if root.is_symlink():
            raise ValueError(f"refusing to remove symlinked tree root {root}")
        unseal(root)
        for current, child_directories, _filenames in os.walk(root):
            for name in child_directories:
                unseal(os.path.join(current, name))
        shutil.rmtree(root)
    except (OSError, ValueError) as exc:
        if primary is None:
            raise
        note = f"cleanup of {root} also failed: {exc!r}"
        if hasattr(primary, "add_note"):
            primary.add_note(note)
        else:
            print(note, file=sys.stderr)


def snapshot_algorithm_source(source_dir, destination, task, max_bytes):
    """Copy only declared AlgorithmBundle source files into a trusted root.

    Solver plumbing may need task-owned files such as ``solve.sh``.  The
    evaluator/training bridge does not: it should receive only the declared
    algorithm source, so search and final-audit use the same source set and
    source identity cannot change through an untracked helper file.
    """
    source = Path(source_dir)
    target_root = Path(destination)
    if target_root.exists() or target_root.is_symlink():
        raise ValueError("algorithm source destination must be fresh")
    names = _configured_algorithm_source_files(task)
    target_root.mkdir(mode=0o700, parents=True)
    try:
        if names == (PYTHON_PROGRAM_SOURCE_TREE_SENTINEL,):
            # The trusted recursive snapshot applies the same symlink-free,
            # bounded tree walk used for provenance.  Task-specific policy is
            # enforced by the evaluator after this copy.
            for relative, data, mode in _source_tree_entries(source, int(max_bytes)):
                # solve.sh is task-owned transport plumbing and must never be
                # exposed to the candidate evaluator as executable source.
                if relative.as_posix() == "solve.sh":
                    continue
                target = target_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(mode & ~0o222)
        else:
            for name in sorted(names):
                data = read_regular_file(
                    source / name,
                    int(max_bytes),
                    label=f"algorithm source file {name}",
                )
                target = target_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o400)
        for directory in sorted(
            (path for path in target_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o500)
        target_root.chmod(0o500)
    except Exception:
        remove_sealed_tree(target_root)
        raise
    return target_root


def _source_tree_entries(source_dir, max_bytes):
    """Yield one bounded, symlink-free snapshot of a candidate source tree."""
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise ValueError("candidate source directory not found")
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise ValueError("candidate source must be a real directory")

    total = 0
    for current, directories, filenames in os.walk(
            source_dir, topdown=True, followlinks=False):
        current = Path(current)
        kept_directories = []
        for name in sorted(directories):
            if name in SOURCE_TREE_IGNORES:
                continue
            child = current / name
            if child.is_symlink() or not child.is_dir():
                relative = child.relative_to(source_dir).as_posix()
                raise ValueError(
                    f"candidate source directory {relative} must not be a link"
                )
            kept_directories.append(name)
        directories[:] = kept_directories

        for name in sorted(filenames):
            if name in SOURCE_TREE_IGNORES:
                continue
            path = current / name
            relative = path.relative_to(source_dir)
            remaining = max_bytes - total
            data = read_regular_file(
                path, max(0, remaining),
                label=f"candidate source file {relative.as_posix()}",
            )
            total += len(data)
            mode = path.stat().st_mode & 0o777
            yield relative, data, mode


def _source_manifest_hash(hashes):
    payload = json.dumps(
        hashes, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def source_tree_hash(source_dir, max_bytes):
    """Hash exactly the source files that are eligible for execution/commit."""
    tree_hash, hashes, _files = read_source_tree(source_dir, max_bytes)
    return tree_hash, hashes


def read_source_tree(source_dir, max_bytes):
    """Read one complete source snapshot for hash validation or export."""
    hashes = {}
    files = {}
    for relative, data, _mode in _source_tree_entries(source_dir, max_bytes):
        name = relative.as_posix()
        hashes[name] = hashlib.sha256(data).hexdigest()
        files[name] = data
    return _source_manifest_hash(hashes), hashes, files


def snapshot_source_tree(source_dir, trusted_source_dir, max_bytes):
    """Seal candidate source bytes in a parent-controlled directory."""
    trusted_source_dir = Path(trusted_source_dir)
    if trusted_source_dir.exists():
        shutil.rmtree(trusted_source_dir)
    trusted_source_dir.mkdir(parents=True)
    hashes = {}
    try:
        for relative, data, mode in _source_tree_entries(source_dir, max_bytes):
            destination = trusted_source_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            destination.chmod(mode & ~0o222)
            hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    except Exception:
        shutil.rmtree(trusted_source_dir, ignore_errors=True)
        raise
    return _source_manifest_hash(hashes), hashes


def _snapshot_artifact(artifact, trusted_dir, max_bytes):
    """Copy a validated candidate artifact into a fresh trusted directory."""
    data = read_regular_file(
        artifact, max_bytes, label="solution.json",
    )
    trusted_dir = Path(trusted_dir)
    trusted_dir.mkdir(parents=True, exist_ok=True)
    snapshot = trusted_dir / "solution.snapshot.json"
    if snapshot.exists() or snapshot.is_symlink():
        snapshot.unlink()
    snapshot.write_bytes(data)
    snapshot.chmod(0o444)
    return snapshot, data


def _trusted_score(
        task, snapshot_path, cancel_event=None, *, evaluation_request=None,
        trusted_dir=None, candidate_source_dir=None):
    """Run the trusted evaluator for one frozen artifact.

    Legacy evaluators receive exactly the historical ``solution.json`` (and,
    when configured, the evaluation request).  Algorithm-bundle tasks may
    additionally receive a parent-controlled, sealed source directory.  The
    directory is passed as an explicit argv value rather than imported here.
    Trusted evaluation and any candidate training run in separate worker
    processes; host-level permissions belong to the deployment.
    """
    timeout_s = float(getattr(task, "evaluator_timeout_s", 300))
    output_mb = int(getattr(task, "max_output_mb", 64))
    command = [sys.executable, str(Path(task.evaluator).resolve()), str(Path(snapshot_path).resolve())]
    request_sha256 = None
    request_path = None
    if evaluation_request is not None:
        if trusted_dir is None:
            raise ValueError("trusted_dir is required when passing an evaluation request")
        request_path = Path(trusted_dir).resolve() / "evaluation_request.json"
        request_sha256 = write_evaluation_request(request_path, evaluation_request)
        command.append(str(request_path))
    if candidate_source_dir is not None:
        command.extend(("--candidate-source", str(Path(candidate_source_dir).resolve())))
    try:
        result = run_worker_subprocess(
            command, timeout_s=timeout_s, cancel_event=cancel_event,
            work_dir=None if trusted_dir is None else Path(trusted_dir),
            log_bytes=max(1, output_mb * 1024 * 1024),
        )
    finally:
        if request_path is not None and request_path.exists():
            request_path.chmod(0o600)
            request_path.unlink()
    elapsed = result["elapsed_s"]
    stdout = result["stdout"]
    stderr = result["stderr"]
    if result["timed_out"]:
        return None, "crash", {}, "evaluator timed out", elapsed, None, None, request_sha256
    if result["cancelled"]:
        return None, "cancelled", {}, "evaluator cancelled", elapsed, None, None, request_sha256
    returncode = result["returncode"]
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    try:
        result = json.loads(line)
    except ValueError:
        note = f"evaluator produced no verdict: {stderr.strip()[:300]}"
        return None, "crash", {}, note, elapsed, None, None, request_sha256
    if not isinstance(result, dict):
        return (
            None, "crash", {}, "evaluator verdict must be an object", elapsed,
            None, None, request_sha256,
        )
    if "error" in result:
        return (
            None, "crash", {},
            f"evaluator rejected solution: {result['error']}",
            elapsed, None, None, request_sha256,
        )
    if returncode != 0:
        note = f"evaluator exited with status {returncode}: {stderr.strip()[:300]}"
        return None, "crash", {}, note, elapsed, None, None, request_sha256
    normalized = result.get("normalized_solution")
    if normalized is None and result.get("normalized_A") is not None:
        # Backward compatibility for task evaluators using the original
        # normalized_A response contract.
        normalized = {"A": result["normalized_A"]}
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return (
            None, "crash", {}, "evaluator metrics must be an object", elapsed,
            None, None, request_sha256,
        )
    try:
        score = float(result["score"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return (
            None, "crash", {}, "evaluator score must be numeric", elapsed,
            None, None, request_sha256,
        )
    if not math.isfinite(score):
        return (
            None, "crash", {}, "evaluator score must be finite", elapsed,
            None, None, request_sha256,
        )
    return (
        score, "ok", metrics, "", elapsed,
        normalized, result.get("evidence"), request_sha256,
    )


def evaluate_trusted_artifact(
        task, artifact_path, trusted_dir, evaluation_request,
        cancel_event=None, candidate_source_dir=None):
    """Evaluate an already-frozen artifact without running candidate code."""
    trusted_dir = Path(trusted_dir)
    if trusted_dir.exists():
        shutil.rmtree(trusted_dir)
    trusted_dir.mkdir(parents=True)
    max_artifact_bytes = int(getattr(
        task, "max_artifact_bytes", DEFAULT_MAX_ARTIFACT_BYTES,
    ))
    try:
        snapshot, snapshot_bytes = _snapshot_artifact(
            artifact_path, trusted_dir, max_artifact_bytes,
        )
    except (OSError, ValueError) as exc:
        return {
            "score": None,
            "status": "crash",
            "metrics": {},
            "note": f"could not freeze trusted artifact: {exc}",
            "artifact_sha256": None,
            "request_sha256": None,
        }
    artifact_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    (
        score, status, metrics, note, evaluator_seconds, normalized, evidence,
        request_sha256,
    ) = _trusted_score(
        task, snapshot, cancel_event,
        evaluation_request=evaluation_request,
        trusted_dir=trusted_dir,
        candidate_source_dir=candidate_source_dir,
    )
    metrics = dict(metrics)
    metrics.update({
        "evaluator_seconds": evaluator_seconds,
        "artifact_sha256": artifact_sha256,
        "evaluation_request_sha256": request_sha256,
        "evaluation_stage": evaluation_request["stage"],
        "evaluation_suite_id": evaluation_request["suite_id"],
    })
    return {
        "score": score,
        "status": status,
        "metrics": metrics,
        "note": note,
        "artifact_sha256": artifact_sha256,
        "request_sha256": request_sha256,
        "normalized_solution": normalized,
        "evidence": evidence,
    }


def _apply_formalization_verdict(task, normalized, evidence, metrics):
    """Run an optional task-owned proof gate and promote only trusted claims."""
    research = (
        normalized.get("research")
        if isinstance(normalized, dict) else None
    )
    if not isinstance(research, dict):
        return
    request = research.get("formalization")
    if request is None:
        return
    formal_started = time.perf_counter()
    verifier = getattr(task, "verify_formalization", None)
    config = getattr(task, "formalization", {}) or {}
    if not callable(verifier):
        verdict = {
            "target": "lean4",
            "status": "unavailable",
            "reason": "task_formalizer_not_configured",
        }
    else:
        try:
            verdict = verifier(
                request,
                research.get("claims", []),
                runner=getattr(task, "formal_runner", None),
                trusted_files=getattr(task, "formal_spec_files", {}),
                command_prefix=tuple(
                    config.get(
                        "command_prefix", ["lake", "env", "lean"],
                    )
                ),
                toolchain=config.get("toolchain"),
                mathlib_revision=config.get("mathlib_revision"),
                allowed_axioms=tuple(config.get(
                    "allowed_axioms",
                    ["Classical.choice", "Quot.sound", "propext"],
                )),
            )
        except Exception as exc:
            verdict = {
                "target": "lean4",
                "status": "infrastructure_error",
                "reason": "formal_verifier_exception",
                "failure": {
                    "phase": "trusted_parent",
                    "detail": repr(exc)[:4000],
                },
            }
    metrics["formal_verifier_seconds"] = (
        time.perf_counter() - formal_started
    )

    if not isinstance(verdict, dict) or verdict.get("status") not in {
        "not_submitted",
        "unavailable",
        "rejected",
        "verified",
        "infrastructure_error",
    }:
        verdict = {
            "target": "lean4",
            "status": "infrastructure_error",
            "reason": "invalid_formal_verifier_verdict",
        }
    research_evidence = evidence.setdefault("research", {})
    claims = {
        item.get("id"): item
        for item in research_evidence.get("claims", [])
        if isinstance(item, dict)
    }
    requested_proofs = {}
    for item in request.get("proofs", []):
        if (
            isinstance(item, dict)
            and isinstance(item.get("claim_id"), str)
            and isinstance(item.get("term"), str)
        ):
            requested_proofs[item["claim_id"]] = hashlib.sha256(
                item["term"].strip().encode("utf-8")
            ).hexdigest()
    expected_binding = None
    binding_error = None
    validator = getattr(task, "validate_formalization_request", None)
    wrapper_builder = getattr(task, "build_formalization_wrapper", None)
    audit_builder = getattr(task, "build_formalization_audit", None)
    if (
        callable(validator)
        and callable(wrapper_builder)
        and callable(audit_builder)
    ):
        try:
            sealed_request, theorem_types = validator(
                request,
                research.get("claims", []),
                allow_sealed_hashes=True,
            )
            wrapper, theorem_names = wrapper_builder(
                sealed_request,
                theorem_types,
                trusted_spec_source=getattr(
                    task, "formal_spec_files", {}
                ).get("OpenHyraSumDiff/Spec.lean"),
            )
            audit = audit_builder(theorem_names)
            expected_binding = {
                "wrapper_sha256": hashlib.sha256(wrapper).hexdigest(),
                "audit_sha256": hashlib.sha256(audit).hexdigest(),
                "theorem_types": dict(theorem_types),
                "theorem_names": dict(theorem_names),
            }
        except Exception as exc:
            binding_error = repr(exc)[:4_000]
    if verdict.get("status") == "verified":
        promotion_error = None
        verified_payload = verdict.get("verified_claim_ids")
        verified_ids_valid = (
            isinstance(verified_payload, list)
            and all(isinstance(item, str) for item in verified_payload)
            and len(verified_payload) == len(set(verified_payload))
        )
        if (
            not requested_proofs
            or not verified_ids_valid
            or set(verified_payload) != set(requested_proofs)
        ):
            promotion_error = "formal_verifier_claim_set_mismatch"
        expected_request_hash = metrics.get(
            "formalization_request_sha256"
        )
        if (
            promotion_error is None
            and (
                not isinstance(expected_request_hash, str)
                or verdict.get("request_sha256") != expected_request_hash
            )
        ):
            promotion_error = "formal_verifier_request_hash_mismatch"
        verdict_proofs = verdict.get("proofs")
        verdict_proof_map = {}
        if isinstance(verdict_proofs, list):
            for item in verdict_proofs:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("claim_id"), str)
                    or not isinstance(item.get("proof_sha256"), str)
                    or item["claim_id"] in verdict_proof_map
                ):
                    verdict_proof_map = {}
                    break
                verdict_proof_map[item["claim_id"]] = item["proof_sha256"]
        if (
            promotion_error is None
            and verdict_proof_map != requested_proofs
        ):
            promotion_error = "formal_verifier_proof_hash_mismatch"
        require_environment_binding = bool(
            config.get("mathlib_revision")
        )
        if (
            promotion_error is None
            and require_environment_binding
            and expected_binding is None
        ):
            promotion_error = "formal_claim_binding_unavailable"
        if (
            promotion_error is None
            and expected_binding is not None
            and (
                verdict.get("wrapper_sha256")
                != expected_binding["wrapper_sha256"]
                or verdict.get("audit_sha256")
                != expected_binding["audit_sha256"]
                or verdict.get("theorem_types")
                != expected_binding["theorem_types"]
                or verdict.get("theorem_names")
                != expected_binding["theorem_names"]
            )
        ):
            promotion_error = "formal_claim_binding_mismatch"
        runner_identity = getattr(task, "formal_runner_identity", None)
        spec_sha256 = getattr(task, "formal_spec_sha256", None)
        runtime_attestation = verdict.get("runtime_attestation")
        if (
            promotion_error is None
            and require_environment_binding
            and (
                not isinstance(spec_sha256, str)
                or not _is_sha256(spec_sha256)
                or not isinstance(runner_identity, dict)
                or not _is_sha256(runner_identity.get("sha256"))
                or not isinstance(runtime_attestation, dict)
                or any(
                    not _is_sha256(runtime_attestation.get(field))
                    for field in (
                        "environment_sha256",
                        "lean_binary_sha256",
                        "mathlib_tree_sha256",
                    )
                )
                or runtime_attestation.get("toolchain")
                != config.get("toolchain")
                or runtime_attestation.get("mathlib_revision")
                != config.get("mathlib_revision")
            )
        ):
            promotion_error = "formal_environment_binding_mismatch"
        if (
            promotion_error is None
            and (
                any(claim_id not in claims for claim_id in requested_proofs)
                or any(
                    claims[claim_id].get("status") == "refuted"
                    for claim_id in requested_proofs
                )
            )
        ):
            promotion_error = (
                "formal_proof_conflicts_with_trusted_refutation"
            )
        if promotion_error is not None:
            verdict = {
                **verdict,
                "status": "infrastructure_error",
                "reason": promotion_error,
                **(
                    {
                        "failure": {
                            "phase": "trusted_parent_binding",
                            "detail": binding_error,
                        },
                    }
                    if binding_error is not None else {}
                ),
            }
    runner_identity = getattr(task, "formal_runner_identity", None)
    trusted_environment = {
        "spec_sha256": getattr(task, "formal_spec_sha256", None),
        "runner_sha256": (
            runner_identity.get("sha256")
            if isinstance(runner_identity, dict) else None
        ),
        "toolchain": (
            (getattr(task, "formalization", {}) or {}).get("toolchain")
        ),
        "mathlib_revision": (
            (getattr(task, "formalization", {}) or {}).get(
                "mathlib_revision"
            )
        ),
        "runtime_attestation": (
            verdict.get("runtime_attestation")
            if isinstance(verdict.get("runtime_attestation"), dict)
            else None
        ),
    }
    verdict = {
        **verdict,
        "trusted_environment": trusted_environment,
    }
    metrics["formal_spec_sha256"] = trusted_environment["spec_sha256"]
    metrics["formal_runner_sha256"] = trusted_environment["runner_sha256"]
    metrics["formal_toolchain"] = trusted_environment["toolchain"]
    metrics["formal_mathlib_revision"] = trusted_environment[
        "mathlib_revision"
    ]
    runtime_attestation = trusted_environment["runtime_attestation"] or {}
    metrics["formal_environment_sha256"] = runtime_attestation.get(
        "environment_sha256"
    )
    metrics["formal_lean_binary_sha256"] = runtime_attestation.get(
        "lean_binary_sha256"
    )
    metrics["formal_mathlib_tree_sha256"] = runtime_attestation.get(
        "mathlib_tree_sha256"
    )

    research_evidence["formalization"] = verdict
    metrics["formalization_status"] = verdict.get("status", "infrastructure_error")
    for source, destination in (
        ("request_sha256", "formalization_request_sha256"),
        ("wrapper_sha256", "formal_wrapper_sha256"),
        ("audit_sha256", "formal_audit_sha256"),
    ):
        if verdict.get(source):
            metrics[destination] = verdict[source]
    proof_hashes = sorted(requested_proofs.values())
    if proof_hashes:
        metrics["proof_sha256"] = hashlib.sha256(
            json.dumps(proof_hashes, separators=(",", ":")).encode()
        ).hexdigest()

    verified_ids = set(verdict.get("verified_claim_ids", []))
    has_refutation = (
        any(
            isinstance(metrics.get(field), int)
            and not isinstance(metrics.get(field), bool)
            and metrics.get(field) > 0
            for field in (
                "refuted_claim_count",
                "refuted_obligation_count",
                "refuted_certificate_count",
            )
        )
        or any(
            claim.get("status") == "refuted"
            for claim in claims.values()
        )
        or (
            isinstance(research_evidence.get("construction"), dict)
            and research_evidence["construction"].get("status")
            == "contains_refutation"
        )
        or any(
            isinstance(item, dict)
            and item.get("status") == "refuted"
            for item in research_evidence.get("certificates", [])
        )
    )
    if verdict.get("status") == "verified":
        # Promotion is atomic: all ids, hashes and refutation checks above
        # succeed before any claim status is changed.
        for claim_id in verified_ids:
            claim = claims.get(claim_id)
            claim["status"] = "formal_checked"
            claim["formalization_request_sha256"] = verdict.get(
                "request_sha256"
            )
        if has_refutation:
            research_evidence["status"] = (
                "formal_checked_with_refutation"
            )
            metrics["research_rank"] = max(
                metrics.get("research_rank", 0), 70
            )
            metrics["evidence_level"] = (
                "formal_checked_with_refutation"
            )
        else:
            research_evidence["status"] = "formal_checked"
            metrics["research_rank"] = max(
                metrics.get("research_rank", 0), 80
            )
            metrics["evidence_level"] = "formal_checked"
    elif verdict.get("status") == "infrastructure_error":
        if has_refutation:
            research_evidence["status"] = "contains_refutation"
        else:
            research_evidence["status"] = "infrastructure_error"
            metrics["research_rank"] = -1

    formally_checked = [
        item for item in claims.values()
        if item.get("status") == "formal_checked"
    ]
    refuted = [
        item for item in claims.values()
        if item.get("status") == "refuted"
    ]
    bounded = [
        item for item in claims.values()
        if item.get("status") == "bounded_supported"
    ]
    metrics["formally_checked_claim_count"] = len(formally_checked)
    metrics["refuted_claim_count"] = len(refuted)
    metrics["bounded_supported_claim_count"] = len(bounded)
    metrics["formal_checked_claim_templates"] = sorted({
        item.get("template")
        for item in formally_checked
        if isinstance(item.get("template"), str)
    })
    metrics["formal_checked_targets"] = [
        {
            "claim_id": item.get("id"),
            "template": item.get("template"),
            "target": item.get("target"),
        }
        for item in sorted(
            formally_checked,
            key=lambda claim: str(claim.get("id")),
        )
        if isinstance(item.get("target"), dict)
    ]


def run_solution(solution_dir: Path, sandbox_dir: Path, task):
    """Run a candidate, kill its process group, snapshot output, then score."""
    total_started = time.perf_counter()
    cancel_event = getattr(task, "cancel_event", None)
    if cancel_event is not None and cancel_event.is_set():
        return None, "cancelled", "candidate cancelled before solver launch", {
            "solver_seconds": 0.0,
        }
    sandbox_dir = Path(sandbox_dir)
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    trusted_dir = trusted_artifact_dir(sandbox_dir)
    if trusted_dir.exists():
        shutil.rmtree(trusted_dir)
    trusted_dir.mkdir(parents=True)
    max_source_bytes = int(
        getattr(task, "max_source_bytes", 0)
        or int(getattr(task, "max_output_mb", 64)) * 1024 * 1024
    )
    try:
        source_snapshot_sha256, _source_hashes = snapshot_source_tree(
            solution_dir, trusted_dir / "source", max_source_bytes,
        )
        shutil.copytree(trusted_dir / "source", sandbox_dir)
    except (OSError, ValueError) as exc:
        return None, "crash", f"could not seal candidate source: {exc}", {}
    log_path = sandbox_dir / "run.log"
    solver_command = ["bash", getattr(task, "solve_entrypoint", "solve.sh")]
    result = run_worker_subprocess(
        solver_command, timeout_s=float(task.timeout_s), work_dir=sandbox_dir,
        extra_env={"OPENHYRA_PYTHON": os.fspath(task.python_bin)},
        cancel_event=cancel_event,
    )
    log = "\n".join(part for part in (result["stdout"], result["stderr"]) if part)
    log_path.write_text(log, encoding="utf-8")
    log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
    base_metrics = {
        "solver_seconds": result["elapsed_s"],
        "source_snapshot_sha256": source_snapshot_sha256,
        "research_fallback": False,
    }
    if result["cancelled"]:
        return None, "cancelled", (
            f"cancelled solver process group\n{log_tail}"
        ).strip(), base_metrics
    if result["timed_out"]:
        return None, "timeout", (
            f"killed process group after {task.timeout_s}s\n{log_tail}"
        ).strip(), base_metrics
    if result["returncode"] != 0:
        return None, "crash", log_tail, base_metrics

    artifact = sandbox_dir / "solution.json"
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

    candidate_source_dir = None
    if getattr(task, "candidate_mode", "legacy") in {
        "algorithm_bundle", "python_program",
    }:
        # The evaluator receives only the task-declared candidate program
        # files.  For ``python_program`` these are executable source rather
        # than parameters for a registered trusted runner.
        try:
            candidate_source_dir = snapshot_algorithm_source(
                trusted_dir / "source",
                trusted_dir / "algorithm_source",
                task,
                max_source_bytes,
            )
        except (OSError, ValueError) as exc:
            return None, "crash", (
                log_tail + f"\ncould not prepare algorithm source: {exc}"
            ).strip(), base_metrics

    (
        score, status, metrics, note, evaluator_seconds, normalized, evidence,
        request_sha256,
    ) = _trusted_score(
        task, snapshot, cancel_event,
        evaluation_request=getattr(task, "search_evaluation_request", None),
        trusted_dir=trusted_dir,
        candidate_source_dir=candidate_source_dir,
    )
    if normalized is not None and evidence is not None:
        _apply_formalization_verdict(
            task, normalized, evidence, metrics,
        )
    evaluated_artifact_sha256 = candidate_artifact_sha256
    if normalized is not None:
        evaluated_bytes = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        evaluated = trusted_dir / "evaluated_solution.json"
        evaluated.write_bytes(evaluated_bytes)
        evaluated.chmod(0o444)
        evaluated_artifact_sha256 = hashlib.sha256(evaluated_bytes).hexdigest()
    if evidence is not None:
        evidence_bytes = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        evidence_path = trusted_dir / "evidence.json"
        evidence_path.write_bytes(evidence_bytes)
        evidence_path.chmod(0o444)
        metrics["evidence_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    metrics.update(base_metrics)
    metrics.update({
        "evaluator_seconds": evaluator_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "artifact_sha256": evaluated_artifact_sha256,
    })
    request = getattr(task, "search_evaluation_request", None)
    if request is not None:
        metrics.update({
            "evaluation_request_sha256": request_sha256,
            "evaluation_stage": request["stage"],
            "evaluation_suite_id": request["suite_id"],
        })
    if note:
        log_tail = (log_tail + "\n[evaluator] " + note).strip()
    return score, status, log_tail, metrics
