"""One-shot private acceptance audit for frozen Experience Bank artifacts."""

import hashlib
import json
import math
import re
import secrets
import shutil
import time
from pathlib import Path

from provenance import build_evaluation_request, sha256_json
from sandbox import (
    evaluate_trusted_artifact,
    read_regular_file,
    source_tree_hash,
)
from stopping import incomplete_contexts, write_termination


FINAL_AUDIT_SCHEMA = "openhyra-final-audit.v1"
FREEZE_MANIFEST_SCHEMA = "openhyra-audit-freeze.v1"
ALGORITHM_BUNDLE_SCHEMA = "openhyra-algorithm-bundle.v1"
DEFAULT_ALGORITHM_SOURCE_FILES = ("train.py", "manifest.json")
SAFE_RECORD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_private_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _algorithm_source_files(task):
    """Return the task-owned source allowlist for an AlgorithmBundle.

    The harness validates this list when it constructs a real ``Task``.  The
    small amount of validation here is intentional: final-audit orchestration
    also receives lightweight task doubles in tests and must not turn a bad
    task attribute into an arbitrary path traversal.
    """
    configured = getattr(task, "candidate_source_files", None)
    if configured is None:
        configured = getattr(task, "source_files", None)
    if configured is None:
        configured = DEFAULT_ALGORITHM_SOURCE_FILES
    if not isinstance(configured, (list, tuple)) or not configured:
        raise RuntimeError(
            "algorithm bundle task has no configured source_files"
        )
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
            raise RuntimeError(
                "algorithm bundle source_files contains an unsafe path"
            )
        result.append(name)
    return tuple(result)


def _algorithm_bundle_digest(source_dir, task):
    """Compute the canonical AlgorithmBundle digest from sealed source bytes.

    This deliberately mirrors the harness/evaluator contract: generated
    ``solution.json`` and telemetry are not part of an algorithm identity;
    only the configured source files, with their byte sizes and SHA-256s, are
    included.
    """
    files = []
    source_dir = Path(source_dir)
    max_bytes = int(
        getattr(task, "max_source_bytes", 0)
        or int(getattr(task, "max_output_mb", 64)) * 1024 * 1024
    )
    for name in sorted(_algorithm_source_files(task)):
        data = read_regular_file(
            source_dir / name,
            max_bytes,
            label=f"algorithm source file {name}",
        )
        files.append({
            "path": name,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    payload = {"schema": ALGORITHM_BUNDLE_SCHEMA, "files": files}
    return sha256_json(payload)


def _copy_algorithm_source(source_dir, destination, task):
    """Copy and seal exactly the configured AlgorithmBundle source files.

    Returns the canonical bundle digest.  ``read_regular_file`` gives the
    copy the same no-symlink/no-hard-link and byte-limit checks used by the
    normal candidate intake path.
    """
    source_dir = Path(source_dir)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    max_bytes = int(
        getattr(task, "max_source_bytes", 0)
        or int(getattr(task, "max_output_mb", 64)) * 1024 * 1024
    )
    files = []
    try:
        for name in sorted(_algorithm_source_files(task)):
            data = read_regular_file(
                source_dir / name,
                max_bytes,
                label=f"algorithm source file {name}",
            )
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o400)
            files.append({
                "path": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
        # The evaluator only needs to read/execute the sealed source.  Keep
        # every directory non-writable so a candidate subprocess cannot alter
        # the frozen bytes between candidates.
        for directory in sorted(
                (path for path in destination.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts), reverse=True):
            directory.chmod(0o500)
        destination.chmod(0o500)
    except Exception:
        # The caller has already committed to a one-shot audit.  Remove a
        # partial source copy so a failed freeze cannot be mistaken for a
        # complete candidate bundle.
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return sha256_json({"schema": ALGORITHM_BUNDLE_SCHEMA, "files": files})


def _source_artifact_protocol(source_dir):
    """Read the declared artifact protocol from a frozen source manifest.

    This is metadata only; the evaluator remains the authority that fully
    validates the manifest before scoring.  Returning ``None`` for malformed
    JSON lets the normal evaluator failure path preserve its diagnostics.
    """
    try:
        payload = json.loads(
            read_regular_file(
                Path(source_dir) / "manifest.json",
                1024 * 1024,
                label="frozen algorithm manifest",
            )
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    schema = payload.get("schema") if isinstance(payload, dict) else None
    return schema if isinstance(schema, str) and schema else None


def _valid_scored_records(records):
    return [
        record for record in records
        if record.get("status") == "ok"
        and isinstance(record.get("score"), (int, float))
        and not isinstance(record.get("score"), bool)
        and math.isfinite(float(record["score"]))
    ]


def select_top_k(records, direction, top_k):
    """Select search winners deterministically, with record id as tie-breaker."""
    if direction not in {"min", "max"}:
        raise ValueError("search direction must be min or max")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("audit top_k must be a positive integer")
    ordered = sorted(
        _valid_scored_records(records),
        key=lambda record: (
            -float(record["score"])
            if direction == "max" else float(record["score"]),
            str(record.get("id", "")),
        ),
    )
    selected = []
    seen_artifacts = set()
    for record in ordered:
        metrics = record.get("metrics", {})
        # For Python candidates the normalized manifest can be identical while
        # the training algorithm differs.  Deduplicate by the source-bundle
        # identity first; legacy Feature IR records continue to use their
        # normalized artifact hash.
        artifact_hash = metrics.get("algorithm_bundle_sha256") or metrics.get(
            "artifact_sha256"
        )
        if not isinstance(artifact_hash, str) or artifact_hash in seen_artifacts:
            continue
        seen_artifacts.add(artifact_hash)
        selected.append(record)
        if len(selected) == top_k:
            break
    return selected


def _freeze_candidates(
        task, records, destination, manifest_sha256, run_manifest, now):
    """Validate provenance, then copy every selected normalized artifact."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(
            "final audit freeze directory already exists; refusing to reuse it"
        )
    # The freeze contains candidate source code and private audit inputs; it
    # is not a public export directory.  Keep it owner-only while retaining
    # write access for the trusted evaluator's per-candidate working roots.
    destination.mkdir(mode=0o700, parents=True)
    algorithm_mode = (
        getattr(task, "candidate_mode", "legacy") == "algorithm_bundle"
    )
    # Resolve the configured list once, before creating any candidate slots.
    # This makes a malformed task fail before the freeze can be mistaken for a
    # completed snapshot and keeps every candidate on one canonical allowlist.
    algorithm_source_files = (
        _algorithm_source_files(task) if algorithm_mode else ()
    )
    candidates = []
    for rank, record in enumerate(records, start=1):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not SAFE_RECORD_ID.fullmatch(record_id):
            raise RuntimeError("selected audit record has no valid id")
        metadata = record.get("metadata", {})
        if metadata.get("run_manifest_sha256") != manifest_sha256:
            raise RuntimeError(
                f"{record_id} does not belong to the frozen run manifest"
            )
        if metadata.get("task_provenance") != run_manifest.get("task"):
            raise RuntimeError(
                f"{record_id} task provenance does not match the run manifest"
            )
        if metadata.get("source_sha256") != run_manifest.get("source_sha256"):
            raise RuntimeError(
                f"{record_id} harness provenance does not match the run manifest"
            )
        metrics = record.get("metrics", {})
        expected_artifact_hash = metrics.get("artifact_sha256")
        expected_source_hash = metrics.get("source_snapshot_sha256")
        if not isinstance(expected_artifact_hash, str):
            raise RuntimeError(f"{record_id} lacks artifact provenance")
        if not isinstance(expected_source_hash, str):
            raise RuntimeError(f"{record_id} lacks source provenance")

        record_dir = Path(record.get("path", ""))
        expected_record_dir = (
            Path(task.run_dir) / "eb" / "solutions" / record_id
        )
        if record_dir.resolve() != expected_record_dir.resolve():
            raise RuntimeError(
                f"{record_id} record path is outside its Experience Bank slot"
            )
        source_hash, _source_files = source_tree_hash(
            record_dir,
            int(
                getattr(task, "max_source_bytes", 0)
                or int(getattr(task, "max_output_mb", 64)) * 1024 * 1024
            ),
        )
        if source_hash != expected_source_hash:
            raise RuntimeError(
                f"{record_id} source provenance mismatch before audit freeze"
            )
        artifact = record_dir / "solution.json"
        artifact_bytes = read_regular_file(
            artifact,
            int(getattr(task, "max_artifact_bytes", 1024 * 1024)),
            label=f"{record_id} normalized solution.json",
        )
        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
        if artifact_hash != expected_artifact_hash:
            raise RuntimeError(
                f"{record_id} artifact provenance mismatch before audit freeze"
            )
        frozen_dir = destination / record_id
        frozen_dir.mkdir(mode=0o700)
        frozen_artifact = frozen_dir / "solution.json"
        frozen_artifact.write_bytes(artifact_bytes)
        frozen_artifact.chmod(0o400)
        candidate = {
            "rank": rank,
            "id": record_id,
            "search_score": float(record["score"]),
            "artifact_sha256": artifact_hash,
            "source_snapshot_sha256": source_hash,
            "frozen_artifact": (
                Path(record_id) / "solution.json"
            ).as_posix(),
        }
        if algorithm_mode:
            metrics_bundle_hash = metrics.get("algorithm_bundle_sha256")
            metadata_bundle_hash = (
                metadata.get("algorithm_bundle_sha256")
                if isinstance(metadata, dict) else None
            )
            declared_hashes = {
                value for value in (metrics_bundle_hash, metadata_bundle_hash)
                if value not in (None, "")
            }
            if len(declared_hashes) > 1:
                raise RuntimeError(
                    f"{record_id} has conflicting algorithm bundle provenance"
                )
            if not declared_hashes:
                raise RuntimeError(
                    f"{record_id} lacks algorithm bundle provenance"
                )
            declared_bundle_hash = next(iter(declared_hashes))
            if (
                not isinstance(declared_bundle_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", declared_bundle_hash)
            ):
                raise RuntimeError(
                    f"{record_id} has an invalid algorithm bundle digest"
                )
            frozen_source = frozen_dir / "source"
            actual_bundle_hash = _copy_algorithm_source(
                record_dir, frozen_source, task,
            )
            if actual_bundle_hash != declared_bundle_hash:
                raise RuntimeError(
                    f"{record_id} algorithm bundle provenance mismatch before "
                    "audit freeze"
                )
            # Keep the allowlist and digest in the freeze manifest itself.  A
            # later audit process can therefore locate the source without
            # consulting the mutable EB record or task configuration.
            candidate.update({
                "candidate_mode": "algorithm_bundle",
                "source_files": list(algorithm_source_files),
                "algorithm_bundle_sha256": actual_bundle_hash,
                "artifact_protocol": _source_artifact_protocol(frozen_source),
                "frozen_source": (
                    Path(record_id) / "source"
                ).as_posix(),
            })
            frozen_dir.chmod(0o500)
        candidates.append(candidate)
    manifest = {
        "schema": FREEZE_MANIFEST_SCHEMA,
        "frozen_at": now,
        "run_manifest_sha256": manifest_sha256,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    freeze_path = destination / "manifest.json"
    freeze_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    freeze_path.chmod(0o400)
    return manifest, sha256_json(manifest), _file_sha256(freeze_path)


def _seed_commitment(seed, freeze_manifest_sha256):
    material = (
        "openhyra-final-audit-seed.v1\0"
        + freeze_manifest_sha256
        + "\0"
        + str(seed)
    ).encode()
    return hashlib.sha256(material).hexdigest()


def _audit_termination(previous, report, report_path):
    winner = report.get("winner") or {}
    return {
        "reason": (
            "final_audit_complete"
            if report.get("status") == "complete" else "final_audit_failed"
        ),
        "terminal": True,
        "requested_by": "user",
        "accepted_by": "harness",
        "search_termination": previous,
        "prior_termination_reason": previous.get("reason"),
        "final_audit_status": report.get("status"),
        "final_audit_payload_sha256": sha256_json(report),
        "final_audit_file_sha256": _file_sha256(report_path),
        "audit_seed_commitment": report.get("seed_commitment"),
        "freeze_manifest_sha256": report.get("freeze_manifest_sha256"),
        "audit_winner_id": winner.get("id"),
        "audit_winner_score": winner.get("score"),
    }


def run_final_audit(task, eb, run_manifest, *, seed_factory=None, now=None):
    """Freeze search Top-K, generate a fresh private seed, and audit once.

    No result is committed to the Experience Bank. The final audit report is
    deliberately private (mode 0600), and only its seed commitment is copied
    into the public termination summary.
    """
    run_dir = Path(task.run_dir)
    report_path = run_dir / "final_audit.json"
    freeze_dir = run_dir / "final_audit_artifacts"
    if report_path.exists() or report_path.is_symlink():
        raise RuntimeError("final audit already exists; refusing to run it again")
    if freeze_dir.exists() or freeze_dir.is_symlink():
        raise RuntimeError(
            "final audit artifacts already exist; refusing a potentially "
            "seed-contaminated retry"
        )
    termination_path = run_dir / "termination.json"
    if not termination_path.is_file():
        raise RuntimeError(
            "run is not complete; final audit requires a search termination record"
        )
    try:
        previous_termination = json.loads(termination_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("search termination record is invalid") from exc
    prior_reason = previous_termination.get("reason")
    clean_state = (
        (prior_reason == "iteration_limit"
         and previous_termination.get("terminal") is False)
        or (prior_reason == "agent_converged"
            and previous_termination.get("terminal") is True)
    )
    if not clean_state:
        raise RuntimeError(
            "final audit requires a clean completed search run"
        )

    records = eb.records()
    incomplete = incomplete_contexts(records)
    if incomplete:
        raise RuntimeError(
            "final audit refuses incomplete Context(s): "
            + ", ".join(map(str, incomplete))
        )
    config = (getattr(task, "evaluation", {}) or {}).get("audit_stage")
    if not isinstance(config, dict):
        raise RuntimeError("task has no configured private audit stage")
    selected = select_top_k(records, task.direction, config.get("top_k"))
    if not selected:
        raise RuntimeError("final audit has no successful scored candidates")
    manifest_sha256 = run_manifest.get("manifest_sha256")
    if not isinstance(manifest_sha256, str):
        raise RuntimeError("run manifest has no trusted digest")
    timestamp = now or _timestamp()

    # The freeze completes before seed_factory is called. Tests can inject a
    # seed_factory that verifies this ordering without exposing production seed.
    report = {
        "schema": FINAL_AUDIT_SCHEMA,
        "status": "freezing",
        "created_at": timestamp,
        "run_id": task.run_id,
        "task": task.name,
        "protocol": task.protocol,
        "run_manifest_sha256": manifest_sha256,
        "search_termination": previous_termination,
        "freeze_manifest": None,
        "freeze_manifest_sha256": None,
        "freeze_manifest_file_sha256": None,
        "seed": None,
        "seed_commitment": None,
        "evaluation_request": None,
        "evaluation_request_sha256": None,
        "candidates": [],
        "winner": None,
    }
    if getattr(task, "candidate_mode", "legacy") == "algorithm_bundle":
        report.update({
            "candidate_mode": "algorithm_bundle",
            "source_files": list(_algorithm_source_files(task)),
            "artifact_protocol": getattr(task, "artifact_protocol", None),
        })
    _write_private_json(report_path, report)

    try:
        (
            freeze_manifest,
            freeze_manifest_sha256,
            freeze_manifest_file_sha256,
        ) = _freeze_candidates(
            task, selected, freeze_dir, manifest_sha256, run_manifest,
            timestamp,
        )
        report.update({
            "status": "frozen",
            "freeze_manifest": freeze_manifest,
            "freeze_manifest_sha256": freeze_manifest_sha256,
            "freeze_manifest_file_sha256": freeze_manifest_file_sha256,
        })
        _write_private_json(report_path, report)

        seed_factory = seed_factory or (lambda: secrets.randbits(63))
        seed = seed_factory()
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= (1 << 63) - 1
        ):
            raise ValueError("private audit seed must be a 63-bit integer")
        request = build_evaluation_request(task, "audit", seed)
        if request is None:
            raise RuntimeError("task has no configured private audit request")
        report.update({
            "status": "running",
            "seed": seed,
            "seed_commitment": _seed_commitment(
                seed, freeze_manifest_sha256,
            ),
            "evaluation_request": request,
            "evaluation_request_sha256": sha256_json(request),
        })
        _write_private_json(report_path, report)

        results = []
        for candidate in freeze_manifest["candidates"]:
            candidate_source_dir = None
            if candidate.get("candidate_mode") == "algorithm_bundle":
                # ``evaluate_trusted_artifact`` recreates its trusted working
                # directory on every call.  The frozen source therefore lives
                # beside (not inside) that directory and remains available for
                # the evaluator subprocess.
                candidate_source_dir = (
                    freeze_dir / candidate["frozen_source"]
                )
                try:
                    observed_bundle_hash = _algorithm_bundle_digest(
                        candidate_source_dir, task,
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    raise RuntimeError(
                        f"{candidate['id']} frozen algorithm source is invalid: "
                        f"{exc}"
                    ) from exc
                if observed_bundle_hash != candidate.get(
                        "algorithm_bundle_sha256"):
                    raise RuntimeError(
                        f"{candidate['id']} frozen algorithm source changed "
                        "before audit"
                    )
            evaluate_args = (
                task,
                freeze_dir / candidate["frozen_artifact"],
                freeze_dir / ".trusted" / candidate["id"],
                request,
            )
            if candidate_source_dir is None:
                # Preserve the historical positional call shape for legacy
                # evaluators and lightweight test doubles.
                result = evaluate_trusted_artifact(*evaluate_args)
            else:
                result = evaluate_trusted_artifact(
                    *evaluate_args,
                    candidate_source_dir=candidate_source_dir,
                )
            if candidate_source_dir is not None:
                try:
                    after_bundle_hash = _algorithm_bundle_digest(
                        candidate_source_dir, task,
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    raise RuntimeError(
                        f"{candidate['id']} frozen algorithm source became "
                        f"unreadable during audit: {exc}"
                    ) from exc
                if after_bundle_hash != candidate.get(
                        "algorithm_bundle_sha256"):
                    raise RuntimeError(
                        f"{candidate['id']} frozen algorithm source changed "
                        "during audit"
                    )
            normalized = result.get("normalized_solution")
            evidence = result.get("evidence")
            normalized_sha256 = (
                sha256_json(normalized) if normalized is not None else None
            )
            evidence_sha256 = sha256_json(evidence)
            result_record = {
                "rank": candidate["rank"],
                "id": candidate["id"],
                "search_score": candidate["search_score"],
                "artifact_sha256": candidate["artifact_sha256"],
                "status": result["status"],
                "score": result["score"],
                "metrics": result["metrics"],
                "note": result["note"],
                "normalized_solution_sha256": normalized_sha256,
                "evidence_sha256": evidence_sha256,
                "evidence": evidence,
            }
            if candidate.get("candidate_mode") == "algorithm_bundle":
                evaluated_bundle_hash = result.get("metrics", {}).get(
                    "algorithm_bundle_sha256"
                )
                evaluated_artifact_protocol = result.get("metrics", {}).get(
                    "artifact_protocol"
                ) or candidate.get("artifact_protocol")
                if (
                    result.get("status") == "ok"
                    and evaluated_bundle_hash != candidate.get(
                        "algorithm_bundle_sha256")
                ):
                    raise RuntimeError(
                        f"{candidate['id']} evaluator algorithm provenance "
                        "does not match the frozen source"
                    )
                if evaluated_bundle_hash is None:
                    # A failed evaluator may not have had a chance to emit
                    # metrics.  Preserve the frozen identity in the private
                    # report while leaving the candidate failure visible to
                    # the normal all-candidates gate below.
                    result["metrics"] = dict(result.get("metrics", {}))
                    result["metrics"]["algorithm_bundle_sha256"] = (
                        candidate["algorithm_bundle_sha256"]
                    )
                    evaluated_bundle_hash = candidate[
                        "algorithm_bundle_sha256"
                    ]
                    result_record["metrics"] = result["metrics"]
                result_record.update({
                    "candidate_mode": "algorithm_bundle",
                    "source_files": candidate.get("source_files", []),
                    "algorithm_bundle_sha256": evaluated_bundle_hash,
                    "frozen_source": candidate["frozen_source"],
                    # This comes from the evaluator-loaded frozen
                    # manifest, not the task's default artifact protocol.
                    "artifact_protocol": evaluated_artifact_protocol,
                })
            results.append(result_record)
            report["candidates"] = results
            _write_private_json(report_path, report)
            if result.get("artifact_sha256") != candidate["artifact_sha256"]:
                raise RuntimeError(
                    f"{candidate['id']} changed after the audit freeze"
                )
            if result["status"] == "ok" and normalized_sha256 is None:
                raise RuntimeError(
                    f"{candidate['id']} audit verdict omitted normalized_solution"
                )
            if (
                result["status"] == "ok"
                and normalized_sha256 != candidate["artifact_sha256"]
            ):
                raise RuntimeError(
                    f"{candidate['id']} normalized audit artifact differs "
                    "from the frozen search artifact"
                )

        successful = [
            item for item in results
            if item["status"] == "ok"
            and item["score"] is not None
            and math.isfinite(float(item["score"]))
        ]
        audit_direction = config.get("direction", task.direction)
        if audit_direction not in {"min", "max"}:
            raise RuntimeError("private audit direction must be min or max")
        if len(successful) == len(results):
            winner = (min if audit_direction == "min" else max)(
                successful,
                key=lambda item: (
                    item["score"],
                    # Reverse only score; deterministic ties prefer lower id.
                ),
            )
            tied = [
                item for item in successful
                if item["score"] == winner["score"]
            ]
            winner = min(tied, key=lambda item: item["id"])
            report["winner"] = {
                "id": winner["id"],
                "score": winner["score"],
                "search_score": winner["search_score"],
            }
            report["status"] = "complete"
            report["completed_at"] = now or _timestamp()
        else:
            report["status"] = "failed"
            report["failure"] = "one_or_more_audit_candidates_failed"
            report["completed_at"] = now or _timestamp()
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        report["completed_at"] = now or _timestamp()
        _write_private_json(report_path, report)
        write_termination(
            termination_path,
            _audit_termination(previous_termination, report, report_path),
        )
        raise

    _write_private_json(report_path, report)
    write_termination(
        termination_path,
        _audit_termination(previous_termination, report, report_path),
    )
    if report["status"] != "complete":
        raise RuntimeError(
            f"private audit failed closed; inspect {report_path}"
        )
    return report
