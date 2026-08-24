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
        artifact_hash = record.get("metrics", {}).get("artifact_sha256")
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
    destination.mkdir(parents=True)
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
        frozen_dir.mkdir()
        frozen_artifact = frozen_dir / "solution.json"
        frozen_artifact.write_bytes(artifact_bytes)
        frozen_artifact.chmod(0o400)
        candidates.append({
            "rank": rank,
            "id": record_id,
            "search_score": float(record["score"]),
            "artifact_sha256": artifact_hash,
            "source_snapshot_sha256": source_hash,
            "frozen_artifact": (
                Path(record_id) / "solution.json"
            ).as_posix(),
        })
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
            result = evaluate_trusted_artifact(
                task,
                freeze_dir / candidate["frozen_artifact"],
                freeze_dir / ".trusted" / candidate["id"],
                request,
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
