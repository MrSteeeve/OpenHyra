"""Export compact, independently auditable experiment bundles."""

import csv
import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

from provenance import git_metadata, sha256_file
from sandbox import read_regular_file, read_source_tree


SUMMARY_FIELDS = [
    "id", "parent", "iteration", "status", "description", "n", "sums",
    "diffs", "span", "max_abs", "score", "solver_seconds", "evaluator_seconds",
    "total_seconds", "set_hash", "evidence_level", "research_claim_count",
    "research_rank", "bounded_supported_claim_count", "refuted_claim_count",
    "formally_checked_claim_count", "verified_obligation_count",
    "refuted_obligation_count", "construction_sha256",
    "verified_certificate_count", "refuted_certificate_count",
    "formalization_status", "formalization_request_sha256", "proof_sha256",
    "formal_wrapper_sha256", "formal_audit_sha256",
    "formal_verifier_seconds",
    "formal_spec_sha256", "formal_runner_sha256", "formal_toolchain",
    "formal_mathlib_revision", "formal_environment_sha256",
    "formal_lean_binary_sha256", "formal_mathlib_tree_sha256",
    "formal_checked_claim_templates",
    "formal_checked_targets", "research_sha256", "evidence_sha256",
    "artifact_sha256",
    "source_snapshot_sha256",
    "candidate_count", "candidate_index", "candidate_seed", "duplicate_of",
    "numeric_duplicate_of", "attempt_index", "attempt_kind", "repair_of",
    "run_manifest_sha256", "editable_file_sha256",
]


def _summary_key_segment(value):
    """Escape separators so flattened column names cannot collide."""
    return str(value).replace("\\", "\\\\").replace(".", "\\.")


def _summary_cell(value):
    if isinstance(value, (list, tuple, Mapping)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return value


def _flatten_summary_mapping(mapping, namespace):
    """Flatten nested JSON mappings into deterministic namespaced columns."""
    if not isinstance(mapping, Mapping):
        return {}
    flattened = {}

    def visit(value, path):
        if isinstance(value, Mapping) and value:
            for key in sorted(value, key=lambda item: str(item)):
                visit(value[key], path + (_summary_key_segment(key),))
            return
        name = ".".join((namespace, *path))
        flattened[name] = _summary_cell(value)

    for key in sorted(mapping, key=lambda item: str(item)):
        visit(mapping[key], (_summary_key_segment(key),))
    return flattened


def _summary_fieldnames(records):
    dynamic = set()
    for record in records:
        dynamic.update(_flatten_summary_mapping(
            record.get("metrics", {}), "metrics",
        ))
        dynamic.update(_flatten_summary_mapping(
            record.get("metadata", {}), "metadata",
        ))
    return [*SUMMARY_FIELDS, *sorted(dynamic)]


def export_bundle(task, eb, destination, *, root, run_manifest):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {destination}")
    records = eb.records()
    allowed = set(task.editable_files) | {
        "solve.sh", "solution.json", "evidence.json", "PROPOSAL.md", "run.log",
    }
    max_output_bytes = int(getattr(task, "max_output_mb", 64)) * 1024 * 1024
    max_artifact_bytes = int(
        getattr(task, "max_artifact_bytes", 1024 * 1024)
    )
    validated_files = {}
    for record in records:
        source = Path(record["path"])
        try:
            source_hash, _source_hashes, source_files = read_source_tree(
                source, max_output_bytes,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"{record['id']} source snapshot is unsafe: {exc}"
            ) from exc
        files = {
            name: data for name, data in source_files.items()
            if name in allowed
        }
        for name in ("solution.json", "evidence.json", "run.log"):
            path = source / name
            try:
                os.lstat(path)
            except FileNotFoundError:
                continue
            limit = max_artifact_bytes if name == "solution.json" else max_output_bytes
            try:
                files[name] = read_regular_file(
                    path, limit, label=f"{record['id']} {name}",
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc

        metrics = record.get("metrics", {})
        for name, hash_field in (
            ("solution.json", "artifact_sha256"),
            ("evidence.json", "evidence_sha256"),
        ):
            expected = metrics.get(hash_field)
            if expected and name not in files:
                raise RuntimeError(
                    f"{record['id']} is missing {name} required by {hash_field}"
                )
            if not expected and name in files:
                raise RuntimeError(
                    f"{record['id']} has {name} without a trusted hash"
                )
            if expected:
                actual = hashlib.sha256(files[name]).hexdigest()
                if actual != expected:
                    raise RuntimeError(
                        f"{record['id']} {name} does not match its trusted hash"
                    )

        editable_hashes = record.get("metadata", {}).get(
            "editable_file_sha256",
        )
        for name in task.editable_files:
            expected = (
                editable_hashes.get(name)
                if isinstance(editable_hashes, dict) else None
            )
            if expected and name not in files:
                raise RuntimeError(
                    f"{record['id']} is missing hashed editable file {name}"
                )
            if name in files and not expected:
                raise RuntimeError(
                    f"{record['id']} has editable file {name} "
                    "without a trusted hash"
                )
            if expected:
                actual = hashlib.sha256(files[name]).hexdigest()
                if actual != expected:
                    raise RuntimeError(
                        f"{record['id']} editable file {name} "
                        "does not match its trusted hash"
                    )
        expected_source_hash = metrics.get("source_snapshot_sha256")
        if source_files and not expected_source_hash:
            raise RuntimeError(
                f"{record['id']} has source files without a trusted snapshot hash"
            )
        if expected_source_hash and source_hash != expected_source_hash:
            raise RuntimeError(
                f"{record['id']} source snapshot does not match its trusted hash"
            )
        validated_files[record["id"]] = files
    destination.mkdir(parents=True)

    normalized_records = []
    for record in records:
        item = dict(record)
        item["path"] = f"solutions/{record['id']}"
        normalized_records.append(item)
    with open(destination / "records.jsonl", "w") as stream:
        for record in normalized_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    analyses = eb.root / "analyses"
    if analyses.exists():
        shutil.copytree(analyses, destination / "analyses")
    termination = task.run_dir / "termination.json"
    if termination.is_file():
        shutil.copy2(termination, destination / "termination.json")
    final_audit = task.run_dir / "final_audit.json"
    try:
        os.lstat(final_audit)
    except FileNotFoundError:
        pass
    else:
        try:
            final_audit_data = read_regular_file(
                final_audit,
                max_output_bytes,
                label="final audit",
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        (destination / "final_audit.json").write_bytes(final_audit_data)
    output_solutions = destination / "solutions"
    output_solutions.mkdir()
    for record in records:
        target = output_solutions / record["id"]
        target.mkdir()
        for name, data in validated_files[record["id"]].items():
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            output.chmod(0o755 if name == "solve.sh" else 0o644)

    with open(destination / "summary.tsv", "w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=_summary_fieldnames(records),
            delimiter="\t",
        )
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics", {})
            metadata = record.get("metadata", {})
            writer.writerow({
                "id": record["id"],
                "parent": record.get("parent"),
                "iteration": metadata.get("iteration"),
                "status": record["status"],
                "description": record["description"],
                "n": metrics.get("n"),
                "sums": metrics.get("sums"),
                "diffs": metrics.get("diffs"),
                "span": metrics.get("span"),
                "max_abs": metrics.get("max_abs"),
                "score": record.get("score"),
                "solver_seconds": metrics.get("solver_seconds"),
                "evaluator_seconds": metrics.get("evaluator_seconds"),
                "total_seconds": metrics.get("total_seconds"),
                "set_hash": metrics.get("set_hash"),
                "evidence_level": metrics.get("evidence_level"),
                "research_claim_count": metrics.get("research_claim_count"),
                "research_rank": metrics.get("research_rank"),
                "bounded_supported_claim_count": metrics.get(
                    "bounded_supported_claim_count"
                ),
                "refuted_claim_count": metrics.get("refuted_claim_count"),
                "formally_checked_claim_count": metrics.get(
                    "formally_checked_claim_count"
                ),
                "verified_obligation_count": metrics.get(
                    "verified_obligation_count"
                ),
                "refuted_obligation_count": metrics.get(
                    "refuted_obligation_count"
                ),
                "construction_sha256": metrics.get("construction_sha256"),
                "verified_certificate_count": metrics.get(
                    "verified_certificate_count"
                ),
                "refuted_certificate_count": metrics.get(
                    "refuted_certificate_count"
                ),
                "formalization_status": metrics.get(
                    "formalization_status"
                ),
                "formalization_request_sha256": metrics.get(
                    "formalization_request_sha256"
                ),
                "proof_sha256": metrics.get("proof_sha256"),
                "formal_wrapper_sha256": metrics.get(
                    "formal_wrapper_sha256"
                ),
                "formal_audit_sha256": metrics.get(
                    "formal_audit_sha256"
                ),
                "formal_verifier_seconds": metrics.get(
                    "formal_verifier_seconds"
                ),
                "formal_spec_sha256": metrics.get("formal_spec_sha256"),
                "formal_runner_sha256": metrics.get("formal_runner_sha256"),
                "formal_toolchain": metrics.get("formal_toolchain"),
                "formal_mathlib_revision": metrics.get(
                    "formal_mathlib_revision"
                ),
                "formal_environment_sha256": metrics.get(
                    "formal_environment_sha256"
                ),
                "formal_lean_binary_sha256": metrics.get(
                    "formal_lean_binary_sha256"
                ),
                "formal_mathlib_tree_sha256": metrics.get(
                    "formal_mathlib_tree_sha256"
                ),
                "formal_checked_claim_templates": json.dumps(
                    metrics.get("formal_checked_claim_templates"),
                    sort_keys=True,
                ) if metrics.get("formal_checked_claim_templates") else None,
                "formal_checked_targets": json.dumps(
                    metrics.get("formal_checked_targets"),
                    sort_keys=True,
                ) if metrics.get("formal_checked_targets") else None,
                "research_sha256": metrics.get("research_sha256"),
                "evidence_sha256": metrics.get("evidence_sha256"),
                "artifact_sha256": metrics.get("artifact_sha256"),
                "source_snapshot_sha256": metrics.get(
                    "source_snapshot_sha256"
                ),
                "candidate_count": metadata.get("candidate_count"),
                "candidate_index": metadata.get("candidate_index"),
                "candidate_seed": metadata.get("candidate_seed"),
                "duplicate_of": metadata.get("duplicate_of"),
                "numeric_duplicate_of": metadata.get(
                    "numeric_duplicate_of"
                ),
                "attempt_index": metadata.get("attempt_index"),
                "attempt_kind": metadata.get("attempt_kind"),
                "repair_of": metadata.get("repair_of"),
                "run_manifest_sha256": metadata.get("run_manifest_sha256"),
                "editable_file_sha256": json.dumps(
                    metadata.get("editable_file_sha256"), sort_keys=True,
                ) if metadata.get("editable_file_sha256") else None,
                **_flatten_summary_mapping(metrics, "metrics"),
                **_flatten_summary_mapping(metadata, "metadata"),
            })

    snapshot_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest = {
        "schema_version": 3,
        "task": task.name,
        "protocol": task.protocol,
        "run_id": task.run_id,
        "run_manifest_sha256": run_manifest["manifest_sha256"],
        "run": run_manifest,
        "snapshot_at": snapshot_at,
        "export_git": git_metadata(root),
        "termination_sha256": (
            sha256_file(termination) if termination.is_file() else None
        ),
        "final_audit_sha256": (
            sha256_file(destination / "final_audit.json")
            if (destination / "final_audit.json").is_file() else None
        ),
        "record_count": len(records),
        "context_count": len({
            record.get("metadata", {}).get("iteration")
            for record in records
            if isinstance(record.get("metadata", {}).get("iteration"), int)
        }),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (destination / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return destination
