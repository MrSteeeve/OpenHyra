"""Export compact, independently auditable experiment bundles."""

import csv
import json
import shutil
import time
from pathlib import Path

from provenance import git_metadata, sha256_file


SUMMARY_FIELDS = [
    "id", "parent", "iteration", "status", "description", "n", "sums",
    "diffs", "span", "score", "solver_seconds", "evaluator_seconds",
    "total_seconds", "set_hash", "artifact_sha256", "candidate_count",
    "candidate_index", "candidate_seed", "duplicate_of", "attempt_index",
    "repair_of", "run_manifest_sha256", "editable_file_sha256",
]


def export_bundle(task, eb, destination, *, root, run_manifest):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {destination}")
    destination.mkdir(parents=True)
    records = eb.records()

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
    output_solutions = destination / "solutions"
    output_solutions.mkdir()
    allowed = set(task.editable_files) | {
        "solve.sh", "solution.json", "PROPOSAL.md", "run.log",
    }
    for record in records:
        source = Path(record["path"])
        target = output_solutions / record["id"]
        target.mkdir()
        for path in source.iterdir() if source.exists() else ():
            if path.is_file() and path.name in allowed:
                shutil.copy2(path, target / path.name)

    with open(destination / "summary.tsv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, delimiter="\t")
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
                "score": record.get("score"),
                "solver_seconds": metrics.get("solver_seconds"),
                "evaluator_seconds": metrics.get("evaluator_seconds"),
                "total_seconds": metrics.get("total_seconds"),
                "set_hash": metrics.get("set_hash"),
                "artifact_sha256": metrics.get("artifact_sha256"),
                "candidate_count": metadata.get("candidate_count"),
                "candidate_index": metadata.get("candidate_index"),
                "candidate_seed": metadata.get("candidate_seed"),
                "duplicate_of": metadata.get("duplicate_of"),
                "attempt_index": metadata.get("attempt_index"),
                "repair_of": metadata.get("repair_of"),
                "run_manifest_sha256": metadata.get("run_manifest_sha256"),
                "editable_file_sha256": json.dumps(
                    metadata.get("editable_file_sha256"), sort_keys=True,
                ) if metadata.get("editable_file_sha256") else None,
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
