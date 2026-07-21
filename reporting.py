"""Export compact, independently auditable experiment bundles."""

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_version(command):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (result.stdout or result.stderr).strip().splitlines()[0]


def _git_metadata(root):
    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True,
            timeout=10, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


SUMMARY_FIELDS = [
    "id", "parent", "iteration", "status", "description", "n", "sums",
    "diffs", "span", "score", "solver_seconds", "evaluator_seconds",
    "total_seconds", "set_hash", "artifact_sha256",
]


def export_bundle(task, eb, destination, *, root, backend, model, workers,
                  trial_seed, started_at):
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
    output_solutions = destination / "solutions"
    output_solutions.mkdir()
    allowed = {"solver.py", "solve.sh", "solution.json", "PROPOSAL.md", "run.log"}
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
            })

    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    manifest = {
        "schema_version": 1,
        "task": task.name,
        "protocol": task.protocol,
        "run_id": task.run_id,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "trial_seed": trial_seed,
        "backend": backend,
        "model": model,
        "workers": workers,
        "eval_concurrency": task.eval_concurrency,
        "task_config_sha256": sha256_file(task.dir / "task.json"),
        "task_description_sha256": sha256_file(task.dir / "TASK.md"),
        "evaluator_sha256": sha256_file(task.evaluator),
        "git": _git_metadata(root),
        "environment": {
            "python": sys.version,
            "numpy": numpy_version,
            "platform": platform.platform(),
            "backend_cli": _command_version([backend, "--version"]),
        },
        "record_count": len(records),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return destination
