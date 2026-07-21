"""Experience Bank (EB): stores every proposed solution, its score and logs.

Mirrors the Hyra tech report: solutions are folders with a solve.sh entry;
each run's artifacts and evaluation result are committed back into the bank.
"""

import json
import shutil
import time
from pathlib import Path


class ExperienceBank:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.solutions_dir = self.root / "solutions"
        self.records_path = self.root / "records.jsonl"
        self.solutions_dir.mkdir(parents=True, exist_ok=True)

    def records(self):
        if not self.records_path.exists():
            return []
        with open(self.records_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def best(self):
        scored = [r for r in self.records() if r["score"] is not None]
        return min(scored, key=lambda r: r["score"]) if scored else None

    def next_id(self):
        return f"sol_{len(self.records()):04d}"

    def commit(self, sol_id, src_dir, score, status, description, parent, log_tail, metrics=None):
        """Copy solution folder into the bank and append a record."""
        dst = self.solutions_dir / sol_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_dir, dst, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".git"))
        record = {
            "id": sol_id,
            "parent": parent,
            "score": score,
            "status": status,  # "ok" | "crash" | "timeout"
            "description": description,
            "path": str(dst),
            "log_tail": log_tail,
            "metrics": metrics or {},  # parsed train.py summary (tokens, steps, MFU, ...)
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.records_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
