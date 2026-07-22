"""Experience Bank (EB): stores every proposed solution, its score and logs.

Mirrors the Hyra tech report: solutions are folders with a solve.sh entry;
each run's artifacts and evaluation result are committed back into the bank.
Thread-safe: commit/next_id are serialized so concurrent Proposal workers
cannot collide on ids or interleave JSONL writes.
"""

import json
import os
import shutil
import threading
import time
from pathlib import Path


class ExperienceBank:
    def __init__(self, root: Path, direction: str = "min"):
        assert direction in ("min", "max")
        self.root = Path(root)
        self.direction = direction
        self.solutions_dir = self.root / "solutions"
        self.records_path = self.root / "records.jsonl"
        self.solutions_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _records_unlocked(self):
        if not self.records_path.exists():
            return []
        with open(self.records_path) as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def records(self):
        with self._lock:
            return self._records_unlocked()

    def snapshot(self):
        """Return one consistent (version, records) view of completed work."""
        with self._lock:
            records = self._records_unlocked()
            return len(records), records

    def best(self):
        with self._lock:
            scored = [r for r in self._records_unlocked() if r["score"] is not None]
            if not scored:
                return None
            pick = min if self.direction == "min" else max
            return pick(scored, key=lambda r: r["score"])

    def is_improvement(self, score, other):
        if score is None:
            return False
        if other is None:
            return True
        return score < other if self.direction == "min" else score > other

    def commit(self, src_dir, score, status, description, parent, log_tail,
               metrics=None, metadata=None):
        """Copy solution folder into the bank and append a record. Returns the record."""
        with self._lock:
            sol_id = f"sol_{len(self._records_unlocked()):04d}"
            dst = self.solutions_dir / sol_id
            if dst.exists():
                shutil.rmtree(dst)
            if Path(src_dir).exists():
                shutil.copytree(src_dir, dst, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".git"))
            record = {
                "id": sol_id,
                "parent": parent,
                "score": score,
                "status": status,  # "ok" | "crash" | "timeout" | "violation" | "rejected" (preflight lint)
                "description": description,
                "path": str(dst),
                "log_tail": log_tail,
                "metrics": metrics or {},
                "metadata": metadata or {},
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(self.records_path, "a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return record
