"""OpenHyra harness: a minimal reproduction of the Hyra loop from the tech report.

    Context Agent -> inspirations -> Proposal Agents (xN) -> sandbox + trusted
    evaluator -> Experience Bank

Tasks are plugins under tasks/<name>/ (task.json + TASK.md + evaluator.py +
seed_solution/). Evaluation runs behind a semaphore sized by the task
(GPU tasks: 1); with --workers N proposal generation overlaps evaluation,
approximating the report's asynchronous producer-consumer pipeline.

Usage:
    python harness.py --task sums_diffs --init          # run + commit the seed
    python harness.py --task sums_diffs --iterations 5 [--workers 2]
    python harness.py --task sums_diffs --status
"""

import argparse
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eb import ExperienceBank
from context_agent import build_inspiration
from proposal_agent import propose
from sandbox import run_solution

ROOT = Path(__file__).resolve().parent


class Task:
    def __init__(self, name):
        self.dir = ROOT / "tasks" / name
        if not self.dir.exists():
            sys.exit(f"Unknown task {name!r}. Available: "
                     + ", ".join(sorted(p.name for p in (ROOT / 'tasks').iterdir() if p.is_dir())))
        cfg = json.loads((self.dir / "task.json").read_text())
        self.name = cfg["name"]
        self.direction = cfg["direction"]
        self.metric = cfg.get("metric", "score")
        self.editable_files = cfg["editable_files"]
        self.timeout_s = cfg.get("sandbox_timeout_s", 660)
        self.eval_concurrency = cfg.get("eval_concurrency", 1)
        self.max_training_seconds = cfg.get("max_training_seconds")
        self.fallback_directions = cfg.get("fallback_directions", [])
        self.description = (self.dir / "TASK.md").read_text()
        self.evaluator = self.dir / "evaluator.py"
        if not self.evaluator.exists():
            sys.exit(f"Task {name!r} has no trusted evaluator.py — refusing to run "
                     f"(candidate-reported scores are never accepted).")
        self.seed_dir = self.dir / "seed_solution"
        self.python_bin = sys.executable
        self.run_dir = ROOT / "runs" / self.name


def solution_files(d):
    skip = {".venv", "__pycache__", ".git"}
    out = {}
    for p in Path(d).rglob("*"):
        if p.is_file() and not (set(p.relative_to(d).parts) & skip):
            rel = str(p.relative_to(d))
            if rel not in {"run.log", "train.log", "solution.json", "PROPOSAL.md"}:
                out[rel] = p.read_bytes()
    return out


def check_frozen(parent_dir, draft_dir, editable):
    """Whitelist check: only the editable files may differ between parent and draft."""
    before, after = solution_files(parent_dir), solution_files(draft_dir)
    violations = []
    for rel in sorted(set(before) | set(after)):
        if rel in editable:
            continue
        if before.get(rel) != after.get(rel):
            violations.append(rel)
    return violations


def iterate(task, eb, iteration, eval_sem):
    """One full Hyra loop iteration. Returns the committed record."""
    parent, prompt, direction = build_inspiration(task, eb, iteration)
    short_dir = " ".join(direction.split())[:160]
    print(f"[context] iter {iteration}: parent = {parent['id']}, next = {short_dir}")

    draft = task.run_dir / "drafts" / f"iter_{iteration:04d}"
    ok, description = propose(Path(parent["path"]), draft, prompt, task.editable_files)
    print(f"[proposal] iter {iteration}: {description}" if ok else f"[proposal] iter {iteration} FAILED: {description}")
    if not ok:
        return eb.commit(draft, None, "crash", description, parent["id"], "")

    violations = check_frozen(parent["path"], draft, task.editable_files)
    if violations:
        msg = f"modified non-editable file(s) {violations}: {description}"
        print(f"[integrity] iter {iteration} REJECTED — {msg}")
        return eb.commit(draft, None, "violation", msg, parent["id"], "")

    sandbox = task.run_dir / "sandboxes" / f"iter_{iteration:04d}"
    with eval_sem:
        print(f"[sandbox] iter {iteration}: running solve.sh + trusted evaluator ...")
        score, status, log_tail, metrics = run_solution(draft, sandbox, task)
    if (sandbox / "run.log").exists():
        shutil.copy(sandbox / "run.log", draft / "run.log")
    if (sandbox / "solution.json").exists():
        shutil.copy(sandbox / "solution.json", draft / "solution.json")

    if (status == "ok" and task.max_training_seconds
            and metrics.get("training_seconds", 0) > task.max_training_seconds):
        print(f"[integrity] iter {iteration} REJECTED — training ran "
              f"{metrics['training_seconds']:.0f}s > {task.max_training_seconds}s budget")
        score, status = None, "violation"
        description = f"exceeded training budget ({metrics['training_seconds']:.0f}s): {description}"

    prev_best = eb.best()
    rec = eb.commit(draft, score, status, description, parent["id"], log_tail, metrics)
    improved = eb.is_improvement(score, prev_best["score"] if prev_best else None)
    best = eb.best()
    verdict = "IMPROVED — new best" if improved else "kept in bank, best unchanged"
    print(f"[eb] {rec['id']}: {task.metric}={score} status={status} -> {verdict} "
          f"(best: {best['id']} @ {best['score']:.6f})")
    return rec


def init_seed(task, eb):
    """Run the seed solution through the trusted pipeline and commit it."""
    sandbox = task.run_dir / "sandboxes" / "seed"
    print(f"[seed] running {task.seed_dir} through sandbox + evaluator ...")
    score, status, log_tail, metrics = run_solution(task.seed_dir, sandbox, task)
    if status != "ok":
        sys.exit(f"Seed run failed ({status}):\n{log_tail}")
    if (sandbox / "solution.json").exists():
        shutil.copy(sandbox / "solution.json", task.seed_dir / "solution.json")
    rec = eb.commit(task.seed_dir, score, status, "seed solution", None, log_tail, metrics)
    print(f"[eb] seeded {rec['id']}: {task.metric}={score}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sums_diffs")
    ap.add_argument("--init", action="store_true", help="run and commit the seed solution")
    ap.add_argument("--iterations", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent proposal workers (evaluation stays behind the task's semaphore)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    task = Task(args.task)
    eb = ExperienceBank(task.run_dir / "eb", direction=task.direction)

    if args.status:
        for r in eb.records():
            score = f"{r['score']:.6f}" if r["score"] is not None else "   -    "
            print(f"{r['id']}  {score}  {r['status']:9s}  {r['description']}")
        best = eb.best()
        if best:
            print(f"\nbest: {best['id']} @ {best['score']:.6f}  ({task.metric}, {task.direction})")
        return

    if args.init:
        if eb.records():
            sys.exit("Experience bank already seeded; delete the run dir to reset.")
        init_seed(task, eb)

    if args.iterations <= 0:
        return
    if not eb.records():
        sys.exit("Experience bank is empty; run with --init first.")

    eval_sem = threading.Semaphore(task.eval_concurrency)
    start = len([r for r in eb.records() if r["parent"] is not None])
    if args.workers <= 1:
        for i in range(start, start + args.iterations):
            iterate(task, eb, i, eval_sem)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(iterate, task, eb, i, eval_sem)
                       for i in range(start, start + args.iterations)]
            for f in futures:
                f.result()


if __name__ == "__main__":
    main()
