"""OpenHyra: Context producer -> Proposal workers -> Evaluator workers."""

import argparse
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
from pathlib import Path

from context_agent import build_inspiration, finalize_analysis
from eb import ExperienceBank
from llm_backend import SUPPORTED_BACKENDS
from proposal_agent import propose
from reporting import export_bundle
from sandbox import run_solution

ROOT = Path(__file__).resolve().parent
STOP = object()


class Task:
    def __init__(self, name, run_id="default"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
            sys.exit("--run-id must contain only letters, numbers, dot, dash, underscore")
        self.dir = ROOT / "tasks" / name
        if not self.dir.exists():
            available = ", ".join(sorted(
                p.name for p in (ROOT / "tasks").iterdir() if p.is_dir()
            ))
            sys.exit(f"Unknown task {name!r}. Available: {available}")
        cfg = json.loads((self.dir / "task.json").read_text())
        self.name = cfg["name"]
        self.protocol = cfg["protocol"]
        self.direction = cfg["direction"]
        self.metric = cfg.get("metric", "score")
        self.editable_files = cfg["editable_files"]
        self.timeout_s = cfg.get("sandbox_timeout_s", 660)
        self.eval_concurrency = cfg.get("eval_concurrency", 1)
        self.max_training_seconds = cfg.get("max_training_seconds")
        self.max_memory_mb = cfg.get("max_memory_mb", 1024)
        self.max_output_mb = cfg.get("max_output_mb", 64)
        self.fallback_directions = cfg.get("fallback_directions", [])
        self.description = (self.dir / "TASK.md").read_text()
        self.evaluator = self.dir / "evaluator.py"
        if not self.evaluator.exists():
            sys.exit(f"Task {name!r} has no trusted evaluator.py — refusing to run")
        self.seed_dir = self.dir / "seed_solution"
        self.python_bin = sys.executable
        self.run_id = run_id
        self.run_dir = ROOT / "runs" / self.name / run_id


def solution_files(directory):
    skip = {".venv", "__pycache__", ".git", ".tmp"}
    output = {}
    for path in Path(directory).rglob("*"):
        if path.is_file() and not (set(path.relative_to(directory).parts) & skip):
            relative = str(path.relative_to(directory))
            if relative not in {
                "run.log", "train.log", "solution.json",
                "solution.snapshot.json", "PROPOSAL.md",
            }:
                output[relative] = path.read_bytes()
    return output


def check_frozen(parent_dir, draft_dir, editable):
    before, after = solution_files(parent_dir), solution_files(draft_dir)
    return [
        relative for relative in sorted(set(before) | set(after))
        if relative not in editable and before.get(relative) != after.get(relative)
    ]


def _record_metadata(task, context_meta, backend, model):
    return {
        **context_meta,
        "protocol": task.protocol,
        "run_id": task.run_id,
        "backend": backend,
        "model": model,
    }


def _evaluate_and_commit(item, task, eb, backend, model, print_lock):
    iteration = item["iteration"]
    parent = item["parent"]
    draft = item["draft"]
    description = item["description"]
    metadata = _record_metadata(task, item["context_meta"], backend, model)

    if item.get("failure"):
        score, status, log_tail, metrics = None, item["failure_status"], item["failure"], {}
    else:
        sandbox = task.run_dir / "sandboxes" / f"iter_{iteration:04d}"
        with print_lock:
            print(f"[sandbox] iter {iteration}: running candidate + trusted evaluator ...")
        score, status, log_tail, metrics = run_solution(draft, sandbox, task)
        if (sandbox / "run.log").exists():
            shutil.copy2(sandbox / "run.log", draft / "run.log")
        snapshot = sandbox / "evaluated_solution.json"
        if not snapshot.exists():
            snapshot = sandbox / "solution.snapshot.json"
        if snapshot.exists():
            shutil.copy2(snapshot, draft / "solution.json")

    previous_best = eb.best()
    record = eb.commit(
        draft, score, status, description, parent["id"], log_tail,
        metrics=metrics, metadata=metadata,
    )
    finalize_analysis(eb, iteration, record["id"])
    best = eb.best()
    improved = eb.is_improvement(
        score, previous_best["score"] if previous_best else None,
    )
    with print_lock:
        verdict = "IMPROVED" if improved else "best unchanged"
        best_text = f"{best['id']} @ {best['score']:.9f}" if best else "none"
        print(
            f"[eb] iter {iteration} -> {record['id']}: score={score} "
            f"status={status}, {verdict} (best: {best_text})"
        )
    return record


def run_pipeline(task, eb, iterations, workers, backend, model, trial_seed):
    """Run a bounded three-stage asynchronous producer-consumer pipeline."""
    inspiration_queue = queue.Queue(maxsize=max(1, workers))
    candidate_queue = queue.Queue(maxsize=max(1, workers + task.eval_concurrency))
    max_inflight = max(1, workers + task.eval_concurrency)
    inflight = threading.Semaphore(max_inflight)
    active_directions = {}
    active_lock = threading.Lock()
    print_lock = threading.Lock()
    errors = queue.Queue()
    start = len([record for record in eb.records() if record["parent"] is not None])

    def context_producer():
        try:
            for iteration in range(start, start + iterations):
                inflight.acquire()
                try:
                    with active_lock:
                        reserved = tuple(active_directions.values())
                    parent, prompt, direction, context_meta = build_inspiration(
                        task, eb, iteration, backend=backend, model=model,
                        active_directions=reserved,
                        trial_seed=trial_seed + iteration,
                    )
                    with active_lock:
                        active_directions[iteration] = direction
                    with print_lock:
                        short = " ".join(direction.split())[:180]
                        print(
                            f"[context] iter {iteration}: EB v{context_meta['eb_version']}, "
                            f"parent={parent['id']}, next={short}"
                        )
                    inspiration_queue.put({
                        "iteration": iteration,
                        "parent": parent,
                        "prompt": prompt,
                        "context_meta": context_meta,
                    })
                except Exception:
                    inflight.release()
                    raise
        except Exception as exc:
            errors.put(("context", exc))
        finally:
            for _ in range(workers):
                inspiration_queue.put(STOP)

    def proposal_worker():
        while True:
            item = inspiration_queue.get()
            if item is STOP:
                inspiration_queue.task_done()
                break
            iteration = item["iteration"]
            parent = item["parent"]
            draft = task.run_dir / "drafts" / f"iter_{iteration:04d}"
            try:
                ok, description = propose(
                    Path(parent["path"]), draft, item["prompt"], task.editable_files,
                    backend=backend, model=model,
                )
                failure = None
                failure_status = None
                if not ok:
                    failure, failure_status = description, "crash"
                else:
                    violations = check_frozen(parent["path"], draft, task.editable_files)
                    if violations:
                        failure = f"modified non-editable file(s): {violations}"
                        failure_status = "violation"
                with print_lock:
                    label = description if ok else f"FAILED: {description}"
                    print(f"[proposal] iter {iteration}: {label}")
                candidate_queue.put({
                    **item,
                    "draft": draft,
                    "description": description,
                    "failure": failure,
                    "failure_status": failure_status,
                })
            except Exception as exc:
                candidate_queue.put({
                    **item,
                    "draft": draft,
                    "description": f"proposal worker exception: {exc}",
                    "failure": repr(exc),
                    "failure_status": "crash",
                })
            finally:
                inspiration_queue.task_done()

    def evaluator_worker():
        while True:
            item = candidate_queue.get()
            if item is STOP:
                candidate_queue.task_done()
                break
            try:
                _evaluate_and_commit(item, task, eb, backend, model, print_lock)
            except Exception as exc:
                errors.put((f"evaluator iter {item['iteration']}", exc))
            finally:
                with active_lock:
                    active_directions.pop(item["iteration"], None)
                inflight.release()
                candidate_queue.task_done()

    evaluators = [
        threading.Thread(target=evaluator_worker, name=f"evaluator-{index}")
        for index in range(task.eval_concurrency)
    ]
    proposals = [
        threading.Thread(target=proposal_worker, name=f"proposal-{index}")
        for index in range(workers)
    ]
    producer = threading.Thread(target=context_producer, name="context-producer")
    for thread in evaluators + proposals + [producer]:
        thread.start()
    producer.join()
    inspiration_queue.join()
    for thread in proposals:
        thread.join()
    for _ in evaluators:
        candidate_queue.put(STOP)
    candidate_queue.join()
    for thread in evaluators:
        thread.join()
    if not errors.empty():
        stage, exc = errors.get()
        raise RuntimeError(f"pipeline failure in {stage}: {exc}")


def init_seed(task, eb):
    sandbox = task.run_dir / "sandboxes" / "seed"
    print(f"[seed] validating official SimpleTES seed under {task.protocol} ...")
    score, status, log_tail, metrics = run_solution(task.seed_dir, sandbox, task)
    if status != "ok":
        sys.exit(f"Seed run failed ({status}):\n{log_tail}")
    seed_candidate = task.run_dir / "drafts" / "seed"
    if seed_candidate.exists():
        shutil.rmtree(seed_candidate)
    shutil.copytree(
        task.seed_dir, seed_candidate,
        ignore=shutil.ignore_patterns("solution.json", "run.log", "__pycache__"),
    )
    snapshot = sandbox / "evaluated_solution.json"
    if not snapshot.exists():
        snapshot = sandbox / "solution.snapshot.json"
    if snapshot.exists():
        shutil.copyfile(snapshot, seed_candidate / "solution.json")
    if (sandbox / "run.log").exists():
        shutil.copy2(sandbox / "run.log", seed_candidate / "run.log")
    record = eb.commit(
        seed_candidate, score, status, "official SimpleTES 17-element seed",
        None, log_tail, metrics,
        metadata={"protocol": task.protocol, "run_id": task.run_id},
    )
    print(f"[eb] seeded {record['id']}: {task.metric}={score:.12f}")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sums_diffs")
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS,
                        default=os.environ.get("OPENHYRA_BACKEND", "claude"))
    parser.add_argument("--model", default=os.environ.get("OPENHYRA_MODEL"))
    parser.add_argument("--trial-seed", type=int, default=0)
    parser.add_argument("--export-bundle")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.iterations < 0 or args.workers < 1:
        parser.error("--iterations must be >= 0 and --workers must be >= 1")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    task = Task(args.task, args.run_id)
    eb = ExperienceBank(task.run_dir / "eb", direction=task.direction)
    if args.status:
        for record in eb.records():
            score = f"{record['score']:.12f}" if record["score"] is not None else "-"
            iteration = record.get("metadata", {}).get("iteration", "seed")
            print(f"{record['id']}  iter={iteration}  {score}  {record['status']}")
        best = eb.best()
        if best:
            print(f"best: {best['id']} @ {best['score']:.12f}")
        return

    if args.init:
        if eb.records():
            sys.exit(f"run {args.run_id!r} is already initialized")
        init_seed(task, eb)
    if args.iterations:
        if not eb.records():
            sys.exit("Experience Bank is empty; use --init first")
        run_pipeline(
            task, eb, args.iterations, args.workers,
            args.backend, args.model, args.trial_seed,
        )
    if args.export_bundle:
        destination = export_bundle(
            task, eb, args.export_bundle, root=ROOT,
            backend=args.backend, model=args.model, workers=args.workers,
            trial_seed=args.trial_seed, started_at=started_at,
        )
        print(f"[bundle] exported {destination}")


if __name__ == "__main__":
    main()
