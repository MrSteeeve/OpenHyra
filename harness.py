"""OpenHyra: Context producer -> Proposal workers -> Evaluator workers."""

import argparse
import ast
import hashlib
import json
import os
import queue
import re
import shutil
import sys
import threading
from pathlib import Path

from context_agent import (
    CANDIDATE_SEED_TOKEN,
    build_inspiration,
    finalize_analysis,
    record_stop_review,
)
from eb import ExperienceBank
from llm_backend import SUPPORTED_BACKENDS
from proposal_agent import propose, repair_candidate
from provenance import (
    RunLock,
    build_run_manifest,
    load_run_manifest,
    validate_run_manifest,
    write_run_manifest,
)
from reporting import export_bundle
from sandbox import run_solution, trusted_artifact_dir
from stopping import StopController, StopPolicy, stopping_evidence, write_termination

ROOT = Path(__file__).resolve().parent
STOP = object()
MIN_CANDIDATES_PER_CONTEXT = 1
MAX_STORED_LOG_CHARS = 6000
REPAIRABLE_STATUSES = {"crash", "timeout"}


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
        self.candidates_per_context = cfg.get(
            "candidates_per_context", MIN_CANDIDATES_PER_CONTEXT,
        )
        self.candidate_repair_attempts = cfg.get("candidate_repair_attempts", 0)
        if self.candidates_per_context < MIN_CANDIDATES_PER_CONTEXT:
            sys.exit("task candidates_per_context must be >= 1")
        if self.candidate_repair_attempts < 0:
            sys.exit("task candidate_repair_attempts must be >= 0")
        self.max_training_seconds = cfg.get("max_training_seconds")
        self.max_memory_mb = cfg.get("max_memory_mb", 1024)
        self.max_output_mb = cfg.get("max_output_mb", 64)
        self.max_artifact_bytes = cfg.get("max_artifact_bytes", 1024 * 1024)
        self.evaluator_timeout_s = cfg.get("evaluator_timeout_s", 300)
        self.evaluator_max_memory_mb = cfg.get("evaluator_max_memory_mb", 512)
        self.fallback_directions = cfg.get("fallback_directions", [])
        self.engineering_invariants = cfg.get("engineering_invariants", [])
        self.description = (self.dir / "TASK.md").read_text()
        self.evaluator = self.dir / "evaluator.py"
        if not self.evaluator.exists():
            sys.exit(f"Task {name!r} has no trusted evaluator.py — refusing to run")
        self.seed_dir = self.dir / "seed_solution"
        self.python_bin = sys.executable
        self.run_id = run_id
        self.run_dir = ROOT / "runs" / self.name / run_id
        self.run_manifest = None


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


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _same_expression(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _guard_proves_nonempty_range(test, start, stop):
    for node in ast.walk(test):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
            continue
        left, operation, right = node.left, node.ops[0], node.comparators[0]
        if (isinstance(operation, ast.Gt) and
                _same_expression(left, stop) and _same_expression(right, start)):
            return True
        if (isinstance(operation, ast.Lt) and
                _same_expression(left, start) and _same_expression(right, stop)):
            return True
    return False


def _known_solver_issues(draft_dir, editable_files):
    """Catch previously observed deterministic runtime hazards before launch.

    Lints every editable Python file — the rules are generic Python hazards,
    not task-specific ones, so this stays valid for future task plugins.
    """
    issues = []
    for name in editable_files:
        if not name.endswith(".py"):
            continue
        issues.extend(_known_file_issues(Path(draft_dir) / name, name))
    return issues


def _known_file_issues(path, name):
    if not path.is_file():
        return [f"{name} is missing"]
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [f"{name} cannot be parsed: {exc}"]

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    clamped = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None or not any(
            isinstance(child, ast.Call) and _call_name(child.func) in {"min", "max"}
            for child in ast.walk(value)
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        clamped.update(target.id for target in targets if isinstance(target, ast.Name))

    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Pow):
            continue
        exponent = node.right.value if isinstance(node.right, ast.Constant) else None
        base = node.left
        if not (isinstance(exponent, float) and not exponent.is_integer()):
            continue
        if not (isinstance(base, ast.BinOp) and isinstance(base.op, ast.Sub) and
                isinstance(base.right, ast.Name)):
            continue
        variable = base.right.id
        if ("progress" in variable.lower() or "fraction" in variable.lower()) and variable not in clamped:
            issues.append(
                f"{name} line {node.lineno}: fractional power of (constant - {variable}) "
                "without clamping the time-derived value to [0, 1]"
            )

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node.func) == "randrange" and
                len(node.args) >= 2):
            continue
        start, stop = node.args[0], node.args[1]
        if (isinstance(start, ast.Constant) and isinstance(stop, ast.Constant) and
                isinstance(start.value, int) and isinstance(stop.value, int) and
                stop.value > start.value):
            continue
        guarded = False
        ancestor = parents.get(node)
        while ancestor is not None:
            if isinstance(ancestor, (ast.If, ast.While)):
                in_body = any(
                    child is node or any(descendant is node for descendant in ast.walk(child))
                    for child in ancestor.body
                )
                if in_body and _guard_proves_nonempty_range(ancestor.test, start, stop):
                    guarded = True
                    break
            ancestor = parents.get(ancestor)
        if not guarded:
            issues.append(
                f"{name} line {node.lineno}: dynamic randrange(start, stop) lacks an explicit "
                "enclosing guard proving stop > start"
            )
    return issues


def _record_metadata(task, context_meta, backend, model):
    metadata = {
        **context_meta,
        "protocol": task.protocol,
        "run_id": task.run_id,
        "backend": backend,
        "model": model,
    }
    manifest = getattr(task, "run_manifest", None) or {}
    if manifest:
        metadata.update({
            "run_manifest_sha256": manifest.get("manifest_sha256"),
            "source_sha256": manifest.get("source_sha256"),
            "task_provenance": manifest.get("task"),
        })
    return metadata


def _editable_hashes(directory, editable_files):
    hashes = {}
    for name in editable_files:
        path = Path(directory) / name
        hashes[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() else None
        )
    return hashes


def _next_context_iteration(records):
    """Count Context rounds independently from the number of EB records."""
    iterations = [
        record.get("metadata", {}).get("iteration")
        for record in records
    ]
    iterations = [iteration for iteration in iterations if isinstance(iteration, int)]
    return max(iterations, default=-1) + 1


def _candidate_seed(context_seed, candidate_index):
    return context_seed * 1_000_003 + candidate_index


def _candidate_prompt(prompt, candidate_index, candidate_count, seed):
    prompt = prompt.replace(CANDIDATE_SEED_TOKEN, str(seed))
    return prompt + f"""

## Local candidate identity

This is candidate {candidate_index + 1} of {candidate_count} generated from the
same Context briefing. Produce your own concrete implementation/parameterization;
all {candidate_count} candidates are evaluated independently, and every outcome
is committed to the Experience Bank, including failures and low scores.
"""


def _evaluate_candidate(item, task, print_lock):
    iteration = item["iteration"]
    candidate_index = item["candidate_index"]
    draft = item["draft"]

    if item.get("failure"):
        score, status, log_tail, metrics = None, item["failure_status"], item["failure"], {}
    else:
        sandbox = task.run_dir / "sandboxes" / f"iter_{iteration:04d}" / f"cand_{candidate_index:02d}"
        with print_lock:
            print(
                f"[sandbox] iter {iteration} candidate {candidate_index + 1}/"
                f"{item['candidate_count']}: running candidate + trusted evaluator ..."
            )
        score, status, log_tail, metrics = run_solution(draft, sandbox, task)
        if (sandbox / "run.log").exists():
            shutil.copy2(sandbox / "run.log", draft / "run.log")
        trusted = trusted_artifact_dir(sandbox)
        snapshot = trusted / "evaluated_solution.json"
        if not snapshot.exists():
            snapshot = trusted / "solution.snapshot.json"
        if snapshot.exists():
            shutil.copy2(snapshot, draft / "solution.json")

    return {
        "item": item,
        "score": score,
        "status": status,
        "log_tail": log_tail,
        "metrics": metrics,
    }


def _stored_log(log_tail):
    return (log_tail or "")[-MAX_STORED_LOG_CHARS:]


def _evaluate_candidate_with_repair(item, task, backend, model, print_lock):
    """Return immutable initial/repair attempts, each backed by its own draft."""
    item = {**item, "attempt_index": 0, "repair_of": None}
    result = _evaluate_candidate(item, task, print_lock)
    results = [result]
    current_item = item
    repair_budget = getattr(task, "candidate_repair_attempts", 0)

    for repair_index in range(repair_budget):
        runtime_failure = (
            not current_item.get("failure") and
            result["status"] in REPAIRABLE_STATUSES
        )
        rejected_preflight = bool(current_item.get("repairable"))
        if not (runtime_failure or rejected_preflight):
            break
        with print_lock:
            print(
                f"[repair] iter {item['iteration']} candidate "
                f"{item['candidate_index'] + 1}/{item['candidate_count']}: "
                f"attempt {repair_index + 1}/{repair_budget} after {result['status']}"
            )
        repair_draft = item["draft"].with_name(
            f"{item['draft'].name}_repair_{repair_index + 1:02d}"
        )
        ok, note = repair_candidate(
            current_item["draft"], repair_draft, result.get("log_tail", ""),
            task.editable_files,
            backend=backend, model=model,
        )
        repair_item = {
            **item,
            "draft": repair_draft,
            "attempt_index": repair_index + 1,
            "failure": None,
            "failure_status": None,
            "repairable": False,
            "repair_note": note,
            "preflight_notes": [],
        }
        if not ok:
            repair_item.update({
                "failure": note,
                "failure_status": "crash",
            })
        else:
            violations = check_frozen(
                item["parent"]["path"], repair_draft, task.editable_files,
            )
            issues = _known_solver_issues(repair_draft, task.editable_files)
            if violations:
                repair_item.update({
                    "failure": f"repair modified non-editable file(s): {violations}",
                    "failure_status": "violation",
                })
            elif issues:
                feedback = "Engineering preflight rejected the repair:\n- " + "\n- ".join(issues)
                repair_item.update({
                    "failure": feedback,
                    "failure_status": "rejected",
                    "repairable": True,
                    "preflight_notes": [feedback],
                })
        result = _evaluate_candidate(repair_item, task, print_lock)
        results.append(result)
        current_item = repair_item

    return results


def _duplicate_of(result, records):
    """Return the first evaluator-equivalent EB record, if one exists."""
    set_hash = result.get("metrics", {}).get("set_hash")
    if result["status"] != "ok" or not set_hash:
        return None
    return next(
        (record["id"] for record in records
         if record.get("metrics", {}).get("set_hash") == set_hash),
        None,
    )


def _commit_candidate_result(result, task, eb, backend, model, print_lock,
                             parent_id=None, repair_of=None):
    """Commit one candidate outcome without local winner selection."""
    item = result["item"]
    iteration = item["iteration"]
    parent = item["parent"]
    metadata = _record_metadata(task, item["context_meta"], backend, model)
    metadata.update({
        "candidate_count": item["candidate_count"],
        "candidate_index": item["candidate_index"],
        "candidate_seed": item["candidate_seed"],
        "duplicate_of": _duplicate_of(result, eb.records()),
        "attempt_index": item.get("attempt_index", 0),
        "repair_of": repair_of,
        "repair_note": item.get("repair_note"),
        "preflight_notes": item.get("preflight_notes", []),
        "editable_file_sha256": _editable_hashes(
            item["draft"], task.editable_files,
        ),
    })

    previous_best = eb.best()
    record = eb.commit(
        item["draft"], result["score"], result["status"],
        item["description"], parent_id or parent["id"], result["log_tail"],
        metrics=result["metrics"], metadata=metadata,
    )
    best = eb.best()
    improved = eb.is_improvement(
        result["score"], previous_best["score"] if previous_best else None,
    )
    with print_lock:
        verdict = "IMPROVED" if improved else "best unchanged"
        best_text = f"{best['id']} @ {best['score']:.9f}" if best else "none"
        print(
            f"[eb] iter {iteration} candidate {item['candidate_index'] + 1}/"
            f"{item['candidate_count']} attempt {item.get('attempt_index', 0)} "
            f"-> {record['id']}: score={result['score']} "
            f"status={result['status']}, {verdict} (best: {best_text})"
        )
    return record


def run_pipeline(task, eb, iterations, workers, backend, model, trial_seed,
                 candidates_per_context=None, stop_policy=None):
    """Run a bounded three-stage asynchronous producer-consumer pipeline."""
    if candidates_per_context is None:
        candidates_per_context = task.candidates_per_context
    if candidates_per_context < MIN_CANDIDATES_PER_CONTEXT:
        raise ValueError(
            f"candidates_per_context must be >= {MIN_CANDIDATES_PER_CONTEXT}"
        )
    stop_policy = stop_policy or StopPolicy()
    stop_controller = StopController(stop_policy, task.direction)
    inspiration_queue = queue.Queue(maxsize=max(1, workers * candidates_per_context))
    candidate_queue = queue.Queue(maxsize=max(1, workers + task.eval_concurrency))
    # Agent stop decisions must observe all results from the prior Context.
    # Candidate generation/evaluation inside that Context remains concurrent.
    context_window = 1 if stop_policy.enabled else max(1, workers + task.eval_concurrency)
    inflight = threading.Semaphore(context_window)
    active_directions = {}
    active_lock = threading.Lock()
    print_lock = threading.Lock()
    errors = queue.Queue()
    termination_request = {}
    start = _next_context_iteration(eb.records())

    def context_producer():
        try:
            for iteration in range(start, start + iterations):
                inflight.acquire()
                try:
                    with active_lock:
                        reserved = tuple(active_directions.values())
                    evidence_at_decision = (
                        stopping_evidence(
                            eb.records(), direction=task.direction,
                            policy=stop_policy,
                        )
                        if stop_policy.enabled else None
                    )
                    decision, baseline, prompt, direction, context_meta = build_inspiration(
                        task, eb, iteration, backend=backend, model=model,
                        active_directions=reserved,
                        trial_seed=trial_seed + iteration,
                        agent_stop_enabled=stop_policy.enabled,
                        stop_evidence=evidence_at_decision,
                    )
                    if decision.action == "stop":
                        review = stop_controller.review(decision, eb.records())
                        review_payload = review.to_dict()
                        record_stop_review(eb, iteration, review_payload)
                        context_meta["stop_review"] = review_payload
                        with print_lock:
                            verdict = "accepted" if review.accepted else "rejected"
                            reasons = ", ".join(review.reasons) or "all guards passed"
                            print(
                                f"[stop] iter {iteration}: Agent request {verdict} "
                                f"({reasons})"
                            )
                        if review.accepted:
                            termination_request.update({
                                "reason": "agent_converged",
                                "terminal": True,
                                "requested_by": "context_agent",
                                "accepted_by": "stop_controller",
                                "context_decision": decision.to_dict(),
                                "stop_review": review_payload,
                            })
                            inflight.release()
                            break
                        context_meta["effective_context_decision"] = (
                            decision.forced_continue(
                                direction,
                                "Stop request rejected by deterministic evidence guards.",
                            ).to_dict()
                        )
                    with active_lock:
                        active_directions[iteration] = direction
                    with print_lock:
                        short = " ".join(direction.split())[:180]
                        print(
                            f"[context] iter {iteration}: EB v{context_meta['eb_version']}, "
                            f"baseline={baseline['id']}, next={short}"
                        )
                    for candidate_index in range(candidates_per_context):
                        seed = _candidate_seed(context_meta["trial_seed"], candidate_index)
                        inspiration_queue.put({
                            "iteration": iteration,
                            "parent": baseline,
                            "prompt": _candidate_prompt(
                                prompt, candidate_index, candidates_per_context, seed,
                            ),
                            "context_meta": context_meta,
                            "candidate_index": candidate_index,
                            "candidate_count": candidates_per_context,
                            "candidate_seed": seed,
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
            candidate_index = item["candidate_index"]
            parent = item["parent"]
            draft = task.run_dir / "drafts" / f"iter_{iteration:04d}" / f"cand_{candidate_index:02d}"
            try:
                ok, description = propose(
                    Path(parent["path"]), draft, item["prompt"], task.editable_files,
                    backend=backend, model=model,
                )
                failure = None
                failure_status = None
                preflight_notes = []
                if not ok:
                    failure, failure_status = description, "crash"
                else:
                    violations = check_frozen(parent["path"], draft, task.editable_files)
                    if violations:
                        failure = f"modified non-editable file(s): {violations}"
                        failure_status = "violation"
                if failure is None:
                    issues = _known_solver_issues(draft, task.editable_files)
                    if issues:
                        feedback = "Engineering preflight rejected the draft:\n- " + "\n- ".join(issues)
                        preflight_notes.append(feedback)
                        failure = "engineering preflight failed: " + "; ".join(issues)
                        # distinct from "violation" (frozen-file tampering):
                        # a lint reject is benign engineering feedback
                        failure_status = "rejected"
                with print_lock:
                    label = description if failure is None else f"FAILED: {failure}"
                    print(
                        f"[proposal] iter {iteration} candidate {candidate_index + 1}/"
                        f"{item['candidate_count']}: {label}"
                    )
                candidate_queue.put({
                    **item,
                    "draft": draft,
                    "description": description,
                    "failure": failure,
                    "failure_status": failure_status,
                    "repairable": failure_status == "rejected",
                    "preflight_notes": preflight_notes,
                })
            except Exception as exc:
                candidate_queue.put({
                    **item,
                    "draft": draft,
                    "description": f"proposal worker exception: {exc}",
                    "failure": repr(exc),
                    "failure_status": "crash",
                    "repairable": False,
                    "preflight_notes": [],
                })
            finally:
                inspiration_queue.task_done()

    context_completions = {}
    completion_lock = threading.Lock()

    def evaluator_worker():
        while True:
            item = candidate_queue.get()
            if item is STOP:
                candidate_queue.task_done()
                break
            records = []
            try:
                try:
                    results = _evaluate_candidate_with_repair(
                        item, task, backend, model, print_lock,
                    )
                except Exception as exc:
                    results = [{
                        "item": item,
                        "score": None,
                        "status": "crash",
                        "log_tail": repr(exc),
                        "metrics": {},
                    }]
                parent_id = item["parent"]["id"]
                repair_of = None
                for result in results:
                    record = _commit_candidate_result(
                        result, task, eb, backend, model, print_lock,
                        parent_id=parent_id, repair_of=repair_of,
                    )
                    records.append(record)
                    parent_id = record["id"]
                    repair_of = record["id"]
            except Exception as exc:
                errors.put((f"evaluator iter {item['iteration']}", exc))
            finally:
                context_finished = False
                result_ids = None
                with completion_lock:
                    state = context_completions.setdefault(
                        item["iteration"], {"finished": 0, "result_ids": []},
                    )
                    state["finished"] += 1
                    state["result_ids"].extend(record["id"] for record in records)
                    if state["finished"] == item["candidate_count"]:
                        result_ids = list(state["result_ids"])
                        context_completions.pop(item["iteration"])
                        context_finished = True
                if context_finished:
                    finalize_analysis(eb, item["iteration"], result_ids)
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
    if termination_request:
        return termination_request
    return {
        "reason": "iteration_limit",
        "terminal": False,
        "requested_by": "harness",
        "accepted_by": "harness",
    }


def _termination_payload(task, eb, stop_policy, outcome, requested_iterations):
    records = eb.records()
    evidence = (
        outcome.get("stop_review", {}).get("evidence")
        or stopping_evidence(records, direction=task.direction, policy=stop_policy)
    )
    best = eb.best()
    candidate_attempts = sum(
        isinstance(record.get("metadata", {}).get("iteration"), int)
        for record in records
    )
    payload = {
        **outcome,
        "run_id": task.run_id,
        "requested_iterations": requested_iterations,
        "completed_contexts": evidence["completed_contexts"],
        "candidate_attempts": candidate_attempts,
        "best_id": best["id"] if best else None,
        "best_score": best["score"] if best else None,
        "contexts_since_meaningful_improvement": evidence[
            "contexts_since_meaningful_improvement"
        ],
        "stopping_policy": stop_policy.to_dict(),
        "evidence": evidence,
    }
    return payload


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
    trusted = trusted_artifact_dir(sandbox)
    snapshot = trusted / "evaluated_solution.json"
    if not snapshot.exists():
        snapshot = trusted / "solution.snapshot.json"
    if snapshot.exists():
        shutil.copyfile(snapshot, seed_candidate / "solution.json")
    if (sandbox / "run.log").exists():
        shutil.copy2(sandbox / "run.log", seed_candidate / "run.log")
    manifest = getattr(task, "run_manifest", None) or {}
    seed_metadata = {
        "protocol": task.protocol,
        "run_id": task.run_id,
        "run_manifest_sha256": manifest.get("manifest_sha256"),
        "source_sha256": manifest.get("source_sha256"),
        "task_provenance": manifest.get("task"),
        "editable_file_sha256": _editable_hashes(
            seed_candidate, task.editable_files,
        ),
    }
    record = eb.commit(
        seed_candidate, score, status, "official SimpleTES 17-element seed",
        None, log_tail, metrics,
        metadata=seed_metadata,
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
    parser.add_argument(
        "--candidates-per-context", type=int,
        help="independent candidates per Context; every outcome is committed",
    )
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS,
                        default=os.environ.get("OPENHYRA_BACKEND", "claude"))
    parser.add_argument("--model", default=os.environ.get("OPENHYRA_MODEL"))
    parser.add_argument("--trial-seed", type=int, default=0)
    parser.add_argument(
        "--agent-stop", action="store_true",
        help="allow Context to request stopping, subject to deterministic guards",
    )
    parser.add_argument("--min-contexts-before-stop", type=int, default=6)
    parser.add_argument("--stop-patience", type=int, default=4)
    parser.add_argument("--stop-min-delta", type=float, default=0.0001)
    parser.add_argument("--stop-recent-window", type=int, default=4)
    parser.add_argument("--stop-min-successful-candidates", type=int, default=4)
    parser.add_argument("--export-bundle")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if (args.iterations < 0 or args.workers < 1 or
            (args.candidates_per_context is not None and
             args.candidates_per_context < MIN_CANDIDATES_PER_CONTEXT)):
        parser.error(
            "--iterations must be >= 0; --workers must be >= 1; "
            f"--candidates-per-context must be >= {MIN_CANDIDATES_PER_CONTEXT}"
        )

    task = Task(args.task, args.run_id)
    try:
        stop_policy = StopPolicy(
            enabled=args.agent_stop,
            min_contexts_before_stop=args.min_contexts_before_stop,
            stop_patience=args.stop_patience,
            meaningful_delta=args.stop_min_delta,
            recent_window=args.stop_recent_window,
            min_successful_candidates=args.stop_min_successful_candidates,
        )
    except ValueError as exc:
        parser.error(str(exc))
    eb = ExperienceBank(task.run_dir / "eb", direction=task.direction)
    if args.status:
        for record in eb.records():
            score = f"{record['score']:.12f}" if record["score"] is not None else "-"
            iteration = record.get("metadata", {}).get("iteration", "seed")
            print(f"{record['id']}  iter={iteration}  {score}  {record['status']}")
        best = eb.best()
        if best:
            print(f"best: {best['id']} @ {best['score']:.12f}")
        termination_path = task.run_dir / "termination.json"
        if termination_path.is_file():
            termination = json.loads(termination_path.read_text())
            print(
                f"last termination: {termination.get('reason')} "
                f"(terminal={termination.get('terminal')})"
            )
        return

    candidates_per_context = (
        task.candidates_per_context
        if args.candidates_per_context is None
        else args.candidates_per_context
    )
    manifest_path = task.run_dir / "run_manifest.json"
    lock = RunLock(task.run_dir / "run.lock")
    try:
        lock.acquire()
        if args.init:
            if eb.records():
                sys.exit(f"run {args.run_id!r} is already initialized")
            task.run_manifest = build_run_manifest(
                task, ROOT, backend=args.backend, model=args.model,
                workers=args.workers,
                candidates_per_context=candidates_per_context,
                trial_seed=args.trial_seed,
                stopping_policy=stop_policy.to_dict(),
            )
            write_run_manifest(manifest_path, task.run_manifest)
            init_seed(task, eb)
        elif args.iterations:
            if not eb.records():
                sys.exit("Experience Bank is empty; use --init first")
            recorded = load_run_manifest(manifest_path)
            current = build_run_manifest(
                task, ROOT, backend=args.backend, model=args.model,
                workers=args.workers,
                candidates_per_context=candidates_per_context,
                trial_seed=args.trial_seed,
                stopping_policy=stop_policy.to_dict(),
            )
            task.run_manifest = validate_run_manifest(recorded, current)

        if args.iterations:
            try:
                outcome = run_pipeline(
                    task, eb, args.iterations, args.workers,
                    args.backend, args.model, args.trial_seed,
                    candidates_per_context=candidates_per_context,
                    stop_policy=stop_policy,
                )
            except KeyboardInterrupt:
                outcome = {
                    "reason": "user_interrupt",
                    "terminal": True,
                    "requested_by": "user",
                    "accepted_by": "harness",
                }
                write_termination(
                    task.run_dir / "termination.json",
                    _termination_payload(
                        task, eb, stop_policy, outcome, args.iterations,
                    ),
                )
                raise
            except Exception as exc:
                outcome = {
                    "reason": "pipeline_error",
                    "terminal": True,
                    "requested_by": "harness",
                    "accepted_by": "harness",
                    "error": repr(exc),
                }
                write_termination(
                    task.run_dir / "termination.json",
                    _termination_payload(
                        task, eb, stop_policy, outcome, args.iterations,
                    ),
                )
                raise
            write_termination(
                task.run_dir / "termination.json",
                _termination_payload(
                    task, eb, stop_policy, outcome, args.iterations,
                ),
            )
        if args.export_bundle:
            if task.run_manifest is None:
                task.run_manifest = load_run_manifest(manifest_path)
            destination = export_bundle(
                task, eb, args.export_bundle, root=ROOT,
                run_manifest=task.run_manifest,
            )
            print(f"[bundle] exported {destination}")
    except RuntimeError as exc:
        sys.exit(str(exc))
    finally:
        lock.release()


if __name__ == "__main__":
    main()
