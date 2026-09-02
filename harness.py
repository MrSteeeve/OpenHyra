"""OpenHyra: Context producer -> Proposal workers -> Evaluator workers."""

import argparse
import ast
import hashlib
import importlib.util
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
    MAX_PROPOSAL_PROMPT_CHARS,
    PROPOSAL_IDENTITY_RESERVE_CHARS,
    build_inspiration,
    finalize_analysis,
    record_stop_review,
)
from auditing import run_final_audit
from eb import ExperienceBank
from external_formal_runner import build_external_formal_runner
from llm_backend import SUPPORTED_BACKENDS
from proposal_agent import (
    propose,
    repair_candidate,
    revise_research_candidate,
)
from provenance import (
    RunLock,
    build_run_manifest,
    load_run_manifest,
    validate_run_manifest,
    write_run_manifest,
)
from reporting import export_bundle
from sandbox import (
    read_regular_file,
    run_solution,
    snapshot_source_tree,
    source_tree_hash,
    trusted_artifact_dir,
)
from stopping import (
    StopController,
    StopPolicy,
    incomplete_contexts,
    stopping_evidence,
    write_termination,
)

try:
    from harness_v5 import V5Bridge, adapt_bermudan_metrics, get_metrics_adapter
    from schemas_v5 import ExperimentPlan
except ImportError:
    V5Bridge = None
    adapt_bermudan_metrics = None
    ExperimentPlan = None

ROOT = Path(__file__).resolve().parent
STOP = object()
MIN_CANDIDATES_PER_CONTEXT = 1
MAX_STORED_LOG_CHARS = 6000
REPAIRABLE_STATUSES = {"crash", "timeout"}
V5_PACKET_TRUNCATION_MARKER = "\n\n[V5 packet clipped to fit the prompt budget]"


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
        self.seed_description = cfg.get(
            "seed_description", f"official {self.name} seed",
        )
        if (
            not isinstance(self.seed_description, str)
            or not self.seed_description.strip()
            or len(self.seed_description) > 256
        ):
            sys.exit("task seed_description must be bounded non-empty text")
        self.seed_description = self.seed_description.strip()
        self.editable_files = cfg["editable_files"]
        self.timeout_s = cfg.get("sandbox_timeout_s", 660)
        self.eval_concurrency = cfg.get("eval_concurrency", 1)
        self.candidates_per_context = cfg.get(
            "candidates_per_context", MIN_CANDIDATES_PER_CONTEXT,
        )
        self.candidate_repair_attempts = cfg.get("candidate_repair_attempts", 0)
        self.research_revision_attempts = cfg.get(
            "research_revision_attempts", 0,
        )
        phases = cfg.get("allowed_context_phases")
        if phases is not None and (
            not isinstance(phases, list)
            or not phases
            or any(
                not isinstance(phase, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", phase)
                for phase in phases
            )
            or len(set(phases)) != len(phases)
        ):
            sys.exit(
                "task allowed_context_phases must be a non-empty list of "
                "unique bounded strings"
            )
        self.allowed_context_phases = tuple(phases) if phases else None
        instructions = cfg.get("candidate_instructions")
        if instructions is not None and not (
            isinstance(instructions, str)
            or (
                isinstance(instructions, list)
                and all(isinstance(item, str) for item in instructions)
            )
        ):
            sys.exit(
                "task candidate_instructions must be text or a list of text"
            )
        self.candidate_instructions = instructions
        evaluation = cfg.get("evaluation", {})
        if not isinstance(evaluation, dict):
            sys.exit("task evaluation must be an object")
        unknown_evaluation = set(evaluation) - {"search_stage", "audit_stage"}
        if unknown_evaluation:
            sys.exit(
                "task evaluation contains unsupported stage(s): "
                + ", ".join(sorted(unknown_evaluation))
            )
        for stage in ("search_stage", "audit_stage"):
            value = evaluation.get(stage)
            if value is not None and not isinstance(value, dict):
                sys.exit(f"task evaluation.{stage} must be an object")
        audit_stage = evaluation.get("audit_stage")
        if audit_stage is not None:
            top_k = audit_stage.get("top_k")
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
                sys.exit("task evaluation.audit_stage.top_k must be >= 1")
            audit_direction = audit_stage.get("direction", self.direction)
            if audit_direction not in {"min", "max"}:
                sys.exit(
                    "task evaluation.audit_stage.direction must be min or max"
                )
        self.evaluation = evaluation
        self.search_evaluation_request = None
        if self.candidates_per_context < MIN_CANDIDATES_PER_CONTEXT:
            sys.exit("task candidates_per_context must be >= 1")
        if self.candidate_repair_attempts < 0:
            sys.exit("task candidate_repair_attempts must be >= 0")
        if self.research_revision_attempts < 0:
            sys.exit("task research_revision_attempts must be >= 0")
        self.max_training_seconds = cfg.get("max_training_seconds")
        self.max_memory_mb = cfg.get("max_memory_mb", 1024)
        self.max_output_mb = cfg.get("max_output_mb", 64)
        self.max_artifact_bytes = cfg.get("max_artifact_bytes", 1024 * 1024)
        self.evaluator_timeout_s = cfg.get("evaluator_timeout_s", 300)
        self.evaluator_max_memory_mb = cfg.get("evaluator_max_memory_mb", 512)
        self.fallback_directions = cfg.get("fallback_directions", [])
        self.engineering_invariants = cfg.get("engineering_invariants", [])
        formalization = cfg.get("formalization", {})
        self.formalization = formalization
        self.required_formal_claims = tuple(
            formalization.get("required_claim_templates", [])
        )
        self.formal_runner = None
        self.formal_runner_identity = None
        self.verify_formalization = None
        self.validate_formalization_request = None
        self.build_formalization_wrapper = None
        self.build_formalization_audit = None
        self.formal_spec_files = {}
        self.formal_spec_sha256 = None
        formalizer_name = formalization.get("module")
        if formalizer_name:
            formalizer_path = self.dir / formalizer_name
            if not formalizer_path.is_file():
                sys.exit(
                    f"Task {name!r} formalization module is missing: "
                    f"{formalizer_name}"
                )
            module_name = (
                "openhyra_task_"
                + re.sub(r"[^A-Za-z0-9_]", "_", name)
                + "_formalization"
            )
            module_spec = importlib.util.spec_from_file_location(
                module_name, formalizer_path,
            )
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
            self.verify_formalization = module.verify_formalization_request
            self.validate_formalization_request = getattr(
                module, "validate_formalization_request", None
            )
            self.build_formalization_wrapper = getattr(
                module, "build_formalization_wrapper", None
            )
            self.build_formalization_audit = getattr(
                module, "build_formalization_audit", None
            )
        spec_dir_name = formalization.get("spec_dir")
        if spec_dir_name:
            spec_dir = self.dir / spec_dir_name
            if not spec_dir.is_dir():
                sys.exit(
                    f"Task {name!r} formalization spec is missing: "
                    f"{spec_dir_name}"
                )
            for path in sorted(spec_dir.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(spec_dir).as_posix()
                    self.formal_spec_files[relative] = path.read_bytes()
            spec_hashes = {
                name: hashlib.sha256(data).hexdigest()
                for name, data in sorted(self.formal_spec_files.items())
            }
            self.formal_spec_sha256 = hashlib.sha256(
                json.dumps(
                    spec_hashes,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
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
                "solution.snapshot.json", "evidence.json", "PROPOSAL.md",
            }:
                output[relative] = path.read_bytes()
    return output


def check_frozen(parent_dir, draft_dir, editable):
    before, after = solution_files(parent_dir), solution_files(draft_dir)
    return [
        relative for relative in sorted(set(before) | set(after))
        if relative not in editable and before.get(relative) != after.get(relative)
    ]


def _remove_generated_path(path):
    """Remove one reserved harness-generated path without following links."""
    path = Path(path)
    try:
        path.lstat()
    except FileNotFoundError:
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _copy_generated_file(source, destination):
    """Replace an untrusted destination with one trusted regular-file copy."""
    source = Path(source)
    data = read_regular_file(
        source, 64 * 1024 * 1024, label=f"generated file {source.name}",
    )
    _remove_generated_path(destination)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def _source_limit_bytes(task):
    return int(
        getattr(task, "max_source_bytes", 0)
        or int(getattr(task, "max_output_mb", 64)) * 1024 * 1024
    )


def _validate_hashed_file(path, expected_hash, *, label, max_bytes):
    """Fail closed unless one regular file matches its trusted digest."""
    data = read_regular_file(path, max_bytes, label=label)
    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return data


def _assemble_commit_snapshot(source_dir, sandbox, task, metrics):
    """Build one parent-controlled directory used for both validation and EB."""
    sandbox = Path(sandbox)
    trusted = trusted_artifact_dir(sandbox)
    commit_dir = trusted / "commit"
    _remove_generated_path(commit_dir)

    trusted_source = trusted / "source"
    if trusted_source.is_dir():
        shutil.copytree(trusted_source, commit_dir)
    elif Path(source_dir).exists():
        snapshot_source_tree(
            source_dir, commit_dir, _source_limit_bytes(task),
        )
    else:
        commit_dir.mkdir(parents=True)
    # EB records are future Proposal baselines. Restore owner write permission
    # only on configured editable files in this parent-controlled copy; the
    # sealed execution source remains read-only.
    for name in getattr(task, "editable_files", ()):
        editable = commit_dir / name
        if editable.is_file() and not editable.is_symlink():
            editable.chmod(editable.stat().st_mode | 0o200)

    source_hash, _source_files = source_tree_hash(
        commit_dir, _source_limit_bytes(task),
    )
    expected_source_hash = metrics.get("source_snapshot_sha256")
    if expected_source_hash and source_hash != expected_source_hash:
        raise RuntimeError(
            "sealed source hash no longer matches the evaluated source"
        )
    metrics.setdefault("source_snapshot_sha256", source_hash)

    artifact = trusted / "evaluated_solution.json"
    if not artifact.exists():
        artifact = trusted / "solution.snapshot.json"
    expected_artifact_hash = metrics.get("artifact_sha256")
    if expected_artifact_hash:
        _validate_hashed_file(
            artifact, expected_artifact_hash,
            label="trusted solution.json",
            max_bytes=int(getattr(task, "max_artifact_bytes", 1024 * 1024)),
        )
        _copy_generated_file(artifact, commit_dir / "solution.json")
    elif artifact.exists() or artifact.is_symlink():
        raise RuntimeError("trusted solution artifact is missing its hash")

    evidence = trusted / "evidence.json"
    expected_evidence_hash = metrics.get("evidence_sha256")
    if expected_evidence_hash:
        _validate_hashed_file(
            evidence, expected_evidence_hash,
            label="trusted evidence.json",
            max_bytes=int(getattr(task, "max_output_mb", 64)) * 1024 * 1024,
        )
        _copy_generated_file(evidence, commit_dir / "evidence.json")
    elif evidence.exists() or evidence.is_symlink():
        raise RuntimeError("trusted evidence artifact is missing its hash")

    log_path = sandbox / "run.log"
    if log_path.exists() or log_path.is_symlink():
        _copy_generated_file(log_path, commit_dir / "run.log")

    if expected_artifact_hash:
        _validate_hashed_file(
            commit_dir / "solution.json", expected_artifact_hash,
            label="commit solution.json",
            max_bytes=int(getattr(task, "max_artifact_bytes", 1024 * 1024)),
        )
    if expected_evidence_hash:
        _validate_hashed_file(
            commit_dir / "evidence.json", expected_evidence_hash,
            label="commit evidence.json",
            max_bytes=int(getattr(task, "max_output_mb", 64)) * 1024 * 1024,
        )
    return commit_dir


def _seal_candidate_source(source_dir, destination, task):
    """Capture exactly one proposal state before any validation or execution."""
    destination = Path(destination)
    _remove_generated_path(destination)
    if not Path(source_dir).exists():
        destination.mkdir(parents=True)
        return source_tree_hash(
            destination, _source_limit_bytes(task),
        )[0]
    source_hash, _source_files = snapshot_source_tree(
        source_dir, destination, _source_limit_bytes(task),
    )
    return source_hash


def _controlled_failure_result(item, task, exc, *, cancelled=False):
    """Represent intake failure without copying any rejected candidate bytes."""
    attempt_index = item.get("attempt_index", 0)
    name = f"cand_{item['candidate_index']:02d}"
    if attempt_index:
        kind = item.get("attempt_kind", "repair")
        name += f"_{kind}_{attempt_index:02d}"
    commit_dir = (
        task.run_dir / "rejected_sources"
        / f"iter_{item['iteration']:04d}" / name
    )
    _remove_generated_path(commit_dir)
    commit_dir.mkdir(parents=True)
    source_hash = source_tree_hash(
        commit_dir, _source_limit_bytes(task),
    )[0]
    status = "cancelled" if cancelled else "crash"
    note = (
        "evaluation cancelled by user interrupt"
        if cancelled else f"candidate intake failed closed: {exc!r}"
    )
    failed_item = {
        **item,
        "commit_dir": commit_dir,
        "failure": note,
        "failure_status": status,
        "repairable": False,
    }
    return {
        "item": failed_item,
        "score": None,
        "status": status,
        "log_tail": note,
        "metrics": {"source_snapshot_sha256": source_hash},
    }


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


def ensure_run_resumable(task, eb):
    """Fail closed when continuing would overwrite or skip terminal state."""
    if (
        (task.run_dir / "final_audit.json").exists()
        or (task.run_dir / "final_audit_artifacts").exists()
    ):
        raise RuntimeError(
            f"run {task.run_id!r} has entered final audit; start a new --run-id"
        )
    termination_path = task.run_dir / "termination.json"
    if termination_path.is_file():
        try:
            termination = json.loads(termination_path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"cannot validate existing termination record: {exc}"
            ) from exc
        if termination.get("terminal") is True:
            reason = termination.get("reason", "unknown")
            raise RuntimeError(
                f"run {task.run_id!r} already terminated ({reason}); "
                "start a new --run-id"
            )

    incomplete = incomplete_contexts(eb.records())
    if incomplete:
        labels = ", ".join(str(iteration) for iteration in incomplete)
        raise RuntimeError(
            f"run {task.run_id!r} contains incomplete Context(s): {labels}; "
            "automatic replay is not implemented, so start a new --run-id"
        )


def _candidate_seed(context_seed, candidate_index):
    return context_seed * 1_000_003 + candidate_index


def _candidate_prompt(prompt, candidate_index, candidate_count, seed):
    prompt = prompt.replace(CANDIDATE_SEED_TOKEN, str(seed))
    candidate_prompt = prompt + f"""

## Local candidate identity

This is candidate {candidate_index + 1} of {candidate_count} generated from the
same Context briefing. Produce your own concrete implementation/parameterization;
all {candidate_count} candidates are evaluated independently, and every outcome
is committed to the Experience Bank, including failures and low scores.
"""
    if len(candidate_prompt) > MAX_PROPOSAL_PROMPT_CHARS:
        raise ValueError("candidate Proposal prompt exceeds character limit")
    return candidate_prompt


def _evaluate_candidate(item, task, print_lock):
    item = dict(item)
    iteration = item["iteration"]
    candidate_index = item["candidate_index"]
    draft = item["draft"]
    attempt_index = item.get("attempt_index", 0)
    sandbox_name = f"cand_{candidate_index:02d}"
    if attempt_index:
        kind = item.get("attempt_kind", "repair")
        sandbox_name += f"_{kind}_{attempt_index:02d}"
    sandbox = (
        task.run_dir / "sandboxes" / f"iter_{iteration:04d}" / sandbox_name
    )
    sealed = (
        task.run_dir / "sealed_sources" / f"iter_{iteration:04d}" / sandbox_name
    )
    source_snapshot_sha256 = _seal_candidate_source(
        draft, sealed, task,
    )

    if item.get("failure"):
        score, status, log_tail, metrics = (
            None,
            item["failure_status"],
            item["failure"],
            {"source_snapshot_sha256": source_snapshot_sha256},
        )
    else:
        violations = check_frozen(
            item["parent"]["path"], sealed, task.editable_files,
        )
        issues = _known_solver_issues(sealed, task.editable_files)
        if violations:
            score, status, log_tail, metrics = (
                None,
                "violation",
                f"sealed proposal modified non-editable file(s): {violations}",
                {"source_snapshot_sha256": source_snapshot_sha256},
            )
        elif issues:
            score, status, log_tail, metrics = (
                None,
                "rejected",
                "sealed proposal failed engineering preflight: "
                + "; ".join(issues),
                {"source_snapshot_sha256": source_snapshot_sha256},
            )
        else:
            with print_lock:
                print(
                    f"[sandbox] iter {iteration} candidate "
                    f"{candidate_index + 1}/{item['candidate_count']}: "
                    "running candidate + trusted evaluator ..."
                )
            score, status, log_tail, metrics = run_solution(
                sealed, sandbox, task,
            )
    commit_dir = _assemble_commit_snapshot(
        sealed, sandbox, task, metrics,
    )
    item["commit_dir"] = commit_dir

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
    item = {
        **item,
        "attempt_index": 0,
        "attempt_kind": "initial",
        "repair_of": None,
    }
    cancel_event = getattr(task, "cancel_event", None)
    try:
        result = _evaluate_candidate(item, task, print_lock)
    except Exception as exc:
        result = _controlled_failure_result(
            item, task, exc,
            cancelled=cancel_event is not None and cancel_event.is_set(),
        )
    results = [result]
    current_item = result["item"]
    repair_budget = getattr(task, "candidate_repair_attempts", 0)

    for repair_index in range(repair_budget):
        if cancel_event is not None and cancel_event.is_set():
            break
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
            current_item["commit_dir"],
            repair_draft,
            result.get("log_tail", ""),
            task.editable_files,
            backend=backend, model=model, cancel_event=cancel_event,
        )
        repair_item = {
            **item,
            "draft": repair_draft,
            "attempt_index": repair_index + 1,
            "attempt_kind": "runtime_repair",
            "failure": None,
            "failure_status": None,
            "repairable": False,
            "repair_note": note,
            "preflight_notes": [],
        }
        if cancel_event is not None and cancel_event.is_set():
            repair_item.update({
                "failure": "repair cancelled by user interrupt",
                "failure_status": "cancelled",
            })
        elif not ok:
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
        try:
            result = _evaluate_candidate(repair_item, task, print_lock)
        except Exception as exc:
            result = _controlled_failure_result(
                repair_item, task, exc,
                cancelled=cancel_event is not None and cancel_event.is_set(),
            )
        results.append(result)
        current_item = result["item"]

    research_budget = getattr(task, "research_revision_attempts", 0)
    for revision_index in range(research_budget):
        if cancel_event is not None and cancel_event.is_set():
            break
        metrics = result.get("metrics", {})
        formal_status = metrics.get("formalization_status")
        needs_revision = (
            metrics.get("refuted_obligation_count", 0) > 0
            or metrics.get("refuted_claim_count", 0) > 0
            or metrics.get("refuted_certificate_count", 0) > 0
            or formal_status in {"rejected", "infrastructure_error"}
        )
        if not needs_revision:
            break
        evidence_path = current_item.get("commit_dir", Path()) / "evidence.json"
        try:
            evidence_feedback = read_regular_file(
                evidence_path,
                int(getattr(task, "max_output_mb", 64)) * 1024 * 1024,
                label="trusted research evidence",
            ).decode("utf-8", errors="replace")
        except (OSError, ValueError):
            evidence_feedback = json.dumps(
                {
                    "formalization_status": formal_status,
                    "refuted_obligation_count": metrics.get(
                        "refuted_obligation_count", 0,
                    ),
                    "refuted_claim_count": metrics.get(
                        "refuted_claim_count", 0,
                    ),
                    "refuted_certificate_count": metrics.get(
                        "refuted_certificate_count", 0,
                    ),
                },
                sort_keys=True,
            )
        with print_lock:
            print(
                f"[research-revision] iter {item['iteration']} candidate "
                f"{item['candidate_index'] + 1}/{item['candidate_count']}: "
                f"attempt {revision_index + 1}/{research_budget}"
            )
        revision_draft = item["draft"].with_name(
            f"{item['draft'].name}_research_{revision_index + 1:02d}"
        )
        ok, note = revise_research_candidate(
            current_item["commit_dir"],
            revision_draft,
            evidence_feedback,
            task.editable_files,
            backend=backend,
            model=model,
            cancel_event=cancel_event,
        )
        revision_item = {
            **item,
            "draft": revision_draft,
            "attempt_index": len(results),
            "attempt_kind": "research_revision",
            "failure": None,
            "failure_status": None,
            "repairable": False,
            "repair_note": note,
            "preflight_notes": [],
        }
        if cancel_event is not None and cancel_event.is_set():
            revision_item.update({
                "failure": "research revision cancelled by user interrupt",
                "failure_status": "cancelled",
            })
        elif not ok:
            revision_item.update({
                "failure": note,
                "failure_status": "crash",
            })
        else:
            violations = check_frozen(
                item["parent"]["path"],
                revision_draft,
                task.editable_files,
            )
            issues = _known_solver_issues(
                revision_draft, task.editable_files,
            )
            if violations:
                revision_item.update({
                    "failure": (
                        "research revision modified non-editable file(s): "
                        f"{violations}"
                    ),
                    "failure_status": "violation",
                })
            elif issues:
                feedback = (
                    "Engineering preflight rejected the research revision:\n- "
                    + "\n- ".join(issues)
                )
                revision_item.update({
                    "failure": feedback,
                    "failure_status": "rejected",
                    "repairable": True,
                    "preflight_notes": [feedback],
                })
        try:
            result = _evaluate_candidate(
                revision_item, task, print_lock,
            )
        except Exception as exc:
            result = _controlled_failure_result(
                revision_item,
                task,
                exc,
                cancelled=(
                    cancel_event is not None and cancel_event.is_set()
                ),
            )
        results.append(result)
        current_item = result["item"]

    return results


def _duplicate_of(result, records):
    """Return the first fully equivalent numeric+research record."""
    candidate_hash = result.get("metrics", {}).get("candidate_hash")
    if result["status"] != "ok" or not candidate_hash:
        return None
    return next(
        (record["id"] for record in records
         if record.get("metrics", {}).get("candidate_hash") == candidate_hash),
        None,
    )


def _numeric_duplicate_of(result, records):
    """Return the first record with the same normalized finite set."""
    set_hash = result.get("metrics", {}).get("set_hash")
    if result["status"] != "ok" or not set_hash:
        return None
    return next(
        (
            record["id"]
            for record in records
            if record.get("metrics", {}).get("set_hash") == set_hash
        ),
        None,
    )


def _build_v5_prompt_section(v5_context):
    """Render the V5 Context packet for Context and Proposal prompts."""
    sections = []
    portfolio_text = v5_context.get("portfolio_text", "")
    if portfolio_text:
        sections.append(f"## V5 Portfolio Context\n\n{portfolio_text}")
    analysis_text = v5_context.get("analysis_text", "")
    if analysis_text:
        sections.append(f"## V5 Island Analysis\n\n{analysis_text}")
    return "\n\n".join(sections)


def _clip_v5_packet(text, limit):
    """Keep a packet inside the caller's remaining prompt budget."""
    text = str(text or "")
    limit = max(0, int(limit))
    if len(text) <= limit:
        return text
    if limit <= len(V5_PACKET_TRUNCATION_MARKER):
        return text[:limit]
    return (
        text[: limit - len(V5_PACKET_TRUNCATION_MARKER)].rstrip()
        + V5_PACKET_TRUNCATION_MARKER
    )


def _inject_v5_context(prompt, v5_section, max_total_chars=None):
    """Insert V5 Context material immediately before the assignment marker."""
    if not v5_section:
        return prompt
    if max_total_chars is not None:
        available = max(0, int(max_total_chars) - len(prompt) - 2)
        v5_section = _clip_v5_packet(v5_section, available)
        if not v5_section:
            return prompt
    marker = "## Your assignment"
    insertion = f"{v5_section.rstrip()}\n\n"
    marker_index = prompt.find(marker)
    if marker_index < 0:
        return f"{prompt.rstrip()}\n\n{v5_section.rstrip()}"
    return prompt[:marker_index] + insertion + prompt[marker_index:]


def _inject_v5_proposal(prompt, proposal_text, max_total_chars=None):
    """Insert the V5 Proposal packet immediately before candidate identity."""
    if not proposal_text:
        return prompt
    marker = "## Local candidate identity"
    section_prefix = "## V5 Proposal Context\n\n"
    proposal_text = str(proposal_text)
    if max_total_chars is not None:
        available = max(
            0,
            int(max_total_chars) - len(prompt) - len(section_prefix) - 2,
        )
        proposal_text = _clip_v5_packet(proposal_text, available)
        if not proposal_text:
            return prompt
    section = f"{section_prefix}{proposal_text.rstrip()}"
    marker_index = prompt.find(marker)
    if marker_index < 0:
        return f"{prompt.rstrip()}\n\n{section}"
    return prompt[:marker_index] + f"{section}\n\n" + prompt[marker_index:]


def _build_experiment_plan(
    iteration,
    island_epoch_id,
    direction,
    decision,
    baseline,
    task,
    candidates_per_context,
):
    """Build a deterministic, validated ExperimentPlan for one Context round."""
    plan_fields = {
        "action": decision.action,
        "target_island_epoch_id": island_epoch_id,
        "generation_operator": "local_mutation",
        "parent_ids": [baseline["id"]],
        "inspiration_ids": [],
        "analogy_hypothesis_id": None,
        "implementation_intent": direction,
        "negative_constraints": list(
            getattr(task, "engineering_invariants", []) or []
        ),
        "success_criterion": (
            decision.success_criterion or "trusted_evaluator_score_improves"
        ),
        "budget": {
            "candidate_count": candidates_per_context,
            "sandbox_seconds_per_cell": int(getattr(task, "timeout_s", 0)),
            "max_artifact_bytes": int(
                getattr(task, "max_artifact_bytes", 1024 * 1024)
            ),
        },
    }
    plan_hash = hashlib.sha256(
        json.dumps(
            {
                "iteration": iteration,
                **plan_fields,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    plan = ExperimentPlan(
        id=f"plan_{iteration:04d}_{plan_hash[:12]}",
        **plan_fields,
    )
    plan.validate()
    return plan


def _parent_source_text(parent_path, editable_files, max_chars=48_000):
    """Render the editable parent source that belongs in a ProposalPacket."""
    chunks = []
    for filename in editable_files:
        path = Path(parent_path) / filename
        if not path.is_file():
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.append(f"# FILE: {filename}\n{source}")
    return _clip_v5_packet("\n\n".join(chunks), max_chars)


def _commit_candidate_result(result, task, eb, backend, model, print_lock,
                             parent_id=None, repair_of=None, v5_bridge=None):
    """Commit one candidate outcome without local winner selection."""
    item = result["item"]
    commit_dir = item.get("commit_dir")
    if commit_dir is None:
        raise RuntimeError("candidate has no parent-controlled commit snapshot")
    iteration = item["iteration"]
    parent = item["parent"]
    if not result.get("metrics", {}).get("evidence_sha256"):
        # evidence.json is reserved for trusted evaluator output. A Proposal
        # Agent may not smuggle an evidence file into a failed/unscored record.
        _remove_generated_path(commit_dir / "evidence.json")
    metrics = result.get("metrics", {})
    expected_source_hash = metrics.get("source_snapshot_sha256")
    if expected_source_hash:
        actual_source_hash, _source_files = source_tree_hash(
            commit_dir, _source_limit_bytes(task),
        )
        if actual_source_hash != expected_source_hash:
            raise RuntimeError(
                "commit source no longer matches its sealed snapshot hash"
            )
    if metrics.get("artifact_sha256"):
        _validate_hashed_file(
            commit_dir / "solution.json", metrics["artifact_sha256"],
            label="commit solution.json",
            max_bytes=int(getattr(task, "max_artifact_bytes", 1024 * 1024)),
        )
    if metrics.get("evidence_sha256"):
        _validate_hashed_file(
            commit_dir / "evidence.json", metrics["evidence_sha256"],
            label="commit evidence.json",
            max_bytes=int(getattr(task, "max_output_mb", 64)) * 1024 * 1024,
        )
    metadata = _record_metadata(task, item["context_meta"], backend, model)
    metadata.update({
        "candidate_count": item["candidate_count"],
        "candidate_index": item["candidate_index"],
        "candidate_seed": item["candidate_seed"],
        "duplicate_of": _duplicate_of(result, eb.records()),
        "numeric_duplicate_of": _numeric_duplicate_of(
            result, eb.records(),
        ),
        "attempt_index": item.get("attempt_index", 0),
        "attempt_kind": item.get("attempt_kind", "initial"),
        "repair_of": repair_of,
        "repair_note": item.get("repair_note"),
        "preflight_notes": item.get("preflight_notes", []),
        "editable_file_sha256": _editable_hashes(
            commit_dir, task.editable_files,
        ),
    })

    previous_best = eb.best()
    record = eb.commit(
        commit_dir, result["score"], result["status"],
        item["description"], parent_id or parent["id"], result["log_tail"],
        metrics=result["metrics"], metadata=metadata,
    )
    if v5_bridge is not None:
        v5_island = metadata.get("island_epoch_id", "island_00_epoch_00")
        raw_metrics = result.get("metrics", {})
        adapter = get_metrics_adapter(task.protocol)
        try:
            v5_metrics = adapter(raw_metrics)
        except Exception as exc:
            print(
                f"[v5] warning: metrics adapter failed for {record['id']}: "
                f"{exc!r}",
                file=sys.stderr,
            )
            v5_bridge._log_sync_error(record["id"], "metrics_adapter", exc)
            v5_metrics = raw_metrics
        experiment_plan_id = metadata.get("experiment_plan_id")
        if experiment_plan_id:
            v5_metrics["experiment_plan_id"] = experiment_plan_id
        v5_bridge.on_candidate_evaluated(
            record_id=record["id"],
            island_epoch_id=v5_island,
            score=result["score"],
            status=result["status"],
            description=item["description"],
            parent_ids=[parent_id] if parent_id else [],
            metrics=v5_metrics,
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
                 candidates_per_context=None, stop_policy=None, v5_bridge=None):
    """Run a bounded three-stage asynchronous producer-consumer pipeline."""
    if candidates_per_context is None:
        candidates_per_context = task.candidates_per_context
    if candidates_per_context < MIN_CANDIDATES_PER_CONTEXT:
        raise ValueError(
            f"candidates_per_context must be >= {MIN_CANDIDATES_PER_CONTEXT}"
        )
    stop_policy = stop_policy or StopPolicy()
    stop_controller = StopController(
        stop_policy,
        task.direction,
        required_formal_claims=getattr(
            task, "required_formal_claims", (),
        ),
    )
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
    cancel_event = threading.Event()
    task.cancel_event = cancel_event
    start = _next_context_iteration(eb.records())

    def context_producer():
        try:
            for iteration in range(start, start + iterations):
                acquired = False
                while not cancel_event.is_set():
                    if inflight.acquire(timeout=0.2):
                        acquired = True
                        break
                if not acquired:
                    break
                try:
                    with active_lock:
                        reserved = tuple(active_directions.values())
                    evidence_at_decision = (
                        stopping_evidence(
                            eb.records(), direction=task.direction,
                            policy=stop_policy,
                            required_formal_claims=getattr(
                                task, "required_formal_claims", (),
                            ),
                        )
                        if stop_policy.enabled else None
                    )
                    v5_context = None
                    v5_context_section = ""
                    island_epoch_id = None
                    v5_warnings = []
                    if v5_bridge is not None:
                        try:
                            island_epoch_id = v5_bridge.pick_island(iteration)
                            v5_context = v5_bridge.build_context(island_epoch_id)
                            v5_context_section = _build_v5_prompt_section(
                                v5_context
                            )
                        except Exception as exc:
                            v5_warnings.append({
                                "stage": "build_context",
                                "error": repr(exc),
                            })
                            print(
                                f"[v5] warning: build_context failed at round "
                                f"{iteration}: {exc!r}; continuing with legacy "
                                "context",
                                file=sys.stderr,
                            )
                    decision, baseline, prompt, direction, context_meta = build_inspiration(
                        task, eb, iteration, backend=backend, model=model,
                        active_directions=reserved,
                        trial_seed=trial_seed + iteration,
                        agent_stop_enabled=stop_policy.enabled,
                        stop_evidence=evidence_at_decision,
                        cancel_event=cancel_event,
                        v5_context_prompt=v5_context_section,
                    )
                    if v5_bridge is not None:
                        context_meta["island_epoch_id"] = island_epoch_id
                        context_meta["v5_status"] = (
                            "degraded" if v5_warnings else "ready"
                        )
                        context_meta["v5_warnings"] = v5_warnings
                        if v5_context is not None:
                            context_meta["v5_context_provenance"] = {}
                            for provenance_key, context_key in (
                                ("portfolio", "portfolio_provenance"),
                                ("analysis", "analysis_provenance"),
                            ):
                                provenance = v5_context.get(context_key)
                                if provenance is None or isinstance(provenance, dict):
                                    serialized = provenance
                                else:
                                    serialized = vars(provenance)
                                context_meta["v5_context_provenance"][
                                    provenance_key
                                ] = serialized
                    if cancel_event.is_set():
                        inflight.release()
                        break
                    effective_decision = decision
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
                        effective_decision = decision.forced_continue(
                            direction,
                            "Stop request rejected by deterministic evidence guards.",
                        )
                    try:
                        if v5_bridge is not None and island_epoch_id is not None:
                            prompt = _inject_v5_context(
                                prompt,
                                v5_context_section,
                                max_total_chars=(
                                    MAX_PROPOSAL_PROMPT_CHARS
                                    - PROPOSAL_IDENTITY_RESERVE_CHARS
                                ),
                            )
                            experiment_plan = _build_experiment_plan(
                                iteration,
                                island_epoch_id,
                                direction,
                                effective_decision,
                                baseline,
                                task,
                                candidates_per_context,
                            )
                            v5_bridge.event_store.append_plan_event(
                                experiment_plan
                            )
                            context_meta["experiment_plan_id"] = experiment_plan.id
                            context_meta["experiment_plan"] = experiment_plan.to_dict()
                    except Exception as exc:
                        if v5_bridge is not None:
                            v5_warnings.append({
                                "stage": "experiment_plan",
                                "error": repr(exc),
                            })
                            context_meta["v5_status"] = "degraded"
                            print(
                                f"[v5] warning: experiment plan failed at round "
                                f"{iteration}: {exc!r}; continuing without a "
                                "frozen plan",
                                file=sys.stderr,
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
                proposal_prompt = item["prompt"]
                if (
                    v5_bridge is not None
                    and "experiment_plan" in item["context_meta"]
                ):
                    try:
                        plan = ExperimentPlan.from_dict(
                            item["context_meta"]["experiment_plan"]
                        )
                        proposal_context = v5_bridge.build_proposal_context(
                            plan,
                            _parent_source_text(
                                parent["path"], task.editable_files,
                            ),
                            candidate_seed=item["candidate_seed"],
                        )
                        proposal_prompt = _inject_v5_proposal(
                            proposal_prompt,
                            proposal_context.get("proposal_text", ""),
                            max_total_chars=MAX_PROPOSAL_PROMPT_CHARS,
                        )
                    except Exception as exc:
                        item["context_meta"].setdefault(
                            "v5_warnings", []
                        ).append({
                            "stage": "build_proposal_context",
                            "error": repr(exc),
                        })
                        item["context_meta"]["v5_status"] = "degraded"
                        print(
                            f"[v5] warning: build_proposal_context failed at "
                            f"round {iteration}, candidate {candidate_index}: "
                            f"{exc!r}",
                            file=sys.stderr,
                        )
                ok, description = propose(
                    Path(parent["path"]), draft, proposal_prompt, task.editable_files,
                    backend=backend, model=model, cancel_event=cancel_event,
                )
                failure = None
                failure_status = None
                preflight_notes = []
                if cancel_event.is_set():
                    failure = "proposal cancelled by user interrupt"
                    failure_status = "cancelled"
                elif not ok:
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
                cancelled = cancel_event.is_set()
                candidate_queue.put({
                    **item,
                    "draft": draft,
                    "description": f"proposal worker exception: {exc}",
                    "failure": (
                        "proposal cancelled by user interrupt"
                        if cancelled else repr(exc)
                    ),
                    "failure_status": "cancelled" if cancelled else "crash",
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
                    cancelled = cancel_event.is_set()
                    results = [
                        _controlled_failure_result(
                            item, task, exc, cancelled=cancelled,
                        )
                    ]
                parent_id = item["parent"]["id"]
                repair_of = None
                for result in results:
                    record = _commit_candidate_result(
                        result, task, eb, backend, model, print_lock,
                        parent_id=parent_id, repair_of=repair_of,
                        v5_bridge=v5_bridge,
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
                    try:
                        finalize_analysis(eb, item["iteration"], result_ids)
                        if v5_bridge is not None:
                            try:
                                v5_bridge.on_context_complete(item["iteration"])
                            except Exception as exc:
                                print(
                                    f"[v5] warning: on_context_complete failed "
                                    f"at round {item['iteration']}: {exc!r}",
                                    file=sys.stderr,
                                )
                    except Exception as exc:
                        errors.put((f"finalize iter {item['iteration']}", exc))
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

    interrupted = False

    def request_cancel():
        nonlocal interrupted
        if not cancel_event.is_set():
            with print_lock:
                print("[cancel] interrupt received; cancelling active work and waiting for cleanup ...")
        interrupted = True
        cancel_event.set()

    def join_threads(threads):
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                if not thread.is_alive():
                    continue
                try:
                    thread.join(timeout=0.2)
                except KeyboardInterrupt:
                    request_cancel()

    def put_control(target_queue, item):
        while True:
            try:
                target_queue.put(item, timeout=0.2)
                return
            except queue.Full:
                continue
            except KeyboardInterrupt:
                request_cancel()

    join_threads([producer])
    join_threads(proposals)
    for _ in evaluators:
        put_control(candidate_queue, STOP)
    join_threads(evaluators)
    if cancel_event.is_set() or interrupted:
        return {
            "reason": "user_interrupt",
            "terminal": True,
            "requested_by": "user",
            "accepted_by": "harness",
            "cleanup": "all pipeline threads joined before termination",
        }
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
        or stopping_evidence(
            records,
            direction=task.direction,
            policy=stop_policy,
            required_formal_claims=getattr(
                task, "required_formal_claims", (),
            ),
        )
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
    print(f"[seed] validating {task.seed_description} under {task.protocol} ...")
    score, status, log_tail, metrics = run_solution(task.seed_dir, sandbox, task)
    if status != "ok":
        sys.exit(f"Seed run failed ({status}):\n{log_tail}")
    seed_candidate = _assemble_commit_snapshot(
        task.seed_dir, sandbox, task, metrics,
    )
    if metrics.get("artifact_sha256"):
        _validate_hashed_file(
            seed_candidate / "solution.json", metrics["artifact_sha256"],
            label="seed solution.json",
            max_bytes=int(getattr(task, "max_artifact_bytes", 1024 * 1024)),
        )
    if metrics.get("evidence_sha256"):
        _validate_hashed_file(
            seed_candidate / "evidence.json", metrics["evidence_sha256"],
            label="seed evidence.json",
            max_bytes=int(getattr(task, "max_output_mb", 64)) * 1024 * 1024,
        )
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
        seed_candidate, score, status, task.seed_description,
        None, log_tail, metrics,
        metadata=seed_metadata,
    )
    print(f"[eb] seeded {record['id']}: {task.metric}={score:.12f}")
    return record


def _validate_manifest_for_final_audit(task, manifest_path):
    """Rebuild current provenance using the search settings frozen at init."""
    recorded = load_run_manifest(manifest_path)
    search = recorded.get("search", {})
    required = {
        "backend", "workers", "candidates_per_context", "trial_seed",
    }
    if not required.issubset(search):
        raise RuntimeError("run manifest lacks frozen search settings")
    current = build_run_manifest(
        task, ROOT,
        backend=search["backend"],
        model=search.get("model"),
        workers=search["workers"],
        candidates_per_context=search["candidates_per_context"],
        trial_seed=search["trial_seed"],
        stopping_policy=recorded.get("stopping_policy", {}),
        v5_config=recorded.get("v5", {}),
    )
    return validate_run_manifest(recorded, current)


def _extract_baseline_scores(record: dict) -> dict[str, float]:
    """Extract per-instance baseline scores from a seed record's metrics.

    Returns a dict mapping instance_id → mean baseline score, suitable
    for BehaviorProfiler initialization. Returns empty dict if the
    metrics don't contain the expected summaries.
    """
    metrics = record.get("metrics", {})
    summaries = metrics.get("summaries")
    if not isinstance(summaries, list) or not summaries:
        return {}
    baselines: dict[str, float] = {}
    counts: dict[str, int] = {}
    for s in summaries:
        iid = s.get("instance_id", "")
        baseline_lb = s.get("baseline_lower_bound")
        if iid and baseline_lb is not None:
            baselines[iid] = baselines.get(iid, 0.0) + float(baseline_lb)
            counts[iid] = counts.get(iid, 0) + 1
    for iid, count in counts.items():
        if count > 1:
            baselines[iid] /= count
    return baselines


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
    parser.add_argument(
        "--formal-runner",
        help=(
            "absolute path to a trusted isolated formal-runner executable; "
            "its hash is frozen in run provenance"
        ),
    )
    parser.add_argument("--export-bundle")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--v5", action="store_true",
                        help="enable v5 island scheduling and structured retrieval")
    parser.add_argument(
        "--final-audit", action="store_true",
        help="freeze search Top-K and run the configured one-shot private audit",
    )
    args = parser.parse_args()
    if (args.iterations < 0 or args.workers < 1 or
            (args.candidates_per_context is not None and
             args.candidates_per_context < MIN_CANDIDATES_PER_CONTEXT)):
        parser.error(
            "--iterations must be >= 0; --workers must be >= 1; "
            f"--candidates-per-context must be >= {MIN_CANDIDATES_PER_CONTEXT}"
        )
    if args.final_audit and (args.init or args.iterations or args.status):
        parser.error(
            "--final-audit is a separate one-shot action and cannot be "
            "combined with --init, --iterations, or --status"
        )

    task = Task(args.task, args.run_id)
    if args.formal_runner:
        if not callable(task.verify_formalization):
            parser.error(
                f"task {task.name!r} has no configured formal proof gate"
            )
        try:
            (
                task.formal_runner,
                task.formal_runner_identity,
            ) = build_external_formal_runner(args.formal_runner)
        except ValueError as exc:
            parser.error(str(exc))
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
    v5_bridge = None
    v5_config = {}
    if getattr(args, 'v5', False):
        if V5Bridge is None:
            sys.exit("--v5 requires harness_v5 module")
        v5_bridge = V5Bridge(task.run_dir)
        v5_config = {
            "enabled": True,
            "num_islands": 4,
        }
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
        if v5_bridge is not None:
            diagnostics = v5_bridge.get_island_diagnostics()
            event_count = len(
                v5_bridge.event_store.read_experiment_events()
            )
            plan_count = len(v5_bridge.event_store.read_plan_events())
            print(
                "v5: "
                f"sync={diagnostics['sync_status']} "
                f"events={event_count} plans={plan_count} "
                f"active_islands={diagnostics['active_islands']} "
                f"profiles={diagnostics['profiles_cached']} "
                f"unresolved_sync_errors={diagnostics['unresolved_sync_errors']}"
            )
            print(
                "v5 island sizes: "
                + ", ".join(
                    f"{epoch}={size}"
                    for epoch, size in sorted(
                        diagnostics["island_sizes"].items()
                    )
                )
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
                v5_config=v5_config,
            )
            write_run_manifest(manifest_path, task.run_manifest)
            task.search_evaluation_request = task.run_manifest.get(
                "search", {}
            ).get("evaluation_request")
            init_seed(task, eb)
            if v5_bridge is not None:
                seed = eb.records()[0]
                seed_metrics_adapter = get_metrics_adapter(task.protocol)
                seed_v5_metrics = seed_metrics_adapter(
                    seed.get("metrics", {})
                )
                v5_bridge.record_seed(
                    record_id=seed["id"],
                    score=seed["score"],
                    metrics=seed_v5_metrics,
                )
                seed_baselines = _extract_baseline_scores(seed)
                v5_bridge.initialize(
                    [seed["id"]],
                    frozen_baseline_score=seed["score"],
                    base_proposal_seed=args.trial_seed,
                    baseline_scores=seed_baselines or None,
                    probe_suite_sha256=seed.get("metrics", {}).get(
                        "evaluation_request_sha256", ""
                    ),
                )
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
                v5_config=v5_config,
            )
            task.run_manifest = validate_run_manifest(recorded, current)
            task.search_evaluation_request = task.run_manifest.get(
                "search", {}
            ).get("evaluation_request")
            if v5_bridge is not None:
                seeds = [r["id"] for r in eb.records() if r.get("metadata", {}).get("iteration") is None]
                seed_record = eb.records()[0]
                seed_metrics_adapter = get_metrics_adapter(task.protocol)
                seed_v5_metrics = seed_metrics_adapter(
                    seed_record.get("metrics", {})
                )
                v5_bridge.record_seed(
                    record_id=seed_record["id"],
                    score=seed_record["score"],
                    metrics=seed_v5_metrics,
                )
                seed_baselines = _extract_baseline_scores(seed_record)
                v5_bridge.initialize(
                    seeds or [seed_record["id"]],
                    frozen_baseline_score=seed_record["score"],
                    base_proposal_seed=args.trial_seed,
                    baseline_scores=seed_baselines or None,
                    probe_suite_sha256=seed_record.get("metrics", {}).get(
                        "evaluation_request_sha256", ""
                    ),
                )
                reconciliation = v5_bridge._reconcile(
                    legacy_record_ids=[r["id"] for r in eb.records()],
                )
                if reconciliation.get("missing_in_v5"):
                    print(
                        f"[v5] warning: {len(reconciliation['missing_in_v5'])} "
                        "legacy records missing from V5 events",
                        file=sys.stderr,
                    )

        if args.iterations:
            ensure_run_resumable(task, eb)
            try:
                outcome = run_pipeline(
                    task, eb, args.iterations, args.workers,
                    args.backend, args.model, args.trial_seed,
                    candidates_per_context=candidates_per_context,
                    stop_policy=stop_policy,
                    v5_bridge=v5_bridge,
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
            if v5_bridge is not None:
                v5_bridge.save_state()
            write_termination(
                task.run_dir / "termination.json",
                _termination_payload(
                    task, eb, stop_policy, outcome, args.iterations,
                ),
            )
        if args.final_audit:
            if not eb.records():
                sys.exit("Experience Bank is empty; use --init first")
            task.run_manifest = _validate_manifest_for_final_audit(
                task, manifest_path,
            )
            report = run_final_audit(task, eb, task.run_manifest)
            winner = report.get("winner")
            if winner:
                print(
                    f"[audit] complete: winner={winner['id']} "
                    f"score={winner['score']}"
                )
            else:
                print("[audit] failed: no successful private candidate")
        if args.export_bundle:
            if v5_bridge is not None and v5_bridge.sync_status == "degraded":
                sys.exit(
                    "[v5] export refused: V5 sync status is 'degraded' — "
                    "check v5/sync_errors.jsonl for failed event recordings. "
                    "Resolve gaps before exporting."
                )
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
