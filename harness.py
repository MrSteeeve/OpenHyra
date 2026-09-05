"""OpenHyra: Context producer -> Proposal workers -> Evaluator workers."""

import argparse
import ast
import hashlib
import importlib.util
import inspect
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
from feedback import BeliefReducer, FeedbackPacket, ProblemStateLog, render_feedback_context
from llm_backend import SUPPORTED_BACKENDS
from matched_control import ControlPair, MatchedControlBuilder
from mechanism_hypotheses import (
    candidate_hypotheses,
    hypothesis_to_analogy,
    load_mechanism_design,
    mechanism_generation_operator,
    matched_control_enabled,
)
from intervention_router import AcquisitionRouter, PendingHypothesisQueue
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
CANDIDATE_MODES = frozenset({"legacy", "algorithm_bundle", "python_program"})
PROGRAM_CANDIDATE_MODES = frozenset({"algorithm_bundle", "python_program"})
ALGORITHM_BUNDLE_SCHEMA = "openhyra-algorithm-bundle.v1"
DEFAULT_ALGORITHM_SOURCE_FILES = ("train.py", "manifest.json")
DEFAULT_PYTHON_PROGRAM_SOURCE_FILES = ("algorithm.py", "manifest.json")


def _is_program_candidate(task):
    return getattr(task, "candidate_mode", "legacy") in PROGRAM_CANDIDATE_MODES


def _safe_relative_source_files(value, *, label):
    """Normalize a task-owned candidate file allowlist.

    The list is configuration, not candidate input.  Keeping this validation
    in the harness ensures every copy/freeze path uses the same lexical
    interpretation and cannot be widened by a submitted manifest.
    """
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list of relative paths")
    result = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or "\x00" in item
            or Path(item).is_absolute()
            or ".." in Path(item).parts
            or item in result
        ):
            raise ValueError(f"{label} contains an unsafe or duplicate path")
        # Candidate bundle paths are POSIX names even on the host.  Reject
        # backslashes so a Windows-looking path cannot bypass a later join.
        if "\\" in item:
            raise ValueError(f"{label} contains an unsafe path separator")
        result.append(item)
    return tuple(result)


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
        candidate_cfg = cfg.get("candidate", {})
        if candidate_cfg is None:
            candidate_cfg = {}
        if not isinstance(candidate_cfg, dict):
            sys.exit("task candidate must be an object")
        # ``candidate_mode`` is intentionally optional.  Existing tasks keep
        # the historical solve.sh/solution.json path byte-for-byte; new tasks
        # can opt into either a data-artifact bundle or a complete Python
        # program explicitly.
        candidate_mode = cfg.get(
            "candidate_mode", candidate_cfg.get("mode", "legacy"),
        )
        if candidate_mode not in CANDIDATE_MODES:
            sys.exit(
                "task candidate_mode must be one of: "
                + ", ".join(sorted(CANDIDATE_MODES))
            )
        self.candidate_mode = candidate_mode
        # Open Python candidates may submit a bounded source tree.  The
        # manifest remains deliberately minimal; this task-owned switch keeps
        # the tree policy out of candidate input and lets legacy bundles retain
        # their exact two-file contract.
        source_tree = cfg.get("source_tree", candidate_cfg.get("source_tree", False))
        if not isinstance(source_tree, bool):
            sys.exit("task source_tree must be boolean")
        self.candidate_source_tree = source_tree

        configured_editable = cfg.get("editable_files")
        if configured_editable is None and candidate_mode in PROGRAM_CANDIDATE_MODES:
            default_sources = (
                DEFAULT_ALGORITHM_SOURCE_FILES
                if candidate_mode == "algorithm_bundle"
                else DEFAULT_PYTHON_PROGRAM_SOURCE_FILES
            )
            configured_editable = candidate_cfg.get(
                "editable_files", list(default_sources),
            )
        if configured_editable is None:
            # Preserve the old error shape for malformed legacy task specs.
            sys.exit("task editable_files is required")
        try:
            self.editable_files = list(_safe_relative_source_files(
                configured_editable, label="task editable_files",
            ))
        except ValueError as exc:
            sys.exit(str(exc))

        configured_sources = cfg.get(
            "source_files",
            candidate_cfg.get("source_files", configured_editable),
        )
        try:
            self.candidate_source_files = _safe_relative_source_files(
                configured_sources, label="task source_files",
            )
        except ValueError as exc:
            sys.exit(str(exc))
        if not set(self.editable_files).issubset(self.candidate_source_files):
            sys.exit("task editable_files must be a subset of task source_files")
        if (
            candidate_mode == "algorithm_bundle"
            and set(self.candidate_source_files)
            != set(DEFAULT_ALGORITHM_SOURCE_FILES)
        ):
            sys.exit(
                "algorithm_bundle v1 source_files must be exactly train.py "
                "and manifest.json"
            )
        if (
            candidate_mode == "python_program"
            and "manifest.json" not in self.candidate_source_files
        ):
            sys.exit("python_program source_files must include manifest.json")
        if candidate_mode == "python_program" and self.candidate_source_tree:
            # The explicit entrypoint and manifest are required anchors; other
            # source files are admitted by the recursive policy in sandbox and
            # the trusted evaluator.
            if "algorithm.py" not in self.candidate_source_files:
                sys.exit("python_program source_tree must include algorithm.py")
        self.candidate_entrypoint = cfg.get(
            "entrypoint",
            candidate_cfg.get(
                "entrypoint",
                (
                    "train.py" if candidate_mode == "algorithm_bundle"
                    else "algorithm.py" if candidate_mode == "python_program"
                    else "solve.sh"
                ),
            ),
        )
        if (
            not isinstance(self.candidate_entrypoint, str)
            or not self.candidate_entrypoint
            or Path(self.candidate_entrypoint).is_absolute()
            or ".." in Path(self.candidate_entrypoint).parts
            or "\\" in self.candidate_entrypoint
        ):
            sys.exit("task entrypoint must be a safe relative path")
        if (
            candidate_mode == "algorithm_bundle"
            and self.candidate_entrypoint != "train.py"
        ):
            sys.exit(
                "algorithm_bundle v1 currently requires entrypoint train.py"
            )
        if (
            candidate_mode == "python_program"
            and self.candidate_entrypoint not in self.candidate_source_files
        ):
            sys.exit("python_program entrypoint must be included in source_files")
        self.solve_entrypoint = cfg.get(
            "solve_entrypoint",
            candidate_cfg.get("solve_entrypoint", "solve.sh"),
        )
        if (
            not isinstance(self.solve_entrypoint, str)
            or not self.solve_entrypoint
            or Path(self.solve_entrypoint).is_absolute()
            or ".." in Path(self.solve_entrypoint).parts
            or "\\" in self.solve_entrypoint
        ):
            sys.exit("task solve_entrypoint must be a safe relative path")
        self.artifact_protocol = cfg.get(
            "artifact_protocol",
            candidate_cfg.get(
                "artifact_protocol",
                (
                    "openhyra-policy-spec.v1"
                    if candidate_mode == "algorithm_bundle"
                    else "openhyra-python-program.v1"
                    if candidate_mode == "python_program"
                    else self.protocol
                ),
            ),
        )
        if (
            not isinstance(self.artifact_protocol, str)
            or not self.artifact_protocol.strip()
            or len(self.artifact_protocol) > 256
        ):
            sys.exit("task artifact_protocol must be bounded non-empty text")
        configured_protocols = cfg.get(
            "artifact_protocols",
            candidate_cfg.get("artifact_protocols", [self.artifact_protocol]),
        )
        if (
            not isinstance(configured_protocols, (list, tuple))
            or not configured_protocols
            or any(
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 256
                for value in configured_protocols
            )
        ):
            sys.exit("task artifact_protocols must be bounded non-empty text")
        self.artifact_protocols = tuple(dict.fromkeys(
            value.strip() for value in configured_protocols
        ))
        if self.artifact_protocol not in self.artifact_protocols:
            sys.exit("task artifact_protocol must be in artifact_protocols")
        self.bundle_subdir = cfg.get(
            "bundle_subdir", candidate_cfg.get("bundle_subdir", "."),
        )
        if (
            not isinstance(self.bundle_subdir, str)
            or not self.bundle_subdir
            or Path(self.bundle_subdir).is_absolute()
            or ".." in Path(self.bundle_subdir).parts
            or "\\" in self.bundle_subdir
        ):
            sys.exit("task bundle_subdir must be a safe relative path")
        self.timeout_s = cfg.get("sandbox_timeout_s", 660)
        self.eval_concurrency = cfg.get("eval_concurrency", 1)
        self.candidates_per_context = cfg.get(
            "candidates_per_context", MIN_CANDIDATES_PER_CONTEXT,
        )
        self.candidate_repair_attempts = cfg.get("candidate_repair_attempts", 0)
        self.research_revision_attempts = cfg.get(
            "research_revision_attempts", 0,
        )
        # Optional open-ended algorithm-design surface.  The generic harness
        # remains unchanged for legacy tasks; AlgorithmBundle tasks can expose
        # a small portfolio of mechanism hypotheses and paired control arms.
        # The design object is task configuration, not candidate data.
        self.mechanism_design = load_mechanism_design(self)
        self.mechanism_directions = tuple(
            item.to_dict() for item in self.mechanism_design.directions
        )
        self.matched_control_enabled = matched_control_enabled(self)
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
        # Adaptive feedback is opt-in per task.  It serializes Context rounds
        # while preserving proposal/evaluator concurrency inside a round.
        # Accept a few explicit aliases so task authors can migrate without a
        # schema flag explosion; unknown values fall back to the legacy loop.
        feedback_mode = cfg.get(
            "feedback_mode",
            cfg.get("context_feedback_mode", "static"),
        )
        self.feedback_mode = (
            feedback_mode.strip().lower()
            if isinstance(feedback_mode, str) else "static"
        )
        self.adaptive_feedback = bool(
            cfg.get("adaptive_feedback", False)
            or self.feedback_mode in {"adaptive", "directional", "closed_loop"}
        )
        self.context_barrier = bool(
            cfg.get("context_barrier", self.adaptive_feedback)
        )
        if self.adaptive_feedback:
            self.context_barrier = True
        configured_phase_semantics = cfg.get("phase_semantics", {})
        self.phase_semantics = (
            dict(configured_phase_semantics)
            if isinstance(configured_phase_semantics, dict) else {}
        )
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


def check_frozen(parent_dir, draft_dir, editable, *, allow_source_tree=False):
    before, after = solution_files(parent_dir), solution_files(draft_dir)
    source_extensions = {".py", ".json", ".toml"}

    def is_source_tree_file(relative):
        path = Path(relative)
        # solve.sh and other task plumbing remain parent-owned even when a
        # Python candidate is allowed to add helper modules/configuration.
        return (
            path.name != "solve.sh"
            and not any(part.startswith(".") for part in path.parts)
            and path.suffix in source_extensions
        )

    return [
        relative for relative in sorted(set(before) | set(after))
        if (
            relative not in editable
            and not (allow_source_tree and is_source_tree_file(relative))
            and before.get(relative) != after.get(relative)
        )
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
    failed_metrics = {"source_snapshot_sha256": source_hash}
    # A controlled failure has no usable bundle, but retaining the configured
    # mode lets downstream provenance distinguish it from a legacy solver
    # crash without inventing a digest.
    if _is_program_candidate(task):
        failed_metrics["candidate_mode"] = task.candidate_mode
    return {
        "item": failed_item,
        "score": None,
        "status": status,
        "log_tail": note,
        "metrics": failed_metrics,
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


def _candidate_preflight_issues(task, draft_dir):
    """Keep legacy solver hints out of the open Python program space."""
    if getattr(task, "candidate_mode", "legacy") == "python_program":
        return []
    return _known_solver_issues(draft_dir, task.editable_files)


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


def _v5_metrics_input(task, raw_metrics):
    """Add trusted AlgorithmBundle identity fields before V5 adaptation.

    The legacy evaluator historically reported ``protocol`` as its task
    protocol, while V5 mechanism cards use that field for the artifact wire
    protocol.  Keep the old path untouched and, for Python candidates, derive
    the card identity from evaluator output or the task-owned default.  The
    returned copy prevents V5 bookkeeping from mutating the Experience Bank
    metrics.
    """
    payload = dict(raw_metrics or {})
    if not _is_program_candidate(task):
        return payload
    artifact_protocol = payload.get("artifact_protocol") or getattr(
        task, "artifact_protocol", ""
    )
    if artifact_protocol:
        payload["artifact_protocol"] = artifact_protocol
        # MechanismCardBuilder consumes ``protocol`` as the artifact protocol.
        # This is a trusted projection, not a candidate-supplied score field.
        payload["protocol"] = artifact_protocol
    payload.setdefault(
        "entrypoint", getattr(task, "candidate_entrypoint", "train.py")
    )
    payload.setdefault("candidate_mode", task.candidate_mode)
    return payload


def _annotate_feedback_lineage(metrics, item):
    """Attach harness-owned mechanism/arm ids to an evaluator feedback packet.

    Domain feedback is generated by the evaluator before the harness knows the
    Context slot id.  Copying the packet here links its observations to the
    typed hypothesis without changing any measured value or score.
    """
    if not isinstance(metrics, dict):
        return metrics
    packet = metrics.get("feedback_packet")
    mechanism = item.get("mechanism") if isinstance(item, dict) else None
    if not isinstance(packet, dict) or not isinstance(mechanism, dict):
        return metrics
    mechanism_id = mechanism.get("id") or item.get("mechanism_id")
    if not mechanism_id:
        return metrics
    payload = dict(packet)
    source_mechanism_id = payload.get("mechanism_id", "")
    payload["mechanism_id"] = str(mechanism_id)
    lineage = {
        "mechanism_id": str(mechanism_id),
        "source_mechanism_id": source_mechanism_id,
        "hypothesis_id": item.get("hypothesis_id", ""),
        "matched_pair_id": item.get("matched_pair_id", ""),
        "matched_arm": item.get("matched_arm", ""),
    }
    directional = []
    for raw in packet.get("directional", []) if isinstance(packet.get("directional"), list) else []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row["mechanism_id"] = str(mechanism_id)
        directional.append(row)
    if directional:
        payload["directional"] = directional
    updated = dict(metrics)
    updated["feedback_packet"] = payload
    updated["feedback_packet_mechanism_id"] = str(mechanism_id)
    # Keep harness lineage beside (rather than inside) the strict packet
    # schema; FeedbackPacket.from_dict must remain able to replay the packet.
    updated["feedback_lineage"] = lineage
    return updated


def _editable_hashes(directory, editable_files):
    hashes = {}
    for name in editable_files:
        path = Path(directory) / name
        hashes[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() else None
        )
    return hashes


def _algorithm_bundle_digest(directory, task):
    """Return the canonical digest for a configured AlgorithmBundle.

    The digest intentionally covers only the task-owned ``source_files``
    allowlist (normally ``train.py`` and ``manifest.json``), not harness
    outputs such as ``solve.sh`` or ``solution.json``.  This matches the V5
    AlgorithmBundle provenance schema and gives EB/V5 a stable code identity
    independent of generated artifacts.
    """
    if getattr(task, "candidate_mode", "legacy") != "algorithm_bundle":
        return None
    source_files = tuple(getattr(task, "candidate_source_files", ()))
    if not source_files:
        raise ValueError("algorithm bundle has no configured source_files")
    files = []
    root = Path(directory)
    for name in sorted(source_files):
        path = root / name
        # ``read_regular_file`` rejects symlinks and hard-link surprises and
        # applies the same source-size budget as the sealing path.
        data = read_regular_file(
            path,
            _source_limit_bytes(task),
            label=f"algorithm source file {name}",
        )
        files.append({
            "path": name,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    payload = {
        "schema": ALGORITHM_BUNDLE_SCHEMA,
        "files": files,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return digest


def _validate_algorithm_bundle_source(directory, task):
    """Validate required bundle files before launching candidate/evaluator."""
    if getattr(task, "candidate_mode", "legacy") != "algorithm_bundle":
        return None
    digest = _algorithm_bundle_digest(directory, task)
    entrypoint = getattr(task, "candidate_entrypoint", "train.py")
    if entrypoint not in getattr(task, "candidate_source_files", ()):
        raise ValueError(
            "algorithm entrypoint must be included in task source_files"
        )
    return digest


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


def _mechanism_hypothesis_id(iteration, parent_id, mechanism_id, pair_index=0):
    """Build a stable id for one mechanism/parent comparison.

    The id is harness-owned metadata.  It lets V5 and the matched-control
    report join the two arms without trusting a candidate-authored label.
    """
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(mechanism_id or "mechanism"))
    slug = slug.strip("-")[:28] or "mechanism"
    digest = hashlib.sha256(
        f"{iteration}|{parent_id}|{mechanism_id}|{pair_index}".encode()
    ).hexdigest()[:10]
    return f"analogy_{int(iteration):04d}_{slug}_{digest}"


def _mechanism_slots(task, context_meta, iteration, baseline, candidate_count):
    """Return harness-owned mechanism/arm metadata for candidate slots.

    A configured portfolio plus Context-generated hypotheses is an open
    starting space: the Context can append a new mechanism id, while the
    deterministic slot mapping ensures every proposal has a reproducible
    lineage.  Paired mode uses two slots per mechanism (guided/control) with a
    shared seed and parent; legacy tasks return empty metadata and retain the
    historical seed sequence.
    """
    # `selected_mechanism_candidates` is produced by the deterministic
    # acquisition router.  Fall back to the complete Context portfolio for
    # legacy metadata and custom callers.
    context_hypotheses = context_meta.get(
        "selected_mechanism_candidates",
        context_meta.get("mechanism_candidates", ()),
    )
    try:
        pair_enabled = bool(matched_control_enabled(task))
    except Exception:
        pair_enabled = bool(getattr(task, "matched_control_enabled", False))
    if not pair_enabled and not context_hypotheses:
        # A task may still have a configured portfolio without controls.
        try:
            has_design = bool(load_mechanism_design(task).active)
        except Exception:
            has_design = False
        if not has_design:
            return [{} for _ in range(candidate_count)]

    pair_count = (candidate_count + 1) // 2 if pair_enabled else candidate_count
    selected = candidate_hypotheses(
        task,
        context_hypotheses=context_hypotheses,
        candidate_count=pair_count,
        iteration=iteration,
    )
    if not selected:
        return [{} for _ in range(candidate_count)]

    trial_seed = int(context_meta.get("trial_seed", 0))
    parent_id = str(baseline.get("id", ""))
    slots = []
    for candidate_index in range(candidate_count):
        if pair_enabled:
            pair_index = candidate_index // 2
            unpaired_tail = (
                candidate_count % 2 == 1
                and candidate_index == candidate_count - 1
            )
            arm = "guided" if candidate_index % 2 == 0 else "control"
            mechanism = dict(selected[min(pair_index, len(selected) - 1)])
            pair_seed = _candidate_seed(trial_seed, pair_index)
            pair_id = (
                f"pair_{int(iteration):04d}_{pair_index:02d}_"
                f"{mechanism.get('id', 'mechanism')}"
            ) if not unpaired_tail else ""
        else:
            pair_index = candidate_index
            arm = "guided"
            mechanism = dict(selected[min(candidate_index, len(selected) - 1)])
            pair_seed = _candidate_seed(trial_seed, candidate_index)
            pair_id = ""
        mechanism_id = mechanism.get("id", "mechanism")
        mechanism["hypothesis_id"] = _mechanism_hypothesis_id(
            iteration, parent_id, mechanism_id, pair_index,
        )
        slots.append({
            "mechanism": mechanism,
            "mechanism_id": mechanism_id,
            "hypothesis_id": mechanism["hypothesis_id"],
            "matched_pair_id": pair_id,
            "matched_arm": arm,
            "matched_seed": pair_seed,
            "matched_control_enabled": pair_enabled and bool(pair_id),
        })
    return slots


def _candidate_prompt(
    prompt, candidate_index, candidate_count, seed, *, slot=None,
):
    prompt = prompt.replace(CANDIDATE_SEED_TOKEN, str(seed))
    slot = slot or {}
    candidate_prompt = prompt + f"""

## Local candidate identity

This is candidate {candidate_index + 1} of {candidate_count} generated from the
same Context briefing. Produce your own concrete implementation/parameterization;
all {candidate_count} candidates are evaluated independently, and every outcome
is committed to the Experience Bank, including failures and low scores.
"""
    mechanism = slot.get("mechanism") if isinstance(slot, dict) else None
    if isinstance(mechanism, dict) and mechanism.get("id"):
        candidate_prompt += f"""

## Mechanism slot

Mechanism id: `{mechanism.get('id')}`
Family: {mechanism.get('family', 'general')}
Intervention scope: {mechanism.get('intervention_scope', mechanism.get('scope', 'mechanism'))}
Intervention operator: {mechanism.get('intervention_operator', mechanism.get('operator', 'replace'))}
Target slice: {mechanism.get('target_slice', ', '.join(mechanism.get('target_slices', []) if isinstance(mechanism.get('target_slices'), list) else []))}
Proposed mechanism: {mechanism.get('mechanism', '')}
Prediction: {mechanism.get('prediction', '')}
Falsifier: {mechanism.get('failure_condition', '')}
Matched control: {mechanism.get('matched_control', '')}
Next probe: {mechanism.get('next_probe', '')}
Evidence ids: {', '.join(str(value) for value in mechanism.get('evidence_ids', []) if value)}
"""
        arm = slot.get("matched_arm", "guided")
        pair_id = slot.get("matched_pair_id")
        if arm == "control" and pair_id:
            candidate_prompt += f"""
This is the CONTROL arm for matched pair `{pair_id}`. Keep the same parent,
candidate seed, data boundary, and compute budget, but remove or neutralize the
focal mechanism. Make the control executable and describe it in PROPOSAL.md;
do not claim a transfer result before evaluation.
"""
        else:
            candidate_prompt += """
This is the GUIDED arm. Implement or materially instantiate the proposed
mechanism, while keeping the artifact protocol and evaluator-owned boundary
unchanged. You may synthesize a genuinely new structure when the evidence
supports it; record the falsifiable prediction and control in PROPOSAL.md.
"""
    if len(candidate_prompt) > MAX_PROPOSAL_PROMPT_CHARS:
        raise ValueError("candidate Proposal prompt exceeds character limit")
    return candidate_prompt


def _final_candidate_record(records):
    """Choose the scored terminal attempt for one proposal slot."""
    if not records:
        return None
    scored = [
        record for record in records
        if record.get("status") in {"ok", "early_stopped"}
        and record.get("score") is not None
    ]
    return scored[-1] if scored else records[-1]


def _append_jsonl_once(path, payload, key="pair_id"):
    """Append one small research result, keeping reruns idempotent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identifier = payload.get(key)
    if identifier and path.is_file():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(existing, dict) and existing.get(key) == identifier:
                    return False
        except OSError:
            pass
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
    return True


def _finalize_matched_controls(
    task, iteration, completed_items, eb, v5_bridge=None,
):
    """Score guided/control pairs after both proposals have been evaluated.

    The evaluator remains the sole source of scores.  This function only
    contrasts two already-committed EB records that share a parent, seed, and
    mechanism slot, then writes the compact result for V5/reports.
    """
    groups = {}
    for entry in completed_items or ():
        item = entry.get("item", {})
        pair_id = item.get("matched_pair_id")
        arm = item.get("matched_arm")
        if not pair_id or arm not in {"guided", "control"}:
            continue
        record = _final_candidate_record(entry.get("records", []))
        if record is None:
            continue
        groups.setdefault(pair_id, {})[arm] = {
            "item": item,
            "record": record,
        }
    if not groups:
        return []

    all_records = {record["id"]: record for record in eb.records()}
    results = []
    for pair_id, arms in sorted(groups.items()):
        guided = arms.get("guided")
        control = arms.get("control")
        if guided is None or control is None:
            continue
        guided_item = guided["item"]
        control_item = control["item"]
        mechanism = guided_item.get("mechanism") or control_item.get("mechanism")
        if not isinstance(mechanism, dict):
            continue
        parent_id = str(
            guided_item.get("parent", {}).get("id")
            or control_item.get("parent", {}).get("id")
            or ""
        )
        if not parent_id:
            continue
        hypothesis_id = guided_item.get("hypothesis_id") or control_item.get(
            "hypothesis_id"
        )
        try:
            hypothesis = hypothesis_to_analogy(
                mechanism,
                target_parent_id=parent_id,
                source_record_ids=[parent_id],
                analogy_id=hypothesis_id,
                metric=getattr(task, "metric", "paired_lower_bound_lcb"),
                matched_control={
                    "enabled": True,
                    "description": mechanism.get("matched_control", ""),
                    "same_parent": True,
                    "same_seed": True,
                    "same_compute_budget": True,
                },
            )
            pair = MatchedControlBuilder.build_pair(
                hypothesis,
                parent_id=parent_id,
                seed=int(
                    guided_item.get("matched_seed", guided_item.get("candidate_seed", 0))
                ),
            )
            pair.guided_record_id = guided["record"]["id"]
            pair.control_record_id = control["record"]["id"]
            pair.guided_score = guided["record"].get("score")
            pair.control_score = control["record"].get("score")
            pair.baseline_score = all_records.get(parent_id, {}).get("score")
            pair.control_metadata = dict(hypothesis.matched_control)
            pair.guided_parent_id = str(
                guided_item.get("parent", {}).get("id") or parent_id
            )
            pair.control_parent_id = str(
                control_item.get("parent", {}).get("id") or parent_id
            )
            if guided_item.get("matched_seed") is not None:
                pair.guided_seed = int(guided_item["matched_seed"])
            if control_item.get("matched_seed") is not None:
                pair.control_seed = int(control_item["matched_seed"])
            # The evaluator summaries are already keyed by instance/repeat;
            # retain their paired deltas so the reported SE/CI is computed
            # across cells instead of from a relative scalar ratio.
            MatchedControlBuilder.attach_per_cell_summaries(
                pair,
                guided["record"].get("metrics"),
                control["record"].get("metrics"),
            )
            # Match the evaluator-owned training-data budget when available.
            # Wall-clock time remains a reported outcome rather than a target:
            # different algorithm families may legitimately finish earlier.
            guided_budget = guided["record"].get("metrics", {}).get(
                "total_training_path_budget"
            )
            control_budget = control["record"].get("metrics", {}).get(
                "total_training_path_budget"
            )
            if isinstance(guided_budget, (int, float)) and not isinstance(
                guided_budget, bool
            ):
                pair.guided_compute_budget = float(guided_budget)
            if isinstance(control_budget, (int, float)) and not isinstance(
                control_budget, bool
            ):
                pair.control_compute_budget = float(control_budget)
            # Keep the pair's shared seed and parent explicit even when a
            # candidate omitted optional metadata in a legacy replay.
            pair.shared_parent_id = parent_id
            pair.shared_seed = int(
                guided_item.get("matched_seed", guided_item.get("candidate_seed", 0))
            )
            analogy_result = MatchedControlBuilder.evaluate_pair(
                pair, direction=getattr(task, "direction", "max")
            )
        except (TypeError, ValueError, KeyError) as exc:
            # A malformed optional mechanism annotation should not discard the
            # already committed candidate records.  Keep the round useful and
            # surface the omission in the run log.
            print(
                f"[matched-control] pair {pair_id} could not be summarized: {exc!r}",
                file=sys.stderr,
            )
            continue

        payload = {
            "schema": "openhyra-matched-control.v1",
            "pair_id": pair_id,
            "iteration": iteration,
            "mechanism_id": mechanism.get("id", ""),
            "hypothesis": hypothesis.to_dict(),
            "pair": pair.to_dict(),
            "result": analogy_result.to_dict(),
        }
        _append_jsonl_once(
            Path(getattr(task, "run_dir", eb.root.parent))
            / "research" / "matched_controls.jsonl",
            payload,
        )
        if v5_bridge is not None:
            try:
                if hasattr(v5_bridge, "record_hypothesis"):
                    v5_bridge.record_hypothesis(hypothesis)
                if hasattr(v5_bridge, "record_analogy_result"):
                    v5_bridge.record_analogy_result(analogy_result, hypothesis)
            except Exception as exc:
                # Candidate/EB commits remain valid if optional V5 indexing is
                # unavailable; the bridge records its own warning/diagnostic.
                print(
                    f"[v5] warning: matched-control pair {pair_id} sync failed: {exc!r}",
                    file=sys.stderr,
                )
        results.append(payload)
    return results


def _update_acquisition_from_round(
    router, task, iteration, completed_items, matched_results=()
):
    """Fold trusted round outcomes into the pending hypothesis queue.

    This is intentionally a thin bridge: the evaluator owns scores and the
    queue owns only hypothesis lifecycle.  Paired results take precedence over
    a raw parent comparison because they isolate the focal intervention.
    """
    if router is None:
        return
    paired_by_mechanism = {}
    for payload in matched_results or ():
        if not isinstance(payload, dict):
            continue
        mechanism_id = payload.get("mechanism_id")
        result = payload.get("result", {})
        if mechanism_id and isinstance(result, dict):
            paired_by_mechanism[str(mechanism_id)] = result
    direction = getattr(task, "direction", "max")
    for entry in completed_items or ():
        item = entry.get("item", {}) if isinstance(entry, dict) else {}
        mechanism = item.get("mechanism")
        if not isinstance(mechanism, dict):
            continue
        mechanism_id = mechanism.get("id")
        if not mechanism_id:
            continue
        # Controls are not evidence for the focal mechanism.  A paired result
        # is finalized once for the guided arm only.
        if item.get("matched_arm") == "control":
            continue
        result = paired_by_mechanism.get(str(mechanism_id))
        result_id = None
        improved = None
        reason = "round completed without a decisive comparison"
        if result is not None:
            result_id = result.get("guided_record_id")
            verdict = result.get("verdict")
            if verdict == "transfer_supported":
                improved = True
                reason = "matched-control transfer supported"
            elif verdict in {"transfer_refuted", "execution_failed", "invalid_control"}:
                improved = False
                reason = f"matched-control verdict: {verdict}"
            else:
                reason = f"matched-control verdict: {verdict or 'inconclusive'}"
        else:
            records = entry.get("records", []) if isinstance(entry, dict) else []
            final = _final_candidate_record(records)
            parent_score = item.get("parent", {}).get("score")
            if final is not None and final.get("score") is not None and parent_score is not None:
                if direction == "min":
                    improved = float(final["score"]) < float(parent_score)
                else:
                    improved = float(final["score"]) > float(parent_score)
                result_id = final.get("id")
                reason = "guided candidate improved against its parent" if improved else "guided candidate did not improve its parent"
        router.observe_result(
            str(mechanism_id), improved=improved, result_id=result_id,
            iteration=iteration, reason=reason,
        )


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
    bundle_digest = None
    bundle_error = None
    if getattr(task, "candidate_mode", "legacy") == "algorithm_bundle":
        try:
            bundle_digest = _validate_algorithm_bundle_source(sealed, task)
        except (OSError, ValueError) as exc:
            bundle_error = str(exc)

    if item.get("failure"):
        score, status, log_tail, metrics = (
            None,
            item["failure_status"],
            item["failure"],
            {
                "source_snapshot_sha256": source_snapshot_sha256,
                **({"algorithm_bundle_sha256": bundle_digest}
                   if bundle_digest else {}),
            },
        )
    elif bundle_error:
        score, status, log_tail, metrics = (
            None,
            "rejected",
            f"algorithm bundle preflight failed: {bundle_error}",
            {"source_snapshot_sha256": source_snapshot_sha256},
        )
    else:
        violations = check_frozen(
            item["parent"]["path"], sealed, task.editable_files,
            allow_source_tree=bool(getattr(task, "candidate_source_tree", False)),
        )
        issues = _candidate_preflight_issues(task, sealed)
        if violations:
            score, status, log_tail, metrics = (
                None,
                "violation",
                f"sealed proposal modified non-editable file(s): {violations}",
                {
                    "source_snapshot_sha256": source_snapshot_sha256,
                    **({"algorithm_bundle_sha256": bundle_digest}
                       if bundle_digest else {}),
                },
            )
        elif issues:
            score, status, log_tail, metrics = (
                None,
                "rejected",
                "sealed proposal failed engineering preflight: "
                + "; ".join(issues),
                {
                    "source_snapshot_sha256": source_snapshot_sha256,
                    **({"algorithm_bundle_sha256": bundle_digest}
                       if bundle_digest else {}),
                },
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
            if bundle_digest:
                metrics = dict(metrics)
                metrics.setdefault(
                    "algorithm_bundle_sha256", bundle_digest,
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
            candidate_mode=getattr(task, "candidate_mode", "legacy"),
            entrypoint=getattr(task, "candidate_entrypoint", None),
            artifact_protocol=getattr(task, "artifact_protocol", None),
            source_files=getattr(task, "candidate_source_files", None),
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
                allow_source_tree=bool(getattr(task, "candidate_source_tree", False)),
            )
            issues = _candidate_preflight_issues(task, repair_draft)
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
            candidate_mode=getattr(task, "candidate_mode", "legacy"),
            entrypoint=getattr(task, "candidate_entrypoint", None),
            artifact_protocol=getattr(task, "artifact_protocol", None),
            source_files=getattr(task, "candidate_source_files", None),
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
                allow_source_tree=bool(getattr(task, "candidate_source_tree", False)),
            )
            issues = _candidate_preflight_issues(task, revision_draft)
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
    feedback_text = v5_context.get("feedback_text", "")
    if feedback_text:
        sections.append(
            "## Evaluator Directional Feedback (public)\n\n"
            + str(feedback_text)
        )
    state = v5_context.get("problem_state_dict")
    if isinstance(state, dict):
        # Keep the proposal prompt small; the Context call receives the full
        # structured state separately and can use this compact version for
        # immediate intervention routing.
        compact = {
            "state_version": state.get("state_version"),
            "cells": state.get("cells", {}),
        }
        try:
            state_text = json.dumps(
                compact, ensure_ascii=False, sort_keys=True,
            )
        except (TypeError, ValueError):
            state_text = ""
        if state_text:
            sections.append("## Problem State (public)\n\n" + state_text[:12_000])
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


def _mechanism_generation_operator(mechanism):
    """Map a proposed mechanism to the V5 generation operator vocabulary.

    This is deliberately a small, deterministic vocabulary bridge.  It keeps
    the existing ExperimentPlan schema useful while letting each candidate
    carry the operator implied by its own mechanism slot rather than inheriting
    the Context round's primary plan label.
    """
    return mechanism_generation_operator(mechanism)


def _secondary_program_parent(
    eb, primary, direction, *, entrypoint="algorithm.py",
):
    """Choose a distinct, behaviorally complementary executable parent."""
    alternatives = []
    primary_id = str(primary.get("id", ""))
    primary_metrics = primary.get("metrics", {})
    primary_candidate_hash = primary_metrics.get("candidate_hash")
    primary_rates = primary_metrics.get("per_instance_exercise_rates", {})
    if not isinstance(primary_rates, dict):
        primary_rates = {}
    try:
        primary_interface = json.loads(
            (Path(primary.get("path", "")) / "manifest.json").read_text()
        ).get("interface")
    except (OSError, ValueError, AttributeError):
        primary_interface = None
    for record in eb.records():
        score = record.get("score")
        path = Path(record.get("path", ""))
        metadata = record.get("metadata", {})
        metrics = record.get("metrics", {})
        if (
            record.get("id") == primary_id
            or record.get("status") != "ok"
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not path.is_dir()
            or not (path / entrypoint).is_file()
            or (isinstance(metadata, dict) and bool(metadata.get("duplicate_of")))
            or (
                isinstance(primary_candidate_hash, str)
                and primary_candidate_hash
                and metrics.get("candidate_hash") == primary_candidate_hash
            )
        ):
            continue
        try:
            interface = json.loads(
                (path / "manifest.json").read_text()
            ).get("interface")
        except (OSError, ValueError, AttributeError):
            continue
        if interface != primary_interface:
            continue
        rates = metrics.get("per_instance_exercise_rates", {})
        common = (
            sorted(set(primary_rates).intersection(rates))
            if isinstance(rates, dict) else []
        )
        behavior_distance = (
            sum(
                abs(float(primary_rates[key]) - float(rates[key]))
                for key in common
            ) / len(common)
            if common else 0.0
        )
        alternatives.append((behavior_distance, record))
    alternatives.sort(
        key=lambda item: (
            -item[0],
            -float(item[1]["score"])
            if direction == "max" else float(item[1]["score"]),
            str(item[1].get("id", "")),
        )
    )
    return alternatives[0][1] if alternatives else None


def _build_experiment_plan(
    iteration,
    island_epoch_id,
    direction,
    decision,
    baseline,
    task,
    candidates_per_context,
    mechanism=None,
    mechanism_hypotheses=None,
):
    """Build a deterministic, validated ExperimentPlan for one Context round."""
    generation_operator = _mechanism_generation_operator(mechanism)
    analogy_hypothesis_id = None
    inspiration_ids = []
    implementation_intent = direction
    if isinstance(mechanism, dict) and mechanism.get("id"):
        analogy_hypothesis_id = mechanism.get("hypothesis_id") or mechanism.get("id")
        implementation_intent = (
            f"{direction}; mechanism={mechanism.get('id')}: "
            f"{mechanism.get('mechanism', '')}"
        ).strip()
        for item in mechanism_hypotheses or ():
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_record_id") or item.get("source_id")
            if isinstance(source_id, str) and source_id:
                inspiration_ids.append(source_id)
        if not inspiration_ids and baseline.get("id"):
            inspiration_ids.append(baseline["id"])
        inspiration_ids = list(dict.fromkeys(inspiration_ids))
    plan_fields = {
        "action": decision.action,
        "target_island_epoch_id": island_epoch_id,
        "generation_operator": generation_operator,
        "parent_ids": [baseline["id"]],
        "inspiration_ids": inspiration_ids,
        "analogy_hypothesis_id": analogy_hypothesis_id,
        "implementation_intent": implementation_intent,
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
    # Feedback-aware schemas add typed intervention fields as keyword-only
    # extensions.  Construct them opportunistically so this bridge remains
    # compatible with archived v1 schemas during a rolling upgrade.
    typed_source = mechanism if isinstance(mechanism, dict) else {}
    typed_fields = {
        "phase": getattr(decision, "phase", None),
        "intervention_scope": typed_source.get(
            "intervention_scope", typed_source.get("scope")
        ) or getattr(decision, "intervention_scope", None),
        "intervention_operator": typed_source.get(
            "intervention_operator", typed_source.get("operator")
        ) or getattr(decision, "intervention_operator", None),
        "target_slice": typed_source.get("target_slice") or getattr(
            decision, "target_slice", None
        ),
        "prediction": typed_source.get("prediction") or getattr(
            decision, "prediction", None
        ),
        "falsifier": typed_source.get("failure_condition") or typed_source.get(
            "falsifier"
        ) or getattr(decision, "falsifier", None),
        "evidence_ids": typed_source.get("evidence_ids") or list(
            getattr(decision, "evidence_ids", ()) or ()
        ),
        "next_probe": typed_source.get("next_probe") or getattr(
            decision, "next_probe", None
        ),
        "state_version": typed_source.get("state_version") or getattr(
            decision, "state_version", None
        ),
        "state_hash": typed_source.get("state_hash") or getattr(
            decision, "state_hash", None
        ),
    }
    plan_hash = hashlib.sha256(
        json.dumps(
            {
                "iteration": iteration,
                **plan_fields,
                "typed_intervention": typed_fields,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    plan_kwargs = {
        "id": f"plan_{iteration:04d}_{plan_hash[:12]}",
        **plan_fields,
    }
    try:
        accepted_fields = set(inspect.signature(ExperimentPlan).parameters)
    except (TypeError, ValueError):
        accepted_fields = set()
    for key, value in typed_fields.items():
        if key in accepted_fields and value not in (None, "", []):
            plan_kwargs[key] = value
    plan = ExperimentPlan(**plan_kwargs)
    plan.validate()
    return plan


def _parent_source_text(parent_path, editable_files, max_chars=48_000,
                        source_files=None):
    """Render parent code for a ProposalPacket using the task allowlist.

    Algorithm bundles normally expose the same files as ``editable_files``;
    ``source_files`` can additionally include frozen helper modules when a
    protocol explicitly permits them.  Legacy callers retain their original
    positional behavior.
    """
    chunks = []
    filenames = source_files if source_files is not None else editable_files
    for filename in filenames:
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
    metrics = _annotate_feedback_lineage(
        result.get("metrics", {}), item
    )
    # Keep the enriched sidecar on the committed result object as well as the
    # local variable so EB, V5, and round reducers see the same packet.
    result["metrics"] = metrics
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
    lineage_parent_ids = [parent_id or parent["id"]]
    secondary_parent = item.get("secondary_parent")
    if (
        isinstance(secondary_parent, dict)
        and secondary_parent.get("id")
        and secondary_parent["id"] not in lineage_parent_ids
    ):
        lineage_parent_ids.append(secondary_parent["id"])
    metadata["parent_ids"] = lineage_parent_ids
    metadata["lineage_semantics"] = "proposal_ancestry"
    # Mechanism/arm labels are assigned by the Harness from task + Context
    # configuration.  They are useful lineage metadata for the EB and are not
    # inferred from candidate-authored prose.
    for key in (
        "mechanism_id", "hypothesis_id", "matched_pair_id",
        "matched_arm", "matched_seed", "matched_control_enabled",
    ):
        if key in item:
            metadata[key] = item[key]
    slot_plans = item.get("context_meta", {}).get("experiment_plans", {})
    slot_hypothesis_id = item.get("hypothesis_id")
    if (
        isinstance(slot_plans, dict)
        and slot_hypothesis_id in slot_plans
        and isinstance(slot_plans[slot_hypothesis_id], dict)
    ):
        # Keep the EB/V5 event tied to the same slot-specific plan that the
        # Proposal packet received.  The primary round plan remains available
        # for legacy callers, but must not mask a distinct mechanism.
        slot_plan = dict(slot_plans[slot_hypothesis_id])
        metadata["experiment_plan_id"] = slot_plan.get(
            "id", metadata.get("experiment_plan_id", "")
        )
        metadata["experiment_plan"] = slot_plan
    mechanism = item.get("mechanism")
    if isinstance(mechanism, dict):
        metadata["mechanism"] = dict(mechanism)
        # A round has one primary ExperimentPlan, but each Proposal slot can
        # represent a different mechanism family.  Preserve that per-candidate
        # operator in the existing EB/V5 lineage fields so non-primary ideas
        # remain distinguishable in later Context retrieval.
        metadata["generation_operator"] = _mechanism_generation_operator(
            mechanism
        )
    if _is_program_candidate(task):
        # The evaluator validates the sealed source manifest and reports its
        # schema (MLP, linear, or expression).  Preserve that actual protocol
        # in EB metadata; the task default is only a fallback for preflight or
        # failed candidates that never reached the evaluator.
        actual_artifact_protocol = metrics.get("artifact_protocol") or (
            task.artifact_protocol
        )
        metadata.update({
            "candidate_mode": task.candidate_mode,
            "entrypoint": task.candidate_entrypoint,
            "solve_entrypoint": task.solve_entrypoint,
            "artifact_protocol": actual_artifact_protocol,
            "source_files": list(task.candidate_source_files),
        })
        if task.candidate_mode == "algorithm_bundle":
            metadata["algorithm_bundle_sha256"] = metrics.get(
                "algorithm_bundle_sha256"
            ) or ""

    previous_best = eb.best()
    record = eb.commit(
        commit_dir, result["score"], result["status"],
        item["description"], parent_id or parent["id"], result["log_tail"],
        metrics=result["metrics"], metadata=metadata,
    )
    if v5_bridge is not None:
        v5_island = metadata.get("island_epoch_id", "island_00_epoch_00")
        raw_metrics = _v5_metrics_input(task, result.get("metrics", {}))
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
        for key in (
            "generation_operator", "inspiration_ids",
            "mechanism_id", "hypothesis_id", "matched_pair_id",
            "matched_arm", "matched_seed", "matched_control_enabled",
        ):
            if key in metadata:
                v5_metrics[key] = metadata[key]
        v5_bridge.on_candidate_evaluated(
            record_id=record["id"],
            island_epoch_id=v5_island,
            score=result["score"],
            status=result["status"],
            description=item["description"],
            parent_ids=list(metadata.get("parent_ids", ())),
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
    context_window = (
        1
        if stop_policy.enabled
        or bool(getattr(task, "context_barrier", False))
        or bool(getattr(task, "adaptive_feedback", False))
        else max(1, workers + task.eval_concurrency)
    )
    inflight = threading.Semaphore(context_window)
    active_directions = {}
    active_lock = threading.Lock()
    print_lock = threading.Lock()
    errors = queue.Queue()
    termination_request = {}
    cancel_event = threading.Event()
    task.cancel_event = cancel_event
    start = _next_context_iteration(eb.records())
    adaptive_feedback = bool(
        getattr(task, "adaptive_feedback", False)
        or getattr(task, "feedback_mode", "") in {"adaptive", "directional", "closed_loop"}
    )
    # Keep the queue scoped to this run.  Legacy tasks do not instantiate it,
    # preserving their exact Context/Proposal behavior and prompt footprint.
    pending_queue = None
    acquisition_router = None
    configured_design = getattr(task, "mechanism_design", None)
    if isinstance(configured_design, dict):
        mechanism_design_active = bool(
            configured_design.get("enabled", True)
            and configured_design.get("directions")
        )
    else:
        mechanism_design_active = bool(
            getattr(configured_design, "active", False)
        )
    if adaptive_feedback or mechanism_design_active:
        pending_queue = PendingHypothesisQueue(
            Path(task.run_dir) / "v5" / "pending_hypotheses.json"
        )
        acquisition_router = AcquisitionRouter(pending_queue)

    # The feedback loop is useful even when a caller deliberately leaves V5
    # disabled.  Keep a tiny local packet log/state in that case so the next
    # Context still receives evaluator-owned directional evidence; V5 runs
    # continue to use their richer bridge as the single writer.
    local_feedback_log = None
    local_feedback_reducer = None
    local_feedback_state = None
    local_feedback_packet_ids: set[str] = set()
    local_feedback_lock = threading.RLock()
    if adaptive_feedback and v5_bridge is None:
        try:
            local_feedback_log = ProblemStateLog(
                Path(task.run_dir) / "v5" / "feedback_packets.jsonl"
            )
            local_feedback_reducer = BeliefReducer()
            existing_packets = local_feedback_log.read()
            local_feedback_packet_ids = {
                packet.packet_id for packet in existing_packets
            }
            public_packets = [
                packet for packet in existing_packets
                if packet.data.get("split") != "private"
            ]
            local_feedback_state = local_feedback_reducer.rebuild(
                public_packets, state_id="problem-state"
            )
        except (OSError, ValueError, TypeError):
            # A missing/corrupt optional sidecar should not prevent the legacy
            # candidate loop from running.  The current round can rebuild it.
            local_feedback_log = ProblemStateLog(
                Path(task.run_dir) / "v5" / "feedback_packets.jsonl"
            )
            local_feedback_reducer = BeliefReducer()
            local_feedback_state = local_feedback_reducer.rebuild(
                [], state_id="problem-state"
            )

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
                    target_record_ids = []
                    feedback_state = None
                    v5_warnings = []
                    if v5_bridge is not None:
                        try:
                            island_epoch_id = v5_bridge.pick_island(iteration)
                            try:
                                target_record_ids = list(
                                    v5_bridge.island_scheduler.get_members(
                                        island_epoch_id
                                    )
                                )
                            except (AttributeError, KeyError, RuntimeError):
                                target_record_ids = []
                            # Feedback/state implementations are deliberately
                            # duck-typed so V5 can resume with older bridges.
                            for state_method in (
                                "get_problem_state",
                                "get_feedback_state",
                                "problem_state",
                            ):
                                getter = getattr(v5_bridge, state_method, None)
                                if getter is None:
                                    continue
                                try:
                                    feedback_state = getter() if callable(getter) else getter
                                except Exception:
                                    feedback_state = None
                                if feedback_state is not None:
                                    break
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
                    elif local_feedback_log is not None:
                        # Snapshot the local state and its compact text under
                        # one lock.  The evaluator updates both only at the
                        # completed-round barrier below.
                        with local_feedback_lock:
                            feedback_state = local_feedback_state
                            try:
                                local_packets = local_feedback_log.read()
                                local_feedback_text = render_feedback_context(
                                    local_packets
                                )
                            except (OSError, ValueError, TypeError):
                                local_feedback_text = ""
                        if local_feedback_text:
                            v5_context_section = (
                                "## Evaluator Directional Feedback (public)\n\n"
                                + local_feedback_text
                            )
                    decision, baseline, prompt, direction, context_meta = build_inspiration(
                        task, eb, iteration, backend=backend, model=model,
                        active_directions=reserved,
                        trial_seed=trial_seed + iteration,
                        agent_stop_enabled=stop_policy.enabled,
                        stop_evidence=evidence_at_decision,
                        cancel_event=cancel_event,
                        v5_context_prompt=v5_context_section,
                        target_island_epoch_id=island_epoch_id,
                        target_record_ids=target_record_ids,
                        feedback_state=feedback_state,
                        pending_queue=pending_queue,
                        acquisition_router=acquisition_router,
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
                    # Freeze the round's open mechanism portfolio before any
                    # Proposal worker starts.  The mapping is deterministic,
                    # while the Context-generated list may contain genuinely
                    # new structures beyond the task's seed directions.
                    mechanism_slots = _mechanism_slots(
                        task,
                        context_meta,
                        iteration,
                        baseline,
                        candidates_per_context,
                    )
                    context_meta["matched_control_enabled"] = bool(
                        any(slot.get("matched_pair_id") for slot in mechanism_slots)
                    )
                    context_meta["mechanism_slots"] = [
                        {
                            key: value for key, value in slot.items()
                            if key != "mechanism"
                        }
                        | ({"mechanism": dict(slot["mechanism"])} if isinstance(slot.get("mechanism"), dict) else {})
                        for slot in mechanism_slots
                    ]
                    mechanism_hypotheses = []
                    seen_hypothesis_ids = set()
                    for slot in mechanism_slots:
                        mechanism = slot.get("mechanism")
                        hypothesis_id = slot.get("hypothesis_id")
                        if not isinstance(mechanism, dict) or not hypothesis_id:
                            continue
                        if hypothesis_id in seen_hypothesis_ids:
                            continue
                        seen_hypothesis_ids.add(hypothesis_id)
                        mechanism_hypotheses.append({
                            **dict(mechanism),
                            "hypothesis_id": hypothesis_id,
                            "source_record_id": baseline.get("id", ""),
                        })
                    # Preserve the rest of the Context portfolio as
                    # preregistered ideas even when this round has fewer
                    # execution slots than proposed mechanisms.  The slots
                    # remain the only items that produce candidates; keeping
                    # the unselected ideas here lets V5 and the next Context
                    # round see the full creative branch rather than silently
                    # collapsing it to the first pair.
                    for portfolio_index, raw_mechanism in enumerate(
                        context_meta.get("mechanism_candidates", [])
                    ):
                        if not isinstance(raw_mechanism, dict):
                            continue
                        mechanism = dict(raw_mechanism)
                        mechanism_id = mechanism.get("id")
                        if not mechanism_id:
                            continue
                        if any(
                            item.get("id") == mechanism_id
                            for item in mechanism_hypotheses
                        ):
                            continue
                        hypothesis_id = _mechanism_hypothesis_id(
                            iteration,
                            baseline.get("id", ""),
                            mechanism_id,
                            100 + portfolio_index,
                        )
                        mechanism["hypothesis_id"] = hypothesis_id
                        mechanism_hypotheses.append({
                            **mechanism,
                            "source_record_id": baseline.get("id", ""),
                        })
                    context_meta["mechanism_portfolio_count"] = len(
                        mechanism_hypotheses
                    )
                    context_meta["mechanism_hypotheses"] = mechanism_hypotheses
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
                            for item in mechanism_hypotheses:
                                try:
                                    hypothesis = hypothesis_to_analogy(
                                        item,
                                        target_parent_id=baseline["id"],
                                        source_record_ids=[baseline["id"]],
                                        analogy_id=item.get("hypothesis_id"),
                                        metric=getattr(
                                            task, "metric", "paired_lower_bound_lcb"
                                        ),
                                    )
                                    if hasattr(v5_bridge, "record_hypothesis"):
                                        v5_bridge.record_hypothesis(hypothesis)
                                except (TypeError, ValueError, KeyError) as exc:
                                    v5_warnings.append({
                                        "stage": "mechanism_hypothesis",
                                        "error": repr(exc),
                                    })
                            # Build one V5 plan projection per distinct
                            # mechanism slot.  The first remains the round's
                            # primary plan for compatibility, while Proposal
                            # workers use the matching projection below.
                            experiment_plans = {}
                            appended_plan_ids = set()
                            primary_plan = None
                            for slot in mechanism_slots:
                                mechanism = slot.get("mechanism")
                                hypothesis_id = slot.get("hypothesis_id")
                                if hypothesis_id in experiment_plans:
                                    continue
                                slot_plan = _build_experiment_plan(
                                    iteration,
                                    island_epoch_id,
                                    direction,
                                    effective_decision,
                                    baseline,
                                    task,
                                    candidates_per_context,
                                    mechanism=(
                                        mechanism
                                        if isinstance(mechanism, dict)
                                        else None
                                    ),
                                    mechanism_hypotheses=mechanism_hypotheses,
                                )
                                if primary_plan is None:
                                    primary_plan = slot_plan
                                if slot_plan.id not in appended_plan_ids:
                                    v5_bridge.event_store.append_plan_event(
                                        slot_plan
                                    )
                                    appended_plan_ids.add(slot_plan.id)
                                if hypothesis_id:
                                    experiment_plans[hypothesis_id] = (
                                        slot_plan.to_dict()
                                    )
                            if primary_plan is None:
                                slot_plan = _build_experiment_plan(
                                    iteration,
                                    island_epoch_id,
                                    direction,
                                    effective_decision,
                                    baseline,
                                    task,
                                    candidates_per_context,
                                    mechanism=None,
                                    mechanism_hypotheses=mechanism_hypotheses,
                                )
                                primary_plan = slot_plan
                            if primary_plan.id not in appended_plan_ids:
                                v5_bridge.event_store.append_plan_event(primary_plan)
                            experiment_plan = primary_plan
                            context_meta["experiment_plan_id"] = experiment_plan.id
                            context_meta["experiment_plan"] = experiment_plan.to_dict()
                            context_meta["experiment_plans"] = experiment_plans
                            context_meta["generation_operator"] = (
                                experiment_plan.generation_operator
                            )
                            context_meta["inspiration_ids"] = list(
                                experiment_plan.inspiration_ids
                            )
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
                            f"baseline={baseline['id']}, "
                            f"mechanisms={context_meta.get('mechanism_portfolio_count', 0)}, "
                            f"next={short}"
                        )
                    secondary_program_parent = (
                        _secondary_program_parent(
                            eb,
                            baseline,
                            task.direction,
                            entrypoint=getattr(
                                task, "candidate_entrypoint", "algorithm.py"
                            ),
                        )
                        if getattr(task, "candidate_mode", "legacy") == "python_program"
                        else None
                    )
                    for candidate_index in range(candidates_per_context):
                        slot = (
                            mechanism_slots[candidate_index]
                            if candidate_index < len(mechanism_slots) else {}
                        )
                        seed = int(slot.get("matched_seed")) if slot.get(
                            "matched_seed"
                        ) is not None else _candidate_seed(
                            context_meta["trial_seed"], candidate_index
                        )
                        mechanism = slot.get("mechanism")
                        wants_crossover = (
                            isinstance(mechanism, dict)
                            and mechanism.get(
                                "intervention_operator",
                                mechanism.get("operator"),
                            ) == "ast_crossover"
                            and slot.get("matched_arm") != "control"
                        )
                        inspiration_queue.put({
                            "iteration": iteration,
                            "parent": baseline,
                            "prompt": _candidate_prompt(
                                prompt,
                                candidate_index,
                                candidates_per_context,
                                seed,
                                slot=slot,
                            ),
                            # Each candidate gets its own shallow context copy;
                            # Proposal/V5 warnings from one worker must not
                            # mutate the metadata seen by its siblings.
                            "context_meta": {
                                **context_meta,
                                "v5_warnings": list(
                                    context_meta.get("v5_warnings", [])
                                ),
                                "mechanism_candidates": [
                                    dict(candidate)
                                    for candidate in context_meta.get(
                                        "mechanism_candidates", []
                                    )
                                    if isinstance(candidate, dict)
                                ],
                            },
                            "candidate_index": candidate_index,
                            "candidate_count": candidates_per_context,
                            "candidate_seed": seed,
                            **(
                                {"secondary_parent": secondary_program_parent}
                                if wants_crossover
                                and secondary_program_parent is not None
                                else {}
                            ),
                            **slot,
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
                        plan_payload = item["context_meta"]["experiment_plan"]
                        # Each mechanism slot gets its own plan projection.
                        # The round's primary plan remains as a legacy fallback,
                        # but a non-primary Proposal must not receive a
                        # contradictory hypothesis/operator in its V5 packet.
                        slot_plans = item["context_meta"].get(
                            "experiment_plans", {}
                        )
                        slot_hypothesis_id = item.get("hypothesis_id")
                        if (
                            isinstance(slot_plans, dict)
                            and slot_hypothesis_id in slot_plans
                        ):
                            plan_payload = slot_plans[slot_hypothesis_id]
                        plan = ExperimentPlan.from_dict(
                            plan_payload
                        )
                        proposal_context = v5_bridge.build_proposal_context(
                            plan,
                            _parent_source_text(
                                parent["path"], task.editable_files,
                                source_files=getattr(
                                    task, "candidate_source_files", None,
                                ),
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
                    candidate_mode=getattr(task, "candidate_mode", "legacy"),
                    entrypoint=getattr(task, "candidate_entrypoint", None),
                    artifact_protocol=getattr(task, "artifact_protocol", None),
                    source_files=getattr(task, "candidate_source_files", None),
                    allow_no_change=(
                        _is_program_candidate(task)
                        and item.get("matched_arm") == "control"
                        and bool(item.get("matched_control_enabled"))
                    ),
                    intervention=(
                        {
                            **item["mechanism"],
                            "matched_arm": item.get("matched_arm", ""),
                            "slot": candidate_index,
                            **(
                                {
                                    "secondary_parent_id": item[
                                        "secondary_parent"
                                    ].get("id", ""),
                                    "secondary_parent_path": item[
                                        "secondary_parent"
                                    ].get("path", ""),
                                }
                                if isinstance(
                                    item.get("secondary_parent"), dict
                                )
                                else {}
                            ),
                        }
                        if isinstance(item.get("mechanism"), dict)
                        else item.get("context_meta", {}).get(
                            "context_decision", {}
                        ).get("intervention")
                    ),
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
                    violations = check_frozen(
                        parent["path"], draft, task.editable_files,
                        allow_source_tree=bool(
                            getattr(task, "candidate_source_tree", False)
                        ),
                    )
                    if violations:
                        failure = f"modified non-editable file(s): {violations}"
                        failure_status = "violation"
                if failure is None:
                    issues = _candidate_preflight_issues(task, draft)
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
        nonlocal local_feedback_state
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
                        item["iteration"],
                        {"finished": 0, "result_ids": [], "items": []},
                    )
                    state["finished"] += 1
                    state["result_ids"].extend(record["id"] for record in records)
                    state["items"].append({"item": item, "records": list(records)})
                    if state["finished"] == item["candidate_count"]:
                        result_ids = list(state["result_ids"])
                        completed_items = list(state["items"])
                        context_completions.pop(item["iteration"])
                        context_finished = True
                if context_finished:
                    try:
                        finalize_analysis(eb, item["iteration"], result_ids)
                        matched_results = _finalize_matched_controls(
                            task,
                            item["iteration"],
                            completed_items,
                            eb,
                            v5_bridge=v5_bridge,
                        )
                        _update_acquisition_from_round(
                            acquisition_router,
                            task,
                            item["iteration"],
                            completed_items,
                            matched_results,
                        )
                        feedback_packets = []
                        for completed in completed_items:
                            for committed in completed.get("records", []):
                                packet = committed.get("metrics", {}).get(
                                    "feedback_packet"
                                )
                                if isinstance(packet, dict):
                                    feedback_packets.append(packet)
                        # Optional feedback reducer hook.  It is called only
                        # after the complete round (including matched arms),
                        # which gives the next Context a coherent state
                        # version and preserves the adaptive barrier.
                        if v5_bridge is not None:
                            state_updater = getattr(
                                v5_bridge, "update_problem_state", None
                            ) or getattr(v5_bridge, "append_feedback", None)
                            if callable(state_updater):
                                try:
                                    state_updater(
                                        iteration=item["iteration"],
                                        result_ids=result_ids,
                                        matched_results=matched_results,
                                        records=eb.records(),
                                        feedback_packets=feedback_packets,
                                    )
                                except TypeError:
                                    # Compatibility with a reducer accepting a
                                    # single payload positional argument.
                                    state_updater({
                                        "iteration": item["iteration"],
                                        "result_ids": result_ids,
                                        "matched_results": matched_results,
                                        "feedback_packets": feedback_packets,
                                    })
                        elif local_feedback_log is not None and local_feedback_reducer is not None:
                            # Non-V5 adaptive runs use the same packet schema
                            # and reducer, but keep their state local to this
                            # pipeline.  Only public evaluator packets enter
                            # the Context-facing belief state.
                            with local_feedback_lock:
                                for raw_packet in feedback_packets:
                                    try:
                                        packet = FeedbackPacket.from_dict(raw_packet)
                                        packet.validate()
                                    except (TypeError, ValueError, KeyError):
                                        continue
                                    if packet.data.get("split") == "private":
                                        continue
                                    if packet.packet_id in local_feedback_packet_ids:
                                        continue
                                    local_feedback_log.append(packet)
                                    local_feedback_packet_ids.add(packet.packet_id)
                                try:
                                    local_feedback_state = local_feedback_reducer.rebuild(
                                        [
                                            packet for packet in local_feedback_log.read()
                                            if packet.data.get("split") != "private"
                                        ],
                                        state_id="problem-state",
                                    )
                                except (OSError, ValueError, TypeError):
                                    # Keep the last coherent snapshot if an
                                    # optional sidecar is torn during shutdown.
                                    pass
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
    if getattr(task, "candidate_mode", "legacy") == "algorithm_bundle":
        try:
            metrics = {
                **metrics,
                "algorithm_bundle_sha256": _validate_algorithm_bundle_source(
                    seed_candidate, task,
                ),
            }
        except (OSError, ValueError) as exc:
            sys.exit(f"Seed algorithm bundle is invalid: {exc}")
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
    if _is_program_candidate(task):
        actual_artifact_protocol = metrics.get("artifact_protocol") or (
            task.artifact_protocol
        )
        seed_metadata.update({
            "candidate_mode": task.candidate_mode,
            "entrypoint": task.candidate_entrypoint,
            "solve_entrypoint": task.solve_entrypoint,
            "artifact_protocol": actual_artifact_protocol,
            "source_files": list(task.candidate_source_files),
        })
        if task.candidate_mode == "algorithm_bundle":
            seed_metadata["algorithm_bundle_sha256"] = metrics.get(
                "algorithm_bundle_sha256"
            ) or ""
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
    direct = metrics.get("baseline_scores")
    if isinstance(direct, dict) and direct:
        try:
            return {
                str(instance_id): float(value)
                for instance_id, value in direct.items()
                if isinstance(instance_id, str) and isinstance(value, (int, float))
            }
        except (TypeError, ValueError):
            pass
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
    parser.add_argument("--task", default="bermudan_optimal_stopping")
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
        allowed_operators = None
        try:
            if load_mechanism_design(task).active:
                # Keep the legacy operators and expose the small set used by
                # the open portfolio.  These are labels for retrieval, not a
                # second execution or security layer.
                allowed_operators = [
                    "feature_augment", "residualize", "local_mutation",
                    "ablation", "composition", "analogy_transfer",
                    "restart_from_skeleton",
                ]
        except Exception:
            allowed_operators = None
        v5_bridge = V5Bridge(
            task.run_dir,
            allowed_operators=allowed_operators,
            direction=task.direction,
        )
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
                    _v5_metrics_input(task, seed.get("metrics", {}))
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
                    direction=task.direction,
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
                    _v5_metrics_input(task, seed_record.get("metrics", {}))
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
                    direction=task.direction,
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
