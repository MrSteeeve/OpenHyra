"""Small, task-owned vocabulary for open algorithm proposals.

The research loop deliberately keeps this module lightweight.  A task may
declare a portfolio of mechanism directions in ``task.json``; Context uses
the declarations to ask for several hypotheses and Proposal uses them to
produce diverse, executable candidates.  The module does not execute a
candidate or score a result -- those responsibilities stay with the task
evaluator.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


CONFIG_SCHEMA = "openhyra-mechanism-design.v1"
HYPOTHESIS_SCHEMA = "openhyra-algorithm-hypothesis.v1"
MAX_DIRECTIONS = 16
MAX_CONTEXT_HYPOTHESES = 8
MAX_TEXT_CHARS = 480
MAX_FAMILY_CHARS = 80
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _text(value: Any, *, limit: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str):
        return ""
    value = " ".join(value.split()).strip()
    return value[:limit].rstrip()


def _id(value: Any) -> str:
    value = _text(value, limit=64)
    return value if _ID_RE.fullmatch(value) else ""


def _derived_id(*values: Any) -> str:
    """Give an agent idea without a machine id a stable readable identity."""
    parts = [_text(value, limit=160) for value in values]
    basis = "|".join(part for part in parts if part)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", basis.lower()).strip("-")
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:10]
    if not slug:
        slug = "mechanism"
    return f"{slug[:53].rstrip('-')}-{digest}"


def _context_operator(family: str, mechanism: str, scope: str) -> str:
    """Infer an executable operator when Context omitted one.

    Context packets are the only place where omission is repaired.  Task
    portfolios keep their historical aliases (for example ``replace``), but
    a generated open-program hypothesis must carry one of the four operators
    that the Harness can execute without interpreting prose.
    """
    family_text = f"{family} {mechanism}".lower()
    scope_text = scope.lower()
    if scope_text in {"fit", "predict", "subsystem"}:
        return "subsystem_rewrite"
    if any(token in family_text for token in (
        "restart", "generate", "fresh program", "from scratch", "discard",
    )):
        return "whole_program_restart"
    if any(token in family_text for token in (
        "composition", "compose", "ensemble", "residual", "representation",
        "crossover", "combine", "switch",
    )):
        return "ast_crossover"
    return "ast_mutation"


def canonical_program_operator(item: dict[str, Any]) -> str:
    """Return the operator actually supported by the Python Proposal path."""
    value = str(item.get("intervention_operator") or item.get("operator") or "").lower()
    aliases = {
        "restart": "whole_program_restart", "restart_from_skeleton": "whole_program_restart",
        "mutate": "ast_mutation", "local_mutation": "ast_mutation",
        "crossover": "ast_crossover", "compose": "ast_crossover",
        "composition": "ast_crossover", "combine": "ast_crossover",
    }
    value = aliases.get(value, value)
    if value in {"whole_program_restart", "ast_mutation", "ast_crossover", "subsystem_rewrite"}:
        return value
    return _context_operator(str(item.get("family", "")), str(item.get("mechanism", "")),
                             str(item.get("intervention_scope") or item.get("scope") or ""))


@dataclass(frozen=True)
class MechanismHypothesis:
    """A concise, falsifiable algorithm-design hypothesis."""

    id: str
    family: str
    mechanism: str
    prediction: str
    failure_condition: str
    matched_control: str
    target_slices: tuple[str, ...] = ()
    implementation_hint: str = ""
    source: str = "task"
    # Typed intervention metadata is optional so older task portfolios remain
    # valid.  Context/Proposal can use it to distinguish a parameter tweak
    # from a family change without parsing free-form prose.
    intervention_scope: str = "mechanism"
    intervention_operator: str = "replace"
    target_slice: str = ""
    evidence_ids: tuple[str, ...] = ()
    next_probe: str = ""
    state_version: str | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HYPOTHESIS_SCHEMA,
            "id": self.id,
            "family": self.family,
            "mechanism": self.mechanism,
            "prediction": self.prediction,
            "failure_condition": self.failure_condition,
            "matched_control": self.matched_control,
            "target_slices": list(self.target_slices),
            "implementation_hint": self.implementation_hint,
            "source": self.source,
            "intervention_scope": self.intervention_scope,
            "intervention_operator": self.intervention_operator,
            "target_slice": self.target_slice,
            "evidence_ids": list(self.evidence_ids),
            "next_probe": self.next_probe,
            "state_version": self.state_version,
        }


@dataclass(frozen=True)
class MechanismDesign:
    """Task configuration plus a bounded list of mechanism directions."""

    schema: str = CONFIG_SCHEMA
    directions: tuple[MechanismHypothesis, ...] = ()
    critic_questions: tuple[str, ...] = ()
    selection: str = "slot_round_robin"
    enabled: bool = False
    matched_control: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "MechanismDesign":
        return cls()

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.directions)


def _as_mapping(task: Any) -> dict[str, Any] | None:
    """Read a task-owned config without changing the generic Task loader."""
    supplied = getattr(task, "mechanism_design", None)
    if isinstance(supplied, MechanismDesign):
        return {
            "schema": supplied.schema,
            "enabled": supplied.enabled,
            "selection": supplied.selection,
            "directions": [item.to_dict() for item in supplied.directions],
            "critic_questions": list(supplied.critic_questions),
            "matched_control": dict(supplied.matched_control),
        }
    if isinstance(supplied, dict):
        return supplied

    task_dir = getattr(task, "dir", None)
    if task_dir is None:
        return None
    try:
        config_path = Path(task_dir) / "task.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    config = payload.get("mechanism_design")
    return config if isinstance(config, dict) else None


def _normalize_hypothesis(item: Any, *, source: str) -> MechanismHypothesis | None:
    if isinstance(item, MechanismHypothesis):
        return item
    if isinstance(item, str):
        mechanism = _text(item)
        if not mechanism:
            return None
        generated_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", mechanism.lower()).strip("-")
        item = {"id": generated_id[:64], "mechanism": mechanism}
    if not isinstance(item, dict):
        return None
    mechanism = _text(
        item.get("mechanism")
        or item.get("hypothesis")
        or item.get("intervention")
        or item.get("idea")
    )
    if not mechanism:
        return None
    family = _text(item.get("family"), limit=MAX_FAMILY_CHARS) or "general"
    prediction = _text(item.get("prediction"))
    failure = _text(item.get("failure_condition"))
    control = _text(item.get("matched_control"))
    # LLMs frequently omit a slug or use a human-language id.  Derive a
    # stable identifier instead of dropping an otherwise useful new idea.
    identifier = _id(
        item.get("id") or item.get("mechanism_id") or item.get("name")
    ) or _derived_id(
        family, mechanism, prediction,
    )
    # A task direction is useful even when its author omitted prose fields;
    # generated Context hypotheses, however, must carry all three fields.
    if not prediction:
        prediction = "improves the trusted metric on at least one declared slice"
    if not failure:
        failure = "no improvement after the matched-budget comparison"
    if not control:
        control = "same parent, seed, and compute budget with the mechanism removed"
    raw_slices = item.get("target_slices", item.get("target_slice", ()))
    if isinstance(raw_slices, str):
        raw_slices = [raw_slices]
    slices: list[str] = []
    if isinstance(raw_slices, Iterable):
        for value in raw_slices:
            value = _text(value, limit=160)
            if value and value not in slices:
                slices.append(value)
            if len(slices) >= 8:
                break
    implementation = _text(item.get("implementation_hint"), limit=MAX_TEXT_CHARS)
    scope = _text(
        item.get("intervention_scope") or item.get("scope"), limit=64
    ) or "mechanism"
    raw_operator = item.get("intervention_operator")
    if raw_operator in (None, ""):
        raw_operator = item.get("operator")
    operator = _text(raw_operator, limit=64)
    if not operator:
        operator = (
            _context_operator(family, mechanism, scope)
            if source == "context" else "replace"
        )
    if source == "context":
        operator = canonical_program_operator({
            "operator": operator, "family": family,
            "mechanism": mechanism, "scope": scope,
        })
    target_slice = item.get("target_slice", "")
    if isinstance(target_slice, (list, tuple)):
        target_slice = ", ".join(
            _text(value, limit=120) for value in target_slice
            if _text(value, limit=120)
        )
    target_slice = _text(target_slice, limit=240)
    if not target_slice and slices:
        target_slice = ", ".join(slices)
    raw_evidence = item.get("evidence_ids", ())
    if isinstance(raw_evidence, str):
        raw_evidence = [raw_evidence]
    evidence_ids: list[str] = []
    if isinstance(raw_evidence, Iterable):
        for value in raw_evidence:
            value = _text(value, limit=160)
            if value and value not in evidence_ids:
                evidence_ids.append(value)
            if len(evidence_ids) >= 16:
                break
    next_probe = _text(item.get("next_probe"), limit=MAX_TEXT_CHARS)
    state_version = item.get("state_version")
    if isinstance(state_version, bool):
        state_version = None
    elif isinstance(state_version, float) and state_version.is_integer():
        state_version = int(state_version)
    elif state_version is not None and not isinstance(state_version, int):
        state_version = _text(state_version, limit=128) or None
    return MechanismHypothesis(
        id=identifier,
        family=family,
        mechanism=mechanism,
        prediction=prediction,
        failure_condition=failure,
        matched_control=control,
        target_slices=tuple(slices),
        implementation_hint=implementation,
        source=_text(item.get("source"), limit=80) or source,
        intervention_scope=scope,
        intervention_operator=operator,
        target_slice=target_slice,
        evidence_ids=tuple(evidence_ids),
        next_probe=next_probe,
        state_version=state_version,
    )


def normalize_hypotheses(
    values: Any,
    *,
    source: str = "context",
    limit: int = MAX_CONTEXT_HYPOTHESES,
) -> list[dict[str, Any]]:
    """Return compact, deterministic hypothesis records for prompt/context use."""
    if not isinstance(values, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        hypothesis = _normalize_hypothesis(item, source=source)
        if hypothesis is None or hypothesis.id in seen:
            continue
        seen.add(hypothesis.id)
        result.append(hypothesis.to_dict())
        if len(result) >= max(1, int(limit)):
            break
    return result


def load_mechanism_design(task: Any) -> MechanismDesign:
    """Load and normalize the optional mechanism portfolio for ``task``."""
    raw = _as_mapping(task)
    if not raw:
        return MechanismDesign.empty()
    schema = _text(raw.get("schema"), limit=128) or CONFIG_SCHEMA
    enabled = raw.get("enabled", True) is not False
    selection = _text(raw.get("selection"), limit=80) or "slot_round_robin"
    raw_directions = raw.get("directions", raw.get("mechanism_directions", []))
    directions: list[MechanismHypothesis] = []
    seen: set[str] = set()
    if isinstance(raw_directions, (list, tuple)):
        for item in raw_directions:
            direction = _normalize_hypothesis(item, source="task")
            if direction is None or direction.id in seen:
                continue
            seen.add(direction.id)
            directions.append(direction)
            if len(directions) >= MAX_DIRECTIONS:
                break
    raw_critics = raw.get("critic_questions", raw.get("critic_prompts", []))
    critics: list[str] = []
    if isinstance(raw_critics, (list, tuple)):
        for item in raw_critics:
            question = _text(item, limit=MAX_TEXT_CHARS)
            if question and question not in critics:
                critics.append(question)
            if len(critics) >= 8:
                break
    if not critics and directions:
        critics = [
            "What observable result would distinguish this mechanism from a larger or longer-trained baseline?",
            "Which contract slice is most likely to falsify the proposed advantage?",
            "Does the implementation preserve the same parent, data boundary, and compute budget?",
        ]
    raw_control = raw.get("matched_control", {})
    control = dict(raw_control) if isinstance(raw_control, dict) else {}
    # Only a small set of scalar control settings is consumed here.  Other
    # task-owned annotations remain available to the caller through task.json.
    control["enabled"] = control.get("enabled") is True
    return MechanismDesign(
        schema=schema,
        directions=tuple(directions),
        critic_questions=tuple(critics),
        selection=selection,
        enabled=enabled,
        matched_control=control,
    )


def _combined_hypotheses(
    design: MechanismDesign,
    context_hypotheses: Iterable[dict[str, Any]] = (),
) -> list[MechanismHypothesis]:
    # Context-generated ideas lead the next round: this is what makes the
    # proposal space genuinely open rather than a fixed menu.  Task-declared
    # directions remain the deterministic fallback and diversity reservoir.
    result: list[MechanismHypothesis] = []
    seen: set[str] = set()
    for item in context_hypotheses:
        hypothesis = _normalize_hypothesis(item, source="context")
        if hypothesis is None or hypothesis.id in seen:
            continue
        result.append(hypothesis)
        seen.add(hypothesis.id)
        if len(result) >= MAX_DIRECTIONS:
            break
    for item in design.directions:
        if item.id in seen:
            continue
        result.append(item)
        seen.add(item.id)
        if len(result) >= MAX_DIRECTIONS:
            break
    return result


def candidate_hypotheses(
    task: Any,
    context_hypotheses: Iterable[dict[str, Any]] = (),
    candidate_count: int | None = None,
    iteration: int = 0,
    *,
    state: Any = None,
    acquisition_router: Any = None,
) -> list[dict[str, Any]]:
    """Choose one deterministic mechanism hypothesis for each candidate slot.

    Context-generated hypotheses lead the pool, while task directions provide a
    stable diversity reservoir and fallback.  A round rotates through the
    pool so repeated rounds do not always start with the same mechanism.
    Returned records are plain dictionaries, which keeps the helper convenient
    for plans and prompt metadata without coupling callers to a schema class.
    """
    design = load_mechanism_design(task)
    if not design.active and not context_hypotheses:
        return []
    try:
        count = int(candidate_count)
    except (TypeError, ValueError):
        count = int(getattr(task, "candidates_per_context", 4) or 4)
    count = max(1, count)
    if acquisition_router is not None:
        # The router is optional and imported lazily to keep this vocabulary
        # module lightweight for legacy tasks.
        selected = acquisition_router.select(
            list(context_hypotheses or ()),
            count=count,
            state=(
                state.to_dict() if hasattr(state, "to_dict") else state
            ),
            iteration=iteration,
        )
        if selected:
            return selected
    pool = _combined_hypotheses(design, context_hypotheses)
    if not pool:
        return []
    try:
        round_number = max(0, int(iteration))
    except (TypeError, ValueError):
        round_number = 0
    offset = 0 if context_hypotheses else (round_number * count) % len(pool)
    return [
        pool[(offset + slot) % len(pool)].to_dict()
        for slot in range(count)
    ]


def mechanism_generation_operator(
    hypothesis: MechanismHypothesis | dict[str, Any],
) -> str:
    """Map one mechanism hypothesis to the small V5 operator vocabulary."""
    normalized = _normalize_hypothesis(hypothesis, source="context")
    if normalized is None:
        return "local_mutation"
    if isinstance(hypothesis, dict):
        raw_explicit = hypothesis.get(
            "intervention_operator", hypothesis.get("operator")
        )
    else:
        raw_explicit = hypothesis.intervention_operator
    explicit = (
        raw_explicit.lower()
        if isinstance(raw_explicit, str) and raw_explicit
        else ""
    )
    if explicit in {"restart", "whole_program_restart"}:
        return "restart_from_skeleton"
    if explicit in {"ast_crossover", "crossover", "compose"}:
        return "composition"
    if explicit in {"ast_mutation", "mutate"}:
        return "local_mutation"
    if explicit in {"subsystem_rewrite"}:
        return "local_mutation"
    family = normalized.family.lower()
    text = normalized.mechanism.lower()
    if any(token in family or token in text for token in ("ablat", "control")):
        return "ablation"
    if (
        family in {"representation", "inductive_bias"}
        or any(
            token in family or token in text
            for token in (
                "composition", "compose", "ensemble", "switch",
                "symbolic", "residual",
            )
        )
    ):
        return "composition"
    if any(token in family or token in text for token in ("analogy", "transfer")):
        return "analogy_transfer"
    return "local_mutation"


def matched_control_enabled(task: Any) -> bool:
    """Return whether this task requests paired guided/control proposals."""
    direct = getattr(task, "matched_control_enabled", None)
    if isinstance(direct, bool):
        return direct
    design = load_mechanism_design(task)
    return design.matched_control.get("enabled") is True


def hypothesis_to_analogy(
    hypothesis: MechanismHypothesis | dict[str, Any],
    *,
    target_parent_id: str,
    source_record_ids: Iterable[str] = (),
    analogy_id: str | None = None,
    metric: str = "paired_lower_bound_lcb",
    minimum_effect: float = 0.0,
    non_correspondence: Iterable[str] = (),
    status: str = "preregistered",
    matched_control: dict[str, Any] | None = None,
    as_dict: bool = False,
):
    """Project a mechanism hypothesis onto the existing analogy schema.

    This is an adapter, not a claim that the analogy has transferred.  The
    returned object remains ``preregistered`` until an experiment supplies an
    independent result.  ``as_dict`` is useful when serializing a plan.
    """
    normalized = _normalize_hypothesis(hypothesis, source="context")
    if normalized is None:
        raise ValueError("hypothesis must contain a valid id and mechanism")
    if not isinstance(target_parent_id, str) or not target_parent_id.strip():
        raise ValueError("target_parent_id must be non-empty")
    source_ids = [
        value.strip() for value in source_record_ids
        if isinstance(value, str) and value.strip()
    ]
    identifier = _id(analogy_id) if analogy_id else _id(f"analogy_{normalized.id}")
    if not identifier:
        raise ValueError("analogy_id must be a bounded identifier")
    try:
        effect = float(minimum_effect)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_effect must be numeric") from exc
    if not effect >= 0.0:
        raise ValueError("minimum_effect must be non-negative")

    control = matched_control
    if control is None:
        control = {
            "enabled": True,
            "description": normalized.matched_control,
            "same_parent": True,
            "same_seed": True,
            "same_compute_budget": True,
        }
    elif not isinstance(control, dict):
        raise ValueError("matched_control must be a mapping")
    else:
        control = dict(control)
    # Keep the mechanism identity alongside the existing control description.
    # The Analogy schema intentionally leaves this mapping extensible, so the
    # next Context can understand a result without inspecting candidate prose.
    control.setdefault("mechanism_id", normalized.id)
    control.setdefault("family", normalized.family)
    control.setdefault(
        "generation_operator", mechanism_generation_operator(normalized)
    )

    # Import lazily so the lightweight task prompt path does not depend on the
    # V5 schema module at import time.
    from schemas_v5 import AnalogyHypothesis

    analogy = AnalogyHypothesis(
        id=identifier,
        source_record_ids=source_ids,
        target_parent_id=target_parent_id.strip(),
        relation_mapping=[
            {
                "source_role": normalized.family,
                "target_role": "continuation_policy",
                "shared_relation": normalized.mechanism,
            }
        ],
        non_correspondence=[
            value.strip() for value in non_correspondence
            if isinstance(value, str) and value.strip()
        ],
        transferable_intervention=normalized.mechanism,
        predicted_effect={
            "metric": _text(metric, limit=160) or "paired_lower_bound_lcb",
            "direction": "positive",
            "minimum_effect": effect,
        },
        falsifier=normalized.failure_condition,
        matched_control=dict(control),
        status=status,
    )
    analogy.validate()
    return analogy.to_dict() if as_dict else analogy


def render_context_block(task: Any, *, candidate_count: int | None = None) -> str:
    """Render concise instructions for Context to generate a portfolio."""
    design = load_mechanism_design(task)
    if not design.active:
        return ""
    count = candidate_count or getattr(task, "candidates_per_context", 4) or 4
    rows = []
    for index, item in enumerate(design.directions):
        slices = ", ".join(item.target_slices) if item.target_slices else "any declared slice"
        hint = f"; implementation={item.implementation_hint}" if item.implementation_hint else ""
        rows.append(
            f"{index + 1}. `{item.id}` [{item.family}] {item.mechanism}; "
            f"scope={item.intervention_scope}; operator={item.intervention_operator}; "
            f"predicts: {item.prediction}; falsifier: {item.failure_condition}; "
            f"control: {item.matched_control}; target: {slices}{hint}"
        )
    critics = "\n".join(f"- {question}" for question in design.critic_questions)
    return (
        "\n## Open algorithm-design portfolio\n"
        f"Task schema: `{design.schema}`; this Context round may generate "
        f"multiple structurally different mechanisms for {count} candidate slots.\n"
        "Use the directions below as starting points, not as a closed menu. "
        "Keep each proposed mechanism executable within the task-owned artifact protocol.\n"
        + "\n".join(rows)
        + "\n\nFor the JSON decision, also return `mechanism_candidates`: a list of "
        "2-6 objects with `id`, `family`, `mechanism`, `prediction`, "
        "`failure_condition`, and `matched_control`. Optionally include "
        "`intervention_scope`, `intervention_operator`, `target_slice`, "
        "`evidence_ids`, and `next_probe`. These are hypotheses, "
        "not results; do not claim a score before evaluation.\n"
        "Critic questions to apply before selecting the next direction:\n"
        + critics
        + "\n"
    )


def render_proposal_block(
    task: Any,
    *,
    candidate_count: int | None = None,
    context_hypotheses: Iterable[dict[str, Any]] = (),
) -> str:
    """Render Proposal instructions for diverse, structured candidate edits."""
    design = load_mechanism_design(task)
    # A Context round may invent its own mechanisms even for a task that has
    # no pre-seeded portfolio.  Preserve that genuinely open path instead of
    # requiring a task author to anticipate every useful structure.
    context_hypotheses = tuple(context_hypotheses or ())
    if not design.active and not context_hypotheses:
        return ""
    count = candidate_count or getattr(task, "candidates_per_context", 4) or 4
    combined = _combined_hypotheses(design, context_hypotheses)
    rows = []
    for index, item in enumerate(combined):
        target = ", ".join(item.target_slices) if item.target_slices else "declared task slices"
        rows.append(
            f"{index % max(1, count) + 1}. `{item.id}` [{item.family}] "
            f"scope={item.intervention_scope}; operator={item.intervention_operator}; "
            f"mechanism={item.mechanism}; prediction={item.prediction}; "
            f"failure={item.failure_condition}; matched_control={item.matched_control}; "
            f"target={target}"
        )
    critics = "\n".join(f"- {question}" for question in design.critic_questions)
    return (
        "\n## Open algorithm-design assignment\n"
        f"This Context has {count} independent candidate slots. Preserve diversity: "
        "each slot should implement or materially instantiate a different mechanism "
        "from the portfolio; do not make all candidates width/depth variants.\n"
        "Choose the row matching your Local candidate identity, or synthesize a "
        "clearly named extension. Before editing, state a falsifiable hypothesis "
        "and a matched control in your reasoning.\n"
        + "\n".join(rows)
        + "\n\nRequired first line of `PROPOSAL.md` (one compact JSON object):\n"
        '{"mechanism_id":"...","family":"...","hypothesis":"...",'
        '"prediction":"...","failure_condition":"...",'
        '"matched_control":"..."}\n'
        "The fields describe the proposed mechanism only; evaluator metrics are "
        "added later by the Harness.\n"
        "Lightweight critic pass:\n"
        + critics
        + "\n"
    )


def hypotheses_from_analysis(path: Path) -> list[dict[str, Any]]:
    """Read Context-generated hypotheses if the current analysis has them."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    return normalize_hypotheses(payload.get("mechanism_candidates", []))
