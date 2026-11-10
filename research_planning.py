"""Evidence-conditioned, mechanism-first planning for open program experiments.

This module changes research context, not the trusted evaluator or candidate
language. It does not classify scientific novelty from model declarations.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def research_packet(records: Sequence[Mapping[str, Any]], limit: int = 8) -> dict:
    """Project public records only; private results must never enter planning.

    The complete records remain in the append-only experiment directory. Keep
    failures and unsuccessful exploratory candidates, not only the incumbent.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    public = [r for r in records if r.get("data_role") == "public"]
    selected = public[-limit:]
    rows = [{k: r.get(k) for k in (
        "id", "status", "mechanism", "score", "observations", "failure",
        "source_digest", "structural_claim",
    )} for r in selected]
    return {"schema": "openhyra-research-packet.v1", "records": rows,
            "record_ids": [r["id"] for r in rows],
            "source_records_digest": digest(selected)}


def planning_prompt(*, arm: str, contract: str, packet: Mapping[str, Any],
                    parent_source: str, round_index: int) -> str:
    if arm not in ("parent_visible", "mechanism_first"):
        raise ValueError("unknown planning arm")
    common = f"""You are a research Context Agent solving Bermudan optimal stopping.
Read only this prompt; no tools, files, or network. Produce JSON only.
Round: {round_index}. Fit budget: 256 paths, 30 seconds; Python standard library
and NumPy only. The objective is expected discounted stopping payoff. Scores
are confidence-gap diagnostics with a common evaluator upper bound; lower is
better but changes in uncertainty can change score without changing payoff.
All observed effects are noisy; do not assert a cause from a scalar score.

External contract (identical in both arms):
{contract}

Seed principle: backward least-squares continuation regression using fixed
features and ridge regularization. This is an initial solution, not a language
restriction. You may replace the whole program and use either interface.

Public evidence (failed and non-selected attempts included):
{canonical(packet)}

Return an object with alternatives (exactly 3 objects), selected_id, and
selection_reason. Each alternative must contain id, mechanism, implementation,
representation, learning_objective, solver, evidence_ids (IDs from the packet),
prediction, falsifier, structural_difference, and analogy (object with source,
mapping, non_correspondence). A proposal's prediction is not observed evidence.
Maximum 1200 words total. No hidden-data assumptions. The selected plan must be
implementable within the same budget; do not emit source code.
"""
    if arm == "parent_visible":
        return common + "\nPropose the most promising improvements to this parent.\nParent implementation:\n" + parent_source
    return common + """
Mechanism-first exploration: reason from the task's relations, numerical
evidence, and competing explanations before choosing an implementation.
Provide one local refinement and two alternatives that change a substantive
representation, learning objective, or solving mechanism. Do not merely rename
the seed or vary its regularization/features. No target model family is
prescribed. State where any analogy fails and a discriminating experiment.
Select a nonlocal alternative for its relevance and feasibility, not fashion.
Unsuccessful public experiments should revise the explanation or open another
direction. Exploration is retained separately from the best scoring incumbent;
one poor score does not erase a candidate's evidence. Do not silently deploy a
seed fallback instead of the selected method. Any fallback must be reported.
"""


def parse_plan(text: str, packet: Mapping[str, Any]) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    plan = json.loads(text)
    alternatives = plan.get("alternatives")
    if not isinstance(alternatives, list) or len(alternatives) != 3:
        raise ValueError("expected exactly three alternatives")
    ids = []
    available = set(packet["record_ids"])
    for item in alternatives:
        for key in ("id", "mechanism", "implementation", "representation",
                    "learning_objective", "solver", "prediction", "falsifier",
                    "structural_difference"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"missing plan field: {key}")
        ids.append(item["id"])
        refs = item.get("evidence_ids")
        if not isinstance(refs, list) or not all(isinstance(x, str) and x in available for x in refs):
            raise ValueError("unknown evidence reference")
        analogy = item.get("analogy")
        if not isinstance(analogy, dict) or not all(isinstance(analogy.get(k), str) for k in ("source", "mapping", "non_correspondence")):
            raise ValueError("missing analogy boundaries")
    if len(set(ids)) != len(ids) or plan.get("selected_id") not in ids:
        raise ValueError("ambiguous plan selection")
    if not isinstance(plan.get("selection_reason"), str) or not plan["selection_reason"].strip():
        raise ValueError("missing selection reason")
    return plan


def selected_plan(plan: Mapping[str, Any]) -> dict:
    return next(x for x in plan["alternatives"] if x["id"] == plan["selected_id"])
