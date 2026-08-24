#!/usr/bin/env python3
"""Trusted evaluator for the OpenHyra sum-difference task.

The submitted artifact must contain a finite explicit set with at least two
bounded integers. Set cardinality has no fixed upper bound; the configured
artifact, evaluator time, and evaluator memory budgets provide the operational
limits. The evaluator computes A+A and A-A by exact enumeration.
"""

import hashlib
import json
import math
import re
import sys
from functools import reduce
from math import gcd
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from formalization import (  # noqa: E402
    ArtifactRejected,
    REQUEST_SCHEMA as FORMALIZATION_SCHEMA,
    validate_formalization_request,
)
from research_verifier import (  # noqa: E402
    SCHEMA as CONSTRUCTION_SCHEMA,
    verify_digit_product,
)

MIN_N = 2
MIN_INT = -1_000_000
MAX_INT = 1_000_000
RESEARCH_SCHEMA = "openhyra-research"
RESEARCH_DEFINITION = "sum-diff-exponent"
RESEARCH_SCOPES = {"current_task", "all_finite_integer_sets"}
CLAIM_TEMPLATES = {
    "observation",
    "finite_witness",
    "supporting_lemma",
    "universal_upper_bound",
    "approximating_family",
    "supremum_eq",
    "nonattainment",
}
FORMALIZATION_TARGETS = {"none", "lean4"}
CLAIM_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
MAX_HYPOTHESIS_CHARS = 1_000
MAX_CONSTRUCTION_CHARS = 6_000
MAX_PROOF_SKETCH_CHARS = 8_000
MAX_FALSIFICATION_CHARS = 4_000
MAX_CLAIMS = 16
MAX_CLAIM_STATEMENT_CHARS = 1_000
MAX_CERTIFICATES = 8
MAX_CERTIFICATE_RESIDUES = 256
MAX_CERTIFICATE_MODULUS = 10_000
MAX_RESEARCH_CONTEXT_CHARS = 3_000
MAX_OBLIGATION_LINKS = 32


def fail(msg):
    print(json.dumps({"error": msg}))
    raise SystemExit(0)


def _canonical_json(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _sha256_json(payload):
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _strict_keys(payload, *, required, allowed, path):
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be an object")
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(unknown)}")
    missing = sorted(set(required) - set(payload))
    if missing:
        raise ValueError(f"{path} is missing field(s): {', '.join(missing)}")


def _bounded_text(value, *, path, limit, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    value = value.strip()
    if len(value) > limit:
        raise ValueError(f"{path} exceeds {limit} characters")
    return value


def _clip_text(value, limit):
    if len(value) <= limit:
        return value
    marker = " ...[truncated]... "
    available = limit - len(marker)
    head = (available * 2) // 3
    return value[:head] + marker + value[-(available - head):]


def _strict_int(value, *, path, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} must be <= {maximum}")
    return value


def _normalize_rational_target(raw, *, path):
    _strict_keys(
        raw,
        required={"numerator", "denominator"},
        allowed={"numerator", "denominator"},
        path=path,
    )
    numerator = _strict_int(
        raw["numerator"], path=f"{path}.numerator",
        minimum=-1_000_000, maximum=1_000_000,
    )
    denominator = _strict_int(
        raw["denominator"], path=f"{path}.denominator",
        minimum=1, maximum=1_000_000,
    )
    common = gcd(abs(numerator), denominator)
    return {
        "numerator": numerator // common,
        "denominator": denominator // common,
    }


def _validate_claims(raw_claims, scope, obligation_outcomes):
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ValueError("research.claims must be a non-empty list")
    if len(raw_claims) > MAX_CLAIMS:
        raise ValueError(f"research.claims exceeds {MAX_CLAIMS} entries")

    claims = []
    ids = set()
    for index, raw in enumerate(raw_claims):
        path = f"research.claims[{index}]"
        _strict_keys(
            raw,
            required={
                "id", "template", "statement",
                "depends_on", "obligation_ids",
            },
            allowed={
                "id", "template", "statement", "depends_on",
                "obligation_ids", "target",
            },
            path=path,
        )
        claim_id = raw["id"]
        if not isinstance(claim_id, str) or not CLAIM_ID_RE.fullmatch(claim_id):
            raise ValueError(f"{path}.id has invalid syntax")
        if claim_id in ids:
            raise ValueError(f"duplicate research claim id: {claim_id}")
        ids.add(claim_id)
        template = raw["template"]
        if template not in CLAIM_TEMPLATES:
            raise ValueError(f"{path}.template is not supported")
        statement = _bounded_text(
            raw["statement"],
            path=f"{path}.statement",
            limit=MAX_CLAIM_STATEMENT_CHARS,
        )
        depends_on = raw.get("depends_on", [])
        if not isinstance(depends_on, list) or any(
                not isinstance(item, str) for item in depends_on):
            raise ValueError(f"{path}.depends_on must be a list of claim ids")
        if len(depends_on) != len(set(depends_on)):
            raise ValueError(f"{path}.depends_on contains duplicates")
        obligation_ids = raw["obligation_ids"]
        if not isinstance(obligation_ids, list) or any(
                not isinstance(item, str) for item in obligation_ids):
            raise ValueError(
                f"{path}.obligation_ids must be a list of obligation ids"
            )
        if len(obligation_ids) > MAX_OBLIGATION_LINKS:
            raise ValueError(
                f"{path}.obligation_ids exceeds {MAX_OBLIGATION_LINKS} entries"
            )
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError(f"{path}.obligation_ids contains duplicates")
        unknown_obligations = sorted(
            set(obligation_ids) - set(obligation_outcomes)
        )
        if unknown_obligations:
            raise ValueError(
                f"{path}.obligation_ids has unknown entries: "
                + ", ".join(unknown_obligations)
            )
        formal_template = template in {
            "universal_upper_bound",
            "approximating_family",
            "supremum_eq",
            "nonattainment",
        }
        target = raw.get("target")
        if formal_template and target is None:
            raise ValueError(
                f"{path}.target is required for {template}"
            )
        if not formal_template and target is not None:
            raise ValueError(
                f"{path}.target is only valid for formal claim templates"
            )
        claims.append({
            "id": claim_id,
            "template": template,
            "statement": statement,
            "depends_on": sorted(depends_on),
            "obligation_ids": sorted(obligation_ids),
            **(
                {
                    "target": _normalize_rational_target(
                        target, path=f"{path}.target",
                    ),
                }
                if target is not None else {}
            ),
        })

    graph = {claim["id"]: claim["depends_on"] for claim in claims}
    for claim_id, dependencies in graph.items():
        unknown = sorted(set(dependencies) - set(graph))
        if unknown:
            raise ValueError(
                f"research claim {claim_id} has unknown dependencies: "
                + ", ".join(unknown)
            )
        if claim_id in dependencies:
            raise ValueError(f"research claim {claim_id} depends on itself")

    visiting = set()
    visited = set()

    def visit(claim_id):
        if claim_id in visiting:
            raise ValueError("research claim dependencies must be acyclic")
        if claim_id in visited:
            return
        visiting.add(claim_id)
        for dependency in graph[claim_id]:
            visit(dependency)
        visiting.remove(claim_id)
        visited.add(claim_id)

    for claim_id in graph:
        visit(claim_id)

    claims.sort(key=lambda claim: claim["id"])
    evidence_claims = []
    for claim in claims:
        claim_payload = {
            "definition": RESEARCH_DEFINITION,
            "scope": scope,
            **claim,
        }
        linked = [
            obligation_outcomes[obligation_id]
            for obligation_id in claim["obligation_ids"]
        ]
        linked_obligation_status = (
            "contains_refutation"
            if any(item["status"] == "refuted" for item in linked) else
            "bounded_checked"
            if linked and all(
                item["status"] == "bounded_checked" for item in linked
            ) else
            "not_linked"
        )
        evidence_claims.append({
            "id": claim["id"],
            "template": claim["template"],
            "claim_hash": _sha256_json(claim_payload),
            # Obligation links organize evidence but do not define a trusted
            # logical implication from a finite construction fact to this
            # natural-language claim. Only the formal gate may promote it.
            "status": "unverified",
            "linked_obligation_status": linked_obligation_status,
            "obligation_ids": list(claim["obligation_ids"]),
            **({"target": claim["target"]} if "target" in claim else {}),
        })
    return claims, evidence_claims


def _validate_certificates(raw_certificates):
    if raw_certificates is None:
        return [], []
    if not isinstance(raw_certificates, list):
        raise ValueError("research.certificates must be a list")
    if len(raw_certificates) > MAX_CERTIFICATES:
        raise ValueError(
            f"research.certificates exceeds {MAX_CERTIFICATES} entries"
        )

    certificates = []
    evidence = []
    for index, raw in enumerate(raw_certificates):
        path = f"research.certificates[{index}]"
        _strict_keys(
            raw,
            required={
                "type", "modulus", "residues",
                "expected_sum_count", "expected_diff_count",
            },
            allowed={
                "type", "modulus", "residues",
                "expected_sum_count", "expected_diff_count",
            },
            path=path,
        )
        if raw["type"] != "modular_sum_diff":
            raise ValueError(f"{path}.type is not supported")
        modulus = _strict_int(
            raw["modulus"],
            path=f"{path}.modulus",
            minimum=2,
            maximum=MAX_CERTIFICATE_MODULUS,
        )
        residues = raw["residues"]
        if not isinstance(residues, list) or not residues:
            raise ValueError(f"{path}.residues must be a non-empty list")
        if len(residues) > MAX_CERTIFICATE_RESIDUES:
            raise ValueError(
                f"{path}.residues exceeds {MAX_CERTIFICATE_RESIDUES} entries"
            )
        normalized_residues = [
            _strict_int(
                residue,
                path=f"{path}.residues[{position}]",
                minimum=0,
                maximum=modulus - 1,
            )
            for position, residue in enumerate(residues)
        ]
        if len(normalized_residues) != len(set(normalized_residues)):
            raise ValueError(f"{path}.residues must be distinct")
        normalized_residues.sort()
        expected_sum = _strict_int(
            raw["expected_sum_count"],
            path=f"{path}.expected_sum_count",
            minimum=1,
            maximum=modulus,
        )
        expected_diff = _strict_int(
            raw["expected_diff_count"],
            path=f"{path}.expected_diff_count",
            minimum=1,
            maximum=modulus,
        )
        sums = {
            (left + right) % modulus
            for left in normalized_residues
            for right in normalized_residues
        }
        diffs = {
            (left - right) % modulus
            for left in normalized_residues
            for right in normalized_residues
        }
        matched = len(sums) == expected_sum and len(diffs) == expected_diff
        certificate = {
            "type": "modular_sum_diff",
            "modulus": modulus,
            "residues": normalized_residues,
            "expected_sum_count": expected_sum,
            "expected_diff_count": expected_diff,
        }
        verification = {
            "sum_count": len(sums),
            "diff_count": len(diffs),
            "sum_full": len(sums) == modulus,
            "diff_full": len(diffs) == modulus,
        }
        certificates.append(certificate)
        evidence.append({
            "certificate_hash": _sha256_json(certificate),
            "type": certificate["type"],
            "modulus": modulus,
            "status": "bounded_checked" if matched else "refuted",
            "expected_sum_count": expected_sum,
            "expected_diff_count": expected_diff,
            **verification,
        })
    combined = sorted(
        zip(certificates, evidence),
        key=lambda item: item[1]["certificate_hash"],
    )
    if len({item[1]["certificate_hash"] for item in combined}) != len(combined):
        raise ValueError("research.certificates contains duplicates")
    return (
        [item[0] for item in combined],
        [item[1] for item in combined],
    )


def _validate_construction(raw):
    evidence = verify_digit_product(raw)
    normalized = evidence["construction"]
    outcomes = {
        item["id"]: item
        for item in evidence["obligations"]
    }
    return normalized, evidence, outcomes


def validate_research(raw):
    """Validate untrusted research narrative and derive bounded evidence.

    Natural-language claims and proof sketches remain unverified. Only the
    allowlisted finite certificates are recomputed by this trusted evaluator.
    """
    if raw is None:
        return None, {
            "evidence_level": "numeric",
            "research_rank": 0,
            "research_claim_count": 0,
            "bounded_supported_claim_count": 0,
            "refuted_claim_count": 0,
            "formally_checked_claim_count": 0,
            "verified_obligation_count": 0,
            "refuted_obligation_count": 0,
            "verified_certificate_count": 0,
            "refuted_certificate_count": 0,
            "formalization_status": "not_submitted",
        }, {
            "status": "not_submitted",
            "claims": [],
            "certificates": [],
            "construction": None,
            "formalization": {
                "target": "lean4",
                "status": "not_submitted",
            },
        }
    _strict_keys(
        raw,
        required={
            "schema", "scope", "hypothesis", "construction",
            "claims", "falsification_plan",
        },
        allowed={
            "schema", "scope", "hypothesis", "construction", "claims",
            "proof_sketch", "falsification_plan", "certificates",
            "formalization",
        },
        path="research",
    )
    if raw["schema"] != RESEARCH_SCHEMA:
        raise ValueError(f"research.schema must be {RESEARCH_SCHEMA!r}")
    scope = raw["scope"]
    if scope not in RESEARCH_SCOPES:
        raise ValueError("research.scope is not supported")
    hypothesis = _bounded_text(
        raw["hypothesis"],
        path="research.hypothesis",
        limit=MAX_HYPOTHESIS_CHARS,
    )
    construction, construction_evidence, obligation_outcomes = (
        _validate_construction(raw["construction"])
    )
    proof_sketch = _bounded_text(
        raw.get("proof_sketch"),
        path="research.proof_sketch",
        limit=MAX_PROOF_SKETCH_CHARS,
        optional=True,
    )
    falsification_plan = _bounded_text(
        raw["falsification_plan"],
        path="research.falsification_plan",
        limit=MAX_FALSIFICATION_CHARS,
    )
    claims, evidence_claims = _validate_claims(
        raw["claims"], scope, obligation_outcomes,
    )
    try:
        sealed_formalization, _theorem_types = validate_formalization_request(
            raw.get("formalization"), claims,
        )
    except ArtifactRejected as exc:
        raise ValueError(f"research.formalization rejected: {exc}") from exc
    normalized_formalization = (
        {
            "schema": sealed_formalization["schema"],
            "target": sealed_formalization["target"],
            "proofs": [
                {
                    "claim_id": item["claim_id"],
                    "term": item["term"],
                }
                for item in sealed_formalization["proofs"]
            ],
        }
        if sealed_formalization is not None else None
    )
    certificates, evidence_certificates = _validate_certificates(
        raw.get("certificates"),
    )
    normalized = {
        "schema": RESEARCH_SCHEMA,
        "scope": scope,
        "hypothesis": hypothesis,
        "construction": construction,
        "claims": claims,
        "falsification_plan": falsification_plan,
        "certificates": certificates,
    }
    if normalized_formalization is not None:
        normalized["formalization"] = normalized_formalization
    if proof_sketch is not None:
        normalized["proof_sketch"] = proof_sketch
    research_hash = _sha256_json(normalized)
    verified_count = sum(
        item["status"] == "bounded_checked" for item in evidence_certificates
    )
    refuted_count = sum(
        item["status"] == "refuted" for item in evidence_certificates
    )
    verified_obligations = sum(
        item["status"] == "bounded_checked"
        for item in obligation_outcomes.values()
    )
    refuted_obligations = sum(
        item["status"] == "refuted"
        for item in obligation_outcomes.values()
    )
    bounded_claims = sum(
        item["status"] == "bounded_supported"
        for item in evidence_claims
    )
    refuted_claims = sum(
        item["status"] == "refuted"
        for item in evidence_claims
    )
    certificate_summary = (
        "contains_mixed_certificates" if verified_count and refuted_count else
        "contains_refuted_certificate" if refuted_count else
        "contains_bounded_certificate" if verified_count else
        "none"
    )
    claim_context = "; ".join(
        f"{claim['id']} [{claim['template']}]: {claim['statement']}"
        for claim in claims
    )
    research_context = _clip_text(
        "\n".join(filter(None, (
            f"scope: {scope}",
            f"hypothesis: {hypothesis}",
            (
                "construction: digit_product "
                f"base={construction['base']} "
                f"digits={construction['digits']} "
                f"checked={construction['check_levels']}"
            ),
            f"claims: {claim_context}",
            f"falsification: {falsification_plan}",
            f"proof sketch: {proof_sketch}" if proof_sketch else "",
            (
                "formalization: submitted"
                if normalized_formalization is not None
                else "formalization: not submitted"
            ),
        ))),
        MAX_RESEARCH_CONTEXT_CHARS,
    )
    finite_certificate_context = "; ".join(
        (
            f"{item['status']} standalone {item['type']} "
            f"modulus={item['modulus']} expected(sum={item['expected_sum_count']},"
            f"diff={item['expected_diff_count']}) "
            f"trusted(sum={item['sum_count']},diff={item['diff_count']})"
        )
        for item in evidence_certificates
    )
    obligation_context = "; ".join(
        (
            f"{item['id']}={item['status']}[{item['type']}]"
            + (
                f" counterexample={item.get('counterexample')}"
                if item["status"] == "refuted" else ""
            )
        )
        for item in obligation_outcomes.values()
    )
    certificate_context = "; ".join(
        item for item in (obligation_context, finite_certificate_context)
        if item
    ) or "none"
    formalization_status = (
        "submitted" if normalized_formalization is not None
        else "not_submitted"
    )
    research_rank = (
        5 if (refuted_claims or refuted_obligations or refuted_count) else
        30 if normalized_formalization is not None else
        20 if (verified_obligations or verified_count) else
        10
    )
    metrics = {
        "evidence_level": (
            "proposal_with_refutation"
            if (refuted_claims or refuted_obligations or refuted_count) else
            "formalization_submitted"
            if normalized_formalization is not None else
            "proposal_with_bounded_support"
            if (verified_obligations or verified_count) else
            "proposal"
        ),
        "research_rank": research_rank,
        "research_claim_count": len(claims),
        "bounded_supported_claim_count": bounded_claims,
        "refuted_claim_count": refuted_claims,
        "formally_checked_claim_count": 0,
        "verified_obligation_count": verified_obligations,
        "refuted_obligation_count": refuted_obligations,
        "verified_certificate_count": verified_count,
        "refuted_certificate_count": refuted_count,
        "formalization_status": formalization_status,
        "formalization_request_sha256": (
            sealed_formalization.get("request_sha256")
            if sealed_formalization is not None else None
        ),
        "construction_sha256": construction_evidence[
            "construction_sha256"
        ],
        "research_sha256": research_hash,
        "research_hypothesis": hypothesis,
        "research_context": research_context,
        "certificate_context": certificate_context,
    }
    evidence = {
        "status": (
            "contains_refutation"
            if (refuted_claims or refuted_obligations or refuted_count) else
            "formalization_submitted"
            if normalized_formalization is not None else
            "bounded_supported"
            if (verified_obligations or verified_count) else
            "proposed"
        ),
        "certificate_summary": certificate_summary,
        "research_sha256": research_hash,
        "definition": RESEARCH_DEFINITION,
        "scope": scope,
        "claims": evidence_claims,
        "construction": construction_evidence,
        "certificates": evidence_certificates,
        "proof_sketch_status": "unverified" if proof_sketch else "not_submitted",
        "formalization": (
            {
                "target": "lean4",
                "status": "submitted",
                "request_sha256": sealed_formalization[
                    "request_sha256"
                ],
                "proofs": [
                    {
                        "claim_id": item["claim_id"],
                        "proof_sha256": item["proof_sha256"],
                    }
                    for item in sealed_formalization["proofs"]
                ],
            }
            if normalized_formalization is not None else
            {"target": "lean4", "status": "not_submitted"}
        ),
    }
    return normalized, metrics, evidence


def canonical_values(values):
    """Normalize translation, integer scale, and reflection symmetries."""
    vals = sorted(values)
    shifted = [x - vals[0] for x in vals]
    scale = reduce(gcd, shifted[1:], 0) or 1
    normalized = [x // scale for x in shifted]
    reflected = [normalized[-1] - x for x in reversed(normalized)]
    return min(normalized, reflected)


def canonical_hash(values):
    payload = json.dumps(canonical_values(values), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def evaluate_values(values):
    if not isinstance(values, list) or not values:
        raise ValueError('solution.json must contain a non-empty list "A"')
    normalized = []
    for index, value in enumerate(values):
        integer = _strict_int(value, path=f"A[{index}]")
        if integer < MIN_INT or integer > MAX_INT:
            raise ValueError(f"elements must be in [{MIN_INT}, {MAX_INT}]")
        normalized.append(integer)
    values = sorted(set(normalized))
    n = len(values)
    if n < MIN_N:
        raise ValueError(f"|A| must be at least {MIN_N}, got {n}")
    sumset = {a + b for a in values for b in values}
    diffset = {a - b for a in values for b in values}
    sums, diffs = len(sumset), len(diffset)
    if sums <= n or diffs <= n:
        raise ValueError("both |A+A|/|A| and |A-A|/|A| must be > 1")
    score = math.log(sums / n) / math.log(diffs / n)
    vals = sorted(values)
    return score, {
        "n": n,
        "sums": sums,
        "diffs": diffs,
        "span": vals[-1] - vals[0],
        "set_hash": canonical_hash(vals),
    }, values


def evaluate_submission(data):
    _strict_keys(
        data,
        required={"A"},
        allowed={"A", "research"},
        path="solution.json",
    )
    score, metrics, normalized_values = evaluate_values(data["A"])
    research, research_metrics, research_evidence = validate_research(
        data.get("research"),
    )
    metrics.update(research_metrics)
    research_hash = research_metrics.get("research_sha256")
    metrics["candidate_hash"] = (
        _sha256_json({
            "set_hash": metrics["set_hash"],
            "research_sha256": research_hash,
        })
        if research_hash else metrics["set_hash"]
    )
    normalized = {"A": normalized_values}
    if research is not None:
        normalized["research"] = research
    evidence = {
        "schema": "openhyra-evidence",
        "numeric": {
            "status": "exact",
            "n": metrics["n"],
            "sums": metrics["sums"],
            "diffs": metrics["diffs"],
            "score": score,
            "set_hash": metrics["set_hash"],
        },
        "research": research_evidence,
        "candidate_hash": metrics["candidate_hash"],
    }
    return score, metrics, normalized, evidence


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main():
    target = Path(sys.argv[1])
    solution_path = target / "solution.json" if target.is_dir() else target
    if not solution_path.exists():
        fail("solution.json not found")
    try:
        data = json.loads(
            solution_path.read_text(),
            object_pairs_hook=_unique_object,
        )
        score, metrics, normalized, evidence = evaluate_submission(data)
    except (OSError, RecursionError, ValueError, TypeError) as exc:
        fail(str(exc))
    # Preserve the full IEEE-754 double in the Experience Bank. Formatting is
    # a presentation concern and must not alter parent selection.
    print(json.dumps({
        "score": score,
        "metrics": metrics,
        "normalized_solution": normalized,
        "evidence": evidence,
    }))


if __name__ == "__main__":
    main()
