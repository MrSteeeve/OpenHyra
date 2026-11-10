"""Development-only mechanism hypotheses, with executed public witnesses.

Profiles describe cashflow roles, not market novelty. Constants and positive
scales do not create profiles; public probes can refute a proposed distinction.
"""
from __future__ import annotations

import json

from .search_feedback import choose_parent


TOLERANCE = 1e-8
RESEARCH_GOAL = "feasible-mechanism-discovery.v1"


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def mechanism_profile(artifact):
    """Parameter-erased payment rules; canonicalize commutative compositions.

This is a conservative research taxonomy, not a semantic equivalence prover.
"""
    payload = artifact["contract"]
    body = payload["program"]["body"]
    params = payload.get("params", {})

    def scalar(value):
        return params[value["param"]] if isinstance(value, dict) and "param" in value else value

    def expression(node):
        if isinstance(node, (list, tuple)):
            return [expression(item) for item in node]
        if not isinstance(node, dict) or "param" in node:
            return "constant"
        op, args = node["op"], node.get("args", [])
        if op == "mul_const":
            factor = scalar(args[0])
            if factor == 0:
                return "constant"
            item = expression(args[1])
            return item if factor > 0 else ["negative", item]
        if op == "lin":
            slope = scalar(args[0])
            return "constant" if slope == 0 else ["affine-up" if slope > 0 else "affine-down", expression(args[1])]
        if op in {"add", "and", "or", "max", "min"}:
            parts = []
            for arg in args:
                item = expression(arg)
                parts.extend(item[1:] if isinstance(item, list) and item[0] == op else [item])
            # Repeated additive roles can make tiers; their execution must
            # still differ before a new group is admitted. Boolean/max/min
            # repetitions are idempotent.
            ordered = sorted(parts, key=_key) if op == "add" else sorted(
                {_key(item): item for item in parts}.values(), key=_key)
            return ordered[0] if len(ordered) == 1 else [op, *ordered]
        return [op, *[expression(arg) for arg in args]]

    def contains_termination(node):
        return isinstance(node, dict) and (node.get("op") == "terminate_if" or
            any(contains_termination(arg) for arg in node.get("args", [])))

    payments = []

    def visit(node, guards=(), direction=1):
        op, args = node["op"], node.get("args", [])
        if op in {"premium", "both"}:
            for arg in args:
                visit(arg, guards, direction)
        elif op == "scale":
            scale = scalar(args[0])
            if scale:
                visit(args[1], guards, direction if scale > 0 else -direction)
        elif op == "give":
            visit(args[0], guards, -direction)
        elif op == "funded":
            if scalar(args[0]):
                payments.append(dict(time="termination-or-maturity" if contains_termination(args[2]) else "maturity",
                    guards=list(guards), direction=direction, amount="constant"))
            visit(args[2], guards, direction)
        elif op in {"when", "until", "activate_if"}:
            visit(args[-1], guards + ([op, expression(args[0]), args[1]],), direction)
        elif op in {"pay", "terminate_if"}:
            guard = list(guards)
            if op == "terminate_if":
                guard.append(["termination", expression(args[0]), args[1]])
            payments.append(dict(time="automatic-termination" if op == "terminate_if" else args[1],
                guards=sorted(guard, key=_key), direction=direction,
                amount=expression(args[2] if op == "terminate_if" else args[0])))
        else:
            raise ValueError(f"unclassified cashflow operation: {op}")

    visit(body)
    return sorted(payments, key=_key)


def probe_shape(probes):
    """Remove constant payment offsets and a single positive global scale.

Keep payment-month locations separately: early cash and maturity cash differ.
Do not normalize each leg separately, which would erase relative participation.
"""
    cells = [row["cashflows"] for row in probes]
    columns = list(zip(*cells))
    means = [sum(column) / len(column) for column in columns]
    centered = [[value - mean for value, mean in zip(row, means)] for row in cells]
    size = max(abs(value) for row in centered for value in row)
    return {"shape": [[value / size if size > TOLERANCE else 0. for value in row] for row in centered],
            "payment_months": [i + 1 for i, column in enumerate(columns)
                               if any(abs(value) > TOLERANCE for value in column)]}


def difference_witness(left, right):
    """Find an executed contrast after excluding global payoff amplification."""
    a, b = probe_shape(left), probe_shape(right)
    for i, (row_a, row_b) in enumerate(zip(a["shape"], b["shape"])):
        for month, (x, y) in enumerate(zip(row_a, row_b), 1):
            if abs(x - y) > TOLERANCE or a["payment_months"] != b["payment_months"]:
                return {"probe": left[i]["name"], "spot_ratios": left[i]["spot_ratios"],
                        "month": month, "candidate_cashflows": left[i]["cashflows"],
                        "reference_cashflows": right[i]["cashflows"],
                        "normalized_difference": abs(x - y),
                        "payment_months_differ": a["payment_months"] != b["payment_months"]}
    return None


class MechanismArchive:
    def __init__(self, references):
        self.references = references
        self.groups = []
        self.assignments = {}
        self.parent_uses = {}
        self.unresolved = []

    def add(self, cid, artifact, result):
        if result.score is None:
            return  # Constraint failures remain in experience, not discoveries.
        probes = result.metrics.get("public_cashflow_probes")
        if not probes:
            self.unresolved.append({"candidate_id": cid, "reason": "no executed public cashflow evidence"})
            return
        profile = mechanism_profile(artifact)
        prior = next((group for group in self.groups if group["profile"] == profile), None)
        witnesses = []
        if prior is None:
            for group in self.groups:
                witness = difference_witness(probes, group["first_probes"])
                if witness is None:
                    prior = group
                    break  # A different syntax with identical probes is not promoted.
                witnesses.append({"reference_candidate": group["first_candidate"], **witness})
        if prior is None:
            known = [row["name"] for row in self.references if row["profile"] == profile]
            prior = {"mechanism_id": f"mechanism-{len(self.groups):03d}", "profile": profile,
                "members": [], "first_candidate": cid, "representative": cid,
                "first_probes": probes, "representative_score": result.score,
                "matched_references": known, "contrast_witnesses": witnesses,
                "novelty_status": "reference-profile" if known else "unmatched-profile-hypothesis"}
            self.groups.append(prior)
        prior["members"].append(cid)
        self.assignments[cid] = prior["mechanism_id"]
        if result.score > prior["representative_score"]:
            prior.update(representative=cid, representative_score=result.score)

    def parent(self, results):
        if not self.groups:
            return choose_parent(results, 0, "incumbent", allow_infeasible=True)
        group = min(self.groups, key=lambda group: self.parent_uses.get(group["mechanism_id"], 0))
        mid = group["mechanism_id"]
        self.parent_uses[mid] = self.parent_uses.get(mid, 0) + 1
        return {"candidate_id": group["representative"], "selection": "least-explored-mechanism",
                "mechanism_id": mid, "allocated_proposals": self.parent_uses[mid]}

    def public_context(self):
        return [{key: group[key] for key in ("mechanism_id", "profile", "representative", "matched_references", "novelty_status")}
                for group in self.groups]

    def report(self):
        return {"schema": "stochopia-mechanism-archive.v1", "research_goal": RESEARCH_GOAL,
            "mechanisms": self.groups, "assignments": self.assignments,
            "parent_uses": self.parent_uses, "unresolved_candidates": self.unresolved,
            "scope": "development-only cashflow-rule profiles with finite public witnesses; not a proof of semantic equivalence or market novelty"}


def discovery_brief(index):
    tracks = ("mechanism-explore", "mechanism-refine", "mechanism-recombine")
    return {"track": tracks[index % 3], "instructions": {
        "mechanism-explore": "Propose a distinct cashflow mechanism relative to prior mechanisms when available. Terminal-only innovations and path-dependent innovations are both welcome.",
        "mechanism-refine": "Improve the supplied mechanism's feasibility or economic usefulness while preserving its distinguishing cashflow rules. It need not outperform other mechanisms' mean returns.",
        "mechanism-recombine": "Combine or reorganize cashflow rules into a different economic mechanism; avoid adding legs merely to increase leverage or syntax length.",
        }[tracks[index % 3]] + " Keep the public demand and loss constraints. Explain a concrete contrast scenario and trade-off in rationale. A new name, parameter or global scale alone does not establish a mechanism discovery. Use existing DSL syntax; return one executable contract."}
