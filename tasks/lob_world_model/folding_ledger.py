"""Full proposal receipts and a token-bounded, objective lineage projection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from difflib import unified_diff
import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Any


NLL = "val_nll_total"
EXHAUST = "exhausted_replicate_rate"
CF = "counterfactual_impact_mae"
ACF = "sign_acf_abs_error"
MEM = "memory_gap_64"
_ALIASES = {ACF: (ACF, "direction_acf_mae"), MEM: (MEM, "G_mem(64)", "G_mem_64")}
_LABELS = re.compile(r"(?<![A-Za-z0-9])(?:w2|w6|thin|thick|hawkes|long)(?![A-Za-z0-9])", re.IGNORECASE)
_UNEVALUATED = {"DUPLICATE_REJECTED", "INVALID_PROPOSAL", "PROMPT_CAPACITY_REJECTED",
                "PROVIDER_ERROR", "INTERRUPTED_CALL"}


# --- Symmetric decisions, based on paired estimates when provided ---

def _metric(row: Mapping[str, Any], key: str, *, require_support: bool = False) -> float | None:
    for name in _ALIASES.get(key, (key,)):
        value = row.get(name)
        if isinstance(value, Mapping):
            days = value.get("paired_days")
            inadequate = "paired_days" in value and (not isinstance(days, (int, float)) or days < 6)
            if require_support and (inadequate or ("interval" in value and value["interval"] is None)):
                continue
            value = value.get("estimate")
        if isinstance(value, (int, float)) and isfinite(value):
            return float(value)
    return None


def _delta(parent: Mapping[str, Any], candidate: Mapping[str, Any], paired: Mapping[str, Any] | None,
           key: str) -> float | None:
    if paired is not None:
        return _metric(paired, key, require_support=True)
    left, right = _metric(parent, key), _metric(candidate, key)
    return None if left is None or right is None else right - left


def promotion_decision(parent: Mapping[str, Any], candidate: Mapping[str, Any],
                       paired_deltas: Mapping[str, Any] | None = None) -> str:
    """Return PROMOTED, STEPPING_STONE or REJECTED after contract validity.

    Deltas are candidate minus parent. Prefer paired day estimates; unavailable
    estimates require six paired days and a reported interval when specified;
    unavailable estimates are not imputed, including absent CF response.
    These are fixed decision thresholds, not a significance or equivalence
    test of a confidence interval. NLL in nats, CF in ticks, exhaust in [0,1].
    """
    nll = _delta(parent, candidate, paired_deltas, NLL)
    exhaust = _delta(parent, candidate, paired_deltas, EXHAUST)
    cf = _delta(parent, candidate, paired_deltas, CF)
    if nll is None or nll > .025 or _metric(candidate, NLL) is None:
        return "REJECTED"
    if exhaust is not None and cf is not None and nll <= .005 and exhaust <= .05:
        if cf <= -.010 or (abs(cf) < .010 and (exhaust <= -.10 or nll <= -.008)):
            return "PROMOTED"
    dt = _delta(parent, candidate, paired_deltas, "dt_nll")
    acf = _delta(parent, candidate, paired_deltas, ACF)
    base_acf = _metric(parent, ACF)
    mem = _delta(parent, candidate, paired_deltas, MEM)
    acf_gain = acf is not None and base_acf is not None and base_acf > 0 and acf <= -.25 * base_acf
    if (dt is not None and dt <= -.010 or exhaust is not None and exhaust <= -.10
            or acf_gain or mem is not None and abs(mem) >= .05):
        return "STEPPING_STONE"
    return "REJECTED"


def _structural_diff(parent: str, source: str) -> tuple[str, str]:
    """Return the full unified diff and an objective one-line edit summary."""
    diff = list(unified_diff(parent.splitlines(), source.splitlines(), fromfile="parent", tofile="candidate"))
    added = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]
    removed = [line[1:] for line in diff if line.startswith("-") and not line.startswith("---")]
    calls = r"\b(?:def\s+|class\s+)?([A-Za-z_]\w*)\s*\("
    old_names = set(re.findall(calls, "\n".join(removed)))
    new_names = set(re.findall(calls, "\n".join(added)))
    # Historical models can choose arbitrary identifiers; the optional ledger
    # context never emits literal generator labels or prose from their replies.
    labels = re.compile(r"W[26]|thin|thick|hawkes|long", re.IGNORECASE)
    plus = [name for name in sorted(new_names - old_names) if not labels.search(name)][:5]
    minus = [name for name in sorted(old_names - new_names) if not labels.search(name)][:5]
    summary = f"+{len(added)}/-{len(removed)} lines, calls/defs +{','.join(plus) or '-'} -{','.join(minus) or '-'}"
    return "\n".join(diff), summary


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _component(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("candidate ID must be a single safe component")
    return value


# --- Durable full evidence and compact projection ---

class FoldingLedger:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "ledger.jsonl"
        self.records: list[dict[str, Any]] = []
        if self.path.exists():
            self.records = [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Persist source, prompt, reply, prediction, code diff and paired results once per ID."""
        row = dict(record)
        cand_id = _component(row["cand_id"])
        if any(old["cand_id"] == cand_id for old in self.records):
            raise ValueError(f"ledger already contains {cand_id}")
        sources = {old["cand_id"]: old.get("source") for old in self.records}
        parent_source = row.pop("parent_source", None) or sources.get(row.get("parent_id")) or ""
        source = row.get("source")
        full_diff, summary = _structural_diff(parent_source, source or "")
        row["structural_diff"] = summary if source else "no parsed source"
        row["code_diff"] = full_diff if source else ""
        encoded = _json(row)  # Validate serializability before touching the journal.
        folder = self.root / "candidates" / cand_id
        folder.mkdir(parents=True, exist_ok=True)
        if isinstance(source, str):
            path = folder / "candidate.py"
            if path.exists() and path.read_text() != source:
                raise ValueError(f"candidate receipt conflicts: {cand_id}")
            path.write_text(source, encoding="utf-8")
        for key in ("prompt", "raw_reply", "prediction", "levels", "deltas", "paired_results"):
            if key in row:
                (folder / f"{key}.json").write_text(_json(row[key]) + "\n", encoding="utf-8")
        (folder / "candidate.diff").write_text(row["code_diff"], encoding="utf-8")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(row)
        return row

    def evaluated_sources(self) -> dict[str, str]:
        return {row["cand_id"]: row["source"] for row in self.records
                if row.get("source") and row["outcome"] not in _UNEVALUATED}

    @staticmethod
    def _line(row: Mapping[str, Any]) -> str:
        deltas = row.get("deltas") or {}
        values = []
        for metric in (NLL, "dt_nll", CF, EXHAUST, MEM):
            delta = _metric(deltas, metric, require_support=True)
            values.append("NA" if delta is None else f"{delta:+.4g}")
        capacity = f"{row.get('parameters', 'NA')}/{row.get('state_scalars', 'NA')}"
        line = " | ".join([row["cand_id"], row.get("parent_id") or "-", row["outcome"], capacity,
                           *values, row["structural_diff"]])
        return _LABELS.sub("[label omitted]", line)

    @staticmethod
    def _stone_score(row: Mapping[str, Any]) -> float:
        deltas = row.get("deltas") or {}
        return max(-(_metric(deltas, "dt_nll", require_support=True) or 0) / .010,
                   -(_metric(deltas, EXHAUST, require_support=True) or 0) / .10,
                   abs(_metric(deltas, MEM, require_support=True) or 0) / .05,
                   -(_metric(deltas, ACF, require_support=True) or 0) / .025)

    @staticmethod
    def _taboo(row: Mapping[str, Any]) -> bool:
        if row["outcome"] == "DUPLICATE_REJECTED":
            return True
        if row["outcome"] != "REJECTED":
            return False
        deltas = row.get("deltas") or {}
        return any((_metric(deltas, name, require_support=True) or 0) > bound for name, bound in
                   ((NLL, .005), (EXHAUST, .05), (CF, .010)))

    def render(self, parent_id: str, unfold_ids: Sequence[str] = (), *, max_tokens: int,
               token_count: Callable[[str], int]) -> str:
        """Fold only optional history to a strict remaining-token budget.

        The caller accounts for parent source, residuals, cards, Z block and
        provider framing separately (then checks the complete request). The
        newest L0 rows and current-branch taboos outrank optional node details.
        """
        rows = [row for row in self.records if row["outcome"] != "SEED"]
        by_id = {row["cand_id"]: row for row in self.records}
        ancestors = {parent_id}
        cursor = parent_id
        while cursor in by_id and by_id[cursor].get("parent_id"):
            previous = by_id[cursor]["parent_id"]
            if previous in ancestors:
                raise ValueError("cyclic candidate lineage")
            ancestors.add(previous)
            cursor = previous
        headings = ["[L0 lineage: id | parent | outcome | params/state | ΔNLL | Δdt | ΔCF | Δexhaust | ΔG64 | edit]",
                    "[Branch taboos]", "[Stepping stones]", "[Requested nodes]"]
        sections: list[list[str]] = [[] for _ in headings]

        def text() -> str:
            return "\n".join("\n".join([header, *(lines or ["none"])])
                             for header, lines in zip(headings, sections, strict=True))

        if token_count(text()) > max_tokens:
            return ""

        def offer(index: int, line: str) -> None:
            sections[index].append(line)
            if token_count(text()) > max_tokens:
                sections[index].pop()

        taboos = [row for row in rows if row.get("parent_id") in ancestors and self._taboo(row)]
        for row in reversed(taboos[-5:]):
            offer(1, self._line(row))
        for row in reversed(rows):
            offer(0, self._line(row))
        stones = sorted((row for row in rows if row["outcome"] == "STEPPING_STONE"),
                        key=lambda row: (-self._stone_score(row), row["cand_id"]))
        for row in stones[:3]:
            offer(2, self._line(row))
        for cand_id in dict.fromkeys(unfold_ids[:2]):
            if cand_id in by_id and by_id[cand_id]["outcome"] != "SEED":
                row = by_id[cand_id]
                paired = row.get("deltas") or {}
                details = {key: _metric(paired, key, require_support=True) for key in (NLL, "dt_nll", CF, EXHAUST, ACF, MEM)}
                offer(3, self._line(row) + "\nPaired estimates: " + _json(details))
        return text()
