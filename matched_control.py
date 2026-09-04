from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping

from schemas_v5 import AnalogyHypothesis, AnalogyResult


@dataclass
class ControlPair:
    """One matched pair for an analogy experiment."""

    hypothesis_id: str
    guided_prompt_suffix: str
    control_prompt_suffix: str
    shared_parent_id: str
    shared_seed: int
    guided_record_id: str = ""
    control_record_id: str = ""
    guided_score: float | None = None
    control_score: float | None = None
    baseline_score: float | None = None
    # Optional evaluator-owned per-cell outcomes.  ``paired_cells`` accepts
    # rows such as ``{"cell_id": ..., "guided_delta": ...,
    # "control_delta": ...}``; the explicit delta lists are convenient for
    # callers that already performed the join.  Legacy scalar pairs remain
    # valid and are treated as one observed cell (SE = 0).
    paired_cells: list[dict[str, Any]] = field(default_factory=list)
    guided_cell_deltas: list[float] = field(default_factory=list)
    control_cell_deltas: list[float] = field(default_factory=list)
    paired_cell_deltas: list[float] = field(default_factory=list)
    confidence_level: float = 0.95
    # Optional identity metadata lets the evaluator distinguish a genuine
    # matched control from a malformed pair without introducing another audit
    # subsystem.  Missing legacy fields are not considered invalid.
    guided_parent_id: str | None = None
    control_parent_id: str | None = None
    guided_seed: int | None = None
    control_seed: int | None = None
    guided_compute_budget: float | None = None
    control_compute_budget: float | None = None
    control_metadata: dict[str, Any] = field(default_factory=dict)
    invalid_control_reason: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> ControlPair:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MatchedControlBuilder:
    """Builds matched control pairs from an AnalogyHypothesis."""

    @staticmethod
    def build_pair(
        hypothesis: AnalogyHypothesis,
        parent_id: str,
        seed: int,
    ) -> ControlPair:
        relation_summary = "; ".join(
            f"{m['source_role']}→{m['target_role']} ({m['shared_relation']})"
            for m in hypothesis.relation_mapping
        )
        guided_suffix = (
            f"\n## Transfer insight (from analogy hypothesis {hypothesis.id})\n\n"
            f"Apply this technique from source algorithm(s) "
            f"{', '.join(hypothesis.source_record_ids)}:\n"
            f"{hypothesis.transferable_intervention}\n"
            f"Relation mapping: {relation_summary}\n"
        )

        control_suffix = (
            "\n## Exploration guidance\n\n"
            "Explore a novel improvement to the current approach.\n"
            "Focus on the same problem region as the parent solution.\n"
        )

        return ControlPair(
            hypothesis_id=hypothesis.id,
            guided_prompt_suffix=guided_suffix,
            control_prompt_suffix=control_suffix,
            shared_parent_id=parent_id,
            shared_seed=seed,
        )

    @staticmethod
    def evaluate_pair(
        pair: ControlPair,
        direction: str = "max",
    ) -> AnalogyResult:
        if direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")

        invalid_reason = MatchedControlBuilder.control_invalid_reason(pair)
        if pair.guided_score is None or pair.control_score is None:
            return AnalogyResult(
                analogy_hypothesis_id=pair.hypothesis_id,
                guided_record_id=pair.guided_record_id,
                control_record_id=pair.control_record_id,
                guided_delta=0.0,
                control_delta=0.0,
                transfer_gain=0.0,
                transfer_gain_standard_error=0.0,
                predicted_slice_effect=0.0,
                prediction_direction_correct=False,
                verdict="invalid_control" if invalid_reason else "execution_failed",
                paired_cell_count=0,
                invalid_control_reason=invalid_reason,
                control_valid=False if invalid_reason else None,
            )

        baseline = pair.baseline_score if pair.baseline_score is not None else 0.0
        if direction == "min":
            guided_delta = baseline - pair.guided_score
            control_delta = baseline - pair.control_score
        else:
            guided_delta = pair.guided_score - baseline
            control_delta = pair.control_score - baseline

        # Prefer evaluator-produced per-cell deltas.  If an integration only
        # supplied the scalar summaries, retain a one-cell observation so the
        # reported SE is honest (zero degrees of freedom), rather than the old
        # relative-gain ratio that was mislabeled as a standard error.
        paired_values = MatchedControlBuilder._paired_values(pair)
        if paired_values:
            if direction == "min":
                # Summary rows are conventionally candidate-baseline (a
                # maximisation orientation).  Reverse the paired effect for
                # minimisation tasks before computing its statistics.
                paired_values = [-value for value in paired_values]
            transfer_gain = float(sum(paired_values) / len(paired_values))
            if len(paired_values) > 1:
                mean = transfer_gain
                variance = sum((value - mean) ** 2 for value in paired_values) / (
                    len(paired_values) - 1
                )
                transfer_gain_se = math.sqrt(variance / len(paired_values))
            else:
                transfer_gain_se = 0.0
        else:
            paired_values = [guided_delta - control_delta]
            transfer_gain = paired_values[0]
            transfer_gain_se = 0.0

        confidence = min(max(float(pair.confidence_level), 0.5), 0.999999)
        z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        ci_low = transfer_gain - z * transfer_gain_se
        ci_high = transfer_gain + z * transfer_gain_se
        denominator = max(abs(control_delta), 1e-12)
        relative_gain = transfer_gain / denominator

        prediction_correct = transfer_gain > 0.0
        if invalid_reason:
            verdict = "invalid_control"
        elif ci_low > 0.01:
            verdict = "transfer_supported"
        elif ci_high < 0.0:
            verdict = "transfer_refuted"
        else:
            verdict = "inconclusive"

        return AnalogyResult(
            analogy_hypothesis_id=pair.hypothesis_id,
            guided_record_id=pair.guided_record_id,
            control_record_id=pair.control_record_id,
            guided_delta=guided_delta,
            control_delta=control_delta,
            transfer_gain=transfer_gain,
            transfer_gain_standard_error=transfer_gain_se,
            predicted_slice_effect=transfer_gain,
            prediction_direction_correct=prediction_correct,
            verdict=verdict,
            paired_cell_count=len(paired_values),
            transfer_gain_ci_low=ci_low,
            transfer_gain_ci_high=ci_high,
            relative_transfer_gain=relative_gain,
            invalid_control_reason=invalid_reason,
            control_valid=False if invalid_reason else True,
        )

    @staticmethod
    def control_invalid_reason(pair: ControlPair) -> str | None:
        """Return a concise reason when matched-control identity is broken.

        Explicit mismatches are invalid.  Missing optional metadata is left
        untouched for compatibility with archived scalar pairs.
        """
        reasons: list[str] = []
        if pair.invalid_control_reason:
            reasons.append(str(pair.invalid_control_reason))
        if (
            pair.guided_parent_id is not None
            and pair.control_parent_id is not None
            and pair.guided_parent_id != pair.control_parent_id
        ):
            reasons.append("parent_mismatch")
        if (
            pair.guided_seed is not None
            and pair.control_seed is not None
            and pair.guided_seed != pair.control_seed
        ):
            reasons.append("seed_mismatch")
        if (
            pair.guided_compute_budget is not None
            and pair.control_compute_budget is not None
            and not math.isclose(
                float(pair.guided_compute_budget),
                float(pair.control_compute_budget),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            reasons.append("compute_budget_mismatch")
        metadata = pair.control_metadata
        if isinstance(metadata, Mapping):
            for key, label in (
                ("same_parent", "parent_mismatch"),
                ("same_seed", "seed_mismatch"),
                ("same_compute_budget", "compute_budget_mismatch"),
            ):
                if key in metadata and metadata[key] is False:
                    reasons.append(label)
        return ";".join(dict.fromkeys(reasons)) or None

    @staticmethod
    def _paired_values(pair: ControlPair) -> list[float]:
        if pair.paired_cell_deltas:
            values = pair.paired_cell_deltas
        elif pair.paired_cells:
            values = []
            for cell in pair.paired_cells:
                if not isinstance(cell, Mapping):
                    continue
                value = cell.get("paired_delta", cell.get("transfer_gain"))
                if value is None:
                    guided = cell.get("guided_delta")
                    control = cell.get("control_delta")
                    if guided is not None and control is not None:
                        value = float(guided) - float(control)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
        elif (
            pair.guided_cell_deltas
            and pair.control_cell_deltas
            and len(pair.guided_cell_deltas) == len(pair.control_cell_deltas)
        ):
            values = [
                float(guided) - float(control)
                for guided, control in zip(
                    pair.guided_cell_deltas, pair.control_cell_deltas
                )
                if math.isfinite(float(guided)) and math.isfinite(float(control))
            ]
        else:
            values = []
        return values

    @staticmethod
    def attach_per_cell_summaries(
        pair: ControlPair,
        guided_metrics: Mapping[str, Any] | None,
        control_metrics: Mapping[str, Any] | None,
    ) -> ControlPair:
        """Join evaluator summaries by ``(instance_id, repeat)``.

        The function mutates and returns ``pair`` for ergonomic use in the
        existing finalizer.  It accepts normalized improvement fields emitted
        by the Bermudan evaluator and falls back to lower-bound minus baseline
        values when replaying older records.
        """
        guided_rows = MatchedControlBuilder._summary_rows(guided_metrics)
        control_rows = MatchedControlBuilder._summary_rows(control_metrics)
        common = sorted(set(guided_rows).intersection(control_rows))
        cells: list[dict[str, Any]] = []
        guided_deltas: list[float] = []
        control_deltas: list[float] = []
        paired_deltas: list[float] = []
        for key in common:
            guided = guided_rows[key]
            control = control_rows[key]
            guided_delta = MatchedControlBuilder._row_delta(guided)
            control_delta = MatchedControlBuilder._row_delta(control)
            if guided_delta is None or control_delta is None:
                continue
            paired_delta = guided_delta - control_delta
            cells.append({
                "cell_id": f"{key[0]}::{key[1]}",
                "instance_id": key[0],
                "repeat": key[1],
                "guided_delta": guided_delta,
                "control_delta": control_delta,
                "paired_delta": paired_delta,
            })
            guided_deltas.append(guided_delta)
            control_deltas.append(control_delta)
            paired_deltas.append(paired_delta)
        pair.paired_cells = cells
        pair.guided_cell_deltas = guided_deltas
        pair.control_cell_deltas = control_deltas
        pair.paired_cell_deltas = paired_deltas
        return pair

    @staticmethod
    def _summary_rows(metrics: Mapping[str, Any] | None) -> dict[tuple[str, int], Mapping[str, Any]]:
        if not isinstance(metrics, Mapping):
            return {}
        rows = metrics.get("summaries")
        if not isinstance(rows, list):
            return {}
        result: dict[tuple[str, int], Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            instance_id = row.get("instance_id")
            if not instance_id:
                continue
            try:
                repeat = int(row.get("repeat", 0))
            except (TypeError, ValueError):
                continue
            result[(str(instance_id), repeat)] = row
        return result

    @staticmethod
    def _row_delta(row: Mapping[str, Any]) -> float | None:
        value = row.get("paired_normalized_improvement")
        if value is None:
            candidate = row.get("candidate_lower_bound")
            baseline = row.get("baseline_lower_bound")
            if candidate is None or baseline is None:
                return None
            value = float(candidate) - float(baseline)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None


class ControlPairStore:
    """Persists control pairs as JSONL."""

    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, pair: ControlPair) -> None:
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(pair.to_dict(), sort_keys=True, ensure_ascii=False)
                + "\n"
            )

    def load_all(self) -> list[ControlPair]:
        if not self.store_path.exists():
            return []
        pairs = []
        for line in self.store_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                pairs.append(ControlPair.from_dict(json.loads(line)))
        return pairs

    def find_by_hypothesis(self, hypothesis_id: str) -> list[ControlPair]:
        return [p for p in self.load_all() if p.hypothesis_id == hypothesis_id]
