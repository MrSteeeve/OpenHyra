from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
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
        paired_effect: Mapping[str, Any] | None = None,
    ) -> AnalogyResult:
        """Contrast guided and control arms.

        ``paired_effect`` is the evaluator-cell contrast from
        ``bermudan_research.paired_slice_effects`` (Monte Carlo SE per cell).
        It is the only source of a decisive verdict, so the matched-control
        verdict and the prediction test cannot disagree.  Without it the
        scalar score contrast is reported as inconclusive.
        """
        if direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")

        invalid_reason = MatchedControlBuilder.control_invalid_reason(pair)
        if (
            not invalid_reason
            and isinstance(paired_effect, Mapping)
            and paired_effect.get("status") == "invalid_control"
        ):
            invalid_reason = str(paired_effect.get("reason") or "invalid_control")
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

        observed = (
            isinstance(paired_effect, Mapping)
            and paired_effect.get("status") == "observed"
        )
        if observed:
            # Summary-row effects are candidate-minus-baseline in maximisation
            # orientation; flip them (and the interval) for minimisation so
            # every field and the verdict share the task direction.
            sign = -1.0 if direction == "min" else 1.0
            transfer_gain = sign * float(paired_effect["effect"])
            transfer_gain_se = float(paired_effect["standard_error"])
            ci_low, ci_high = sorted(sign * float(v) for v in paired_effect["ci95"])
            cell_count = len(paired_effect.get("cells", []))
        else:
            transfer_gain = guided_delta - control_delta
            transfer_gain_se = 0.0
            ci_low = ci_high = None
            cell_count = 0
        denominator = max(abs(control_delta), 1e-12)
        relative_gain = transfer_gain / denominator

        prediction_correct = transfer_gain > 0.0
        if invalid_reason:
            verdict = "invalid_control"
        elif not observed:
            verdict = "inconclusive"
        elif ci_low > 0.0:
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
            paired_cell_count=cell_count,
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
            ):
                if key in metadata and metadata[key] is False:
                    reasons.append(label)
        return ";".join(dict.fromkeys(reasons)) or None


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
