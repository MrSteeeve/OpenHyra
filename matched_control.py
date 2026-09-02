from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
                verdict="execution_failed",
            )

        baseline = pair.baseline_score if pair.baseline_score is not None else 0.0
        if direction == "min":
            guided_delta = baseline - pair.guided_score
            control_delta = baseline - pair.control_score
        else:
            guided_delta = pair.guided_score - baseline
            control_delta = pair.control_score - baseline

        transfer_gain = guided_delta - control_delta
        denominator = max(abs(control_delta), 1e-12)
        transfer_gain_se = abs(transfer_gain) / denominator

        prediction_correct = transfer_gain > 0.0
        verdict = (
            "transfer_supported" if transfer_gain > 0.01 else "transfer_refuted"
        )

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
        )


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
