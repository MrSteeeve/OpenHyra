from __future__ import annotations
import hashlib
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
        source_description: str = "",
    ) -> ControlPair:
        guided_suffix = (
            f"\n## Transfer insight (from analogy hypothesis {hypothesis.id})\n\n"
            f"Apply this technique from source algorithm(s) {', '.join(hypothesis.source_record_ids)}:\n"
            f"{hypothesis.transfer_description}\n"
            f"Allowed operators: {', '.join(hypothesis.operators)}\n"
        )
        if source_description:
            guided_suffix += f"\nSource context: {source_description}\n"

        control_suffix = (
            f"\n## Exploration guidance\n\n"
            f"Explore a novel improvement to the current approach.\n"
            f"Focus on the same problem region as the parent solution.\n"
        )

        return ControlPair(
            hypothesis_id=hypothesis.id,
            guided_prompt_suffix=guided_suffix,
            control_prompt_suffix=control_suffix,
            shared_parent_id=parent_id,
            shared_seed=seed,
        )

    @staticmethod
    def evaluate_pair(pair: ControlPair, direction: str = "max") -> AnalogyResult:
        if pair.guided_score is None or pair.control_score is None:
            return AnalogyResult(
                analogy_hypothesis_id=pair.hypothesis_id,
                guided_record_id=pair.guided_record_id,
                control_record_id=pair.control_record_id,
                guided_score=pair.guided_score,
                control_score=pair.control_score,
                verdict="execution_failed",
                effect_size=0.0,
            )

        if direction == "min":
            raw_effect = pair.control_score - pair.guided_score
        else:
            raw_effect = pair.guided_score - pair.control_score

        denominator = max(abs(pair.control_score), 1e-12)
        effect_size = raw_effect / denominator

        verdict = "transfer_supported" if effect_size > 0.01 else "transfer_refuted"

        return AnalogyResult(
            analogy_hypothesis_id=pair.hypothesis_id,
            guided_record_id=pair.guided_record_id,
            control_record_id=pair.control_record_id,
            guided_score=pair.guided_score,
            control_score=pair.control_score,
            verdict=verdict,
            effect_size=effect_size,
        )


class ControlPairStore:
    """Persists control pairs as JSONL."""

    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, pair: ControlPair) -> None:
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(pair.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")

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
