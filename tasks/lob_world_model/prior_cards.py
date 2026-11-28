"""Open reference catalog of ten structural hypotheses, shown in full every round."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re


CARD_PATH = Path(__file__).with_name("prior_cards.json")


@dataclass(frozen=True)
class PriorCard:
    card_id: str
    title: str
    citation: str
    mechanism_formula: str
    regime_boundary: str
    numerical_guardrails: str

    def render(self) -> str:
        # URLs stay in the on-disk catalog; they add no mathematical content to the prompt.
        citation = re.sub(r",\s*https?://[^;]+", "", self.citation)
        return (f"{self.card_id} {self.title} ({citation}): {self.mechanism_formula}\n"
                f"Boundary: {self.regime_boundary}\nGuard: {self.numerical_guardrails}")


def load_cards(path: Path = CARD_PATH) -> tuple[PriorCard, ...]:
    return tuple(PriorCard(**row) for row in json.loads(path.read_text(encoding="utf-8")))


def shuffled_cards(env_id: str, replicate_id: str, round_idx: int) -> list[PriorCard]:
    """All cards in a deterministic order shared by the information arms of one round."""
    cards = list(load_cards())
    random.Random(f"{env_id}:{replicate_id}:{round_idx}").shuffle(cards)
    return cards
