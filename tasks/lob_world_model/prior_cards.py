"""Bounded structural hypotheses awakened exclusively by model residuals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path


ResidualSummary = Mapping[str, float | None]
CARD_PATH = Path(__file__).with_name("prior_cards.json")
GEOMETRY = frozenset({"C01", "C04", "C05", "C07", "C09"})
TIME_MEMORY = frozenset({"C03", "C06", "C08"})


@dataclass(frozen=True)
class PriorCard:
    card_id: str
    title: str
    citation: str
    symptom_key: str
    mechanism_formula: str
    regime_boundary: str
    numerical_guardrails: str

    def render(self) -> str:
        """Return a neutral mechanism, boundary and guardrail, never an environment label."""
        # URLs are retained in the on-disk citation but consume context without
        # adding mathematical information to this closed-world proposal.
        citation = self.citation.split(", https://", maxsplit=1)[0]
        return (f"{self.card_id} {self.title} ({citation}): {self.mechanism_formula}\n"
                f"Boundary: {self.regime_boundary}\nGuard: {self.numerical_guardrails}")


def load_cards(path: Path = CARD_PATH) -> tuple[PriorCard, ...]:
    cards = tuple(PriorCard(**row) for row in json.loads(path.read_text(encoding="utf-8")))
    if tuple(card.card_id for card in cards) != tuple(f"C{i:02}" for i in range(1, 11)):
        raise ValueError("prior library must contain C01–C10 in registered order")
    return cards


def _value(residual_d: ResidualSummary, *keys: str) -> float | None:
    for key in keys:
        x = residual_d.get(key)
        if isinstance(x, (int, float)) and isfinite(x):
            return float(x)
    return None


def _excess(value: float | None, threshold: float, scale: float) -> float:
    """Positive excess in fixed scale units, capped at four; absent reading is no trigger."""
    return 0.0 if value is None else min(4.0, max(0.0, (value - threshold) / scale))


def _fraction(part: float | None, total: float | None) -> float | None:
    return None if part is None or total is None or total <= 0 else part / total


def _urgency(residual_d: ResidualSummary) -> dict[str, float]:
    """Dimensionless, fixed D-only urgency; larger values get an earlier hypothesis slot.

    Waiting-time NLL (nats/event) has its own fixed scale, not a share of
    the total across heads. C06 additionally responds to absolute zero-wait
    rate error and relative CV² error in simulated versus historical flow.
    The other head fractions retain the previous scales. Unavailable fields
    do not act as zero-error evidence or positive symptoms. These are
    scheduling heuristics, not architecture predictions or fitted weights.
    """
    v = residual_d
    total = _value(v, "val_nll_total")
    event = _fraction(_value(v, "event_nll", "action_nll", "nll_action"), total)
    level = _fraction(_value(v, "level_nll", "rel_ticks_nll", "nll_rel_ticks"), total)
    size = _fraction(_value(v, "size_nll", "volume_nll", "nll_volume"), total)
    time_pressure = _excess(_value(v, "dt_nll", "nll_dt"), 1.0, .5)
    event_pressure = _excess(event, .25, .15)
    level_pressure = _excess(level, .25, .15)
    size_pressure = _excess(size, .25, .15)
    exhausted = _excess(_value(v, "exhausted_replicate_rate"), .10, .20)
    acf = _excess(_value(v, "sign_acf_abs_error", "direction_acf_mae"), .02, .10)
    burst_error = _excess(_value(v, "zero_wait_rate_error"), .01, .025)
    cv2_error = _excess(_value(v, "interarrival_cv2_rel_error"), .10, .25)
    gap = _value(v, "memory_gap_64", "G_mem(64)", "G_mem_64")
    weak_memory = _excess(None if gap is None else -gap, -.05, .05)
    cf = _excess(_value(v, "counterfactual_impact_mae"), .05, .25)
    stylized = _excess(_value(v, "stylized_impact_mae"), .10, .50)
    auc = _value(v, "visible_minus_blind_auc")
    weak_auc = 1.0 if auc is not None and auc <= .0 else .0
    relaxation = _value(v, "post_stop_relaxation_delta")
    weak_relax = 1.0 if relaxation is not None and relaxation <= .0 else .0
    nonrecover = _excess(_value(v, "end_over_peak_ratio", "end_over_peak"), 1.05, .25)
    free_drift = _excess(_value(v, "free_abs_drift_ticks", "free_drift"), .10, .50)
    return {
        "C01": weak_auc + .25 * cf,
        "C02": weak_relax + nonrecover + .25 * stylized,
        "C03": .75 * time_pressure + .5 * event_pressure + acf + .5 * weak_memory,
        "C04": 1.25 * exhausted + .5 * event_pressure + .5 * size_pressure,
        "C05": exhausted + level_pressure,
        "C06": time_pressure + 1.5 * burst_error + cv2_error,
        "C07": .5 * stylized + .25 * level_pressure,
        "C08": .65 * time_pressure + .5 * event_pressure + .5 * acf + .25 * cf,
        "C09": free_drift,
        "C10": .5 * weak_memory + .25 * exhausted,
    }


def select_active_cards(residual_d: ResidualSummary, cooldown_map: dict[str, int], max_cards: int = 3) -> list[PriorCard]:
    """Rank only parent residual symptoms and cooling state, never a data regime.

    Allocate the three slots to geometry, time/memory, and global remainder.
    If a family has no positive noncooled evidence, fill its vacancy from
    the highest remaining global score. Equal scores use card ID. Cooldown
    is counted in future completed rounds (2 means skip the next two).
    """
    if not 0 <= max_cards <= 3:
        raise ValueError("at most three competition cards may be activated")
    scores = _urgency(residual_d)
    eligible = [card for card in load_cards() if cooldown_map.get(card.card_id, 0) <= 0
                and scores[card.card_id] > 0]
    eligible.sort(key=lambda card: (-scores[card.card_id], card.card_id))
    selected = []
    for family in (GEOMETRY, TIME_MEMORY):
        if len(selected) == max_cards:
            break
        match = next((card for card in eligible if card.card_id in family), None)
        if match is not None:
            selected.append(match)
    remainder = [card for card in eligible if card not in selected]
    selected.extend(remainder[:max_cards - len(selected)])
    return selected


def advance_cooldown(cooldown_map: dict[str, int], selected_card_ids: Sequence[str], rejected: bool) -> dict[str, int]:
    """Decay one round; explicitly cited cards on an adverse rejected trial cool for two."""
    updated = {card: rounds - 1 for card, rounds in cooldown_map.items() if rounds > 1}
    if rejected:
        registered = {card.card_id for card in load_cards()}
        for card in selected_card_ids:
            if card not in registered:
                raise ValueError(f"unknown card {card}")
            updated[card] = 2
    return updated
