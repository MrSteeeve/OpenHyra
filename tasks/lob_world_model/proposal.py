"""Single-call synthesis output: complete candidate source and ex ante prediction."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Literal


TargetMetric = Literal[
    "val_nll_total", "dt_nll", "event_nll", "level_nll", "size_nll",
    "zero_wait_rate_error", "interarrival_cv2_rel_error", "exhausted_replicate_rate",
    "sign_acf_abs_error", "counterfactual_impact_mae", "free_abs_drift_ticks",
]
Direction = Literal["decrease", "increase"]
ZMetric = Literal[
    "sign_acf_sum_1_32", "flow_vr_32_4", "trade_to_top1_depth", "midprice_shift_rate",
    "cancel_to_limit_ratio", "zero_wait_prob", "interarrival_cv2",
    "book_imbalance_bias", "flow_sign_bias",
]
WeakerDirection = Literal["lower", "higher"]

TARGET_METRICS = (
    "val_nll_total", "dt_nll", "event_nll", "level_nll", "size_nll",
    "zero_wait_rate_error", "interarrival_cv2_rel_error", "exhausted_replicate_rate",
    "sign_acf_abs_error", "counterfactual_impact_mae", "free_abs_drift_ticks",
)
Z_METRICS = (
    "sign_acf_sum_1_32", "flow_vr_32_4", "trade_to_top1_depth", "midprice_shift_rate",
    "cancel_to_limit_ratio", "zero_wait_prob", "interarrival_cv2",
    "book_imbalance_bias", "flow_sign_bias",
)
CARD_IDS = tuple(f"C{index:02d}" for index in range(1, 11))


class ProposalFormatError(ValueError):
    """A proposal does not satisfy the registered one-call output contract."""


@dataclass(frozen=True)
class StructuredPrediction:
    """Ex ante causal chain and machine-falsifiable Z boundary registered before evaluation.

    Diagnosis of exploitable structure -> optional catalog cards (empty means an
    open structure outside the catalog) -> mechanism with prose and structured Z
    failure boundary -> one lower-is-better readout and direction that can refute it.
    """

    data_regime_diagnosis: str
    selected_card_ids: tuple[str, ...]
    regime_hypothesis: str
    target_metric: TargetMetric
    expected_direction: Direction
    weaker_when_z_metric: ZMetric
    weaker_when_z_direction: WeakerDirection
    unfold_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("data_regime_diagnosis", "regime_hypothesis"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ProposalFormatError(f"{field} must be nonempty text")
        if len(self.selected_card_ids) > 3 or any(card not in CARD_IDS for card in self.selected_card_ids):
            raise ProposalFormatError("selected_card_ids must contain at most three known card IDs")
        if len(set(self.selected_card_ids)) != len(self.selected_card_ids):
            raise ProposalFormatError("selected_card_ids must not contain duplicates")
        if self.target_metric not in TARGET_METRICS:
            raise ProposalFormatError("target_metric must be one of the eleven lower-is-better residual readouts")
        if self.expected_direction not in ("decrease", "increase"):
            raise ProposalFormatError("expected_direction must be decrease or increase")
        if self.weaker_when_z_metric not in Z_METRICS:
            raise ProposalFormatError("weaker_when_z_metric must be one of the nine Z profile metrics")
        if self.weaker_when_z_direction not in ("lower", "higher"):
            raise ProposalFormatError("weaker_when_z_direction must be lower or higher")
        if len(self.unfold_ids) > 2 or any(not isinstance(value, str) or not value.strip() for value in self.unfold_ids):
            raise ProposalFormatError("unfold_ids must contain at most two nonempty candidate IDs")
        if len(set(self.unfold_ids)) != len(self.unfold_ids):
            raise ProposalFormatError("unfold_ids must not contain duplicates")

    @classmethod
    def from_dict(cls, value: object) -> StructuredPrediction:
        if not isinstance(value, dict):
            raise ProposalFormatError("prediction must be a JSON object")
        required = {
            "data_regime_diagnosis", "selected_card_ids", "regime_hypothesis",
            "target_metric", "expected_direction", "weaker_when_z_metric", "weaker_when_z_direction",
        }
        if not required.issubset(value) or set(value) - required - {"unfold_ids"}:
            raise ProposalFormatError("prediction fields do not match the registered output protocol")
        card_ids = value["selected_card_ids"]
        unfold_ids = value.get("unfold_ids", [])
        for field, ids in (("selected_card_ids", card_ids), ("unfold_ids", unfold_ids)):
            if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
                raise ProposalFormatError(f"{field} must be a JSON array of strings")
        return cls(
            value["data_regime_diagnosis"], tuple(card_ids), value["regime_hypothesis"],
            value["target_metric"], value["expected_direction"],
            value["weaker_when_z_metric"], value["weaker_when_z_direction"], tuple(unfold_ids),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Proposal:
    source: str
    prediction: StructuredPrediction

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ProposalFormatError("candidate.py must contain complete nonempty Python source")


_FENCE = re.compile(r"^```(json|python|py)[ \t]*\r?\n(.*?)^```[ \t]*\r?$", re.MULTILINE | re.DOTALL | re.IGNORECASE)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in pairs:
        if key in fields:
            raise ProposalFormatError(f"Duplicate JSON field: {key}")
        fields[key] = value
    return fields


def _json(value: str) -> object:
    try:
        return json.loads(value, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProposalFormatError("proposal metadata is not valid JSON") from exc


def parse_proposal(text: str) -> Proposal:
    """Parse one JSON + one Python fence (or one JSON object with candidate_source).

    No repair request, code execution, or inferred missing prediction is performed.
    Syntax and tensor contracts are independently checked by CandidateGate.
    """
    if not isinstance(text, str) or not text.strip():
        raise ProposalFormatError("proposal reply is empty")
    blocks = list(_FENCE.finditer(text.strip()))
    if blocks:
        metadata = [block.group(2) for block in blocks if block.group(1).lower() == "json"]
        python = [block.group(2) for block in blocks if block.group(1).lower() in ("python", "py")]
        if len(metadata) == 1 and len(python) == 1 and len(blocks) == 2:
            return Proposal(python[0].rstrip() + "\n", StructuredPrediction.from_dict(_json(metadata[0])))
        if len(metadata) == 1 and len(blocks) == 1:
            value = _json(metadata[0])
        else:
            raise ProposalFormatError("proposal requires exactly one JSON block and one Python block")
    else:
        value = _json(text)
    if not isinstance(value, dict) or "candidate_source" not in value:
        raise ProposalFormatError("proposal must supply candidate_source or a Python code fence")
    source = value.pop("candidate_source")
    if "prediction" in value:
        if set(value) != {"prediction"}:
            raise ProposalFormatError("unexpected fields beside prediction and candidate_source")
        value = value["prediction"]
    return Proposal(source, StructuredPrediction.from_dict(value))
