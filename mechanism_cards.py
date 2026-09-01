from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from schemas_v5 import AnnotationEvent, MechanismCard


class MechanismCardBuilder:
    @staticmethod
    def from_bundle_manifest(record_id: str, manifest: dict) -> MechanismCard:
        facts = {
            "protocol": manifest.get("artifact_protocol") or "unknown",
            "entrypoint": manifest.get("entrypoint") or "unknown",
            "generation_operator": manifest.get("generation_operator") or "unknown",
        }
        return MechanismCard(
            record_id=record_id,
            deterministic_facts=facts,
            trusted_observations={},
            llm_inferences=[],
        )

    @staticmethod
    def add_trusted_observations(
        card: MechanismCard,
        strong_slices: list,
        weak_slices: list,
        failure_modes: list,
    ) -> MechanismCard:
        observations = deepcopy(card.trusted_observations)
        observations.update(
            {
                "strong_slices": deepcopy(strong_slices),
                "weak_slices": deepcopy(weak_slices),
                "failure_modes": deepcopy(failure_modes),
            }
        )
        return MechanismCard(
            schema=card.schema,
            record_id=card.record_id,
            deterministic_facts=deepcopy(card.deterministic_facts),
            trusted_observations=observations,
            llm_inferences=deepcopy(card.llm_inferences),
        )

    @staticmethod
    def add_llm_inference(
        card: MechanismCard,
        claim: str,
        confidence: float,
        evidence_record_ids: list[str],
        annotation_event_id: str,
    ) -> MechanismCard:
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("confidence must be in [0, 1]")
        inferences = deepcopy(card.llm_inferences)
        inferences.append(
            {
                "claim": claim,
                "confidence": confidence,
                "evidence_record_ids": deepcopy(evidence_record_ids),
                "annotation_event_id": annotation_event_id,
            }
        )
        return MechanismCard(
            schema=card.schema,
            record_id=card.record_id,
            deterministic_facts=deepcopy(card.deterministic_facts),
            trusted_observations=deepcopy(card.trusted_observations),
            llm_inferences=inferences,
        )

    @staticmethod
    def merge_cards(base: MechanismCard, update: MechanismCard) -> MechanismCard:
        if base.record_id != update.record_id:
            raise ValueError("cards must have the same record_id")

        facts = deepcopy(base.deterministic_facts)
        facts.update(deepcopy(update.deterministic_facts))

        observations = deepcopy(base.trusted_observations)
        for key, value in list(observations.items()):
            if isinstance(value, list):
                observations[key] = _deduplicate(value)
        for key, value in update.trusted_observations.items():
            if key in observations and isinstance(observations[key], list) and isinstance(value, list):
                observations[key] = _deduplicate(observations[key] + deepcopy(value))
            else:
                observations[key] = _deduplicate(value) if isinstance(value, list) else deepcopy(value)

        return MechanismCard(
            schema=base.schema,
            record_id=base.record_id,
            deterministic_facts=facts,
            trusted_observations=observations,
            llm_inferences=deepcopy(base.llm_inferences) + deepcopy(update.llm_inferences),
        )


class MechanismCardStore:
    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

    def save(self, card: MechanismCard) -> None:
        card.validate()
        path = self.store_path / f"{card.record_id}.json"
        path.write_text(
            json.dumps(card.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    def load(self, record_id: str) -> MechanismCard | None:
        path = self.store_path / f"{record_id}.json"
        if not path.exists():
            return None
        return MechanismCard.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_all(self) -> list[MechanismCard]:
        return [
            MechanismCard.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.store_path.glob("*.json"))
        ]

    def find_by_tag(self, tag_key: str, tag_value: object) -> list[MechanismCard]:
        matches = []
        for card in self.load_all():
            facts = card.deterministic_facts
            observations = card.trusted_observations
            if (tag_key in facts and _contains(facts[tag_key], tag_value)) or (
                tag_key in observations
                and _contains(observations[tag_key], tag_value)
            ):
                matches.append(card)
        return matches


def _deduplicate(values: list) -> list:
    result = []
    for value in values:
        if value not in result:
            result.append(deepcopy(value))
    return result


def _contains(container: object, value: object) -> bool:
    if isinstance(container, (str, list, tuple, set, dict)):
        return value in container
    return container == value
