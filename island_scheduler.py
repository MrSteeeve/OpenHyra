from __future__ import annotations

import json
import math
import random
from pathlib import Path

from schemas_v5 import IslandEpoch, ExperimentEvent


DEFAULT_NUM_ISLANDS = 4
DEFAULT_REVIEW_INTERVAL = 10
DEFAULT_CULL_FRACTION = 0.5
_STATE_SCHEMA = "openhyra-island-state.v1"


class IslandScheduler:
    def __init__(
        self,
        state_path: Path,
        num_islands: int = DEFAULT_NUM_ISLANDS,
        review_interval: int = DEFAULT_REVIEW_INTERVAL,
        cull_fraction: float = DEFAULT_CULL_FRACTION,
    ):
        if num_islands <= 0:
            raise ValueError("num_islands must be positive")
        if review_interval <= 0:
            raise ValueError("review_interval must be positive")
        if not 0.0 <= cull_fraction < 1.0:
            raise ValueError("cull_fraction must be in [0.0, 1.0)")

        self.state_path = Path(state_path)
        self.num_islands = num_islands
        self.review_interval = review_interval
        self.cull_fraction = cull_fraction
        self._epochs: list[IslandEpoch] = []
        self._epoch_records: dict[str, list[str]] = {}
        self._last_review_round = 0
        self._load()

    @staticmethod
    def _epoch_id(epoch: IslandEpoch) -> str:
        return f"{epoch.island_id}_epoch_{epoch.epoch:02d}"

    def initialize(
        self,
        seed_record_ids: list[str],
        context_round: int,
        base_proposal_seed: int,
    ) -> list[IslandEpoch]:
        if self._epochs:
            raise RuntimeError("islands are already initialized")

        created = []
        for index in range(self.num_islands):
            epoch = IslandEpoch(
                island_id=f"island_{index:02d}",
                epoch=0,
                seed_record_ids=list(seed_record_ids),
                started_after_context_round=context_round,
                proposal_seed=base_proposal_seed + index * 10_000,
                status="active",
            )
            created.append(epoch)
            self._epoch_records[self._epoch_id(epoch)] = []
        self._epochs.extend(created)
        self._persist()
        return list(created)

    def get_active_epochs(self) -> list[IslandEpoch]:
        return [epoch for epoch in self._epochs if epoch.status == "active"]

    def get_epoch(self, island_epoch_id: str) -> IslandEpoch | None:
        return next(
            (
                epoch
                for epoch in self._epochs
                if self._epoch_id(epoch) == island_epoch_id
            ),
            None,
        )

    def assign_candidate(self, island_epoch_id: str, record_id: str) -> None:
        if self.get_epoch(island_epoch_id) is None:
            raise KeyError(f"unknown island epoch: {island_epoch_id}")
        self._epoch_records.setdefault(island_epoch_id, []).append(record_id)
        self._persist()

    def get_island_records(self, island_epoch_id: str) -> list[str]:
        return list(self._epoch_records.get(island_epoch_id, []))

    def sample_island_for_exploration(self, rng_seed: int) -> str:
        active_ids = [self._epoch_id(epoch) for epoch in self.get_active_epochs()]
        if not active_ids:
            raise RuntimeError("no active island epochs")
        return random.Random(rng_seed).choice(active_ids)

    def should_review(self, context_round: int) -> bool:
        return (
            context_round > 0
            and context_round % self.review_interval == 0
            and context_round - self._last_review_round >= self.review_interval
        )

    def run_review(
        self, context_round: int, scores: dict[str, float]
    ) -> dict[str, str]:
        active = self.get_active_epochs()
        if not active:
            raise RuntimeError("no active island epochs")

        best_by_epoch: dict[str, tuple[float, str | None]] = {}
        for epoch in active:
            epoch_id = self._epoch_id(epoch)
            scored_records = [
                (scores[record_id], record_id)
                for record_id in self._epoch_records.get(epoch_id, [])
                if record_id in scores
            ]
            best_by_epoch[epoch_id] = (
                max(scored_records, key=lambda item: item[0])
                if scored_records
                else (-math.inf, None)
            )

        ranked = sorted(
            active,
            key=lambda epoch: (
                -best_by_epoch[self._epoch_id(epoch)][0],
                epoch.started_after_context_round,
                epoch.island_id,
            ),
        )
        cull_count = math.ceil(self.cull_fraction * len(ranked))
        if cull_count == 0:
            self._last_review_round = context_round
            self._persist()
            return {}
        if cull_count >= len(ranked):
            raise RuntimeError("island review must leave at least one survivor")

        survivors = ranked[:-cull_count]
        for survivor in survivors:
            if best_by_epoch[self._epoch_id(survivor)][1] is None:
                raise ValueError("each surviving epoch must have a scored record")

        culled = ranked[-cull_count:]
        review_rng = random.Random(context_round)
        replacements: dict[str, str] = {}
        for old_epoch in culled:
            old_epoch.status = "culled"
            donor = review_rng.choice(survivors)
            donor_record = best_by_epoch[self._epoch_id(donor)][1]
            assert donor_record is not None
            new_epoch = IslandEpoch(
                island_id=old_epoch.island_id,
                epoch=old_epoch.epoch + 1,
                seed_record_ids=[donor_record],
                started_after_context_round=context_round,
                proposal_seed=self._proposal_seed(context_round, old_epoch),
                status="active",
            )
            new_epoch_id = self._epoch_id(new_epoch)
            self._epochs.append(new_epoch)
            self._epoch_records[new_epoch_id] = []
            replacements[self._epoch_id(old_epoch)] = new_epoch_id

        self._last_review_round = context_round
        self._persist()
        return replacements

    @staticmethod
    def _proposal_seed(context_round: int, old_epoch: IslandEpoch) -> int:
        island_number = int(old_epoch.island_id.rsplit("_", 1)[1])
        return context_round * 1_000_000 + island_number * 10_000 + old_epoch.epoch + 1

    def get_all_epochs(self) -> list[IslandEpoch]:
        return list(self._epochs)

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _STATE_SCHEMA,
            "epochs": [epoch.to_dict() for epoch in self._epochs],
            "epoch_records": self._epoch_records,
            "last_review_round": self._last_review_round,
        }
        temp_path = self.state_path.with_name(f".{self.state_path.name}.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.state_path)

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("schema") != _STATE_SCHEMA:
            raise ValueError("unsupported island state schema")
        self._epochs = [IslandEpoch.from_dict(item) for item in payload["epochs"]]
        self._epoch_records = {
            epoch_id: list(record_ids)
            for epoch_id, record_ids in payload["epoch_records"].items()
        }
        self._last_review_round = payload["last_review_round"]


__all__ = [
    "DEFAULT_CULL_FRACTION",
    "DEFAULT_NUM_ISLANDS",
    "DEFAULT_REVIEW_INTERVAL",
    "IslandScheduler",
]
