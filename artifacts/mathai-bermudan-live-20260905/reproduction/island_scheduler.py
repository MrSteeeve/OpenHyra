from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Mapping

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
        direction: str = "max",
    ):
        if num_islands <= 0:
            raise ValueError("num_islands must be positive")
        if review_interval <= 0:
            raise ValueError("review_interval must be positive")
        if not 0.0 <= cull_fraction < 1.0:
            raise ValueError("cull_fraction must be in [0.0, 1.0)")
        if direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")

        self.state_path = Path(state_path)
        self.num_islands = num_islands
        self.review_interval = review_interval
        self.cull_fraction = cull_fraction
        self.direction = direction
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
            # Seed records are actual island members, not only provenance
            # labels.  This matters on resume: retrieval and review must be
            # able to see the baseline through the same membership index as
            # later candidates.
            self._epoch_records[self._epoch_id(epoch)] = list(
                dict.fromkeys(seed_record_ids)
            )
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
        records = self._epoch_records.setdefault(island_epoch_id, [])
        if record_id not in records:
            records.append(record_id)
        self._persist()

    def get_island_records(
        self, island_epoch_id: str, *, include_seeds: bool = True
    ) -> list[str]:
        """Return the complete persisted membership for an epoch.

        ``include_seeds`` is retained as an explicit compatibility keyword;
        new code should leave it at the default because seed/donor records are
        real members of an island, not merely provenance labels.
        """
        records = list(self._epoch_records.get(island_epoch_id, []))
        if include_seeds:
            return records
        epoch = self.get_epoch(island_epoch_id)
        if epoch is None:
            return records
        seed_ids = set(epoch.seed_record_ids)
        return [record_id for record_id in records if record_id not in seed_ids]

    def get_members(self, island_epoch_id: str) -> list[str]:
        """Return the complete persisted membership, including seed donors."""
        return self.get_island_records(island_epoch_id, include_seeds=True)

    def set_direction(self, direction: str) -> None:
        """Set the task comparator used by subsequent reviews.

        The direction is persisted with the scheduler state.  Keeping this as
        a small setter lets a V5 bridge learn the task direction during
        initialization without changing old constructor call sites.
        """
        if direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")
        if self.direction != direction:
            self.direction = direction
            self._persist()

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
        self,
        context_round: int,
        scores: dict[str, float | Mapping[str, object]],
        *,
        direction: str | None = None,
    ) -> dict[str, str]:
        active = self.get_active_epochs()
        if not active:
            raise RuntimeError("no active island epochs")
        review_direction = direction or self.direction
        if review_direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")

        def score_value(record_id: str) -> float | None:
            raw = scores.get(record_id)
            if isinstance(raw, Mapping):
                raw = raw.get("score")
            if raw is None:
                return None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None

        def record_rank_key(item: tuple[float, str]) -> tuple[float, str]:
            # ``min`` over this transformed key gives the desired comparator
            # while preserving the same lower-record-id tie break for both
            # task directions.
            score, record_id = item
            return (
                score if review_direction == "min" else -score,
                record_id,
            )

        best_by_epoch: dict[str, tuple[float, str | None]] = {}
        for epoch in active:
            epoch_id = self._epoch_id(epoch)
            scored_records = [
                (score_value(record_id), record_id)
                for record_id in self._epoch_records.get(epoch_id, [])
                if score_value(record_id) is not None
            ]
            best_by_epoch[epoch_id] = (
                (
                    min(scored_records, key=record_rank_key)
                )
                if scored_records
                else (
                    math.inf if review_direction == "min" else -math.inf,
                    None,
                )
            )

        ranked = sorted(
            active,
            key=lambda epoch: (
                (
                    best_by_epoch[self._epoch_id(epoch)][0]
                    if review_direction == "min"
                    else -best_by_epoch[self._epoch_id(epoch)][0]
                ),
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
            # The donor is the new epoch's first member.  Keeping it in the
            # membership index makes cull/review and Context retrieval agree
            # about what the epoch was seeded from.
            self._epoch_records[new_epoch_id] = [donor_record]
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
            "direction": self.direction,
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
        persisted_direction = payload.get("direction", self.direction)
        if persisted_direction not in {"min", "max"}:
            raise ValueError("island state direction must be 'min' or 'max'")
        self.direction = persisted_direction
        self._epochs = [IslandEpoch.from_dict(item) for item in payload["epochs"]]
        self._epoch_records = {
            epoch_id: list(record_ids)
            for epoch_id, record_ids in payload["epoch_records"].items()
        }
        # Migrate v1 states written before seeds were indexed as members.  A
        # seed may already have generated candidates, so preserve existing
        # order and prepend only missing seed ids.
        for epoch in self._epochs:
            epoch_id = self._epoch_id(epoch)
            existing = self._epoch_records.setdefault(epoch_id, [])
            self._epoch_records[epoch_id] = list(dict.fromkeys(
                [*epoch.seed_record_ids, *existing]
            ))
        self._last_review_round = payload["last_review_round"]


__all__ = [
    "DEFAULT_CULL_FRACTION",
    "DEFAULT_NUM_ISLANDS",
    "DEFAULT_REVIEW_INTERVAL",
    "IslandScheduler",
]
