from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


def _install_import_placeholders() -> None:
    placeholders = {
        "island_scheduler": ("IslandScheduler",),
        "mechanism_cards": ("MechanismCardBuilder", "MechanismCardStore"),
        "analogy_graph": ("AnalogyGraph",),
        "context_retrieval": (
            "ContextRetrieval",
            "PortfolioPacket",
            "AnalysisPacket",
            "ProposalPacket",
        ),
    }
    for module_name, attributes in placeholders.items():
        if importlib.util.find_spec(module_name) is not None:
            continue
        module = types.ModuleType(module_name)
        for attribute in attributes:
            setattr(module, attribute, type(attribute, (), {}))
        sys.modules[module_name] = module


_install_import_placeholders()
import harness_v5  # noqa: E402
from harness_v5 import V5Bridge  # noqa: E402


@dataclass
class _Epoch:
    island_id: str
    epoch: int = 0
    status: str = "active"


class _ObjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)


class _Event:
    def __init__(self, **values):
        self.__dict__.update(values)

    def validate(self) -> None:
        if not self.record_id or not self.island_epoch_id:
            raise ValueError("record and island IDs are required")


class _EventStore:
    def __init__(self, root: Path, object_store: _ObjectStore):
        self.root = Path(root)
        self.object_store = object_store
        self._events = []

    def append_experiment_event(self, event: _Event) -> None:
        self._events.append(event)

    def read_experiment_events(self) -> list[_Event]:
        return list(self._events)


class _Profiler:
    def build_profile(self, **values):
        return SimpleNamespace(**values)


class _BehaviorIndex:
    def __init__(self, boundaries: dict):
        self.boundaries = boundaries


class _IslandScheduler:
    review_interval = 10

    def __init__(self, path: Path, num_islands: int = 4):
        self.path = Path(path)
        self.num_islands = num_islands
        self._epochs: list[_Epoch] = []
        self._records: dict[str, list[str]] = {}
        self.review_calls: list[tuple[int, dict[str, float]]] = []

    @staticmethod
    def _epoch_id(epoch: _Epoch) -> str:
        return f"{epoch.island_id}_epoch_{epoch.epoch:02d}"

    def initialize(
        self,
        seed_record_ids: list[str],
        context_round: int,
        base_proposal_seed: int,
    ) -> list[_Epoch]:
        if self._epochs:
            raise RuntimeError("already initialized")
        self._epochs = [_Epoch(f"island_{index:02d}") for index in range(self.num_islands)]
        for index, epoch in enumerate(self._epochs):
            epoch_id = self._epoch_id(epoch)
            self._records[epoch_id] = []
            if seed_record_ids:
                self._records[epoch_id].append(seed_record_ids[index % len(seed_record_ids)])
        return list(self._epochs)

    def get_active_epochs(self) -> list[_Epoch]:
        return [epoch for epoch in self._epochs if epoch.status == "active"]

    def get_all_epochs(self) -> list[_Epoch]:
        return list(self._epochs)

    def sample_island_for_exploration(self, context_round: int) -> str:
        active = self.get_active_epochs()
        return self._epoch_id(active[context_round % len(active)])

    def assign_candidate(self, island_epoch_id: str, record_id: str) -> None:
        self._records[island_epoch_id].append(record_id)

    def get_island_records(self, island_epoch_id: str) -> list[str]:
        return list(self._records[island_epoch_id])

    def should_review(self, context_round: int) -> bool:
        return context_round > 0 and context_round % self.review_interval == 0

    def run_review(
        self, context_round: int, scores: dict[str, float],
    ) -> dict[str, str]:
        self.review_calls.append((context_round, dict(scores)))
        return {"island_00_epoch_00": "island_00_epoch_01"}


class _MechanismCardBuilder:
    @staticmethod
    def from_bundle_manifest(record_id: str, manifest: dict):
        return SimpleNamespace(record_id=record_id, manifest=dict(manifest))


class _MechanismCardStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._cards = {}

    def save(self, card) -> None:
        self._cards[card.record_id] = card

    def load(self, record_id: str):
        return self._cards[record_id]


class _AnalogyGraph:
    @classmethod
    def load(cls, path: Path):
        return cls()

    def save(self, path: Path) -> None:
        self.saved_path = Path(path)


class _Packet:
    def __init__(self, text: str):
        self._text = text

    def to_text(self) -> str:
        return self._text


class _ContextRetrieval:
    def __init__(self, **values):
        self.values = values

    def build_portfolio(self):
        record_count = sum(len(records) for records in self.values["island_records"].values())
        return _Packet(f"portfolio records={record_count}"), {"kind": "portfolio"}

    def build_analysis(self, target_island_epoch_id: str):
        return _Packet(f"analysis island={target_island_epoch_id}"), {"kind": "analysis"}

    def build_proposal(self, plan, parent_source: str):
        return _Packet(f"proposal parent={parent_source}"), {"kind": "proposal"}


def _bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> V5Bridge:
    replacements = {
        "ObjectStore": _ObjectStore,
        "ExperienceEventStore": _EventStore,
        "ExperimentEvent": _Event,
        "BehaviorProfiler": _Profiler,
        "BehaviorIndex": _BehaviorIndex,
        "IslandScheduler": _IslandScheduler,
        "MechanismCardBuilder": _MechanismCardBuilder,
        "MechanismCardStore": _MechanismCardStore,
        "AnalogyGraph": _AnalogyGraph,
        "ContextRetrieval": _ContextRetrieval,
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(harness_v5, name, replacement)
    return V5Bridge(tmp_path)


def _initialize(bridge: V5Bridge):
    return bridge.initialize(["seed-0", "seed-1", "seed-2", "seed-3"])


def _evaluate(bridge: V5Bridge, record_id: str, island_epoch_id: str, metrics=None):
    bridge.on_candidate_evaluated(
        record_id=record_id,
        island_epoch_id=island_epoch_id,
        score=0.25,
        status="ok",
        description="minimal candidate",
        parent_ids=["seed-0"],
        metrics=metrics or {"metric": "reward"},
    )


def test_initialize_creates_islands(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)

    epochs = _initialize(bridge)

    assert len(epochs) == 4


def test_initialize_idempotent(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    first = _initialize(bridge)

    second = _initialize(bridge)

    assert second == first


def test_pick_island_deterministic(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    _initialize(bridge)

    assert bridge.pick_island(7) == bridge.pick_island(7)


def test_on_candidate_evaluated_writes_event(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    _initialize(bridge)
    island_epoch_id = bridge.pick_island(0)

    _evaluate(bridge, "candidate-event", island_epoch_id)

    events = bridge.event_store.read_experiment_events()
    assert [event.record_id for event in events] == ["candidate-event"]


def test_on_candidate_evaluated_assigns_to_island(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    _initialize(bridge)
    island_epoch_id = bridge.pick_island(1)

    _evaluate(bridge, "candidate-island", island_epoch_id)

    assert "candidate-island" in bridge.island_scheduler.get_island_records(island_epoch_id)


def test_on_candidate_evaluated_builds_card(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    _initialize(bridge)
    island_epoch_id = bridge.pick_island(2)

    _evaluate(bridge, "candidate-card", island_epoch_id)

    assert bridge.card_store.load("candidate-card").record_id == "candidate-card"


def test_on_candidate_evaluated_builds_profile(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    _initialize(bridge)
    island_epoch_id = bridge.pick_island(3)
    metrics = {
        "per_instance_results": {"case-1": 1.2},
        "baseline_scores": {"case-1": 1.0},
        "elapsed_s": 0.1,
        "peak_memory_mb": 4.0,
    }

    _evaluate(bridge, "candidate-profile", island_epoch_id, metrics)

    assert bridge._profiles["candidate-profile"].overall_score == 0.25


def test_on_context_complete_no_review(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    _initialize(bridge)

    assert bridge.on_context_complete(context_round=5) == {}
    assert bridge.island_scheduler.review_calls == []


def test_on_context_complete_triggers_review(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    epochs = _initialize(bridge)
    for index, epoch in enumerate(epochs):
        island_epoch_id = f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
        _evaluate(bridge, f"candidate-{index}", island_epoch_id)

    replacements = bridge.on_context_complete(context_round=10)

    assert replacements == {"island_00_epoch_00": "island_00_epoch_01"}
    assert bridge.island_scheduler.review_calls[0][0] == 10


def test_build_context_returns_portfolio(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    _initialize(bridge)

    context = bridge.build_context()

    assert context["portfolio_text"].startswith("portfolio records=")
    assert context["portfolio"].to_text() == context["portfolio_text"]
