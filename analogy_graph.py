from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from schemas_v5 import AnalogyHypothesis, AnalogyResult


EDGE_TYPES = {
    "structurally_related",
    "behaviorally_near",
    "complementary_on_slice",
    "analogy_hypothesized",
    "transfer_supported",
    "transfer_refuted",
    "composed_from",
    "counterexample_to",
}


class AnalogyEdge:
    def __init__(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        evidence_id: str,
        metadata: dict | None = None,
    ):
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"unknown edge type: {edge_type}")
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.evidence_id = evidence_id
        self.metadata = deepcopy(metadata) if metadata is not None else {}

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "evidence_id": self.evidence_id,
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> AnalogyEdge:
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=data["edge_type"],
            evidence_id=data["evidence_id"],
            metadata=data.get("metadata"),
        )


class AnalogyGraph:
    def __init__(self):
        self.edges: list[AnalogyEdge] = []

    def add_edge(self, edge: AnalogyEdge) -> None:
        key = _edge_key(edge)
        if any(_edge_key(existing) == key for existing in self.edges):
            return
        self.edges.append(edge)

    def add_hypothesis(self, hypothesis: AnalogyHypothesis) -> None:
        metadata = hypothesis.to_dict()
        for source_id in hypothesis.source_record_ids:
            self.add_edge(
                AnalogyEdge(
                    source_id,
                    hypothesis.target_parent_id,
                    "analogy_hypothesized",
                    hypothesis.id,
                    metadata,
                )
            )

    def add_result(
        self, hypothesis: AnalogyHypothesis, result: AnalogyResult
    ) -> None:
        if result.verdict not in {"transfer_supported", "transfer_refuted"}:
            return
        metadata = result.to_dict()
        for source_id in hypothesis.source_record_ids:
            self.add_edge(
                AnalogyEdge(
                    source_id,
                    result.guided_record_id,
                    result.verdict,
                    result.analogy_hypothesis_id,
                    metadata,
                )
            )

    def get_edges_from(self, record_id: str) -> list[AnalogyEdge]:
        return [edge for edge in self.edges if edge.source_id == record_id]

    def get_edges_to(self, record_id: str) -> list[AnalogyEdge]:
        return [edge for edge in self.edges if edge.target_id == record_id]

    def get_edges_by_type(self, edge_type: str) -> list[AnalogyEdge]:
        return [edge for edge in self.edges if edge.edge_type == edge_type]

    def get_transfer_history(self, record_id: str) -> dict[str, list[AnalogyEdge]]:
        involved = [
            edge
            for edge in self.edges
            if edge.source_id == record_id or edge.target_id == record_id
        ]
        return {
            "supported": [
                edge for edge in involved if edge.edge_type == "transfer_supported"
            ],
            "refuted": [
                edge for edge in involved if edge.edge_type == "transfer_refuted"
            ],
            "hypothesized": [
                edge for edge in involved if edge.edge_type == "analogy_hypothesized"
            ],
        }

    def neighbors(
        self, record_id: str, edge_types: set[str] | None = None
    ) -> set[str]:
        allowed = set(edge_types) if edge_types is not None else None
        result = set()
        for edge in self.edges:
            if allowed is not None and edge.edge_type not in allowed:
                continue
            if edge.source_id == record_id:
                result.add(edge.target_id)
            elif edge.target_id == record_id:
                result.add(edge.source_id)
        return result

    def to_dict(self) -> dict:
        return {"edges": [edge.to_dict() for edge in self.edges]}

    @classmethod
    def from_dict(cls, data: dict) -> AnalogyGraph:
        graph = cls()
        for item in data.get("edges", []):
            graph.add_edge(AnalogyEdge.from_dict(item))
        return graph

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> AnalogyGraph:
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _edge_key(edge: AnalogyEdge) -> tuple[str, str, str, str]:
    return edge.source_id, edge.target_id, edge.edge_type, edge.evidence_id
