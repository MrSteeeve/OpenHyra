from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Callable, TypeVar

from object_store import ObjectStore
from schemas_v5 import (
    AnalogyResult,
    AnnotationEvent,
    ExperimentEvent,
    ExperimentPlan,
)


_EventT = TypeVar("_EventT")


class ExperienceEventStore:
    """Append-only typed event logs and content-addressed event objects."""

    def __init__(self, eb_root: Path):
        self.eb_root = Path(eb_root)
        self.events_dir = self.eb_root / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)

        self._experiment_events_path = self.events_dir / "experiment_events.jsonl"
        self._plan_events_path = self.events_dir / "plan_events.jsonl"
        self._annotation_events_path = self.events_dir / "annotation_events.jsonl"
        self._analogy_results_path = self.events_dir / "analogy_results.jsonl"
        for path in (
            self._experiment_events_path,
            self._plan_events_path,
            self._annotation_events_path,
            self._analogy_results_path,
        ):
            path.touch(exist_ok=True)

        objects_dir = self.eb_root / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)
        self._object_store = ObjectStore(objects_dir)
        self._lock = threading.RLock()

    @property
    def object_store(self) -> ObjectStore:
        return self._object_store

    def _append(
        self,
        event: ExperimentEvent | ExperimentPlan | AnnotationEvent | AnalogyResult,
        path: Path,
        object_filename: str,
    ) -> str:
        with self._lock:
            event.validate()
            payload = event.to_dict()
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            return self._object_store.put_json(payload, object_filename)

    def append_experiment_event(self, event: ExperimentEvent) -> str:
        return self._append(
            event,
            self._experiment_events_path,
            "experiment_event.json",
        )

    def append_plan_event(self, plan: ExperimentPlan) -> str:
        return self._append(plan, self._plan_events_path, "experiment_plan.json")

    def append_annotation_event(self, annotation: AnnotationEvent) -> str:
        return self._append(
            annotation,
            self._annotation_events_path,
            "annotation_event.json",
        )

    def append_analogy_result(self, result: AnalogyResult) -> str:
        return self._append(result, self._analogy_results_path, "analogy_result.json")

    def _read(
        self,
        path: Path,
        from_dict: Callable[[dict], _EventT],
    ) -> list[_EventT]:
        events: list[_EventT] = []
        with self._lock:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        events.append(from_dict(payload))
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        print(
                            f"warning: skipping malformed line {line_number} "
                            f"in {path}: {exc}",
                            file=sys.stderr,
                        )
        return events

    def read_experiment_events(self) -> list[ExperimentEvent]:
        return self._read(self._experiment_events_path, ExperimentEvent.from_dict)

    def read_plan_events(self) -> list[ExperimentPlan]:
        return self._read(self._plan_events_path, ExperimentPlan.from_dict)

    def read_annotation_events(self) -> list[AnnotationEvent]:
        return self._read(self._annotation_events_path, AnnotationEvent.from_dict)

    def read_analogy_results(self) -> list[AnalogyResult]:
        return self._read(self._analogy_results_path, AnalogyResult.from_dict)

    def bridge_legacy_record(self, legacy_record: dict) -> ExperimentEvent:
        status_mapping = {
            "crash": "runtime_error",
            "rejected": "static_rejected",
            "violation": "violation",
            "cancelled": "cancelled",
            "timeout": "timeout",
            "ok": "ok",
        }
        legacy_status = legacy_record.get("status", "")
        return ExperimentEvent(
            record_id=legacy_record.get("id", ""),
            algorithm_bundle_sha256="",
            experiment_plan_id="",
            island_epoch_id="",
            status=status_mapping.get(legacy_status, legacy_status),
            score=legacy_record.get("score"),
            score_metric="",
            per_instance_metrics_ref="",
            behavior_profile_ref="",
            runtime_metrics_ref="",
            parent_ids=[],
            inspiration_ids=[],
            created_at="",
        )


__all__ = ["ExperienceEventStore"]
