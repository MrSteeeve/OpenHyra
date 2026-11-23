"""P0 development and verifier files with independent roots and atomic publication."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from os import getpid
from pathlib import Path
from threading import Thread, current_thread
from typing import Any, BinaryIO

from stochopia.common import atomic_write_json
from stochopia.harness.contract import (
    EvaluationTask,
    ExperimentManifest,
    FinalChoice,
    FitRecord,
    FrozenManifest,
    PairedEvalResult,
    SearchTrialRecord,
    SelectionRecord,
    StructuredPrediction,
)


JsonObject = dict[str, Any]

__all__ = [
    "DevelopmentExport", "DevelopmentStore", "JsonObject", "RunBusyError", "SelectorUsageStore",
    "VerifierStore", "read_json", "validate_task_result",
]


# --- Files and process lock ---


class RunBusyError(RuntimeError):
    """The local process lock for a run is held by another worker."""


def _component(value: str) -> str:
    """Accept one nonempty filename component without separators or traversal."""
    if not isinstance(value, str) or not value or value in {".", ".."} or any(
        char in value for char in "/\\\x00"
    ):
        raise ValueError(f"unsafe file key: {value!r}")
    return value


def read_json(path: Path) -> Any:
    """Read a caller-supplied JSON input path."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class _JsonDirectory:
    def __init__(self, root: Path) -> None:
        """Resolve one caller-supplied root before opening any descendant files."""
        self.path = Path(root).expanduser().resolve()
        self._lock_handle: BinaryIO | None = None
        self._lock_owner: tuple[int, Thread] | None = None
        self._lock_depth = 0

    def _file(self, *parts: str) -> Path:
        """Join literal components below the stored root and reject symlink redirects."""
        file = self.path.joinpath(*map(_component, parts))
        if file.resolve() != file:
            raise ValueError(f"store path resolves outside its registered location: {file}")
        return file

    def _read(self, *parts: str) -> JsonObject | None:
        """Read a registered JSON object. A missing entry is distinct from invalid JSON."""
        file = self._file(*parts)
        try:
            payload = read_json(file)
        except FileNotFoundError:
            return None
        if not isinstance(payload, dict):
            raise ValueError(f"expected a JSON object in {file}")
        return payload

    def _seal(self, payload: JsonObject, *parts: str) -> None:
        """Write a JSON object atomically below this root under the caller's lock.

        An identical existing object is a no-op. A different object for an
        already registered key is rejected.
        """
        self._require_lock()
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object for publication")
        file = self._file(*parts)
        if file.exists():
            if read_json(file) != payload:
                raise ValueError(f"existing {file.name} differs from its registered contents")
            return
        atomic_write_json(file, payload)

    def _require_lock(self) -> None:
        if self._lock_depth <= 0 or self._lock_owner != (getpid(), current_thread()):
            raise RuntimeError("store writes require the current thread to own the run lock")

    def _forget_lock(self) -> None:
        self._lock_handle = None
        self._lock_owner = None
        self._lock_depth = 0

    def _close_inherited_lock(self, pid: int) -> None:
        """Drop a forked descriptor without unlocking its parent's open file description."""
        owner = self._lock_owner
        if owner is not None and owner[0] != pid:
            handle = self._lock_handle
            self._forget_lock()
            if handle is not None:
                handle.close()

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold a nonblocking file lock, reentrant on this instance for its owning thread."""
        pid = getpid()
        owner = (pid, current_thread())
        self._close_inherited_lock(pid)
        if self._lock_owner == owner:
            self._lock_depth += 1
            try:
                yield
            finally:
                exiting_pid = getpid()
                if exiting_pid == pid:
                    self._lock_depth -= 1
                else:
                    self._close_inherited_lock(exiting_pid)
            return

        self.path.mkdir(parents=True, exist_ok=True)
        # Keep the lock inode in place so other workers compete on the same file.
        with self._file(".lock").open("a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunBusyError(f"another worker owns {self.path}") from exc
            self._lock_handle = handle
            self._lock_owner = owner
            self._lock_depth = 1
            try:
                yield
            finally:
                if self._lock_owner == owner:
                    self._forget_lock()
                if getpid() == pid:
                    # Explicit unlock also releases the lock if a forked child
                    # is still holding a duplicate of this file descriptor.
                    fcntl.flock(handle, fcntl.LOCK_UN)


class SelectorUsageStore(_JsonDirectory):
    """The mutable budget ledger for one experiment and group under a caller-selected root."""

    def __init__(self, usage_root: Path, exp_id: str, group_id: str) -> None:
        """Bind the ledger to literal experiment and group path components without redirects."""
        root = Path(usage_root).expanduser().resolve()
        folder = root / _component(exp_id) / _component(group_id)
        if folder.resolve() != folder:
            raise ValueError("selector usage ledger cannot redirect the experiment or group directory")
        super().__init__(folder)

    def usage(self) -> JsonObject | None:
        return self._read("usage.json")

    def save_usage(self, payload: JsonObject) -> None:
        """Atomically replace the cumulative budget accounting under the caller's lock."""
        self._require_lock()
        atomic_write_json(self._file("usage.json"), payload)


# --- Development hand-off and round files ---


@dataclass(frozen=True)
class DevelopmentExport:
    experiment: ExperimentManifest
    selections: tuple[SelectionRecord, ...]
    predictions: tuple[StructuredPrediction, ...]
    search_trials: tuple[SearchTrialRecord, ...]
    final_choices: tuple[FinalChoice, ...]
    fits: tuple[FitRecord, ...]
    development_root: Path

    def to_dict(self) -> JsonObject:
        return {
            "experiment": self.experiment.to_dict(),
            "selections": [row.to_dict() for row in self.selections],
            "predictions": [row.to_dict() for row in self.predictions],
            "search_trials": [row.to_dict() for row in self.search_trials],
            "final_choices": [row.to_dict() for row in self.final_choices],
            "fits": [row.to_dict() for row in self.fits],
            "development_root": str(self.development_root),
        }

    @classmethod
    def from_dict(cls, payload: JsonObject) -> DevelopmentExport:
        return cls(
            experiment=ExperimentManifest.from_dict(payload["experiment"]),
            selections=tuple(map(SelectionRecord.from_dict, payload["selections"])),
            predictions=tuple(map(StructuredPrediction.from_dict, payload["predictions"])),
            search_trials=tuple(map(SearchTrialRecord.from_dict, payload["search_trials"])),
            final_choices=tuple(map(FinalChoice.from_dict, payload["final_choices"])),
            fits=tuple(map(FitRecord.from_dict, payload["fits"])),
            development_root=Path(payload["development_root"]),
        )


class DevelopmentStore(_JsonDirectory):
    """Store only training and development-side OpenHyra state for one experiment."""

    def __init__(self, runs_root: Path, exp_id: str) -> None:
        """Bind one experiment ID to runs_root/stochopia_tier0/<exp_id>.

        The caller chooses runs_root. The two appended components cannot
        redirect the registered experiment elsewhere using symlinks.
        """
        parent = Path(runs_root).expanduser().resolve()
        root = parent / "stochopia_tier0" / _component(exp_id)
        if root.resolve() != root:
            raise ValueError("development run cannot be redirected by a symbolic link")
        super().__init__(root)
        self.exp_id = exp_id

    def register(self, manifest: ExperimentManifest) -> None:
        """Seal the validated train/dev manifest for this experiment ID only."""
        if manifest.exp_id != self.exp_id:
            raise ValueError("manifest does not belong to this development store")
        payload = manifest.to_dict()
        with self.lock():
            self._seal(payload, "manifest.json")

    def manifest(self) -> ExperimentManifest:
        return ExperimentManifest.from_dict(read_json(self._file("manifest.json")))

    def rounds(self) -> tuple[int, ...]:
        """List rounds with a sealed preregistration, including pending rounds."""
        directory = self._file("rounds")
        if not directory.exists():
            return ()
        rounds = []
        for path in directory.iterdir():
            if path.is_dir() and path.name.isascii() and path.name.isdecimal():
                index = int(path.name)
                if index > 0 and path.name == str(index) and self.preregistered(index) is not None:
                    rounds.append(index)
        return tuple(sorted(rounds))

    def preregistered(self, round_index: int) -> JsonObject | None:
        return self._read("rounds", str(round_index), "preregistered.json")

    def feedback(self, round_index: int) -> JsonObject | None:
        return self._read("rounds", str(round_index), "feedback.json")

    def preregister(self, round_index: int, plan: JsonObject) -> None:
        """Seal the preregistration for a round under the caller's lock."""
        self._seal(plan, "rounds", str(round_index), "preregistered.json")

    def measurement(self, round_index: int, trial_id: str) -> JsonObject | None:
        """Read a committed trial, whether or not the full round has been published."""
        return self._read("rounds", str(round_index), "measurements", f"{_component(trial_id)}.json")

    def commit_measurement(self, round_index: int, trial_id: str, payload: JsonObject) -> None:
        """Seal one trial's result under the caller's lock."""
        self._seal(payload, "rounds", str(round_index), "measurements", f"{_component(trial_id)}.json")

    def publish_feedback(self, round_index: int, feedback: JsonObject) -> None:
        self._seal(feedback, "rounds", str(round_index), "feedback.json")

    def fit(self, candidate_id: str, repeat: str) -> FitRecord | None:
        payload = self._read("fits", _component(candidate_id), f"{_component(repeat)}.json")
        return None if payload is None else FitRecord.from_dict(payload)

    def save_fit(self, fit: FitRecord) -> None:
        self._seal(fit.to_dict(), "fits", _component(fit.candidate_id), f"{_component(fit.train_repeat_id)}.json")

    def model_directory(self) -> Path:
        """Return the model artifact directory below this development run."""
        return self._file("models")

    def discovery_ledger_path(self, group_id: str) -> Path:
        """Return one group's derived discovery ledger path below this development run."""
        return self._file("discovery", group_id, "ledger.jsonl")


# --- Held-out batch files and result validation ---


class VerifierStore(_JsonDirectory):
    """A physically separate batch root, established before development or at hand-off."""

    def test_config(self) -> JsonObject | None:
        return self._read("test.json")

    def register_config(self, config: JsonObject) -> None:
        self._seal(config, "test.json")

    def frozen(self) -> FrozenManifest:
        return FrozenManifest.from_dict(read_json(self._file("frozen.json")))

    def result(self, task_id: str) -> PairedEvalResult | None:
        payload = self._read("results", f"{_component(task_id)}.json")
        return None if payload is None else PairedEvalResult.from_dict(payload)

    def report(self) -> JsonObject | None:
        return self._read("report.json")

    def register_frozen(self, frozen: FrozenManifest) -> None:
        for task in frozen.tasks:
            _component(task.task_id)
        self._seal(frozen.to_dict(), "frozen.json")

    def commit_result(self, task_id: str, result: PairedEvalResult) -> None:
        self._seal(result.to_dict(), "results", f"{_component(task_id)}.json")

    def publish_report(self, report: JsonObject) -> None:
        self._seal(report, "report.json")


def validate_task_result(
    result: PairedEvalResult, task: EvaluationTask, parent: FitRecord, candidate: FitRecord
) -> None:
    """Check result identity, split and inline observations against the sealed task and fits."""
    result.validate()
    for name in ("task_id", "split", "scenario_id", "parent_fit_id", "candidate_fit_id",
                 "action_ref", "random_inputs_ref"):
        if getattr(result, name) != getattr(task, name):
            raise ValueError(f"result {name} does not match its registered task")
    if result.candidate_id != candidate.candidate_id or result.parent_id != parent.candidate_id:
        raise ValueError("result candidates do not match the registered fits")
    allowed_units = set(task.unit_ids)
    if any(row.unit_id not in allowed_units or row.train_repeat_id != parent.train_repeat_id
           for row in result.observations):
        raise ValueError("result observations are outside the registered units or training repeat")
