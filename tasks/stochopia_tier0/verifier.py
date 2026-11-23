"""Run a registered P0 held-out batch under an exclusive local file lock."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from stochopia.harness.contract import (
    DataSplit, EvaluationTask, FitRecord, FrozenManifest, PairedEvalResult,
)

from .store import DevelopmentExport, VerifierStore, _read_json, validate_task_result
from .store import RunBusyError as VerifierBusyError


Evaluator = Callable[[EvaluationTask, FitRecord, FitRecord, FrozenManifest], PairedEvalResult]
ReportBuilder = Callable[[FrozenManifest, tuple[PairedEvalResult, ...]], dict[str, Any]]


@dataclass(frozen=True)
class VerifierConfig:
    """The held-out day identifiers and seeds registered independently from development."""

    exp_id: str
    batch_id: str
    test_split: DataSplit
    seeds: Mapping[str, int]

    def __post_init__(self) -> None:
        if any(not isinstance(key, str) or not key.strip() for key in (self.exp_id, self.batch_id)):
            raise ValueError("test configuration requires experiment and batch IDs")
        self.test_split.validate()
        if self.seeds.keys() != {"rollout", "delta_ci"} or any(
            type(seed) is not int or seed < 0 for seed in self.seeds.values()
        ):
            raise ValueError("test configuration requires integer rollout and delta_ci seeds")
        object.__setattr__(self, "seeds", MappingProxyType(dict(self.seeds)))

    def to_dict(self) -> dict[str, Any]:
        return {"exp_id": self.exp_id, "batch_id": self.batch_id, "test_split": self.test_split.to_dict(),
                "seeds": dict(self.seeds)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VerifierConfig:
        if payload.keys() != {"exp_id", "batch_id", "test_split", "seeds"}:
            raise ValueError("test configuration must contain experiment ID, batch ID, split and seeds")
        return cls(payload["exp_id"], payload["batch_id"], DataSplit.from_dict(payload["test_split"]), payload["seeds"])


class Verifier:
    """The evaluator sees the sealed manifest and trusted model/data references only."""

    def __init__(self, verifier_root: Path, *, evaluate: Evaluator | None = None,
                 build_report: ReportBuilder | None = None) -> None:
        self.store = VerifierStore(verifier_root)
        self.evaluate = evaluate
        self.build_report = build_report

    def register_config(self, config: VerifierConfig) -> None:
        """Seal the test split and seeds under the independent verifier root."""
        with self.store.lock():
            self.store._seal(config.to_dict(), "test.json")

    def config(self) -> VerifierConfig:
        """Read the pre-registered held-out JSON from this verifier's root only."""
        return VerifierConfig.from_dict(_read_json(self.store._file("test.json")))

    def register(self, frozen: FrozenManifest) -> None:
        """Register a complete batch, requiring any pre-registered test config to match."""
        frozen.validate()
        config = VerifierConfig(frozen.exp_id, frozen.batch_id, frozen.test_split, frozen.seeds)
        with self.store.lock():
            previous = self.store._read("test.json")
            if previous is not None and previous != config.to_dict():
                raise ValueError("frozen manifest disagrees with the pre-registered test configuration")
            self.store._seal(config.to_dict(), "test.json")
            self.store._register_frozen(frozen)

    def frozen(self) -> FrozenManifest:
        return self.store.frozen()

    def result(self, task_id: str) -> PairedEvalResult | None:
        return self.store.result(task_id)

    def run(self) -> dict[str, Any]:
        """Resume uncommitted task IDs and publish the report once all tasks are terminal."""
        with self.store.lock():
            frozen = self.frozen()
            published = self.store.report()
            if published is not None:
                return published
            results = self._complete(frozen)
            if self.build_report is None:
                from stochopia.harness.report import build_report
            else:
                build_report = self.build_report
            published = build_report(frozen, results)
            self.store._publish(published)
            return self.store.report()

    def _complete(self, frozen: FrozenManifest) -> tuple[PairedEvalResult, ...]:
        evaluator = self.evaluate
        if evaluator is None:
            from stochopia.harness.runner import Runner

            runner = Runner(frozen)

            def evaluate(task: EvaluationTask, parent: FitRecord, candidate: FitRecord,
                         _manifest: FrozenManifest) -> PairedEvalResult:
                return runner.run_paired_eval(task, parent, candidate)

            evaluator = evaluate
        fits = {record.fit_id: record for record in frozen.fits}
        results = []
        for task in frozen.tasks:
            parent = fits[task.parent_fit_id]
            candidate = fits[task.candidate_fit_id]
            result = self.store.result(task.task_id)
            if result is None:
                result = evaluator(task, parent, candidate, frozen)
                validate_task_result(result, task, parent, candidate)
                self.store._commit_result(task.task_id, result)
            else:
                validate_task_result(result, task, parent, candidate)
            results.append(result)
        return tuple(results)


def freeze(dev_export: DevelopmentExport, verifier_config: VerifierConfig, verifier_root: Path,
           *, tasks: Iterable[EvaluationTask]) -> FrozenManifest:
    """Seal published dev records plus registered test tasks in a separate batch directory.

    The verifier root and exported run root are canonical paths. Test files
    cannot be inside the development run namespace or above it, including
    through a symbolic link. Freezing consumes the detached export without
    opening the development run. Its experiment ID must match the test config.
    """
    root = Path(verifier_root).expanduser().resolve()
    development = dev_export.development_root.resolve()
    namespace = development.parent if development.parent.name == "stochopia_tier0" else development
    if root.is_relative_to(namespace) or namespace.is_relative_to(root):
        raise ValueError("the development and verifier roots must be separate")
    frozen = FrozenManifest(
        exp_id=verifier_config.exp_id, batch_id=verifier_config.batch_id, experiment=dev_export.experiment,
        test_split=verifier_config.test_split, seeds=verifier_config.seeds, tasks=tuple(tasks),
        selections=dev_export.selections, predictions=dev_export.predictions, search_trials=dev_export.search_trials,
        final_choices=dev_export.final_choices, fits=dev_export.fits,
    )
    Verifier(root).register(frozen)
    return frozen


def run_verifier(verifier_root: Path, *, evaluate: Evaluator | None = None,
                 build_report: ReportBuilder | None = None) -> dict[str, Any]:
    return Verifier(verifier_root, evaluate=evaluate, build_report=build_report).run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register-config", help="Seal test days and seeds independently of development")
    register.add_argument("config", type=Path)
    seal = commands.add_parser("freeze", help="Seal the development export and test tasks")
    seal.add_argument("export", type=Path)
    seal.add_argument("tasks", type=Path)
    verify = commands.add_parser("run", help="Execute the sealed batch or reopen its published report")
    verify.add_argument("--frozen", type=Path, help="Register a supplied FrozenManifest before verifying")
    args = parser.parse_args(argv)
    verifier = Verifier(args.verifier_root)
    if args.command == "register-config":
        verifier.register_config(VerifierConfig.from_dict(_read_json(args.config)))
    elif args.command == "freeze":
        exported = DevelopmentExport.from_dict(_read_json(args.export))
        tasks = map(EvaluationTask.from_dict, _read_json(args.tasks))
        freeze(exported, verifier.config(), args.verifier_root, tasks=tasks)
    else:
        if args.frozen:
            verifier.register(FrozenManifest.from_dict(_read_json(args.frozen)))
        print(json.dumps(verifier.run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
