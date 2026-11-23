"""Register and execute development rounds using the Stochopia P0 contract."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from algorithm_discovery import DiscoveryLedger
from stochopia.common import atomic_write_json
from stochopia.harness.contract import (
    EvaluationTask,
    ExperimentManifest,
    FinalChoice,
    FitRecord,
    PairedEvalResult,
    ReportStatus,
    ScenarioSpec,
    SearchTrialRecord,
    SelectionRecord,
    StructuredPrediction,
)

from .scoring import TrialProjection, append_discovery_events, project_trial, published_feedback, score_development
from .store import DevelopmentExport, DevelopmentStore, JsonObject, read_json, validate_task_result


@dataclass(frozen=True)
class CheckRequest:
    group_id: str
    scenario_id: str
    candidate_id: str
    purpose: Literal["common_prediction", "unselected_check"]
    visible_to_group: bool = False


Decision = tuple[SelectionRecord, Iterable[StructuredPrediction]]
Selector = Callable[[JsonObject, ScenarioSpec], Decision | None]
Request = SelectionRecord | CheckRequest
Outcome = PairedEvalResult | Iterable[PairedEvalResult]
Executor = Callable[[Request], Outcome]
TrialExecutor = Callable[[Request, str], Outcome]
History = tuple[tuple[JsonObject, JsonObject], ...]


def _json_value(value: JsonObject) -> JsonObject:
    return json.loads(json.dumps(value, allow_nan=False))


class Entry:
    """The development boundary. Selectors receive a group projection of this run."""

    def __init__(self, store: DevelopmentStore, manifest: ExperimentManifest | None = None) -> None:
        if manifest is not None:
            store.register(manifest)
        self.store = store
        self.manifest = store.manifest()
        self.groups = {group.group_id: group for group in self.manifest.groups}
        self.scenarios = {row.scenario_id: row for row in self.manifest.scenarios}
        self.candidates = {row.candidate_id: row for row in self.manifest.candidates}
        self.analyses = {row.analysis_id: row for row in self.manifest.analysis}

    # --- Development history and group projection ---

    def _history(self, round_index: int) -> History:
        if round_index < 1:
            raise ValueError("development rounds start at 1")
        rounds = self.store.rounds()
        if any(index > round_index for index in rounds):
            raise ValueError("cannot reopen a round after another has started")
        if tuple(index for index in rounds if index < round_index) != tuple(range(1, round_index)):
            raise ValueError("development rounds must be registered in order")
        history = []
        for index in range(1, round_index):
            plan = self.store.preregistered(index)
            feedback = self.store.feedback(index)
            if plan is None or feedback is None:
                raise ValueError("finish the preceding round before continuing")
            history.append((plan, feedback))
        return tuple(history)

    def _project(self, group_id: str, round_index: int, baseline_metrics: Mapping[str, Any],
                 evidence: Mapping[str, Any], history: History) -> JsonObject:
        """Copy common development metrics and only this group's allowed evidence and earlier feedback.

        Inputs are trusted development JSON, never verifier configuration. Incomplete rounds
        are excluded from history. Editing the returned object cannot mutate sealed history.
        """
        group = self.groups[group_id]
        return _json_value({
            "exp_id": self.manifest.exp_id,
            "group_id": group_id,
            "round_index": round_index,
            "dev_snapshot_id": f"round-{round_index}",
            "baseline_metrics": dict(baseline_metrics),
            "evidence": {key: evidence[key] for key in group.allowed_evidence if key in evidence},
            "feedback": [item for _, feedback in history for item in feedback[group_id]],
            "group": group.to_dict(),
            "prediction_candidate_set": dict(self.manifest.prediction_candidate_set),
            "search_budget": dict(self.manifest.budget["search"]),
        })

    def snapshot(self, group_id: str, round_index: int, *, baseline_metrics: Mapping[str, Any],
                 evidence: Mapping[str, Any] | None = None) -> JsonObject:
        """Read a group's sealed snapshot or project development-only inputs for a new round."""
        prior = self.store.preregistered(round_index)
        if prior is not None:
            return prior["snapshots"][group_id]
        return self._project(group_id, round_index, dict(baseline_metrics), dict(evidence or {}),
                             self._history(round_index))

    # --- Preregistration and budget checks ---

    def preregister_round(self, round_index: int, *, selections: Iterable[SelectionRecord],
                          predictions: Iterable[StructuredPrediction], baseline_metrics: Mapping[str, Any],
                          evidence: Mapping[str, Any] | None = None, checks: Iterable[CheckRequest] = ()) -> None:
        """Seal a supplied plan. Rewriting a sealed or already measured round fails."""
        with self.store.lock():
            history = self._history(round_index)
            snapshots = self._snapshots(round_index, baseline_metrics, evidence, history)
            self._seal_plan(round_index, tuple(selections), tuple(predictions), tuple(checks), snapshots, history)

    def _snapshots(self, round_index: int, baseline: Mapping[str, Any], evidence: Mapping[str, Any] | None,
                   history: History) -> dict[str, JsonObject]:
        return {group_id: self._project(group_id, round_index, dict(baseline), dict(evidence or {}), history)
                for group_id in self.groups}

    @staticmethod
    def _exposed_candidates(history: History) -> set[tuple[str, str, str]]:
        """Read only grouped, published outcomes when deciding what a selector already saw."""
        exposed = set()
        for _, feedback in history:
            for group, items in feedback.items():
                for item in items:
                    trial = item["trial"]
                    if trial["status"] in {"completed", "unavailable"}:
                        exposed.add((group, trial["scenario_id"], trial["candidate_id"]))
        return exposed

    def _validate_selections(self, round_index: int, selections: tuple[SelectionRecord, ...],
                             snapshots: dict[str, JsonObject], history: History) -> None:
        prior = [row for plan, _ in history for row in plan["selections"]]
        selection_ids = {row["selection_id"] for row in prior}
        current_scopes = set()
        exposed = self._exposed_candidates(history)
        for selection in selections:
            selection.validate()
            scenario = self.scenarios.get(selection.scenario_id)
            group = selection.group_id
            scope = (group, selection.scenario_id)
            if (selection.exp_id != self.manifest.exp_id or selection.round_index != round_index
                    or group not in snapshots or scenario is None
                    or selection.dev_snapshot_id != snapshots[group]["dev_snapshot_id"]
                    or selection.parent_id != scenario.parent_id
                    or (selection.candidate_id is not None and selection.candidate_id not in scenario.candidate_ids)):
                raise ValueError("selection does not match this round's group, menu or snapshot")
            if selection.selection_id in selection_ids or scope in current_scopes:
                raise ValueError("duplicate selection ID or group/scenario in this round")
            if (*scope, selection.candidate_id) in exposed:
                raise ValueError("a previously exposed candidate cannot be registered as a new search")
            previous = [row for row in prior if (row["group_id"], row["scenario_id"]) == scope]
            if previous and (selection.sequence <= previous[-1]["sequence"]
                             or selection.menu_id != previous[-1]["menu_id"]):
                raise ValueError("later selections must advance the sequence and keep the registered menu")
            selection_ids.add(selection.selection_id)
            current_scopes.add(scope)
        if round_index == 1 and current_scopes != {(group, scene) for group in self.groups for scene in self.scenarios}:
            raise ValueError("initial selections must cover every group and scenario")
        for scenario in self.scenarios:
            menu_ids = {row.menu_id for row in selections if row.scenario_id == scenario}
            if len(menu_ids) > 1:
                raise ValueError("all groups must use the same registered menu")

    def _validate_predictions(self, round_index: int, predictions: tuple[StructuredPrediction, ...],
                              selected_groups: set[str], snapshots: dict[str, JsonObject], history: History) -> None:
        prior_ids = {row["prediction_id"] for plan, _ in history for row in plan["predictions"]}
        exposed = self._exposed_candidates(history)
        coverage: dict[tuple[str, str, str], set[str]] = {}
        targets = set()
        for row in predictions:
            row.validate()
            scenario = self.scenarios.get(row.scenario_id)
            analysis = self.analyses.get(row.analysis_id)
            group = self.groups.get(row.group_id)
            scope = (row.group_id, row.scenario_id, row.candidate_id)
            target = (*scope, row.analysis_id)
            if scope in exposed:
                raise ValueError("cannot preregister a prediction after seeing its candidate's result")
            if row.prediction_id in prior_ids or target in targets:
                raise ValueError("duplicate prediction ID or analysis target in this round")
            if (row.exp_id != self.manifest.exp_id or row.round_index != round_index
                    or group is None or group.kind == "no_llm" or scenario is None or analysis is None
                    or row.parent_id != scenario.parent_id or row.candidate_id not in scenario.candidate_ids
                    or row.scenario_id != analysis.scenario_id
                    or row.dev_snapshot_id != snapshots[row.group_id]["dev_snapshot_id"]):
                raise ValueError("prediction does not match this round's group, menu, analysis or snapshot")
            if not row.abstained and (row.readout, row.unit, row.window) != (analysis.readout, analysis.unit, analysis.window):
                raise ValueError("prediction readout, unit and window must match the registered analysis")
            if row.group_id not in selected_groups:
                raise ValueError("prediction requires a same-round selection from its group")
            prior_ids.add(row.prediction_id)
            targets.add(target)
            coverage.setdefault(scope, set()).add(row.analysis_id)
        if round_index == 1:
            first_group, *other_groups = [group for group, spec in self.groups.items() if spec.kind != "no_llm"]
            for scene, candidates in self.manifest.prediction_candidate_set.items():
                for candidate in candidates:
                    expected = coverage.get((first_group, scene, candidate))
                    if not expected or any(
                        coverage.get((group, scene, candidate)) != expected for group in other_groups
                    ):
                        raise ValueError("LLM groups must cover the same nonempty common analysis set")

    def _completed_trials(self, history: History) -> tuple[SearchTrialRecord, ...]:
        records = []
        for plan, _ in history:
            for row in plan["trials"]:
                stored = self.store.measurement(plan["round_index"], row["trial_id"])
                if stored is None:
                    raise ValueError(f"published development round is missing trial {row['trial_id']!r}")
                records.append(SearchTrialRecord.from_dict(stored["trial"]))
        return tuple(records)

    def _plan_trials(self, round_index: int, selections: tuple[SelectionRecord, ...],
                     checks: tuple[CheckRequest, ...], history: History) -> list[SearchTrialRecord]:
        requested = [(row.group_id, row.scenario_id, row.candidate_id, "search", True, row.selection_id)
                     for row in selections if row.action == "select"]
        for check in checks:
            scenario = self.scenarios.get(check.scenario_id)
            if (check.group_id not in self.groups or scenario is None
                    or check.candidate_id == scenario.parent_id or check.candidate_id not in scenario.candidate_ids
                    or check.purpose not in {"common_prediction", "unselected_check"}
                    or not isinstance(check.visible_to_group, bool)):
                raise ValueError("check must match a registered group and non-baseline menu candidate")
            if (check.purpose == "common_prediction"
                    and check.candidate_id not in self.manifest.prediction_candidate_set[check.scenario_id]):
                raise ValueError("common check candidate is outside the registered prediction set")
            requested.append((check.group_id, check.scenario_id, check.candidate_id,
                              check.purpose, check.visible_to_group, None))

        counts = {}
        usage = {}
        for row in self._completed_trials(history):
            scope = (row.group_id, row.scenario_id)
            counts[scope] = max(counts.get(scope, 0), row.sequence)
            budget_key = "search" if row.charged_to_search else "final_check"
            for key, value in row.cost.items():
                cost_key = (row.group_id, budget_key, key)
                usage[cost_key] = usage.get(cost_key, 0.0) + value

        trials = []
        for number, (group, scenario, candidate, purpose, charged, selection_id) in enumerate(requested, start=1):
            budget_key = "search" if charged else "final_check"
            for key, limit in self.manifest.budget[budget_key].items():
                cost_key = (group, budget_key, key)
                spent = usage.get(cost_key, 0.0)
                if key == "trials":
                    if spent + 1 > limit:
                        raise ValueError(f"{group} exceeds its {budget_key} trial budget")
                    usage[cost_key] = spent + 1
                elif spent >= limit:
                    raise ValueError(f"{group} has exhausted its {budget_key} {key} budget")
            scope = (group, scenario)
            counts[scope] = counts.get(scope, 0) + 1
            trials.append(SearchTrialRecord(
                trial_id=f"r{round_index}-trial-{number}", exp_id=self.manifest.exp_id, group_id=group,
                scenario_id=scenario, sequence=counts[scope], candidate_id=candidate, purpose=purpose,
                charged_to_search=charged, budget_units=1.0, cost={}, status="pending",
                selection_id=selection_id,
            ))
        return trials

    def _seal_plan(self, round_index: int, selections: tuple[SelectionRecord, ...],
                   predictions: tuple[StructuredPrediction, ...], checks: tuple[CheckRequest, ...],
                   snapshots: dict[str, JsonObject], history: History) -> JsonObject:
        self._validate_selections(round_index, selections, snapshots, history)
        selected_groups = {row.group_id for row in selections}
        self._validate_predictions(round_index, predictions, selected_groups, snapshots, history)
        trials = self._plan_trials(round_index, selections, checks, history)
        plan = {
            "round_index": round_index,
            "snapshots": snapshots,
            "selections": [row.to_dict() for row in selections],
            "predictions": [row.to_dict() for row in predictions],
            "trials": [row.to_dict() for row in trials],
        }
        self.store.preregister(round_index, plan)
        return plan

    # --- Selection, execution, and feedback ---

    def _default_choice(self, snapshot: JsonObject, scenario: ScenarioSpec, history: History) -> Decision | None:
        spec = self.groups[snapshot["group_id"]]
        previous = [row for plan, _ in history for row in plan["selections"]
                    if (row["group_id"], row["scenario_id"]) == (spec.group_id, scenario.scenario_id)]
        measured = [row["trial"] for row in snapshot["feedback"] if row["trial"]["scenario_id"] == scenario.scenario_id]
        if any(trial["development_status"] == ReportStatus.SUPPORTED.value for trial in measured):
            return None
        options = [candidate for candidate in scenario.candidate_ids if candidate != scenario.parent_id]
        if spec.selection_rule in {"seeded-random", "random"}:
            random.Random(f"{self.manifest.seeds['rollout']}:{spec.group_id}:{scenario.scenario_id}").shuffle(options)
        elif spec.selection_rule not in {"fixed", "fixed-order"}:
            raise ValueError(f"provide a selector for no_llm rule {spec.selection_rule!r}")
        used = {trial["candidate_id"] for trial in measured}
        choice = next((candidate for candidate in options if candidate not in used), None)
        consumed = sum(item["trial"]["budget_units"] for item in snapshot["feedback"])
        cap = self.manifest.budget["search"].get("trials")
        if cap is not None and consumed + 1 > cap:
            choice = None
        if previous and choice is None and previous[-1]["action"] != "select":
            return None
        return SelectionRecord(
            selection_id=f"r{snapshot['round_index']}-{spec.group_id}-{scenario.scenario_id}",
            exp_id=self.manifest.exp_id, round_index=snapshot["round_index"], group_id=spec.group_id,
            scenario_id=scenario.scenario_id, sequence=1 if not previous else previous[-1]["sequence"] + 1,
            parent_id=scenario.parent_id, dev_snapshot_id=snapshot["dev_snapshot_id"],
            menu_id=(previous[-1]["menu_id"] if previous else f"{scenario.scenario_id}-menu"),
            action="select" if choice is not None else "keep_parent", candidate_id=choice,
            reason=spec.selection_rule if choice is not None else "No unmeasured candidate within the trial budget",
        ), ()

    def run_round(self, round_index: int, *, selectors: Mapping[str, Selector] | None = None,
                  evaluate: Executor | None = None, baseline_metrics: Mapping[str, Any] | None = None,
                  evidence: Mapping[str, Any] | None = None, checks: Iterable[CheckRequest] = ()) -> JsonObject:
        """Score only registered decisions. An interrupted round reuses its plan and committed trials."""
        with self.store.lock():
            completed = self.store.feedback(round_index)
            if completed is not None:
                return completed
            history = self._history(round_index)
            plan = self.store.preregistered(round_index)
            if plan is None:
                plan = self._select(round_index, selectors or {}, baseline_metrics or {}, evidence, checks, history)
            by_id = {row["selection_id"]: SelectionRecord.from_dict(row) for row in plan["selections"]}
            feedback = {group: [] for group in self.groups}
            real_executor = None if evaluate is not None else self._real_executor(history, plan)
            for row in plan["trials"]:
                trial = SearchTrialRecord.from_dict(row)
                stored = self.store.measurement(round_index, trial.trial_id)
                if stored is None:
                    request = by_id[trial.selection_id] if trial.selection_id else CheckRequest(
                        trial.group_id, trial.scenario_id, trial.candidate_id, trial.purpose, trial.charged_to_search
                    )
                    if evaluate is not None:
                        outcome = evaluate(request)
                    else:
                        outcome = real_executor(request, trial.trial_id)
                    stored = self._measure(trial, outcome)
                    self.store.commit_measurement(round_index, trial.trial_id, stored)
                if trial.charged_to_search:
                    feedback[trial.group_id].append(published_feedback(stored))
            self.store.publish_feedback(round_index, feedback)
            return feedback

    def _select(self, round_index: int, selectors: Mapping[str, Selector], baseline_metrics: Mapping[str, Any],
                evidence: Mapping[str, Any] | None, checks: Iterable[CheckRequest], history: History) -> JsonObject:
        if set(selectors) - self.groups.keys():
            raise ValueError("selector provided for an unregistered group")
        snapshots = self._snapshots(round_index, baseline_metrics, evidence, history)
        selections, predictions = [], []
        for scenario in self.scenarios.values():
            scene_selections = []
            explicit = [group_id for group_id in self.groups if group_id in selectors]
            default = [group_id for group_id, spec in self.groups.items()
                       if spec.kind == "no_llm" and group_id not in selectors]
            for group_id in explicit + default:
                snapshot = _json_value(snapshots[group_id])
                selector = selectors.get(group_id)
                if selector is None:
                    decision = self._default_choice(snapshot, scenario, history)
                    if decision is not None and scene_selections:
                        choice, forecasts = decision
                        decision = replace(choice, menu_id=scene_selections[0].menu_id), forecasts
                else:
                    decision = selector(snapshot, scenario)
                if decision is not None:
                    selection, forecasts = decision
                    forecasts = tuple(forecasts)
                    scope = (group_id, scenario.scenario_id)
                    if (selection.group_id, selection.scenario_id) != scope or any(
                        (forecast.group_id, forecast.scenario_id) != scope for forecast in forecasts
                    ):
                        raise ValueError("selector returned a decision for another group or scenario")
                    scene_selections.append(selection)
                    predictions.extend(forecasts)
            selections.extend(scene_selections)
        return self._seal_plan(round_index, tuple(selections), tuple(predictions), tuple(checks), snapshots, history)

    def _measure(self, trial: SearchTrialRecord, outcome: Outcome) -> JsonObject:
        results = (outcome,) if isinstance(outcome, PairedEvalResult) else tuple(outcome)
        if not results or len({result.task_id for result in results}) != len(results):
            raise ValueError("trial requires distinct registered measurements")
        cost = {"trials": trial.budget_units}
        parent = self.scenarios[trial.scenario_id].parent_id
        dev_units = set(self.manifest.splits["dev"].unit_ids)
        for result in results:
            result.validate()
            if (result.split, result.scenario_id, result.parent_id, result.candidate_id) != (
                "dev", trial.scenario_id, parent, trial.candidate_id
            ) or any(row.unit_id not in dev_units or row.train_repeat_id not in self.manifest.train_seeds
                     for row in result.observations):
                raise ValueError("result is not a registered development measurement for the trial")
            for key, value in result.cost.items():
                if key != "trials":
                    cost[key] = cost.get(key, 0.0) + value
        analyses = tuple(row for row in self.analyses.values() if row.scenario_id == trial.scenario_id)
        paired, status = score_development(analyses, results, self.manifest.splits["dev"].unit_ids)
        terminal = SearchTrialRecord(
            trial_id=trial.trial_id, exp_id=trial.exp_id, group_id=trial.group_id, scenario_id=trial.scenario_id,
            sequence=trial.sequence, candidate_id=trial.candidate_id, purpose=trial.purpose,
            charged_to_search=trial.charged_to_search, budget_units=trial.budget_units, cost=cost,
            status="unavailable" if status is ReportStatus.UNAVAILABLE else "completed",
            selection_id=trial.selection_id, development_status=status,
        )
        return {"trial": terminal.to_dict(), "results": [row.to_dict() for row in results],
                "paired_diffs": {key: delta.to_dict() for key, delta in paired.items()}}

    def native_feedback(self, round_index: int, group_id: str) -> tuple[TrialProjection, ...]:
        """Read one group's typed feedback after the round has been published."""
        if group_id not in self.groups:
            raise ValueError("native feedback requires a registered group")
        feedback = self.store.feedback(round_index)
        if feedback is None:
            raise ValueError("native feedback is available only after the round is published")
        forecasts = []
        for index in range(1, round_index + 1):
            plan = self.store.preregistered(index)
            if plan is None:
                raise ValueError("published round is missing its preregistered records")
            forecasts.extend(
                StructuredPrediction.from_dict(row) for row in plan["predictions"] if row["group_id"] == group_id
            )
        projections = []
        for row in feedback[group_id]:
            trial = SearchTrialRecord.from_dict(row["trial"])
            if trial.group_id != group_id:
                raise ValueError("published feedback contains a trial from another group")
            analyses = tuple(item for item in self.analyses.values() if item.scenario_id == trial.scenario_id)
            projections.append(project_trial(
                trial, self.candidates[trial.candidate_id], row, analyses, forecasts,
                round_index=round_index, seed=self.manifest.seeds["rollout"],
            ))
        return tuple(projections)

    def export_discovery_ledger(self, group_id: str) -> DiscoveryLedger:
        """Derive the native ledger below this development run for one registered group."""
        if group_id not in self.groups:
            raise ValueError("discovery ledger requires a registered group")
        with self.store.lock():
            ledger = DiscoveryLedger(self.store.discovery_ledger_path(group_id))
            rounds = (
                (index, self.native_feedback(index, group_id))
                for index in self.store.rounds() if self.store.feedback(index) is not None
            )
            append_discovery_events(ledger, exp_id=self.manifest.exp_id, group_id=group_id, rounds=rounds)
            return ledger

    def _accounted_fits(self, history: History, current: JsonObject) -> set[tuple[str, str, str]]:
        """Recover logically charged fits separately for each group and budget."""
        charged = set()
        plans = [plan for plan, _ in history] + [current]
        for plan in plans:
            for trial in plan["trials"]:
                payload = self.store.measurement(plan["round_index"], trial["trial_id"])
                if payload is not None:
                    budget = "search" if trial["charged_to_search"] else "final_check"
                    group = trial["group_id"]
                    for result in payload["results"]:
                        charged.add((group, budget, result["parent_fit_id"]))
                        charged.add((group, budget, result["candidate_fit_id"]))
        return charged

    def _real_executor(self, history: History, current: JsonObject) -> TrialExecutor:
        from stochopia.harness.runner import Runner

        runner = Runner(self.manifest)
        accounted_fits = self._accounted_fits(history, current)

        def fit(candidate_id: str, repeat: str) -> FitRecord:
            record = self.store.fit(candidate_id, repeat)
            if record is not None:
                return record
            record = runner.fit_candidate(candidate_id, repeat, self.store.model_directory())
            self.store.save_fit(record)
            return record

        def execute(request: Request, trial_id: str) -> Iterable[PairedEvalResult]:
            scene = self.scenarios[request.scenario_id]
            charged_to_search = isinstance(request, SelectionRecord) or request.visible_to_group
            budget = "search" if charged_to_search else "final_check"
            for repeat_number, repeat in enumerate(self.manifest.train_seeds, start=1):
                parent = fit(scene.parent_id, repeat)
                child = fit(request.candidate_id, repeat)
                task = EvaluationTask(
                    task_id=f"dev-{trial_id}-fit-{repeat_number}", split="dev",
                    scenario_id=scene.scenario_id, parent_fit_id=parent.fit_id, candidate_fit_id=child.fit_id,
                    unit_ids=self.manifest.splits["dev"].unit_ids, action_ref=f"{scene.scenario_id}:action",
                    random_inputs_ref=f"{scene.scenario_id}:dev:rollout",
                )
                # Shared artifacts reduce wall time. Each group still spends the same
                # registered training cost for the same sequence of fit requests.
                new_fit_keys = set()
                train_seconds = 0.0
                for record in (parent, child):
                    key = (request.group_id, budget, record.fit_id)
                    if key not in accounted_fits:
                        new_fit_keys.add(key)
                        train_seconds += float(record.config.get("train_seconds", 0.0))
                result = runner.run_paired_eval(task, parent, child, train_seconds=train_seconds)
                validate_task_result(result, task, parent, child)
                accounted_fits.update(new_fit_keys)
                yield result

        return execute

    # --- Frozen hand-off and command line ---

    def export(self, final_choices: Iterable[FinalChoice], fits: Iterable[FitRecord] | None = None) -> DevelopmentExport:
        with self.store.lock():
            rounds = self.store.rounds()
            history = self._history((rounds[-1] if rounds else 0) + 1)
            if not history:
                raise ValueError("cannot export an experiment without a completed round")
            if fits is None:
                stored_fits = []
                for candidate in self.manifest.candidates:
                    for repeat in self.manifest.train_seeds:
                        record = self.store.fit(candidate.candidate_id, repeat)
                        if record is not None:
                            stored_fits.append(record)
                fits = stored_fits
            return DevelopmentExport(
                self.manifest,
                tuple(SelectionRecord.from_dict(row) for plan, _ in history for row in plan["selections"]),
                tuple(StructuredPrediction.from_dict(row) for plan, _ in history for row in plan["predictions"]),
                self._completed_trials(history), tuple(final_choices), tuple(fits), self.store.path,
            )


def run_dev_round(dev_root: Path, round_index: int, **kwargs: Any) -> JsonObject:
    """Resume one experiment under a canonical runs/stochopia_tier0/<exp_id> root.

    The input is an existing development experiment directory, not the runs
    parent or a verifier batch. Its sealed development manifest must exist.
    """
    root = Path(dev_root).expanduser().absolute()
    if root.parent.name != "stochopia_tier0":
        raise ValueError("expected a registered development experiment directory")
    store = DevelopmentStore(root.parent.parent, root.name)
    return Entry(store).run_round(round_index, **kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--plan", type=Path, required=True, help="JSON selections, predictions and optional checks")
    parser.add_argument("--inputs", type=Path, help="JSON with development baseline_metrics and evidence")
    parser.add_argument("--export", type=Path, help="Write a development export when the plan includes final_choices")
    args = parser.parse_args(argv)
    manifest = ExperimentManifest.from_dict(read_json(args.manifest))
    entry = Entry(DevelopmentStore(args.runs_root, manifest.exp_id), manifest)
    plan = read_json(args.plan)
    inputs = read_json(args.inputs) if args.inputs else {}
    if entry.store.feedback(args.round) is None:
        entry.preregister_round(
            args.round, selections=map(SelectionRecord.from_dict, plan["selections"]),
            predictions=map(StructuredPrediction.from_dict, plan["predictions"]),
            baseline_metrics=inputs.get("baseline_metrics", {}), evidence=inputs.get("evidence", {}),
            checks=tuple(CheckRequest(**row) for row in plan.get("checks", ())),
        )
    entry.run_round(args.round)
    if args.export:
        exported = entry.export(map(FinalChoice.from_dict, plan["final_choices"]))
        atomic_write_json(args.export, exported.to_dict())


if __name__ == "__main__":
    main()
