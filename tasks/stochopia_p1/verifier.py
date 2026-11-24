"""Freeze development choices and report an isolated, once-registered holdout batch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from .paired_stats import EVALUATION_FAILED, VERIFIED_GAIN, assess, failed_evaluation, metric_delta, predictions_from_payload
from .store import (
    GROUPS, IMPACT, NLL, CandidateEvaluationFailed, DevelopmentExport, EvaluationRequest, Evaluator, Json,
    Ledger, ModelRef, PairedEvaluation, validate_dates,
)


SEED_TASK = "shared-seed"


def freeze(
    export: DevelopmentExport, verifier_root: Path, *, test_dates: tuple[int, ...], rollout_seed: int,
) -> Json:
    """Register the initial model, first trial and accepted final choice once.

    The first/final references may point to the same task. Any selected
    candidate is paired against the actual parent from its preregistration.
    The root receives the detached export values and no development writes.
    """
    ledger = Ledger(verifier_root)
    dev_root = export.development_root.expanduser().resolve()
    if ledger.path.is_relative_to(dev_root) or dev_root.is_relative_to(ledger.path):
        raise ValueError("development and verifier roots must be independent")
    plan = export.plan
    validate_dates(test_dates, "test_dates")
    if len(test_dates) < 2 or test_dates[0] <= plan.dev_dates[-1]:
        raise ValueError("held-out dates must follow the preregistered development dates")

    def request(task: str, group: str, parent: ModelRef, candidate: ModelRef) -> Json:
        return asdict(EvaluationRequest(task, group, parent, candidate, plan.train_dates, tuple(test_dates),
                                        plan.train_seeds, rollout_seed, "test"))

    seed = plan.seed_model
    tasks = [{"request": request(SEED_TASK, "shared", seed, seed), "predictions": []}]
    selections: dict[str, Json] = {}
    for group in GROUPS:
        history = export.groups[group]
        if len(history) != plan.rounds:
            raise ValueError("every group must finish development before freezing")
        first = history[0]
        final = next((trial for trial in reversed(history) if trial["verdict"] == VERIFIED_GAIN), None)
        selections[group] = {
            "initial": SEED_TASK,
            "first": first["trial_id"] if first["candidate"] is not None else None,
            "first_development_verdict": first["verdict"],
            "final": final["trial_id"] if final is not None else SEED_TASK,
        }
        registered: set[str] = set()
        for trial in (first, final):
            if trial is None or trial["candidate"] is None or trial["trial_id"] in registered:
                continue
            task = trial["trial_id"]
            registered.add(task)
            tasks.append({"request": request(task, group, ModelRef(**trial["parent"]), ModelRef(**trial["candidate"])),
                          "predictions": trial["predictions"]})

    manifest = {
        "test_dates": list(test_dates), "tasks": tasks, "selections": selections,
        "rules": [asdict(rule) for rule in plan.metric_rules], "nll_guardrail": plan.nll_guardrail,
    }
    with ledger.lock():
        return ledger.seal(manifest, "frozen.json")


def _rates(trials: list[Json]) -> Json:
    verified = sum(row["verdict"] == VERIFIED_GAIN for row in trials)
    failed = sum(row["verdict"] == EVALUATION_FAILED for row in trials)
    predictions = [p for trial in trials for p in trial["prediction_verdicts"]]
    supported = sum(row["status"] == "SUPPORTED" for row in predictions)
    decided = sum(row["status"] in ("SUPPORTED", "REFUTED") for row in predictions)
    return {
        "verified_mechanism_gain_rate": verified / len(trials) if trials else None,
        "prediction_hit_rate": supported / len(predictions) if predictions else None,
        "prediction_decidable_count": decided, "prediction_count": len(predictions),
        "measured_candidates": len(trials) - failed, "failed_candidates": failed,
    }


def _report(manifest: Json, evaluated: Mapping[str, Json]) -> Json:
    seed = evaluated[SEED_TASK]["candidate"]
    groups: dict[str, Json] = {}

    def reference(task: str | None) -> Json | None:
        if task is None:
            return None
        row = evaluated[task]
        evidence = row["candidate"]
        return {"model_id": row["model_id"], "metrics": evidence,
                "initial_nll_delta": metric_delta(seed.get(NLL), evidence.get(NLL)),
                "initial_impact_delta": metric_delta(seed.get(IMPACT), evidence.get(IMPACT)),
                "paired_assessment": row["assessment"]}

    for group, selection in manifest["selections"].items():
        chosen = dict.fromkeys((selection["first"], selection["final"]))
        trials = [evaluated[task]["assessment"] for task in chosen if task not in (None, SEED_TASK)]
        groups[group] = {
            "initial": reference(SEED_TASK), "first": reference(selection["first"]),
            "first_development_verdict": selection["first_development_verdict"],
            "final": reference(selection["final"]), **_rates(trials),
        }

    all_trials = [row["assessment"] for task, row in evaluated.items() if task != SEED_TASK]
    return {"held_out_dates": manifest["test_dates"], "paired_runs": len(evaluated), "groups": groups,
            **_rates(all_trials)}


def run_verifier(verifier_root: Path, *, evaluate: Evaluator) -> Json:
    """Resume uncommitted task IDs and atomically publish a single held-out report."""
    ledger = Ledger(verifier_root)
    with ledger.lock():
        manifest = ledger.read("frozen.json")
        if manifest is None:
            raise ValueError("register the held-out batch before running the verifier")
        report = ledger.read("report.json")
        if report is not None:
            return report
        evaluated: dict[str, Json] = {}
        for item in manifest["tasks"]:
            request = EvaluationRequest.from_dict(item["request"])
            path = ("results", f"{request.trial_id}.json")
            recorded = ledger.read(*path)
            if recorded is None:
                try:
                    pair = evaluate(request)
                except CandidateEvaluationFailed as exc:
                    if request.trial_id == SEED_TASK:
                        raise
                    result = failed_evaluation(predictions_from_payload(item), str(exc))
                else:
                    pair.validate(request)
                    result = pair.to_dict()
                recorded = ledger.seal(result, *path)
            if recorded.get("verdict") == EVALUATION_FAILED:
                evaluated[request.trial_id] = {"model_id": request.candidate.model_id,
                                               "candidate": {}, "assessment": recorded}
                continue
            pair = PairedEvaluation.from_dict(recorded)
            pair.validate(request)
            assessment = None
            if item["predictions"]:
                assessment = assess(pair, request, predictions_from_payload(item))
            evaluated[request.trial_id] = {"model_id": request.candidate.model_id,
                                           "candidate": pair.candidate_evidence, "assessment": assessment}
        return ledger.seal(_report(manifest, evaluated), "report.json")
