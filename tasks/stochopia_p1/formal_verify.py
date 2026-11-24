"""Execute the preselected M4 holdout tasks and publish a detached report."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .formal_manifest import ARM_MODELS, MANIFEST, SYMBOLS, SourceArchive, validate_frozen
from .paired_stats import (
    EVALUATION_FAILED, GUARDRAIL_VIOLATION, VERIFIED_GAIN, assess, failed_evaluation, metric_delta,
    paired_readouts, predictions_from_payload,
)
from .store import (
    CandidateEvaluationFailed, EvaluationRequest, Evaluator, IMPACT, Json, Ledger, NLL,
    NonfiniteCandidateLoss, PairedEvaluation,
)


AUDIT_ROLE = "audit_only_rejected_unverified"


def load_frozen_study(root: Path) -> Json:
    source_ledger = Ledger(root)
    manifest = source_ledger.read(MANIFEST)
    if manifest is None:
        raise ValueError("complete and freeze all development trials before loading holdout")
    validate_frozen(manifest, SourceArchive(source_ledger))
    return manifest


def _request(manifest: Json, task: Json, archive: SourceArchive) -> EvaluationRequest:
    return EvaluationRequest(
        task["task_id"], task["group"], archive.load(task["parent"]), archive.load(task["candidate"]),
        tuple(manifest["data"]["train_dates"]), tuple(manifest["test_dates"]), tuple(manifest["train_seeds"]),
        manifest["test_rollout_seed"], "test",
    )


def _assess_task(pair: PairedEvaluation, request: EvaluationRequest, task: Json) -> Json:
    """Record paired 95% readouts on the same registered day blocks."""
    predictions = predictions_from_payload(task)
    if predictions:
        assessment = assess(pair, request, predictions)
        readouts = assessment["readouts"]
    else:
        assessment = None
        readouts = paired_readouts(pair, request, (NLL, IMPACT))
    return {
        "status": "measured",
        "metrics": dict(pair.candidate_evidence),
        "parent_metrics": dict(pair.parent_evidence),
        "assessment": assessment,
        "readouts": readouts,
    }


def _run_task(
    manifest: Json, task: Json, archive: SourceArchive, ledger: Ledger, evaluate: Evaluator,
) -> Json:
    path = ("results", f"{task['task_id']}.json")
    saved = ledger.read(*path)
    if saved is not None:
        return saved
    request = _request(manifest, task, archive)
    try:
        pair = evaluate(request)
    except CandidateEvaluationFailed as exc:
        if "initial" in task["roles"]:
            raise
        assessment = failed_evaluation(predictions_from_payload(task), str(exc))
        if isinstance(exc, NonfiniteCandidateLoss):
            assessment["verdict"] = GUARDRAIL_VIOLATION
            assessment["guardrails"] = {"finite_nll_by_seed": False, "nll_passed": False}
        record = {
            "status": EVALUATION_FAILED, "metrics": {}, "parent_metrics": {}, "readouts": {},
            "assessment": assessment,
        }
    else:
        pair.validate(request)
        record = _assess_task(pair, request, task)
    return ledger.seal({**record, **task}, *path)


# --- Selection reporting ---

def _prediction_rates(rows: list[Json]) -> Json:
    counts: Json = {}
    for position, role in enumerate(("primary", "guardrail")):
        statuses = []
        for row in rows:
            forecasts = (row["assessment"] or {}).get("prediction_verdicts", [])
            if len(forecasts) > position:
                statuses.append(forecasts[position]["status"])
        supported = statuses.count("SUPPORTED")
        refuted = statuses.count("REFUTED")
        inconclusive = statuses.count("INCONCLUSIVE")
        counts[role] = {
            "supported": supported, "refuted": refuted, "inconclusive": inconclusive,
            "registered": len(statuses), "decidable": supported + refuted,
            "hit_rate": supported / len(statuses) if statuses else None,
        }
    return counts


def _endpoints(selections: Json, evaluated: Mapping[str, Json]) -> Json:
    endpoints = {}
    for key, selection in selections.items():
        base = evaluated[selection["initial"]]["metrics"]
        row = dict(selection)
        for phase in ("initial", "first", "final"):
            task_id = selection[phase]
            if task_id is None:
                row[phase] = None
                continue
            result = evaluated[task_id]
            metrics = result["metrics"]
            row[phase] = {
                "task_id": task_id, "status": result["status"],
                "model_id": result["candidate"]["model_id"], "metrics": metrics,
                "nll_delta_from_initial": metric_delta(base.get(NLL), metrics.get(NLL)),
                "impact_delta_from_initial": metric_delta(base.get(IMPACT), metrics.get(IMPACT)),
                "impact_ci95": result["readouts"].get(IMPACT, {}).get("candidate_interval"),
                "direct_parent_assessment": result["assessment"],
            }
        endpoints[key] = row
    return endpoints


def _audit_summary(rows: list[Json]) -> Json:
    measured = [row for row in rows if row["status"] == "measured"]
    return {
        "candidates": len(rows), "measured": len(measured),
        "survived_benchmark_gate": sum(row["assessment"]["has_benchmark_gain"] for row in measured),
        "survived_mechanism_gate": sum(row["assessment"]["verdict"] == VERIFIED_GAIN for row in measured),
        "failed_evaluations": len(rows) - len(measured),
        "prediction_rates": _prediction_rates(rows),
    }


def _cross_matrix(registration: Json, measured: Mapping[str, Json]) -> Json:
    matrices = {}
    for arm, choices in registration.items():
        cells = {}
        for source, row in choices.items():
            target = row["target_stock"]
            source_cells = {}
            target_tasks = {source: row["diagonal"], target: row["off_diagonal"]}
            for stock, task_id in target_tasks.items():
                result = measured[task_id]
                source_cells[stock] = {
                    "task_id": task_id, "source": result["candidate"],
                    "status": result["status"], "metrics": result["metrics"],
                    "impact_ci95": result["readouts"].get(IMPACT, {}).get("candidate_interval"),
                }
            auxiliary = row["auxiliary"]
            if auxiliary is not None:
                auxiliary = {
                    "dev_score": auxiliary["dev_score"],
                    "diagonal": measured[auxiliary["diagonal"]],
                    "off_diagonal": measured[auxiliary["off_diagonal"]],
                }
            cells[source] = {
                "representative": row["representative"], "dev_score": row["dev_score"],
                "retains_seed": row["retains_seed"], "cells": source_cells, "auxiliary": auxiliary,
            }
        matrices[arm] = cells
    return matrices


def _summarize(manifest: Json, evaluated: Mapping[str, Json]) -> Json:
    endpoint_rows = _endpoints(manifest["selections"], evaluated)
    audit = [row for row in evaluated.values() if AUDIT_ROLE in row["roles"]]
    per_arm = {}
    for stock in SYMBOLS:
        per_arm[stock] = {}
        for arm in ARM_MODELS:
            selected = [
                row for row in endpoint_rows.values() if row["stock"] == stock and row["arm"] == arm
            ]
            task_ids = set()
            for row in selected:
                for endpoint in (row["first"], row["final"]):
                    if endpoint is not None and endpoint["task_id"] != row["initial"]["task_id"]:
                        task_ids.add(endpoint["task_id"])
            assayed = [evaluated[key] for key in sorted(task_ids)]
            verified = sum(row["assessment"]["verdict"] == VERIFIED_GAIN for row in assayed)
            per_arm[stock][arm] = {
                "verified_gain": verified,
                "verified_mechanism_gain_rate": verified / len(assayed) if assayed else None,
                "verified_denominator": len(assayed),
                "prediction_rates": _prediction_rates(assayed),
                "forecast_type": "fixed_placeholder" if arm == "no_llm" else "preregistered_llm",
                "rejected_audit": _audit_summary(
                    [row for row in audit if row["stock"] == stock and row["group"] == arm]
                ),
            }
    return {
        "schema": manifest["schema"], "code": manifest["code"], "test_dates": manifest["test_dates"],
        "train_dates": manifest["data"]["train_dates"], "train_seeds": manifest["train_seeds"],
        "measured_tasks": sum(row["status"] == "measured" for row in evaluated.values()),
        "scheduled_tasks": len(evaluated), "trajectories": endpoint_rows,
        "arm_summary": per_arm, "audit_summary": _audit_summary(audit),
        "audit_only_rejected_unverified": audit,
        "cross_regime": _cross_matrix(manifest["cross_regime"], evaluated),
    }


def verify_frozen_study(root: Path, *, evaluators: Mapping[str, Evaluator]) -> Json:
    """Evaluate each sealed task under a verifier-only lock and resume saved results."""
    if set(evaluators) != set(SYMBOLS):
        raise ValueError("the verifier requires both stock evaluators")
    manifest = load_frozen_study(root)
    archive_ledger = Ledger(root)
    archive = SourceArchive(archive_ledger)
    ledger = Ledger(archive_ledger.file("verification"))
    with ledger.lock():
        report = ledger.read("report.json")
        if report is not None:
            return report
        records = {}
        for task in manifest["tasks"]:
            records[task["task_id"]] = _run_task(
                manifest, task, archive, ledger, evaluators[task["stock"]],
            )
        return ledger.seal(_summarize(manifest, records), "report.json")
