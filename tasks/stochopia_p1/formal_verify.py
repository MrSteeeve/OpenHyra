"""Execute the preselected M4 holdout tasks and publish a detached report."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from stochopia.common import atomic_write_bytes, atomic_write_json

from .formal_manifest import ARM_MODELS, M5_SCHEMA, MANIFEST, SYMBOLS, SourceArchive, validate_frozen
from .paired_stats import (
    EVALUATION_FAILED, GUARDRAIL_VIOLATION, VERIFIED_GAIN, assess, failed_evaluation, metric_delta,
    paired_readouts, predictions_from_payload,
)
from .store import (
    CandidateEvaluationFailed, EvaluationRequest, Evaluator, IMPACT, Json, Ledger, NLL,
    NonfiniteCandidateLoss, PairedEvaluation,
)


AUDIT_ROLE = "audit_only_rejected_unverified"
RESOURCE_RETRY = "retry-v2-resource"
V2_SCHEMA = "stochopia-p1-m4-v2"


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
        manifest.get("analysis_version", "m4"),
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


def _evaluate_task(manifest: Json, task: Json, archive: SourceArchive, evaluate: Evaluator) -> Json:
    """Measure a registered pair without changing any ledger files."""
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
    return {**record, **task}


def _run_task(
    manifest: Json, task: Json, archive: SourceArchive, ledger: Ledger, evaluate: Evaluator,
) -> Json:
    path = ("results", f"{task['task_id']}.json")
    saved = ledger.read(*path)
    if saved is not None:
        return saved
    return ledger.seal(_evaluate_task(manifest, task, archive, evaluate), *path)


# --- Selection reporting ---

def _prediction_rates(rows: list[Json]) -> Json:
    counts: Json = {}
    for position, role in enumerate(("primary", "guardrail")):
        forecasts_for_role = []
        for row in rows:
            forecasts = (row["assessment"] or {}).get("prediction_verdicts", [])
            if len(forecasts) > position:
                forecasts_for_role.append(forecasts[position])
        statuses = [forecast["status"] for forecast in forecasts_for_role]
        supported = statuses.count("SUPPORTED")
        refuted = statuses.count("REFUTED")
        inconclusive = statuses.count("INCONCLUSIVE")
        counts[role] = {
            "supported": supported, "refuted": refuted, "inconclusive": inconclusive,
            "registered": len(statuses), "decidable": supported + refuted,
            "hit_rate": supported / len(statuses) if statuses else None,
        }
        points = [forecast for forecast in forecasts_for_role if forecast.get("point_available")]
        hits = sum(bool(forecast.get("point_in_interval")) for forecast in points)
        counts[role].update(
            point_evaluable=len(points), point_in_interval=hits,
            point_hit_rate=hits / len(points) if points else None,
        )
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
    summary = {
        "schema": manifest["schema"], "code": manifest["code"], "test_dates": manifest["test_dates"],
        "train_dates": manifest["data"]["train_dates"], "train_seeds": manifest["train_seeds"],
        "measured_tasks": sum(row["status"] == "measured" for row in evaluated.values()),
        "scheduled_tasks": len(evaluated), "trajectories": endpoint_rows,
        "arm_summary": per_arm, "audit_summary": _audit_summary(audit),
        "audit_only_rejected_unverified": audit,
        "cross_regime": _cross_matrix(manifest["cross_regime"], evaluated),
    }
    if manifest["schema"] == M5_SCHEMA:
        summary.update(
            planned_test_dates=manifest["planned_test_dates"],
            replacement_test_dates=manifest["replacement_test_dates"],
            excluded_test_dates=manifest["excluded_test_dates"],
            analysis_version="m5",
        )
    return summary


def _retryable_result(row: Json) -> bool:
    return row["status"] == EVALUATION_FAILED and (row["assessment"] or {}).get("verdict") == EVALUATION_FAILED


def _snapshot(ledger: Ledger, source: tuple[str, ...], destination: tuple[str, ...]) -> None:
    """Copy an original published record once, preserving its file bytes."""
    content = ledger.file(*source).read_bytes()
    path = ledger.file(*destination)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("the original resource-failure snapshot differs from its result")
        return
    atomic_write_bytes(path, content)


def _retry_registration(manifest: Json, ledger: Ledger, retry_code: Mapping[str, str]) -> Json:
    """Seal the failed task roster and its unchanged before evidence first."""
    path = (RESOURCE_RETRY, "registration.json")
    previous = ledger.read(*path)
    if previous is not None:
        if previous["frozen_code"] != manifest["code"] or previous["retry_code"] != retry_code:
            raise ValueError("resume the resource retry using its registered code")
        return previous

    report = ledger.read("report.json")
    if report is None:
        raise ValueError("finish the original frozen verification before retrying failed tasks")
    failed_ids = []
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        row = ledger.read("results", f"{task_id}.json")
        if row is None:
            raise ValueError("resource retry requires every original task to have a terminal record")
        if _retryable_result(row):
            if "initial" in task["roles"]:
                raise ValueError("the original registered baseline must already have a measured result")
            failed_ids.append(task_id)
            _snapshot(ledger, ("results", f"{task_id}.json"),
                      (RESOURCE_RETRY, "before", "results", f"{task_id}.json"))
    _snapshot(ledger, ("report.json",), (RESOURCE_RETRY, "before", "report.json"))
    registration = {
        "frozen_code": manifest["code"], "retry_code": dict(retry_code), "task_ids": failed_ids,
        "original_measured_tasks": report["measured_tasks"],
        "resource_limits": {"train_wall_seconds": 180.0, "evaluate_wall_seconds": 1800.0,
                            "cpu_seconds_multiplier": 4},
    }
    return ledger.seal(registration, *path)


def _measure_pending(
    manifest: Json, tasks: list[Json], archive: SourceArchive, ledger: Ledger,
    evaluators: Mapping[str, Evaluator], jobs: int, results_path: tuple[str, ...],
) -> None:
    """Run each pending pair once, sealing only from the lock-owning thread."""
    pending = []
    for task in tasks:
        path = (*results_path, f"{task['task_id']}.json")
        if ledger.read(*path) is None:
            pending.append(task)

    def evaluate(task: Json) -> Json:
        return _evaluate_task(manifest, task, archive, evaluators[task["stock"]])

    # The caller owns the ledger lock. Only its thread seals finished results.
    if jobs == 1:
        for task in pending:
            row = evaluate(task)
            ledger.seal(row, *results_path, f"{task['task_id']}.json")
        return

    first_error: Exception | None = None
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(evaluate, task): task for task in pending}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            else:
                task_id = futures[future]["task_id"]
                ledger.seal(row, *results_path, f"{task_id}.json")
    if first_error is not None:
        raise first_error


def _publish_retry(manifest: Json, ledger: Ledger, registration: Json) -> Json:
    """Republish the completed repair batch, touching only original failures."""
    repaired: set[str] = set(registration["task_ids"])
    records = {}
    recovered = []
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        name = f"{task_id}.json"
        original_path = ledger.file("results", name)
        if task_id in repaired:
            retry_path = ledger.file(RESOURCE_RETRY, "results", name)
            before_path = ledger.file(RESOURCE_RETRY, "before", "results", name)
            retry = retry_path.read_bytes()
            if original_path.read_bytes() not in (before_path.read_bytes(), retry):
                raise ValueError("the result does not match its original or sealed retry record")
            if original_path.read_bytes() != retry:
                atomic_write_bytes(original_path, retry)
        row = ledger.read("results", name)
        if row is None:
            raise ValueError("a registered task has no result record")
        records[task_id] = row
        if task_id in repaired and row["status"] == "measured":
            recovered.append(task_id)
    report = _summarize(manifest, records)
    report["resource_retry"] = {
        **registration, "recovered_task_ids": recovered,
        "remaining_failed_task_ids": sorted(repaired - set(recovered)),
        "measured_tasks_after": report["measured_tasks"],
    }
    atomic_write_json(ledger.file("report.json"), report)
    ledger.seal(report["resource_retry"], RESOURCE_RETRY, "completed.json")
    return report


def _retry_failed_tasks(
    manifest: Json, archive: SourceArchive, ledger: Ledger, evaluators: Mapping[str, Evaluator],
    retry_code: Mapping[str, str], jobs: int,
) -> Json:
    if manifest["schema"] != V2_SCHEMA:
        raise ValueError("the resource repair is registered for M4 v2 only")
    registration = _retry_registration(manifest, ledger, retry_code)
    completed = ledger.read(RESOURCE_RETRY, "completed.json")
    if completed is not None:
        report = ledger.read("report.json")
        if report is None or report.get("resource_retry") != completed:
            raise ValueError("the completed resource retry is missing its published report")
        return report
    selected = set(registration["task_ids"])
    tasks = [task for task in manifest["tasks"] if task["task_id"] in selected]
    if len(tasks) != len(selected):
        raise ValueError("the resource retry contains an unregistered task")
    _measure_pending(manifest, tasks, archive, ledger, evaluators, jobs, (RESOURCE_RETRY, "results"))
    return _publish_retry(manifest, ledger, registration)


def verify_frozen_study(
    root: Path, *, evaluators: Mapping[str, Evaluator], retry_failed: bool = False,
    retry_code: Mapping[str, str] | None = None, jobs: int = 1,
) -> Json:
    """Evaluate each sealed task under a verifier-only lock and resume saved results."""
    if set(evaluators) != set(SYMBOLS):
        raise ValueError("the verifier requires both stock evaluators")
    if jobs < 1 or jobs > 3:
        raise ValueError("the resource retry supports one to three workers")
    if retry_failed and retry_code is None:
        raise ValueError("register the current code for the resource retry")
    manifest = load_frozen_study(root)
    archive_ledger = Ledger(root)
    archive = SourceArchive(archive_ledger)
    ledger = Ledger(archive_ledger.file("verification"))
    with ledger.lock():
        if retry_failed:
            return _retry_failed_tasks(manifest, archive, ledger, evaluators, retry_code, jobs)
        if manifest["schema"] == M5_SCHEMA:
            from .holdout_data import validate_m5_receipt

            receipt = ledger.read("test-data-reconciliation.json")
            if receipt is None:
                raise ValueError("the M5 holdout requires its registered acquisition receipt")
            effective_dates = validate_m5_receipt(manifest, receipt)
            manifest = {
                **manifest,
                "planned_test_dates": manifest["test_dates"],
                "test_dates": list(effective_dates),
                "excluded_test_dates": receipt["excluded_test_dates"],
            }
        report = ledger.read("report.json")
        if report is not None:
            return report
        if manifest["schema"] == M5_SCHEMA:
            _measure_pending(manifest, manifest["tasks"], archive, ledger, evaluators, jobs, ("results",))
            records = {}
            for task in manifest["tasks"]:
                task_id = task["task_id"]
                row = ledger.read("results", f"{task_id}.json")
                if row is None:
                    raise ValueError(f"missing terminal result for frozen task {task_id}")
                records[task_id] = row
            return ledger.seal(_summarize(manifest, records), "report.json")
        records = {}
        for task in manifest["tasks"]:
            records[task["task_id"]] = _run_task(
                manifest, task, archive, ledger, evaluators[task["stock"]],
            )
        return ledger.seal(_summarize(manifest, records), "report.json")
