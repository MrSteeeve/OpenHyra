"""Reduced post-freeze M5 holdout schedule, with one profile per stock/source.

The original study manifest is never rewritten. A separate selection receipt is
sealed before holdout acquisition. Role-specific comparisons reuse the same
profiles without conflating forecasts made against different development parents.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Protocol, runtime_checkable

from .formal_manifest import ARM_MODELS, M5_SCHEMA, MANIFEST, SYMBOLS, SourceArchive, roster
from .paired_stats import (
    DRIFT_LIMIT, EVALUATION_FAILED, GUARDRAIL_VIOLATION, NLL_GUARDRAIL, VERIFIED_GAIN,
    failed_evaluation, paired_readouts, predictions_from_payload,
)
from .store import (
    CandidateEvaluationFailed, EvaluationRequest, IMPACT, Json, Ledger, ModelRef, NLL,
    NonfiniteCandidateLoss, PairedEvaluation,
)


MODE = "m5-closing"
REGISTRATION = "closing-registration.json"
SECONDARY_ROLE = "best_dev_guarded"
PRIMARY_AUDIT_ROLE = "audit_only_rejected_unverified"
NOT_SCHEDULED = "not_scheduled"
NO_GUARDED = "not_scheduled_no_guarded_candidate"
CATEGORIES = ("llm_best", "no_llm_best")


@runtime_checkable
class ClosingProfiler(Protocol):
    def verification_profile(
        self, model: ModelRef, request: EvaluationRequest, *, candidate: bool,
    ) -> Json: ...


# --- Selection on sealed development evidence only ---


def _finite(value: object) -> float | None:
    if type(value) not in (int, float) or not isfinite(value):
        return None
    return float(value)


def _guarded(trial: Json) -> bool:
    """Require the frozen hard-guard flags and their finite numerical readouts.

    NLL candidate-minus-parent <= .05 (same 1e-12 tolerance as assess),
    exhausted-window count change <= 0, candidate mean absolute drift < 20
    ticks or < parent drift when the parent is already at least 20 ticks.
    A missing/nonfinite measurement or uncompiled model cannot be selected.
    """
    if trial.get("candidate") is None or trial.get("program", {}).get("status") != "ready":
        return False
    if _finite(trial.get("candidate_metrics", {}).get(IMPACT)) is None:
        return False
    guards = trial.get("guardrails") or {}
    flags = ("finite_nll_by_seed", "nll_passed", "exhaustion_passed", "drift_passed")
    if not all(guards.get(flag) is True for flag in flags):
        return False
    nll = _finite(guards.get("delta_val_nll_total"))
    exhausted = _finite(guards.get("delta_exhausted_windows"))
    parent_drift = _finite(guards.get("parent_free_abs_drift_ticks"))
    child_drift = _finite(guards.get("child_free_abs_drift_ticks"))
    if nll is None or exhausted is None or parent_drift is None or child_drift is None:
        return False
    return (
        nll <= NLL_GUARDRAIL + 1e-12
        and exhausted <= 0
        and child_drift < max(DRIFT_LIMIT, parent_drift)
    )


def _score(trial: Json) -> float:
    return float(trial["candidate_metrics"][IMPACT])


def _task(task_id: str, stock: str, arm: str, parent: Json, candidate: Json,
          predictions: list[Json], role: str) -> Json:
    return {
        "task_id": task_id, "stock": stock, "group": arm, "parent": parent,
        "candidate": candidate, "predictions": predictions, "roles": [role],
    }


def _profile_id(stock: str, model: Json) -> str:
    return f"{stock}-{model['source_sha256']}"


def _schedule(manifest: Json) -> Json:
    """Construct A, B and C without consulting any holdout data or prior results.

    Ties retain the earliest registered arm, search seed and round, using the
    original formal roster order, not JSON order or candidate score delta.
    Every retained trial has its own parent and forecast even if its executable
    source is identical to another trial. Source profiles themselves are unique.
    """
    if manifest["schema"] != M5_SCHEMA:
        raise ValueError("closing verification is registered for M5 only")
    tasks: dict[str, Json] = {}
    trials: dict[str, Json] = {}
    by_group: dict[tuple[str, str], list[Json]] = {}
    for item in roster():
        group = by_group.setdefault((item.stock, item.arm), [])
        for trial in manifest["trajectories"][item.key]["trials"]:
            row = {**trial, "stock": item.stock, "arm": item.arm, "search_seed": item.search_seed}
            trials[trial["trial_id"]] = row
            group.append(row)

    def add_trial(trial_id: str, role: str) -> None:
        if trial_id in tasks:
            if role not in tasks[trial_id]["roles"]:
                tasks[trial_id]["roles"].append(role)
            return
        row = trials[trial_id]
        tasks[trial_id] = _task(
            trial_id, row["stock"], row["arm"], row["parent"], row["candidate"], row["predictions"], role,
        )

    for stock in SYMBOLS:
        initial = f"{stock}-seed"
        seed = manifest["baselines"][stock]["model"]
        tasks[initial] = _task(initial, stock, "shared", seed, seed, [], "initial")
    for selection in manifest["selections"].values():
        if selection["final"] != selection["initial"]:
            add_trial(selection["final"], "final")
        for trial_id in selection[PRIMARY_AUDIT_ROLE]:
            add_trial(trial_id, PRIMARY_AUDIT_ROLE)

    secondary = []
    best: dict[tuple[str, str], Json] = {}
    for stock in SYMBOLS:
        for arm in ARM_MODELS:
            eligible = [row for row in by_group[stock, arm] if _guarded(row)]
            if not eligible:
                secondary.append({"stock": stock, "arm": arm, "trial_id": None, "status": NO_GUARDED})
                continue
            winner = max(eligible, key=_score)
            trial_id = winner["trial_id"]
            add_trial(trial_id, SECONDARY_ROLE)
            selected = {
                "stock": stock, "arm": arm, "trial_id": trial_id, "search_seed": winner["search_seed"],
                "round": winner["round"], "dev_score": _score(winner),
                "dev_delta": _finite(winner.get("mean_delta_score")), "dev_verdict": winner["verdict"],
                "parent": winner["parent"], "candidate": winner["candidate"], "status": "scheduled",
            }
            secondary.append(selected)
            best[stock, arm] = selected

    cross: Json = {category: {} for category in CATEGORIES}
    for stock in SYMBOLS:
        target = next(symbol for symbol in SYMBOLS if symbol != stock)
        for category in CATEGORIES:
            arms = ("no_llm",) if category == "no_llm_best" else tuple(arm for arm in ARM_MODELS if arm != "no_llm")
            eligible = [best[stock, arm] for arm in arms if (stock, arm) in best]
            if not eligible:
                cross[category][stock] = {"target_stock": target, "representative": None}
                continue
            winner = max(eligible, key=lambda row: row["dev_score"])
            diagonal = winner["trial_id"]
            add_trial(diagonal, "cross_regime_diagonal")
            off_diagonal = f"closing-cross-{category}-{stock}-to-{target}"
            tasks[off_diagonal] = _task(
                off_diagonal, target, winner["arm"], manifest["baselines"][target]["model"],
                winner["candidate"], [], "cross_regime_transfer",
            )
            cross[category][stock] = {
                "target_stock": target, "representative": diagonal, "source_arm": winner["arm"],
                "dev_score": winner["dev_score"], "diagonal": diagonal, "off_diagonal": off_diagonal,
            }

    skipped = []
    for key, selection in manifest["selections"].items():
        first = selection["first"]
        if first is not None and first in tasks:
            add_trial(first, "first")
        else:
            skipped.append({"trajectory": key, "task_id": first, "status": NOT_SCHEDULED})

    # Include reference profiles needed for immediate-parent counterfactuals,
    # even if that parent was not itself selected as a reported endpoint.
    profiles: dict[str, Json] = {}
    for task in tasks.values():
        stock = task["stock"]
        seed = manifest["baselines"][stock]["model"]
        for model in (task["parent"], task["candidate"]):
            identifier = _profile_id(stock, model)
            profiles.setdefault(identifier, {
                "profile_id": identifier, "stock": stock, "model": model,
                "request_task_id": task["task_id"], "is_seed": model["source_sha256"] == seed["source_sha256"],
            })
    return {
        "mode": MODE, "tasks": list(tasks.values()), "profiles": list(profiles.values()),
        "secondary_best_guarded": secondary, "cross_regime": cross, "skipped_round1": skipped,
        "tie_break": "registered arm order, ascending search seed, ascending round",
        "secondary_label": "Post-Freeze Secondary Audit",
        "profile_identity": ["symbol", "source_sha256"], "max_workers": 3,
    }


def _registration_core(root: Path, manifest: Json) -> Json:
    frozen = Ledger(root).file(MANIFEST)
    return {**_schedule(manifest), "frozen_manifest_sha256": sha256(frozen.read_bytes()).hexdigest(),
            "frozen_code": manifest["code"]}


def _validate_registration(root: Path, manifest: Json, record: Json) -> None:
    actual = {key: val for key, val in record.items() if key not in ("registered_at_utc", "closing_code", "frozen_manifest_mtime_utc")}
    if actual != _registration_core(root, manifest):
        raise ValueError("closing roster differs from the sealed development evidence")
    if not record.get("registered_at_utc") or not isinstance(record.get("closing_code"), dict):
        raise ValueError("closing registration is missing its time or code lock")


def register_closing_verification(root: Path, *, code: Mapping[str, str] | None = None) -> Json:
    """Seal reduced A/B/C selection before there is any holdout receipt or result."""
    from .formal_verify import load_frozen_study

    manifest = load_frozen_study(root)
    ledger = Ledger(Ledger(root).file("verification"))
    with ledger.lock():
        previous = ledger.read(REGISTRATION)
        if previous is not None:
            _validate_registration(root, manifest, previous)
            if code is not None and dict(code) != previous["closing_code"]:
                raise ValueError("resume closing verification with its registered code")
            return previous
        if (ledger.file("test-data-reconciliation.json").exists() or ledger.file("report.json").exists()
                or ledger.file("results").exists() or ledger.file("profile-failures").exists()):
            raise ValueError("closing selection must be registered before holdout acquisition and evaluation")
        record = {
            **_registration_core(root, manifest),
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_manifest_mtime_utc": datetime.fromtimestamp(
                Ledger(root).file(MANIFEST).stat().st_mtime, tz=timezone.utc,
            ).isoformat(),
            "closing_code": dict(manifest["code"] if code is None else code),
        }
        return ledger.seal(record, REGISTRATION)


# --- Unique market profiles and independent role-specific estimands ---


def _profile_failure(exc: CandidateEvaluationFailed) -> Json:
    return {"status": EVALUATION_FAILED, "error": str(exc), "nonfinite_loss": isinstance(exc, NonfiniteCandidateLoss)}


def _measure_profiles(manifest: Json, registration: Json, archive: SourceArchive, ledger: Ledger,
                      evaluators: Mapping[str, ClosingProfiler], jobs: int) -> tuple[dict[str, Json], dict[str, Json]]:
    """Dispatch one raw profile per (stock, SHA); do not retry terminal failures.

    Successful profiles are persisted by the production MarketEvaluator in its
    split/calendar-specific external model cache. Failed profiles are sealed
    here so heterogeneous role comparisons cannot trigger redundant failures.
    Only the lock-owning thread publishes records.
    """
    from .formal_verify import _request

    tasks = {task["task_id"]: task for task in registration["tasks"]}
    successes: dict[str, Json] = {}
    failures: dict[str, Json] = {}
    pending = []
    for profile in registration["profiles"]:
        prefix = ("profile-failures", f"{profile['profile_id']}.json")
        failure = ledger.read(*prefix)
        if failure is not None:
            failures[profile["profile_id"]] = failure
        else:
            pending.append(profile)

    def measure(profile: Json) -> tuple[Json | None, Json | None]:
        task = tasks[profile["request_task_id"]]
        request = _request(manifest, task, archive)
        model = archive.load(profile["model"])
        try:
            data = evaluators[profile["stock"]].verification_profile(
                model, request, candidate=not profile["is_seed"],
            )
        except CandidateEvaluationFailed as exc:
            if profile["is_seed"]:
                raise
            return None, _profile_failure(exc)
        return data, None

    error: Exception | None = None
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(measure, profile): profile for profile in pending}
        for future in as_completed(futures):
            profile_id = futures[future]["profile_id"]
            try:
                data, failure = future.result()
            except Exception as exc:
                if error is None:
                    error = exc
                continue
            if failure is not None:
                failures[profile_id] = ledger.seal(failure, "profile-failures", f"{profile_id}.json")
            elif data is not None:
                successes[profile_id] = data
            else:
                raise RuntimeError("market profiler returned no result or terminal failure")
    if error is not None:
        raise error
    return successes, failures


def _pair(parent: Json, candidate: Json) -> PairedEvaluation:
    return PairedEvaluation(
        {int(day): row for day, row in parent["day_metrics"].items()},
        {int(day): row for day, row in candidate["day_metrics"].items()},
        parent["evidence"], candidate["evidence"],
        parent_market=parent.get("market_bootstrap"), candidate_market=candidate.get("market_bootstrap"),
    )


def _role_results(manifest: Json, registration: Json, archive: SourceArchive, ledger: Ledger,
                  profiles: Mapping[str, Json], failures: Mapping[str, Json]) -> dict[str, Json]:
    """Apply original M5 judge to its registered parent, and separately compare to seed.

    IMPACT and NLL are paired by the same original day blocks and 2,000 draws
    (see paired_stats). Never form an interval by subtracting marginal bounds.
    The seed-vs-candidate contrast is descriptive for non-seed parents and
    must not replace a prediction's registered immediate-parent estimand.
    """
    from .formal_verify import _assess_task, _request

    results: dict[str, Json] = {}
    seed_readouts: dict[str, Json] = {}
    for task in registration["tasks"]:
        task_id = task["task_id"]
        path = ("results", f"{task_id}.json")
        existing = ledger.read(*path)
        if existing is not None:
            if any(existing.get(key) != val for key, val in task.items()):
                raise ValueError(f"stored role record differs from registration: {task_id}")
            results[task_id] = existing
            if existing["seed_readouts"]:
                seed_readouts[existing["candidate_profile_id"]] = existing["seed_readouts"]
            continue
        stock = task["stock"]
        seed_model = manifest["baselines"][stock]["model"]
        candidate_id = _profile_id(stock, task["candidate"])
        parent_id = _profile_id(stock, task["parent"])
        seed_id = _profile_id(stock, seed_model)
        request = _request(manifest, task, archive)
        relevant = [failures[identifier] for identifier in dict.fromkeys((parent_id, candidate_id)) if identifier in failures]
        if relevant:
            reason = "; ".join(item["error"] for item in relevant)
            assessment = failed_evaluation(predictions_from_payload(task), reason)
            if any(item["nonfinite_loss"] for item in relevant):
                assessment["verdict"] = GUARDRAIL_VIOLATION
                assessment["guardrails"] = {"finite_nll_by_seed": False, "nll_passed": False}
            record = {
                "status": EVALUATION_FAILED, "metrics": profiles.get(candidate_id, {}).get("evidence", {}),
                "parent_metrics": profiles.get(parent_id, {}).get("evidence", {}), "readouts": {},
                "assessment": assessment,
            }
        else:
            pair = _pair(profiles[parent_id], profiles[candidate_id])
            pair.validate(request)
            record = _assess_task(pair, request, task)
        # Reuse the actual readout when the registered parent is the seed.
        # Otherwise the same raw profiles support a descriptive seed contrast,
        # even if the immediate parent failed on the holdout.
        if candidate_id not in failures and candidate_id not in seed_readouts:
            if parent_id == seed_id and not relevant:
                seed_readouts[candidate_id] = {metric: record["readouts"][metric] for metric in (NLL, IMPACT)}
            elif seed_id not in failures:
                seed_pair = _pair(profiles[seed_id], profiles[candidate_id])
                seed_task = {**task, "parent": seed_model}
                seed_request = _request(manifest, seed_task, archive)
                seed_readouts[candidate_id] = paired_readouts(seed_pair, seed_request, (NLL, IMPACT))
        result = {**record, **task, "seed_readouts": seed_readouts.get(candidate_id, {}),
                  "candidate_profile_id": candidate_id, "parent_profile_id": parent_id}
        results[task_id] = ledger.seal(result, *path)
    return results


# --- Closing report (primary, additional audit, transfer) ---


def _endpoint(row: Json) -> Json:
    readouts = row["seed_readouts"]
    impact = readouts.get(IMPACT, {})
    nll = readouts.get(NLL, {})
    return {
        **row, "model_id": row["candidate"]["model_id"], "direct_parent_assessment": row["assessment"],
        "nll_delta_from_initial": nll.get("estimate"), "impact_delta_from_initial": impact.get("estimate"),
        "impact_ci95": impact.get("candidate_interval"),
    }


def _summary(manifest: Json, registration: Json, results: Mapping[str, Json], failures: Mapping[str, Json]) -> Json:
    from .formal_verify import _audit_summary, _prediction_rates

    trajectories = {}
    for key, selection in manifest["selections"].items():
        row = dict(selection)
        for phase in ("initial", "first", "final"):
            task_id = selection[phase]
            if phase == "first" and task_id not in results:
                row[phase] = {"task_id": task_id, "status": NOT_SCHEDULED, "metrics": {},
                              "readouts": {}, "seed_readouts": {}, "direct_parent_assessment": None}
            else:
                row[phase] = _endpoint(results[task_id])
        trajectories[key] = row
    audit = [row for row in results.values() if PRIMARY_AUDIT_ROLE in row["roles"]]
    secondary = []
    for row in registration["secondary_best_guarded"]:
        result = None if row["trial_id"] is None else results[row["trial_id"]]
        secondary.append({**row, "result": result})
    cross: Json = {category: {} for category in CATEGORIES}
    for category, sources in registration["cross_regime"].items():
        for source_stock, row in sources.items():
            target = row["target_stock"]
            if row["representative"] is None:
                cells = {stock: {"status": NO_GUARDED, "metrics": {}, "readouts": {}} for stock in (source_stock, target)}
            else:
                diagonal = results[row["diagonal"]]
                cells = {
                    source_stock: {**diagonal, "direct_parent_readouts": diagonal["readouts"],
                                   "readouts": diagonal["seed_readouts"], "comparison": "stock_seed"},
                    target: {**results[row["off_diagonal"]], "comparison": "stock_seed"},
                }
                # A failed immediate-parent contrast does not necessarily make
                # the diagonal seed comparison unavailable.
                if diagonal["seed_readouts"]:
                    cells[source_stock]["status"] = "measured"
            cross[category][source_stock] = {**row, "cells": cells}
    summaries = {}
    for stock in SYMBOLS:
        summaries[stock] = {}
        for arm in ARM_MODELS:
            finals = [value for value in results.values() if value["stock"] == stock and value["group"] == arm
                      and "final" in value["roles"]]
            verified = sum(value["assessment"]["verdict"] == VERIFIED_GAIN for value in finals)
            audit_rows = [value for value in audit if value["stock"] == stock and value["group"] == arm]
            summaries[stock][arm] = {
                "verified_gain": verified, "verified_denominator": len(finals),
                "verified_mechanism_gain_rate": verified / len(finals) if finals else None,
                "prediction_rates": _prediction_rates(finals), "rejected_audit": _audit_summary(audit_rows),
                "forecast_type": "fixed_placeholder" if arm == "no_llm" else "preregistered_llm",
            }
    return {
        "schema": M5_SCHEMA, "mode": MODE, "code": manifest["code"], "closing_registration": registration,
        "test_dates": manifest["test_dates"], "planned_test_dates": manifest["planned_test_dates"],
        "replacement_test_dates": manifest["replacement_test_dates"], "excluded_test_dates": manifest["excluded_test_dates"],
        "train_dates": manifest["data"]["train_dates"], "train_seeds": manifest["train_seeds"], "analysis_version": "m5",
        "scheduled_tasks": len(registration["profiles"]), "measured_tasks": len(registration["profiles"]) - len(failures),
        "scheduled_comparisons": len(results), "measured_comparisons": sum(row["status"] == "measured" for row in results.values()),
        "failed_profiles": dict(failures), "tasks": dict(results), "trajectories": trajectories,
        "arm_summary": summaries, "audit_summary": _audit_summary(audit), "audit_only_rejected_unverified": audit,
        "secondary_best_guarded": secondary, "cross_regime": cross,
    }


def verify_closing_study(root: Path, *, evaluators: Mapping[str, ClosingProfiler], jobs: int = 3) -> Json:
    from .formal_verify import load_frozen_study
    from .holdout_data import validate_m5_receipt

    if set(evaluators) != set(SYMBOLS) or not 1 <= jobs <= 3:
        raise ValueError("closing verification requires both stock evaluators and one to three workers")
    manifest = load_frozen_study(root)
    archive_ledger = Ledger(root)
    archive = SourceArchive(archive_ledger)
    ledger = Ledger(archive_ledger.file("verification"))
    with ledger.lock():
        registration = ledger.read(REGISTRATION)
        if registration is None:
            raise ValueError("seal the closing registration before acquiring holdout data")
        _validate_registration(root, manifest, registration)
        receipt = ledger.read("test-data-reconciliation.json")
        if receipt is None:
            raise ValueError("the M5 holdout requires its registered acquisition receipt")
        effective = validate_m5_receipt(manifest, receipt)
        manifest = {**manifest, "planned_test_dates": manifest["test_dates"], "test_dates": list(effective),
                    "excluded_test_dates": receipt["excluded_test_dates"]}
        report = ledger.read("report.json")
        if report is not None:
            if report.get("mode") != MODE or report.get("closing_registration") != registration:
                raise ValueError("the existing verification report does not agree with the closing registration")
            return report
        completed = all(ledger.file("results", f"{task['task_id']}.json").is_file() for task in registration["tasks"])
        if completed:
            # The final report may have been interrupted after the last role
            # was sealed; finish from the records without touching profiles.
            failures = {}
            for profile in registration["profiles"]:
                identifier = profile["profile_id"]
                failure = ledger.read("profile-failures", f"{identifier}.json")
                if failure is not None:
                    failures[identifier] = failure
            profiles = {}
        else:
            profiles, failures = _measure_profiles(manifest, registration, archive, ledger, evaluators, jobs)
        results = _role_results(manifest, registration, archive, ledger, profiles, failures)
        return ledger.seal(_summary(manifest, registration, results, failures), "report.json")
