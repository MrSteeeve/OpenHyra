"""Closing selection is frozen before holdout, with distinct forecasts per role."""

from __future__ import annotations

import csv
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import Lock

import pytest

from tasks.stochopia_p1.closing_verify import MODE, NO_GUARDED, NOT_SCHEDULED, REGISTRATION, _guarded
from tasks.stochopia_p1.formal_archive import write_archive_tables
from tasks.stochopia_p1.formal_manifest import (
    M5_DEV_DATES, M5_EMBARGO_DATES, M5_REPLACEMENT_TEST_DATES, M5_TEST_DATES, M5_TRAIN_DATES,
    MANIFEST, SYMBOLS, SourceArchive, _schedule, roster, study_registration,
)
from tasks.stochopia_p1.formal_verify import register_closing_verification, verify_frozen_study
from tasks.stochopia_p1.store import (
    CandidateEvaluationFailed, EvaluationRequest, IMPACT, Json, Ledger, ModelRef, NLL, PairedEvaluation,
)


def _guards() -> Json:
    return {
        "finite_nll_by_seed": True, "nll_passed": True, "exhaustion_passed": True, "drift_passed": True,
        "delta_val_nll_total": 0.0, "delta_exhausted_windows": 0,
        "parent_free_abs_drift_ticks": 2.0, "child_free_abs_drift_ticks": 2.0,
    }


def _forecasts(*, increase: bool = True) -> list[Json]:
    return [
        {"metric": IMPACT, "direction": "increase" if increase else "decrease", "role": "primary",
         "interval": [0.5, 4.0] if increase else [-4.0, -0.5]},
        {"metric": NLL, "direction": "equivalent", "role": "guardrail", "interval": [-0.05, 0.05],
         "equivalence_band": [-0.05, 0.05]},
    ]


def _model(archive: SourceArchive, name: str, value: float) -> Json:
    return archive.ref({"model_id": name, "source": f"# {value}\n"})


def _draft(root: Path) -> Json:
    ledger = Ledger(root)
    archive = SourceArchive(ledger)
    data = {
        "source_repo": "synthetic", "source_revision": "fixture", "symbols": SYMBOLS,
        "train_dates": M5_TRAIN_DATES, "dev_dates": M5_DEV_DATES, "test_dates": M5_TEST_DATES,
        "embargo_dates": M5_EMBARGO_DATES, "replacement_test_dates": M5_REPLACEMENT_TEST_DATES,
    }
    registration = study_registration(data, {"openhyra_commit": "dev-fixture"}, "fixture")
    seed = _model(archive, "seed-gru", 30.0)
    prior = _model(archive, "prior-promoted", 32.0)
    shared = _model(archive, "shared-source", 34.0)
    rejected = _model(archive, "audit-source", 31.0)
    weak = _model(archive, "no-llm", 33.0)
    generic = _model(archive, "unscheduled", 30.1)
    baselines = {symbol: {"model": seed, "metrics": {IMPACT: 30.0}} for symbol in SYMBOLS}
    trajectories = {}
    for item in roster():
        rows = []
        for index in range(1, 4):
            trial = {
                "round": index, "trial_id": f"{item.key}-r{index}", "parent": seed, "candidate": generic,
                "predictions": _forecasts(), "verdict": "REFUTED", "guardrails": {},
                "candidate_metrics": {IMPACT: 100.0}, "mean_delta_score": 0.0,
                "program": {"status": "ready", "source_sha256": generic["source_sha256"]},
                "usage": [], "requests": [], "compiler": [], "training_budget_consumed": True,
            }
            if item.search_seed == 17 and item.arm == "evidence_gpt":
                if index == 1:
                    trial.update(candidate=prior, verdict="VERIFIED_GAIN", guardrails=_guards(),
                                 candidate_metrics={IMPACT: 32.0}, mean_delta_score=2.0)
                elif index == 2:
                    trial.update(candidate=rejected, parent=prior, verdict="UNVERIFIED_GAIN", guardrails=_guards(),
                                 candidate_metrics={IMPACT: 31.0}, mean_delta_score=1.0)
                else:
                    trial.update(candidate=shared, parent=prior, guardrails=_guards(),
                                 candidate_metrics={IMPACT: 34.0}, mean_delta_score=2.0)
            elif item.search_seed == 17 and item.arm == "metrics_gpt":
                # Both rows tie: only the first wins; its source is shared
                # with the previous arm but its parent and forecast are not.
                trial.update(candidate=shared, guardrails=_guards(), predictions=_forecasts(increase=False),
                             candidate_metrics={IMPACT: 34.0}, mean_delta_score=4.0)
            elif item.search_seed == 17 and item.arm == "no_llm":
                trial.update(candidate=weak, guardrails=_guards(), candidate_metrics={IMPACT: 33.0}, mean_delta_score=3.0)
            elif item.arm == "evidence_ds":
                trial.update(guardrails={**_guards(), "delta_val_nll_total": 0.051})
            elif item.arm == "metrics_ds":
                trial.update(candidate_metrics={IMPACT: None}, guardrails=_guards())
            trial["program"]["source_sha256"] = trial["candidate"]["source_sha256"]
            rows.append(trial)
        trajectories[item.key] = {"stock": item.stock, "arm": item.arm, "search_seed": item.search_seed, "trials": rows}
    manifest = _schedule(registration, baselines, trajectories)
    with ledger.lock():
        ledger.seal(manifest, MANIFEST)
    return manifest


def _receipt(root: Path) -> None:
    ledger = Ledger(root)
    dates = list(M5_TEST_DATES)
    receipt = {
        "planned_test_dates": dates, "replacement_test_dates": list(M5_REPLACEMENT_TEST_DATES),
        "test_dates": dates, "source_revision": "fixture", "excluded_test_dates": [],
        "reconciliation": [
            {"symbol": stock, "date": day, "n_events": 2, "n_snapshots": 1,
             "touch_match_rate": 1.0, "top10_log_vol_mae": 0.0} for day in dates for stock in SYMBOLS
        ],
        "anchors": [
            {"symbol": stock, "date": day, "session": session, "spread_ticks": 1}
            for day in dates for stock in SYMBOLS for session in (0, 1)
        ],
    }
    with ledger.lock():
        ledger.seal(receipt, "verification", "test-data-reconciliation.json")


class SyntheticProfiler:
    def __init__(self, stock: str, calls: Counter[tuple[str, str]], lock: Lock, *, fail_score: float | None = None) -> None:
        self.stock = stock
        self.calls = calls
        self.lock = lock
        self.fail_score = fail_score

    def __call__(self, request: EvaluationRequest) -> PairedEvaluation:
        raise AssertionError("closing must not invoke the old pair-level evaluator")

    def verification_profile(self, model: ModelRef, request: EvaluationRequest, *, candidate: bool) -> Json:
        assert request.split == "test"
        assert request.eval_dates == M5_TEST_DATES
        assert request.train_dates == M5_TRAIN_DATES
        assert request.analysis_version == "m5"
        digest = sha256(model.source.encode()).hexdigest()
        value = float(model.source[2:].strip())
        with self.lock:
            self.calls[self.stock, digest] += 1
        if self.fail_score == value:
            assert candidate
            raise CandidateEvaluationFailed("synthetic candidate resource failure")
        offset = 0.0 if self.stock == SYMBOLS[0] else 5.0
        metrics = {IMPACT: value + offset, NLL: 3.0, "free_abs_drift_ticks": 2.0}
        evidence = {**metrics, "exhausted_windows": 0,
                    "train_seed_nll": {str(seed): 3.0 for seed in request.train_seeds}}
        return {"day_metrics": {str(day): metrics for day in request.eval_dates}, "evidence": evidence}


def test_closing_selection_guarded_ties_and_distinct_roles(tmp_path: Path) -> None:
    original = _draft(tmp_path)
    before = (tmp_path / MANIFEST).read_bytes()
    registered = register_closing_verification(tmp_path, code={"openhyra_commit": "closing-fixture"})
    assert registered["mode"] == MODE
    assert registered["frozen_manifest_sha256"] == sha256(before).hexdigest()
    assert registered["registered_at_utc"].endswith("+00:00")
    assert registered["closing_code"] != original["code"]
    assert register_closing_verification(tmp_path) == registered
    assert (tmp_path / MANIFEST).read_bytes() == before
    assert len(registered["secondary_best_guarded"]) == 10
    assert len(registered["profiles"]) == 10  # seed+prior+audit+shared+weak per target, including transfers
    assert len(registered["tasks"]) > len(registered["profiles"])
    for row in registered["secondary_best_guarded"]:
        if row["arm"] in ("evidence_ds", "metrics_ds"):
            assert row["trial_id"] is None and row["status"] == NO_GUARDED
        elif row["arm"] == "evidence_gpt":
            assert row["round"] == 3
        else:
            assert row["round"] == 1
    for stock in SYMBOLS:
        chosen = registered["cross_regime"]["llm_best"][stock]
        assert chosen["source_arm"] == "evidence_gpt"
        assert chosen["representative"].endswith("r3")
    assert len(registered["skipped_round1"]) == 24
    all_roles = {task["task_id"]: task["roles"] for task in registered["tasks"]}
    assert "first" in all_roles[f"{SYMBOLS[0]}-evidence_gpt-17-r1"]
    assert "final" in all_roles[f"{SYMBOLS[0]}-evidence_gpt-17-r1"]
    assert "audit_only_rejected_unverified" in all_roles[f"{SYMBOLS[0]}-evidence_gpt-17-r2"]
    assert f"{SYMBOLS[0]}-metrics_gpt-17-r2" not in all_roles


@pytest.mark.parametrize(("field", "value"), [
    ("nll_passed", False), ("delta_val_nll_total", 0.051), ("delta_exhausted_windows", 1),
    ("child_free_abs_drift_ticks", 20.0), ("delta_val_nll_total", None),
    ("delta_val_nll_total", float("inf")), ("finite_nll_by_seed", 1),
])
def test_incomplete_or_failed_guard_is_not_eligible(field: str, value: object) -> None:
    trial = {"candidate": {"source_sha256": "dummy"}, "program": {"status": "ready"},
             "guardrails": _guards(), "candidate_metrics": {IMPACT: 40.0}}
    assert _guarded(trial)
    trial["guardrails"][field] = value
    assert not _guarded(trial)


def test_guard_thresholds_and_nonfinite_score() -> None:
    trial = {"candidate": {}, "program": {"status": "ready"}, "guardrails": _guards(),
             "candidate_metrics": {IMPACT: 40.0}}
    trial["guardrails"].update(delta_val_nll_total=0.0500000000001, parent_free_abs_drift_ticks=22.0,
                               child_free_abs_drift_ticks=21.0)
    assert _guarded(trial)
    trial["candidate_metrics"][IMPACT] = float("nan")
    assert not _guarded(trial)
    trial["candidate_metrics"][IMPACT] = None
    assert not _guarded(trial)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("fail_score", [None, 34.0])
def test_closing_measure_profiles_once_and_preserve_forecasts(tmp_path: Path, fail_score: float | None) -> None:
    _draft(tmp_path)
    calls: Counter[tuple[str, str]] = Counter()
    lock = Lock()
    evaluators = {stock: SyntheticProfiler(stock, calls, lock, fail_score=fail_score) for stock in SYMBOLS}
    with pytest.raises(ValueError, match="seal the closing registration"):
        verify_frozen_study(tmp_path, evaluators=evaluators, jobs=3, closing=True)
    registration = register_closing_verification(tmp_path)
    with pytest.raises(ValueError, match="acquisition receipt"):
        verify_frozen_study(tmp_path, evaluators=evaluators, jobs=3, closing=True)
    assert not calls
    _receipt(tmp_path)
    assert register_closing_verification(tmp_path) == registration
    with pytest.raises(ValueError, match="requires closing verification mode"):
        verify_frozen_study(tmp_path, evaluators=evaluators, jobs=3)
    report = verify_frozen_study(tmp_path, evaluators=evaluators, jobs=3, closing=True)
    assert len(calls) == report["scheduled_tasks"] == 10
    assert all(count == 1 for count in calls.values())
    assert report["measured_tasks"] == (10 if fail_score is None else 8)
    assert report["scheduled_comparisons"] > report["scheduled_tasks"]
    assert verify_frozen_study(tmp_path, evaluators=evaluators, jobs=3, closing=True) == report
    assert all(count == 1 for count in calls.values())
    # Simulate a crash between the final role seal and report publication.
    (tmp_path / "verification" / "report.json").unlink()
    assert verify_frozen_study(tmp_path, evaluators=evaluators, jobs=3, closing=True) == report
    assert all(count == 1 for count in calls.values())
    stock = SYMBOLS[0]
    roles = report["tasks"]
    left = roles[f"{stock}-evidence_gpt-17-r3"]
    right = roles[f"{stock}-metrics_gpt-17-r1"]
    assert left["candidate_profile_id"] == right["candidate_profile_id"]
    assert left["parent_profile_id"] != right["parent_profile_id"]
    if fail_score is None:
        assert left["readouts"][IMPACT]["estimate"] == 2.0
        assert right["readouts"][IMPACT]["estimate"] == 4.0
        assert left["assessment"]["prediction_verdicts"][0]["status"] == "SUPPORTED"
        assert right["assessment"]["prediction_verdicts"][0]["status"] == "REFUTED"
        diagonal = report["cross_regime"]["llm_best"][stock]["cells"][stock]
        assert diagonal["readouts"][IMPACT]["estimate"] == 4.0
        target = report["cross_regime"]["llm_best"][stock]["cells"][SYMBOLS[1]]
        assert target["readouts"][IMPACT]["estimate"] == 4.0
        assert target["parent"]["model_id"] == "seed-gru"
    else:
        assert left["status"] == right["status"] == "evaluation_failed"
        assert len(report["failed_profiles"]) == 2
    assert report["trajectories"][f"{stock}-evidence_ds-17"]["first"]["status"] == NOT_SCHEDULED
    assert report["trajectories"][f"{stock}-evidence_ds-17"]["final"]["model_id"] == "seed-gru"
    write_archive_tables(tmp_path)
    assert len(_csv_rows(tmp_path / "secondary-best-guarded.csv")) == 10
    assert len(_csv_rows(tmp_path / "cross-regime.csv")) == 8
    assert len(_csv_rows(tmp_path / "holdout-endpoints.csv")) == 90
    assert len(_csv_rows(tmp_path / "rejected-audit.csv")) == 2


def test_closing_registration_rejects_late_selection_or_tampering(tmp_path: Path) -> None:
    _draft(tmp_path)
    _receipt(tmp_path)
    with pytest.raises(ValueError, match="before holdout acquisition"):
        register_closing_verification(tmp_path)
    ledger = Ledger(tmp_path)
    ledger.file("verification", "test-data-reconciliation.json").unlink()
    record = register_closing_verification(tmp_path)
    invalid = deepcopy(record)
    invalid["secondary_best_guarded"][0]["dev_score"] += 1
    from stochopia.common import atomic_write_json

    atomic_write_json(ledger.file("verification", REGISTRATION), invalid)
    with pytest.raises(ValueError, match="differs from the sealed development evidence"):
        register_closing_verification(tmp_path)


def test_closing_cli_code_migration_restricts_development_changes(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from tasks.stochopia_p1 import formal

    stochopia_root = tmp_path / "Stochopia"
    openhyra_root = tmp_path / "OpenHyra"
    manifest = {
        "schema": formal.M5_SCHEMA,
        "code": {
            "stochopia_commit": "old-s",
            "stochopia_tree": "tree-s",
            "openhyra_commit": "old-o",
            "openhyra_tree": "tree-o",
        },
    }
    code = {"stochopia_commit": "new-s", "openhyra_commit": "new-o"}
    monkeypatch.setattr(formal, "OPENHYRA_ROOT", openhyra_root)
    monkeypatch.setattr(formal, "_git_value", lambda root, value: "tree-s" if root == stochopia_root else "tree-o")
    monkeypatch.setattr(formal.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    changes = {
        stochopia_root: "experiments/p1_m5_closing_report.py\n",
        openhyra_root: "tasks/stochopia_p1/closing_verify.py\n",
    }
    monkeypatch.setattr(formal.subprocess, "check_output", lambda *args, **kwargs: changes[kwargs["cwd"]])
    formal._require_m5_closing_code(manifest, code, stochopia_root)
    changes[openhyra_root] += "tasks/stochopia_p1/paired_stats.py\n"
    with pytest.raises(ValueError, match="outside"):
        formal._require_m5_closing_code(manifest, code, stochopia_root)
