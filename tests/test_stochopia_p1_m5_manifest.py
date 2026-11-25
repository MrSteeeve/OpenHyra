"""Preregister the high-power calendar and freeze before any market-data IO."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tasks.stochopia_p1.formal import _provider_handshakes
from tasks.stochopia_p1.formal_archive import write_archive_tables
from tasks.stochopia_p1.formal_manifest import (
    M5_DEV_DATES, M5_EMBARGO_DATES, M5_EXCLUDED_DATES, M5_REPLACEMENT_TEST_DATES,
    M5_SCHEMA, M5_TEST_DATES, M5_TRAIN_DATES, MANIFEST, ROUNDS, SYMBOLS,
    TEST_DATES, SourceArchive, data_registration, freeze_study, roster, study_registration, validate_frozen,
)
from tasks.stochopia_p1.store import IMPACT, Json, Ledger
from tasks.stochopia_p1.formal_verify import verify_frozen_study
from tasks.stochopia_p1.protocol import ProviderReply
from tasks.stochopia_p1.provider import ProviderAttemptError, ProviderOutageError
from tasks.stochopia_p1.store import EvaluationRequest, PairedEvaluation


def market_metadata() -> Json:
    return {
        "source_repo": "synthetic-fixture", "source_revision": "fixture", "symbols": list(SYMBOLS),
        "train_dates": list(M5_TRAIN_DATES), "dev_dates": list(M5_DEV_DATES),
        "test_dates": list(M5_TEST_DATES), "replacement_test_dates": list(M5_REPLACEMENT_TEST_DATES),
        "embargo_dates": list(M5_EMBARGO_DATES),
    }


def write_market(root: Path, record: Json) -> Path:
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(record), encoding="utf-8")
    return root


def test_m5_registration_uses_exact_new_common_calendar(tmp_path: Path) -> None:
    market = write_market(tmp_path / "market", market_metadata())
    data = data_registration(market)
    assert len(data["train_dates"]) == 36
    assert len(data["dev_dates"]) == 24
    assert len(data["test_dates"]) == 20
    assert len(data["replacement_test_dates"]) == 10
    assert data["dev_dates"][-12:] == list(TEST_DATES)
    assert data["test_dates"][0] == 20170414
    assert data["test_dates"][-1] == 20170512
    assert data["replacement_test_dates"][0] == 20170515
    assert data["replacement_test_dates"][-1] == 20170526
    assert data["embargo_dates"] == [20170306, 20170413]
    assert data["excluded_dates"] == list(M5_EXCLUDED_DATES)
    assert set(data["dev_dates"]).isdisjoint(data["test_dates"])
    assert set(data["dev_dates"]).isdisjoint(data["replacement_test_dates"])
    assert set(data["test_dates"]).isdisjoint(data["replacement_test_dates"])
    plan = study_registration(data, {"openhyra_commit": "fixture"}, "synthetic")
    assert plan["schema"] == M5_SCHEMA
    assert plan["analysis_version"] == "m5"
    assert plan["data"] == data
    assert sorted(market.iterdir()) == [market / "manifest.json"]


@pytest.mark.parametrize(("field", "index", "changed"), [
    ("train_dates", 0, 20170102),
    ("dev_dates", 11, 20170323),
    ("test_dates", 0, 20170413),
    ("replacement_test_dates", -1, 20170529),
    ("embargo_dates", 1, 20170323),
])
def test_m5_calendar_cannot_be_shifted_before_freeze(tmp_path: Path, field: str, index: int, changed: int) -> None:
    record = market_metadata()
    record[field][index] = changed
    market = write_market(tmp_path / "market", record)
    with pytest.raises(ValueError, match=f"preregistered {field}"):
        data_registration(market)


def test_m5_freeze_seals_every_task_and_keeps_replacement_dates_independent(tmp_path: Path) -> None:
    market = write_market(tmp_path / "market", market_metadata())
    artifacts = tmp_path / "artifacts"
    work = tmp_path / "work"
    ledger = Ledger(artifacts)
    seed = {"model_id": "seed-gru", "source": "# synthetic seed\n"}
    plan = study_registration(data_registration(market),
                              {"openhyra_commit": "fixture", "stochopia_commit": "fixture"}, "synthetic")
    with ledger.lock():
        ledger.seal(plan, "development_plan.json")
        for stock in SYMBOLS:
            ledger.seal({"model": seed, "metrics": {IMPACT: 30.0}}, "baselines", f"{stock}.json")

    with pytest.raises(ValueError, match="incomplete formal trial"):
        freeze_study(artifacts, market, development_root=work)
    assert not (artifacts / MANIFEST).exists()
    assert not (artifacts / "sources").exists()

    for item in roster():
        trials = Ledger(item.path(work))
        with trials.lock():
            for index in range(1, ROUNDS + 1):
                trial_id = f"{item.key}-r{index:02d}"
                model = {"model_id": trial_id, "source": f"# synthetic {trial_id}\n"}
                verdict = "REFUTED"
                if item.arm == "evidence_gpt" and index > 1:
                    verdict = "UNVERIFIED_GAIN" if index == 2 else "VERIFIED_GAIN"
                result = {
                    "round": index, "trial_id": trial_id, "parent": seed, "candidate": model,
                    "hypothesis": None,
                    "predictions": [
                        {"metric": IMPACT, "direction": "increase", "interval": [0.5, 2.0], "role": "primary"},
                        {"metric": "val_nll_total", "direction": "equivalent", "interval": [-0.05, 0.05],
                         "role": "guardrail", "equivalence_band": [-0.05, 0.05]},
                    ],
                    "verdict": verdict, "candidate_metrics": {IMPACT: 30.0 + index},
                    "training_budget_consumed": True,
                }
                path = item.round_path(index)
                trials.seal(result, *path, "result.json")
                trials.seal({"status": "ready", "source": model["source"]}, *path, "program.json")

    manifest = freeze_study(artifacts, market, development_root=work)
    assert manifest["schema"] == M5_SCHEMA
    assert manifest["test_dates"] == list(M5_TEST_DATES)
    assert manifest["replacement_test_dates"] == list(M5_REPLACEMENT_TEST_DATES)
    assert len(manifest["trajectories"]) == 30
    assert sum(len(item["trials"]) for item in manifest["trajectories"].values()) == 90
    assert len({item["task_id"] for item in manifest["tasks"]}) == len(manifest["tasks"])
    assert not (artifacts / "verification").exists()
    assert sorted(market.iterdir()) == [market / "manifest.json"]

    trajectory = next(item for item in roster() if item.arm == "evidence_gpt")
    selection = manifest["selections"][trajectory.key]
    assert selection["first"] == f"{trajectory.key}-r01"
    assert selection["final"] == f"{trajectory.key}-r03"
    assert selection["audit_only_rejected_unverified"] == [f"{trajectory.key}-r02"]
    chosen_cross = manifest["cross_regime"]["evidence_gpt"][SYMBOLS[0]]
    fallback_cross = manifest["cross_regime"]["no_llm"][SYMBOLS[0]]
    assert chosen_cross["retains_seed"] is False
    assert fallback_cross["retains_seed"] is True and fallback_cross["auxiliary"] is not None

    frozen_bytes = (artifacts / MANIFEST).read_bytes()
    assert freeze_study(artifacts, market, development_root=work) == manifest
    assert (artifacts / MANIFEST).read_bytes() == frozen_bytes

    invalid = deepcopy(manifest)
    invalid["replacement_test_dates"].pop()
    with pytest.raises(ValueError, match="replacement test calendar"):
        validate_frozen(invalid, SourceArchive(ledger))
    invalid = deepcopy(manifest)
    invalid["test_dates"][0] = 20170324
    with pytest.raises(ValueError, match="primary test calendar"):
        validate_frozen(invalid, SourceArchive(ledger))
    invalid = deepcopy(manifest)
    invalid["analysis_version"] = "m4"
    with pytest.raises(ValueError, match="paired analysis version"):
        validate_frozen(invalid, SourceArchive(ledger))
    invalid = deepcopy(manifest)
    invalid["data"]["dev_dates"][-1] = 20170414
    with pytest.raises(ValueError, match="preregistered dev_dates"):
        validate_frozen(invalid, SourceArchive(ledger))
    invalid = deepcopy(manifest)
    invalid["selections"][trajectory.key]["final"] = selection["initial"]
    with pytest.raises(ValueError, match="sealed development decisions"):
        validate_frozen(invalid, SourceArchive(ledger))

    # The effective holdout calendar can differ only through the frozen
    # replacement rule. A missing receipt prevents every evaluation call.
    effective = (*M5_TEST_DATES[1:], M5_REPLACEMENT_TEST_DATES[0])
    calls = []

    def evaluate(request: EvaluationRequest) -> PairedEvaluation:
        assert request.eval_dates == effective
        assert request.analysis_version == "m5"
        calls.append(request.trial_id)
        parent = {day: {IMPACT: 30.0, "val_nll_total": 3.0, "free_abs_drift_ticks": 2.0}
                  for day in effective}
        candidate = {day: {**row, IMPACT: 31.0} for day, row in parent.items()}
        evidence = {"exhausted_windows": 0, "train_seed_nll": {str(seed): 3.0 for seed in request.train_seeds}}
        return PairedEvaluation(parent, candidate, evidence, evidence)

    evaluators = {stock: evaluate for stock in SYMBOLS}
    with pytest.raises(ValueError, match="acquisition receipt"):
        verify_frozen_study(artifacts, evaluators=evaluators, jobs=3)
    assert calls == []
    receipt = {
        "planned_test_dates": list(M5_TEST_DATES), "replacement_test_dates": list(M5_REPLACEMENT_TEST_DATES),
        "test_dates": list(effective), "source_revision": "fixture",
        "excluded_test_dates": [{"date": M5_TEST_DATES[0], "reason": "source_missing: fixture"}],
        "reconciliation": [
            {"symbol": stock, "date": day, "n_events": 2, "n_snapshots": 1,
             "touch_match_rate": 1.0, "top10_log_vol_mae": 0.0}
            for day in effective for stock in SYMBOLS
        ],
        "anchors": [
            {"symbol": stock, "date": day, "session": session, "spread_ticks": 1}
            for day in effective for stock in SYMBOLS for session in (0, 1)
        ],
    }
    with ledger.lock():
        ledger.seal(receipt, "verification", "test-data-reconciliation.json")
    report = verify_frozen_study(artifacts, evaluators=evaluators, jobs=3)
    assert report["test_dates"] == list(effective)
    assert report["planned_test_dates"] == list(M5_TEST_DATES)
    assert report["measured_tasks"] == len(manifest["tasks"]) == len(calls)
    rates = report["arm_summary"][SYMBOLS[0]]["evidence_gpt"]["prediction_rates"]["primary"]
    assert rates["point_hit_rate"] == 1.0
    assert (artifacts / MANIFEST).read_bytes() == frozen_bytes
    assert verify_frozen_study(artifacts, evaluators=evaluators, jobs=3) == report
    assert len(calls) == len(manifest["tasks"])
    write_archive_tables(artifacts)
    assert "p10_delta_score" in (artifacts / "development-trials.csv").read_text()
    assert "primary_point_in_interval" in (artifacts / "holdout-endpoints.csv").read_text()
    from experiments.p1_m5_report import render

    markdown = render(artifacts, "合成算子测试")
    assert "## 预测证实与幅度校准" in markdown
    assert "20170515" in markdown


def test_m5_handshakes_resume_without_repeating_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks.stochopia_p1 import provider

    ledger = Ledger(tmp_path / "study")
    attempts = []

    def probe(registry: Path, alias: str) -> ProviderReply:
        attempts.append(alias)
        if alias == "deepseek-flash" and attempts.count(alias) == 1:
            raise ProviderAttemptError("offline fixture", retryable=True)
        return ProviderReply('{"ok": true}', 10, 3)

    monkeypatch.setattr(provider, "provider_handshake", probe)
    with ledger.lock():
        with pytest.raises(ProviderOutageError):
            _provider_handshakes(ledger, tmp_path / "registry")
        _provider_handshakes(ledger, tmp_path / "registry")
        _provider_handshakes(ledger, tmp_path / "registry")
    assert attempts == ["gpt-6.2-sol", "deepseek-flash", "deepseek-flash"]
    failed = ledger.read("handshakes", "deepseek-flash", "attempt-001-result.json")
    assert failed["retryable"] and failed["input_tokens"] is None
    assert ledger.read("handshakes", "deepseek-flash", "passed.json")["attempt"] == 2
