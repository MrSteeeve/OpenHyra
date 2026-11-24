"""Exercise the complete M4 development and holdout schedule offline."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from tasks.stochopia_p1.formal import run_development
from tasks.stochopia_p1.formal_archive import write_archive_tables
from tasks.stochopia_p1.formal_manifest import (
    ARM_MODELS, EMBARGO_DATES, MANIFEST, ROUNDS, SYMBOLS, TEST_DATES, TRAIN_SEEDS, Trajectory,
    data_registration, freeze_study, roster, study_registration,
)
from tasks.stochopia_p1.formal_verify import load_frozen_study, verify_frozen_study
from tasks.stochopia_p1.protocol import Messages, ProviderReply, TwoStepProposer
from tasks.stochopia_p1.provider import ProviderAttemptError
from tasks.stochopia_p1.store import EvaluationRequest, Ledger, ModelRef, PairedEvaluation


NLL = "val_nll_total"
IMPACT = "lob_impact_score"
DRIFT = "free_abs_drift_ticks"
CODE_LOCK = {"stochopia_tree": "fixture", "openhyra_tree": "fixture"}


def _days(start: date, count: int) -> list[int]:
    return [int((start + timedelta(days=index)).strftime("%Y%m%d")) for index in range(count)]


def _market_metadata(root: Path) -> Path:
    root.mkdir()
    metadata = {
        "source_repo": "synthetic-fixture", "source_revision": "fixture", "symbols": SYMBOLS,
        "train_dates": _days(date(2017, 1, 3), 36),
        "dev_dates": _days(date(2017, 3, 7), 12),
        "test_dates": TEST_DATES, "embargo_dates": EMBARGO_DATES,
    }
    (root / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def _seed_metrics(symbol: str, model: ModelRef) -> dict[str, object]:
    assert symbol in SYMBOLS and "CandidateLOBWorldModel" in model.source
    return _metrics(30.0)


def _metrics(score: float) -> dict[str, object]:
    return {
        "parameters": 21932, "state_dim": 48,
        IMPACT: score, NLL: 3.0, DRIFT: 2.0, "exhausted_windows": 0,
        "train_seed_nll": {seed: 3.0 for seed in TRAIN_SEEDS},
        "claims": {"C1": {"status": "fail"}, "C2": {"status": "pass"}},
    }


def _gate(source: str) -> str | None:
    return "scripted compile diagnostic" if "bad fixture" in source else None


def _proposal(item: Trajectory, index: int) -> dict[str, object]:
    parent = None
    if item.arm == "evidence_gpt" and index > 1:
        parent = "h1"
    return {
        "hypothesis": {
            "hypothesis_id": f"h{index}", "parent_hypothesis_id": parent,
            "observed_bottleneck": "Score gap", "operator_diagnosis": "Fixture recurrent scale",
            "structural_prescription": "Change fixture scaling", "falsifier": "Missing gain",
        },
        "predictions": [
            {"role": "primary", "metric": IMPACT, "direction": "increase", "interval": [0.9, 1.1]},
            {
                "role": "guardrail", "metric": NLL, "direction": "equivalent",
                "interval": [-0.01, 0.01], "equivalence_band": [-0.02, 0.02],
            },
        ],
    }


class OfflineHarness:
    def __init__(self) -> None:
        self.dev: list[str] = []
        self.holdout: list[str] = []
        self.providers: list[tuple[str, int, bool]] = []
        self.abort_holdout_at: int | None = None

    def proposer(self, item: Trajectory) -> TwoStepProposer:
        index = 0
        drafts = 0

        def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
            nonlocal index, drafts
            if response_json:
                index += 1
                drafts = 0
                payload = json.dumps(_proposal(item, index))
            else:
                drafts += 1
                code_line = f"# {item.key} round {index}"
                if item.arm == "metrics_ds" and index == 2 and drafts == 1:
                    code_line += " bad fixture"
                payload = f"```python\n{code_line}\n```"
            self.providers.append((item.key, index, response_json))
            assert all(str(day) not in messages[-1]["content"] for day in TEST_DATES)
            return ProviderReply(payload, 60, 20)

        return TwoStepProposer(transport, _gate, lambda _usage: None, count_tokens=lambda _messages: 900)

    def evaluator(self, stock: str, request: EvaluationRequest) -> PairedEvaluation:
        assert stock in SYMBOLS
        if request.split == "dev":
            assert set(request.eval_dates).isdisjoint(TEST_DATES)
            self.dev.append(request.trial_id)
        else:
            assert request.eval_dates == TEST_DATES
            if len(self.holdout) == self.abort_holdout_at:
                raise RuntimeError("scripted interruption before next holdout task")
            self.holdout.append(request.trial_id)

        parent = 31.0 if "evidence_gpt-r01" in request.parent.model_id else 30.0
        gain = 0.20
        if request.group == "evidence_gpt":
            if request.trial_id.endswith("r01"):
                gain = 1.0
            elif request.trial_id.endswith("r02"):
                gain = 2.0
        if request.group == "no_llm" and request.trial_id.endswith("r01"):
            gain = 0.70
        if request.parent == request.candidate:
            gain = 0.0

        def panel(score: float) -> dict[int, dict[str, float]]:
            return {
                day: {IMPACT: score, NLL: 3.0, DRIFT: 2.0}
                for day in request.eval_dates
            }

        return PairedEvaluation(panel(parent), panel(parent + gain), _metrics(parent), _metrics(parent + gain))


def test_full_roster_freezes_sources_and_reuses_holdout_records(tmp_path: Path) -> None:
    market = _market_metadata(tmp_path / "market")
    artifacts = tmp_path / "formal"
    work = tmp_path / "working-ledger"
    harness = OfflineHarness()
    evaluators = {
        stock: (lambda request, stock=stock: harness.evaluator(stock, request)) for stock in SYMBOLS
    }

    with pytest.raises(ValueError, match="freeze"):
        verify_frozen_study(artifacts, evaluators=evaluators)
    assert not harness.holdout

    kwargs = {
        "code_lock": CODE_LOCK, "registry_text": "synthetic registry",
        "make_proposer": harness.proposer, "seed_metrics": _seed_metrics,
        "evaluators": evaluators, "gate": _gate, "development_root": work,
    }
    dev = run_development(artifacts, market, **kwargs)
    assert dev == {"trials": 90, "trajectories": 30}
    assert len(harness.dev) == len(set(harness.dev)) == 90
    calls = list(harness.providers)
    assert sum(json_mode for _, _, json_mode in calls) == 72
    assert run_development(artifacts, market, **kwargs) == dev
    assert harness.providers == calls

    frozen = freeze_study(artifacts, market, development_root=work)
    assert (artifacts / MANIFEST).is_file()
    assert not (artifacts / "development").exists()
    assert len(frozen["trajectories"]) == 30
    assert sum(len(row["trials"]) for row in frozen["trajectories"].values()) == 90
    exemplar = next(item for item in roster() if item.arm == "evidence_gpt")
    first, second, third = frozen["trajectories"][exemplar.key]["trials"]
    assert first["verdict"] == "VERIFIED_GAIN"
    assert second["verdict"] == "UNVERIFIED_GAIN"
    assert third["parent"] == first["candidate"]
    assert frozen["selections"][exemplar.key]["final"] == first["trial_id"]
    assert frozen["selections"][exemplar.key]["audit_only_rejected_unverified"] == [second["trial_id"]]
    assert len({row["task_id"] for row in frozen["tasks"]}) == len(frozen["tasks"])
    assert sum(row["task_id"].startswith("cross-") for row in frozen["tasks"]) == len(ARM_MODELS) * len(SYMBOLS)
    assert frozen["cross_regime"]["evidence_gpt"][SYMBOLS[0]]["retains_seed"] is False
    fallback = frozen["cross_regime"]["metrics_ds"][SYMBOLS[0]]
    assert fallback["retains_seed"] is True and fallback["auxiliary"] is not None

    repaired = next(item for item in roster() if item.arm == "metrics_ds")
    attempts = frozen["trajectories"][repaired.key]["trials"][1]["compiler"]
    assert len(attempts) == 2 and attempts[0]["diagnostic"] == "scripted compile diagnostic"
    draft = artifacts / "sources" / f"{attempts[0]['source_sha256']}.py"
    assert "bad fixture" in draft.read_text(encoding="utf-8")
    assert freeze_study(artifacts, market, development_root=work) == frozen

    with pytest.raises(ValueError, match="closed"):
        run_development(artifacts, market, **kwargs)
    assert harness.providers == calls

    work.rename(tmp_path / "detached-development")
    harness.abort_holdout_at = 2
    with pytest.raises(RuntimeError, match="scripted interruption"):
        verify_frozen_study(artifacts, evaluators=evaluators)
    assert len(harness.holdout) == 2
    harness.abort_holdout_at = None
    report = verify_frozen_study(artifacts, evaluators=evaluators)
    assert report["scheduled_tasks"] == len(frozen["tasks"]) == len(harness.holdout)
    assert set(Counter(harness.holdout).values()) == {1}
    assert report["audit_summary"]["candidates"] == 6
    assert report["arm_summary"][SYMBOLS[0]]["evidence_gpt"]["prediction_rates"]["primary"]["hit_rate"] == 1
    assert report["cross_regime"]["metrics_ds"][SYMBOLS[0]]["auxiliary"]["diagonal"]["status"] == "measured"
    assert verify_frozen_study(artifacts, evaluators=evaluators) == report
    assert set(Counter(harness.holdout).values()) == {1}
    write_archive_tables(artifacts)
    with (artifacts / "development-trials.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 90
    with (artifacts / "api-attempts.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == len(calls)
    with (artifacts / "cross-regime.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert sum(row["role"] == "selected" for row in rows) == 20
    assert any(row["role"] == "auxiliary" for row in rows)

    original = draft.read_text(encoding="utf-8")
    draft.write_text(original + "\n# changed", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen candidate source changed"):
        load_frozen_study(artifacts)


def test_freeze_rejects_incomplete_roster_without_sealing(tmp_path: Path) -> None:
    market = _market_metadata(tmp_path / "market")
    artifacts = tmp_path / "formal"
    with pytest.raises(ValueError, match="initialize"):
        freeze_study(artifacts, market)
    ledger = Ledger(artifacts)
    with ledger.lock():
        ledger.seal(study_registration(data_registration(market), CODE_LOCK, "synthetic"), "development_plan.json")
    with pytest.raises(ValueError, match="incomplete formal trial"):
        freeze_study(artifacts, market)
    assert not (artifacts / MANIFEST).exists()
    assert not (artifacts / "sources").exists()


def test_fatal_provider_error_stops_scheduling_further_trajectories(tmp_path: Path) -> None:
    market = _market_metadata(tmp_path / "market")
    started = []
    barrier = Barrier(2, timeout=5)

    def proposer(item: Trajectory) -> TwoStepProposer:
        def transport(_messages: Messages, *, response_json: bool) -> ProviderReply:
            assert response_json
            started.append(item.key)
            barrier.wait()
            raise ProviderAttemptError("Provider HTTP 401")

        return TwoStepProposer(transport, _gate, lambda _usage: None, count_tokens=lambda _messages: 900)

    with pytest.raises(ProviderAttemptError, match="401"):
        run_development(
            tmp_path / "formal", market, code_lock=CODE_LOCK, registry_text="synthetic",
            make_proposer=proposer, seed_metrics=_seed_metrics,
            evaluators={stock: lambda request: pytest.fail(request.trial_id) for stock in SYMBOLS},
            gate=_gate, jobs=2,
        )
    assert sorted(started) == sorted(item.key for item in list(roster())[:2])
