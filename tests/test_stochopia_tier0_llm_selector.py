"""P0 selector fairness, retry limits and an offline frozen release rehearsal."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

STOCHOPIA = Path(os.environ.get("STOCHOPIA_REPO", Path(__file__).resolve().parents[2] / "Stochopia"))
if str(STOCHOPIA) not in sys.path:
    sys.path.append(str(STOCHOPIA))
pytest.importorskip("stochopia.harness.contract", reason="P0 selector tests require the companion Stochopia checkout")
httpx = pytest.importorskip("httpx", reason="P0 selector tests require the Stochopia client dependencies")

from stochopia.harness.contract import (
    AnalysisSpec, ArmKind, CandidateRef, DataSplit, Direction, EvaluationTask, ExperimentManifest,
    FinalChoice, FitRecord, FrozenManifest, GroupSpec, PairedEvalResult, ReportStatus, ScenarioSpec,
    SelectionRecord, StructuredPrediction, UnitObservation,
)
from stochopia.harness.paired_stats import evaluate_pairs

from tasks.stochopia_tier0.entry import CheckRequest, Entry
from tasks.stochopia_tier0.llm_selector import build_selectors, registry_client_factory
from tasks.stochopia_tier0.store import DevelopmentStore
from tasks.stochopia_tier0.verifier import Verifier, VerifierConfig, freeze


# --- Registered inputs and captured model transport ---


def _split(label: str, seed: int) -> DataSplit:
    names = (f"{label}-a", f"{label}-b")
    return DataSplit(
        source={"kind": "simulated", "namespace": label, "world": "tier0-plain"},
        unit_ids=names, unit_seeds={names[0]: seed, names[1]: seed + 1},
    )


@pytest.fixture
def manifest() -> ExperimentManifest:
    groups = (
        GroupSpec(
            group_id="evidence", kind="evidence_llm", model="registry-alias", model_version="pinned-model",
            prompt_template_id="tier0-anonymous-v1", llm_budget={"max_calls": 4, "max_tokens": 48000},
            allowed_evidence=("flow_rmse",),
        ),
        GroupSpec(
            group_id="metrics", kind="metrics_llm", model="registry-alias", model_version="pinned-model",
            prompt_template_id="tier0-anonymous-v1", llm_budget={"max_calls": 4, "max_tokens": 48000},
        ),
        GroupSpec(group_id="fixed", kind="no_llm", selection_rule="fixed-order"),
    )
    scene = ScenarioSpec(
        scenario_id="scene-internal", diagnostic_question="Problem names private-flow-control",
        environment_kind="tier0_sim", environment={"world": "plain", "private": "hidden-match-key"},
        action_plan={"n_children": 2, "spacing": 1, "n_relax": 4}, window="end",
        parent_id="private-base", candidate_ids=("private-base", "private-flow-control", "private-own-price"),
    )
    readout = AnalysisSpec(
        analysis_id="endpoint-internal", scenario_id=scene.scenario_id, readout="end_response", window="end",
        unit="tick", estimator="mean", acceptance={"min_delta": 0.2}, min_effect=0.2,
        equivalence_band=(-0.2, 0.2), min_pairs=2, min_train_repeats=1,
        pairing="day_exhaustive", alpha=0.05, n_resamples=4, seed=13,
    )
    return ExperimentManifest(
        exp_id="selector-ci", scenarios=(scene,),
        candidates=(
            CandidateRef(candidate_id="private-base", backbone="gru", recipe="seed"),
            CandidateRef(
                candidate_id="private-flow-control", backbone="gru", recipe="separate_flow",
                implementation={"secret": "opaque-implementation"}, parent_id="private-base", channels=("flow",),
            ),
            CandidateRef(
                candidate_id="private-own-price", backbone="gru", recipe="separate_flow_own_price",
                parent_id="private-base", channels=("flow", "price"),
            ),
        ),
        groups=groups, splits={"train": _split("train", 101), "dev": _split("dev", 201)},
        train_seeds={"repeat": 11}, train_plan={"steps": 2},
        measurement_version="tier0-v1", fit_version="tier0-v1", seeds={"rollout": 12, "delta_ci": 13},
        budget={"search": {"trials": 2}, "final_check": {"trials": 2}},
        analysis=(readout,),
        prediction_candidate_set={scene.scenario_id: ("private-flow-control", "private-own-price")},
        unselected_sample_rule="all unselected common candidates",
    )


class FakeClient:
    def __init__(self, invalid_attempts: int = 0, *, missing_interval: bool = False) -> None:
        self.invalid_attempts = invalid_attempts
        self.missing_interval = missing_interval
        self.messages: list[list[dict[str, str]]] = []
        self.total_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def chat(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        assert max_tokens > 0
        self.messages.append(messages)
        self.total_usage["calls"] += 1
        self.total_usage["prompt_tokens"] += 100
        self.total_usage["completion_tokens"] += 25
        request = json.loads(messages[-1]["content"])
        answer = _response(request)
        if len(self.messages) <= self.invalid_attempts:
            if self.missing_interval:
                del answer["predictions"][0]["effect_interval"]
            else:
                answer["predictions"][0]["effect_interval"] = [-2, -1]
        return json.dumps(answer)


def _response(request: Mapping[str, Any]) -> dict[str, Any]:
    first = request["targets"][0]["option"]
    predictions = []
    for target in request["targets"]:
        predictions.append({
            "option": target["option"], "target": target["target"], "direction": "increase",
            "effect_interval": [0.5, 1.5],
        })
    return {"action": "select", "option": first, "reason": "Expected positive response", "predictions": predictions}


def _entry(tmp_path: Path, manifest: ExperimentManifest) -> Entry:
    return Entry(DevelopmentStore(tmp_path / "runs", manifest.exp_id), manifest)


# --- Preregistration across registered LLM groups ---


def _abstention_plan(manifest: ExperimentManifest, round_index: int = 1) -> tuple[
    tuple[SelectionRecord, ...], tuple[StructuredPrediction, ...],
]:
    scenario = manifest.scenarios[0]
    analysis = manifest.analysis[0]
    selections = []
    predictions = []
    for group in manifest.groups:
        inputs = {
            "exp_id": manifest.exp_id, "group_id": group.group_id, "scenario_id": scenario.scenario_id,
            "parent_id": scenario.parent_id, "round_index": round_index, "dev_snapshot_id": f"round-{round_index}",
        }
        selection_id = f"r{round_index}-{group.group_id}"
        selections.append(SelectionRecord(
            **inputs, selection_id=selection_id, sequence=round_index, menu_id="registered-menu",
            action="keep_parent", reason="Retain the registered baseline for this test.",
        ))
        if group.kind != "no_llm":
            for candidate in manifest.prediction_candidate_set[scenario.scenario_id]:
                predictions.append(StructuredPrediction(
                    **inputs, prediction_id=f"{selection_id}-{candidate}", explanation_id="registered",
                    candidate_id=candidate, analysis_id=analysis.analysis_id, abstained=True,
                    abstain_reason="No numerical forecast.",
                ))
    return tuple(selections), tuple(predictions)


@pytest.mark.parametrize("llm_count", [1, 3])
def test_entry_preregisters_one_or_three_llm_groups(
    manifest: ExperimentManifest, tmp_path: Path, llm_count: int,
) -> None:
    evidence, metrics, fixed = manifest.groups
    llms = (metrics,) if llm_count == 1 else (evidence, metrics, replace(metrics, group_id="metrics-2"))
    manifest = replace(manifest, groups=(*llms, fixed))
    entry = _entry(tmp_path, manifest)
    selections, predictions = _abstention_plan(manifest)

    entry.preregister_round(1, selections=selections, predictions=predictions, baseline_metrics={})

    saved = entry.store.preregistered(1)
    assert saved is not None
    assert {row["group_id"] for row in saved["predictions"]} == {group.group_id for group in llms}


def test_entry_rejects_intersecting_but_unequal_common_analyses(
    manifest: ExperimentManifest, tmp_path: Path,
) -> None:
    extra = replace(manifest.analysis[0], analysis_id="second-forecast")
    manifest = replace(manifest, analysis=(*manifest.analysis, extra))
    entry = _entry(tmp_path, manifest)
    selections, predictions = _abstention_plan(manifest)
    unmatched = replace(predictions[0], prediction_id="extra-forecast", analysis_id=extra.analysis_id)

    with pytest.raises(ValueError, match="common analysis"):
        entry.preregister_round(1, selections=selections, predictions=(*predictions, unmatched), baseline_metrics={})

    assert entry.store.preregistered(1) is None


def test_entry_requires_same_round_selection_for_new_forecast(
    manifest: ExperimentManifest, tmp_path: Path,
) -> None:
    entry = _entry(tmp_path, manifest)
    selections, predictions = _abstention_plan(manifest)
    entry.preregister_round(1, selections=selections, predictions=predictions, baseline_metrics={})

    def unexpected_evaluation(request: SelectionRecord | CheckRequest) -> PairedEvalResult:
        pytest.fail(f"retaining the baseline must not create a trial: {request}")

    entry.run_round(1, evaluate=unexpected_evaluation)
    selections, predictions = _abstention_plan(manifest, round_index=2)
    with pytest.raises(ValueError, match="same-round selection"):
        entry.preregister_round(
            2, selections=(selections[0],), predictions=(predictions[-1],), baseline_metrics={},
        )


# --- Anonymization, retries and budget enforcement ---


def test_prompt_uses_only_anonymous_inputs_and_same_template(
    manifest: ExperimentManifest, tmp_path: Path,
) -> None:
    entry = _entry(tmp_path, manifest)
    client = FakeClient()
    configured = []

    def factory(model: str, version: str) -> FakeClient:
        configured.append((model, version))
        return client

    selectors = build_selectors(manifest, mode="real", client_factory=factory)
    scene = manifest.scenarios[0]
    baseline = {scene.scenario_id: {"end_response": 0.75, "path": "/test/internal", "private-flow-control": 300}}
    evidence = {"flow_rmse": {scene.scenario_id: 1.5, "path": "hidden-match-key"},
                "mechanism": "separate_flow_own_price", "outside": "super-secret"}
    for group in ("evidence", "metrics"):
        snapshot = entry.snapshot(group, 1, baseline_metrics=baseline, evidence=evidence)
        record, forecasts = selectors[group](snapshot, scene)
        assert record.candidate_id in manifest.prediction_candidate_set[scene.scenario_id]
        assert {row.candidate_id for row in forecasts} == set(manifest.prediction_candidate_set[scene.scenario_id])

    assert configured and set(configured) == {("registry-alias", "pinned-model")}
    assert len(client.messages) == 2
    encoded = json.dumps(client.messages)
    for text in (
        "private-base", "private-flow-control", "private-own-price", "separate_flow", "opaque-implementation",
        "scene-internal", "endpoint-internal", "/test/internal", "hidden-match-key", "super-secret",
        "background_flow", "price_response",
    ):
        assert text not in encoded
    left_messages, right_messages = client.messages
    assert left_messages[:-1] == right_messages[:-1]
    left = json.loads(left_messages[-1]["content"])
    right = json.loads(right_messages[-1]["content"])
    left_evidence = left.pop("evidence")
    right_evidence = right.pop("evidence")
    assert left == right
    assert left_evidence and not right_evidence
    assert set(item["option"] for item in left["options"]) == {"C1", "C2"}


@pytest.mark.parametrize("missing_interval", (True, False))
def test_structured_prediction_retry_returns_specific_error(
    manifest: ExperimentManifest, tmp_path: Path, missing_interval: bool,
) -> None:
    client = FakeClient(1, missing_interval=missing_interval)
    selectors = build_selectors(manifest, mode="real", client_factory=lambda model, version: client)
    entry = _entry(tmp_path, manifest)
    selector = selectors["evidence"]
    record, rows = selector(entry.snapshot("evidence", 1, baseline_metrics={}), manifest.scenarios[0])
    assert record.action == "select"
    assert all(not row.abstained and row.direction is Direction.INCREASE for row in rows)
    assert len(client.messages) == 2
    retry_payload = json.loads(client.messages[1][-1]["content"])
    assert "effect_interval" in retry_payload["validation_feedback"]
    if not missing_interval:
        assert "prediction direction" in retry_payload["validation_feedback"]
    assert selector.usage()["calls"] == 2


def test_budget_exhaustion_blocks_retry_and_survives_new_selector(
    manifest: ExperimentManifest, tmp_path: Path,
) -> None:
    groups = tuple(
        replace(group, llm_budget={**group.llm_budget, "max_calls": 1}) if group.llm_budget else group
        for group in manifest.groups
    )
    manifest = replace(manifest, groups=groups)
    client = FakeClient(2)
    entry = _entry(tmp_path, manifest)
    usage_root = tmp_path / "usage"
    snapshot = entry.snapshot("evidence", 1, baseline_metrics={})
    for invocation in range(2):
        selectors = build_selectors(
            manifest, mode="real", client_factory=lambda model, version: client, usage_root=usage_root,
        )
        selection, rows = selectors["evidence"](snapshot, manifest.scenarios[0])
        assert selection.action == "abstain" and selection.reason
        assert len(rows) == 2 and all(row.abstained and row.abstain_reason for row in rows)
        assert selectors["evidence"].usage()["calls"] == 1
        if invocation == 0:
            assert "effect_interval" in selection.reason
            assert all(row.abstain_reason == selection.reason for row in rows)
    assert len(client.messages) == 1


def test_token_budget_blocks_transport_before_sending(
    manifest: ExperimentManifest, tmp_path: Path,
) -> None:
    groups = tuple(
        replace(group, llm_budget={**group.llm_budget, "max_tokens": 32}) if group.llm_budget else group
        for group in manifest.groups
    )
    manifest = replace(manifest, groups=groups)
    entry = _entry(tmp_path, manifest)
    client = FakeClient()
    selector = build_selectors(manifest, mode="real", client_factory=lambda model, version: client)["metrics"]
    record, forecasts = selector(entry.snapshot("metrics", 1, baseline_metrics={}), manifest.scenarios[0])
    assert record.action == "abstain" and "token" in record.reason.lower()
    assert all(forecast.abstained for forecast in forecasts)
    assert not client.messages and selector.usage()["calls"] == 0


def test_registry_transport_pins_model_and_has_no_hidden_http_retries(
    manifest: ExperimentManifest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = tuple(
        replace(group, llm_budget={**group.llm_budget, "max_calls": 1}) if group.llm_budget else group
        for group in manifest.groups
    )
    manifest = replace(manifest, groups=groups)
    registry = tmp_path / "models.yaml"
    registry.write_text(
        "models:\n  registry-alias:\n    provider: openai-compatible\n"
        "    base_url: https://example.invalid/v1\n    model: floating-alias\n"
        "    api_key_env: STOCHOPIA_TEST_SELECTOR_KEY\n"
    )
    monkeypatch.setenv("STOCHOPIA_TEST_SELECTOR_KEY", "only-used-by-mock")
    requests = []

    def unavailable(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(503, text="arbitrary-provider-error-with-private-details")

    actual_client = httpx.AsyncClient

    def local_http_client(*, timeout: float) -> httpx.AsyncClient:
        return actual_client(timeout=timeout, transport=httpx.MockTransport(unavailable))

    monkeypatch.setattr(httpx, "AsyncClient", local_http_client)
    entry = _entry(tmp_path, manifest)
    selector = build_selectors(
        manifest, mode="real", client_factory=registry_client_factory(registry),
        usage_root=entry.store.path / "selector-usage", retry_limit=2,
    )["metrics"]
    selection, forecasts = selector(entry.snapshot("metrics", 1, baseline_metrics={}), manifest.scenarios[0])
    assert selection.action == "abstain"
    assert all(row.abstained for row in forecasts)
    assert "arbitrary-provider-error" not in selection.reason
    assert len(requests) == selector.usage()["calls"] == 1
    assert requests[0]["model"] == "pinned-model"
    assert selector.usage()["reserved_tokens"] > 0


def test_selector_normalizes_native_scalar_inputs(manifest: ExperimentManifest, tmp_path: Path) -> None:
    client = FakeClient()
    scene = manifest.scenarios[0]
    entry = _entry(tmp_path, manifest)
    selector = build_selectors(
        manifest, mode="real", client_factory=lambda model, version: client,
        retry_limit=np.int64(0), max_output_tokens=np.int64(256),
        count_prompt_tokens=lambda messages: np.int64(200),
    )["metrics"]
    snapshot = entry.snapshot("metrics", 1, baseline_metrics={})
    snapshot["round_index"] = np.int64(1)
    snapshot["baseline_metrics"] = {scene.scenario_id: {
        "end_response": np.float32(1.5), "nll": False, "clamped_rmse": np.float64(np.inf),
    }}

    record, _ = selector(snapshot, scene)

    assert record.action == "select"
    assert json.loads(client.messages[0][-1]["content"])["baseline"] == {"end_response": 1.5}
    assert selector.usage()["calls"] == 1


# --- Offline development, freeze and independent replay ---


def _offline_response(request: Mapping[str, Any]) -> dict[str, Any]:
    """Arrange all four forecast outcomes on registered, hand-checkable replay rows."""
    targets = request["targets"]
    assert len(targets) == 2
    first = targets[0]
    second = targets[1]
    evidence = bool(request["evidence"])
    answer = _response(request)
    if evidence:
        answer["predictions"][1].update(direction="abstain", effect_interval=None)
    else:
        answer["option"] = second["option"]
        answer["predictions"][0].update(direction="decrease", effect_interval=[-1.5, -0.5])
        answer["predictions"][1].update(direction="increase", effect_interval=[0.5, 2.0])
    assert answer["predictions"][0]["option"] == first["option"]
    return answer


def _fits(experiment: ExperimentManifest) -> tuple[FitRecord, ...]:
    return tuple(
        FitRecord(
            fit_id=f"fit-{candidate.candidate_id}", candidate_id=candidate.candidate_id,
            train_repeat_id="repeat", train_data_ref="fixture-train", config={"recipe": candidate.recipe},
            seed=experiment.train_seeds["repeat"], model_ref=f"fixture-{candidate.candidate_id}",
            parent_id=candidate.parent_id,
        )
        for candidate in experiment.candidates
    )


def _fixture_measurement(
    task: EvaluationTask, parent: FitRecord, candidate: FitRecord, experiment: ExperimentManifest,
) -> PairedEvalResult:
    """Provide two paired inputs to the production bootstrap and report.

    The first edit has stable +1 daily effect. The second has paired effects
    0 and +2, straddling the declared minimum without being disjoint from it.
    These are controlled values for checking status wiring, not trained performance.
    """
    scene = experiment.scenarios[0]
    readout = experiment.analysis[0]
    first = experiment.prediction_candidate_set[scene.scenario_id][0]
    candidate_values = (1.0, 1.0) if candidate.candidate_id == first else (0.0, 2.0)
    observations = []
    for unit_id, value in zip(task.unit_ids, candidate_values, strict=True):
        for model, measured in ((parent, 0.0), (candidate, value)):
            observations.append(UnitObservation(
                unit_id=unit_id, scenario_id=scene.scenario_id, train_repeat_id=model.train_repeat_id,
                arm=ArmKind.CLAMPED, window=readout.window, readout=readout.readout, unit=readout.unit,
                candidate_id=model.candidate_id, values={"value": measured},
            ))
    return evaluate_pairs(
        task, scene, parent, candidate, observations, experiment.analysis,
        cost={"train_seconds": 3.0 if task.split == "dev" else 0.0, "eval_seconds": 1.0},
    )


def run_offline_rehearsal(root: Path, experiment: ExperimentManifest) -> dict[str, Any]:
    """Run and persist an offline release rehearsal with trusted fixture observations."""
    store = DevelopmentStore(root / "runs", experiment.exp_id)
    verifier_root = root / "verification"
    config = VerifierConfig(experiment.exp_id, "offline-rehearsal", _split("heldout-oracle", 301),
                            {"rollout": 14, "delta_ci": 15})
    Verifier(verifier_root).register_config(config)
    entry = Entry(store, experiment)
    selectors = build_selectors(
        experiment, mode="offline", offline_responder=_offline_response, usage_root=store.path / "selector-usage",
    )
    scene = experiment.scenarios[0]
    fit_by_candidate = {fit.candidate_id: fit for fit in _fits(experiment)}
    parent = fit_by_candidate[scene.parent_id]
    choices = experiment.prediction_candidate_set[scene.scenario_id]

    def evaluate_development(request: SelectionRecord | CheckRequest) -> PairedEvalResult:
        candidate = fit_by_candidate[request.candidate_id]
        prefix = request.selection_id if isinstance(request, SelectionRecord) else "common-check"
        task = EvaluationTask(
            task_id=f"dev-{prefix}", split="dev", scenario_id=scene.scenario_id,
            parent_fit_id=parent.fit_id, candidate_fit_id=candidate.fit_id,
            unit_ids=experiment.splits["dev"].unit_ids, action_ref="fixture-order", random_inputs_ref="dev-shared",
        )
        return _fixture_measurement(task, parent, candidate, experiment)

    feedback = entry.run_round(
        1, selectors=selectors, evaluate=evaluate_development,
        evidence={"flow_rmse": {scene.scenario_id: 1.5}}, baseline_metrics={scene.scenario_id: {"end_response": 0.0}},
        checks=(CheckRequest("evidence", scene.scenario_id, choices[1], "common_prediction"),),
    )
    final = []
    for group in experiment.groups:
        trial = feedback[group.group_id][0]["trial"]
        final.append(FinalChoice(
            group_id=group.group_id, scenario_id=scene.scenario_id, candidate_id=trial["candidate_id"],
            selection_id=trial["selection_id"], stopping_reason="offline rehearsal planned one search decision",
        ))
    exported = entry.export(final, fits=tuple(fit_by_candidate.values()))
    tasks = []
    for index, candidate_id in enumerate(choices, start=1):
        tasks.append(EvaluationTask(
            task_id=f"frozen-{index}", split="test", scenario_id=scene.scenario_id,
            parent_fit_id=parent.fit_id, candidate_fit_id=fit_by_candidate[candidate_id].fit_id,
            unit_ids=config.test_split.unit_ids, action_ref="fixture-order", random_inputs_ref="test-shared",
        ))
    freeze(exported, config, verifier_root, tasks=tasks)

    def evaluate_frozen(
        task: EvaluationTask, parent: FitRecord, candidate: FitRecord, frozen: FrozenManifest,
    ) -> PairedEvalResult:
        return _fixture_measurement(task, parent, candidate, frozen.experiment)

    verifier = Verifier(verifier_root, evaluate=evaluate_frozen)
    report = verifier.run()
    assert Verifier(verifier_root).run() == report
    return report


def test_offline_three_arm_rehearsal_scores_all_statuses_and_separates_costs(
    manifest: ExperimentManifest, tmp_path: Path,
) -> None:
    report = run_offline_rehearsal(tmp_path, manifest)
    statuses = {row["status"] for row in report["predictions"]}
    assert statuses == {
        ReportStatus.SUPPORTED.value, ReportStatus.REFUTED.value,
        ReportStatus.INCONCLUSIVE.value, ReportStatus.ABSTAINED.value,
    }
    assert set(report["z_to_x"]) == {"evidence", "metrics", "fixed"}
    assert set(report["x_to_delta_y"]) == {"evidence", "metrics"}
    assert all(row["search_budget_used"] == 1 for row in report["z_to_x"].values())
    assert report["budgets"]["development_checks"] == {"trials": 1.0, "train_seconds": 3.0, "eval_seconds": 1.0}
    assert report["budgets"]["frozen_measurement"] == {"train_seconds": 0.0, "eval_seconds": 2.0}
    assert report["x_to_delta_y"]["evidence"]["coverage_rate"] == 0.5
