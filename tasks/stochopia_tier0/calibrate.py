"""Calibrate the two Tier 0 diagnostic tasks using simulated training and development days."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stochopia.common import atomic_write_json
from stochopia.harness.contract import (
    AnalysisSpec, ArmKind, CandidateRef, DataSplit, EvaluationTask, ExperimentManifest,
    GroupSpec, PairedDelta, PairedEvalResult, ReportStatus, ScenarioSpec, UnitObservation,
)
from stochopia.harness.paired_stats import estimate_delta
from stochopia.harness.report import assess_candidate
from stochopia.harness.runner import Runner, classify_diagnosis
from stochopia.stats import derive_seed

from .store import DevelopmentStore


_EVIDENCE = ("flow_rmse", "price_rmse", "dominance_margin", "background_bias")
_CASES = {
    "case-a": {"channel": "flow", "effective": "x01", "control": "a_self", "joint": "a_joint", "primary": "free_rmse"},
    "case-b": {"channel": "price", "effective": "x12", "control": "x13", "joint": "b_joint", "primary": "clamped_rmse"},
}
_DIAGNOSTIC_EFFECT = 0.20
_DIAGNOSTIC_MARGIN = 0.15
_WORLD_NAME = "p0-tier0-common-rng"
_STEPS, _TRAIN_DAYS, _DEV_DAYS, _TEST_DAYS = 200, 64, 32, 64
_LR, _N_RESAMPLES, _INFO_DRIFT, _INFORMED_FRACTION = 0.001, 2000, 1.80, 0.25
_TRAIN_SEEDS = (103, 137, 211, 293, 367)


# --- Registered training, scenarios and thresholds ---


def _split(split: str, count: int, namespace: str) -> DataSplit:
    unit_ids = tuple(f"{split}-{index:03d}" for index in range(count))
    full_namespace = f"{namespace}-{split}"
    seeds = {}
    for index, unit_id in enumerate(unit_ids):
        seeds[unit_id] = derive_seed("stochopia.sim.world", _WORLD_NAME, full_namespace, index)
    return DataSplit(
        source={"kind": "simulated", "world": _WORLD_NAME, "namespace": full_namespace},
        unit_ids=unit_ids, unit_seeds=seeds,
    )


def _analysis(scene: str, min_days: int, min_repeats: int, n_resamples: int) -> tuple[AnalysisSpec, ...]:
    readouts = (
        ("background_bias", "execution", "signed_trade", "mean", 0.10, 0.025),
        ("clamped_rmse", "full", "tick", "mean_path_rmse", 0.50, 0.10),
        ("free_rmse", "full", "tick", "mean_path_rmse", 2.00 if scene == "case-a" else 0.50, 0.25),
        ("end_response", "end", "tick", "mean", 0.75, 0.25),
        ("nll", "full", "nats", "mean", 0.03, 0.02),
        ("background_bias", "recovery", "signed_trade", "mean", 0.10, 0.025),
        ("free_end_response", "end", "tick", "mean", 0.75, 0.25),
    )
    configs = []
    for readout, window, unit, estimator, min_effect, equivalence in readouts:
        acceptance: dict[str, str | float] = {"role": "prediction"}
        if readout == _CASES[scene]["primary"]:
            acceptance = {"role": "primary", "max_delta": -min_effect}
        elif (scene, readout, window) == ("case-a", "background_bias", "execution"):
            acceptance = {"role": "guardrail", "max_delta": -min_effect}
        elif readout == "nll":
            acceptance = {"role": "guardrail", "max_delta": 0.06}
        key = (readout, window) if window == "recovery" or readout == "free_end_response" else (readout,)
        configs.append(AnalysisSpec(
            analysis_id=":".join((scene, *key)), scenario_id=scene, readout=readout, window=window, unit=unit,
            estimator=estimator, acceptance=acceptance, min_effect=min_effect,
            equivalence_band=(-equivalence, equivalence), min_pairs=min_days, min_train_repeats=min_repeats,
            pairing="day", alpha=0.05, n_resamples=n_resamples,
            seed=derive_seed("stochopia-tier0-calibration-bootstrap", scene, *key),
        ))
    return tuple(configs)


def build_manifest(
    exp_id: str | None = None, *, namespace: str, steps: int = _STEPS, lr: float = _LR,
    n_train_days: int = _TRAIN_DAYS, n_dev_days: int = _DEV_DAYS, n_test_days: int = _TEST_DAYS,
    train_seeds: Sequence[int] = _TRAIN_SEEDS, n_resamples: int = _N_RESAMPLES, hidden: int = 32,
    windows_per_day: int = 6, info_drift: float = _INFO_DRIFT, informed_fraction: float = _INFORMED_FRACTION,
    search_seed: int | None = None, formal_models: Mapping[str, tuple[str, str]] | None = None,
) -> ExperimentManifest:
    """Register the two development worlds and their pure-GRU candidate recipes.

    The retained test count is a plan only. The manifest contains train/dev day IDs.
    A search seed registers one formal block with an independently rotated menu.
    """
    price_environment: dict[str, Any] = {
        "n_trades": 2048, "burn_in": 256, "noise_std": 0.50, "info_drift": info_drift,
        "flow": {"n_slots": 2, "alpha": 1.24, "max_len": 2048, "informed_frac": informed_fraction},
        "kernel": {"beta": 0.28, "tau_max": 96.0, "n_scales": 6},
    }
    flow_environment = dict(price_environment)
    flow_environment["info_drift"] = 0.0
    flow_environment["flow"] = {**price_environment["flow"], "informed_frac": 0.0}
    diagnosis = {"min_effect": _DIAGNOSTIC_EFFECT, "dominance_margin": _DIAGNOSTIC_MARGIN, "min_pairs": 2}
    action = {"children": 12, "spacing": 2, "relax": 18, "context": 48, "windows_per_day": windows_per_day}
    scenes = (
        ScenarioSpec(
            scenario_id="case-a", diagnostic_question="Which registered modification reduces the paired response error?",
            environment_kind="tier0_sim", environment={"world": "plain", "world_overrides": flow_environment, "diagnosis": diagnosis},
            action_plan=action, window="full", parent_id="x00", candidate_ids=("x00", "x01", "a_self", "a_joint"),
        ),
        ScenarioSpec(
            scenario_id="case-b", diagnostic_question="Which registered modification reduces the paired response error?",
            environment_kind="tier0_sim", environment={"world": "informed", "world_overrides": price_environment, "diagnosis": diagnosis},
            action_plan=action, window="full", parent_id="x11", candidate_ids=("x11", "x12", "x13", "b_joint"),
        ),
    )
    candidates = (
        CandidateRef(candidate_id="x00", backbone="gru", recipe="anonymous"),
        CandidateRef(candidate_id="x01", backbone="gru", recipe="separate_flow", parent_id="x00", channels=("flow",)),
        CandidateRef(candidate_id="a_self", backbone="gru", recipe="self", parent_id="x00", channels=("price",)),
        CandidateRef(candidate_id="a_joint", backbone="gru", recipe="separate_flow_own_price", parent_id="x00",
                     channels=("flow", "price")),
        CandidateRef(candidate_id="x02", backbone="gru", recipe="anonymous", parent_id="x00",
                     channels=("flow", "price"), implementation={"train_plan_overrides": {"layers": 2}}),
        CandidateRef(candidate_id="x11", backbone="gru", recipe="separate_flow", parent_id="x00", channels=("flow",)),
        CandidateRef(
            candidate_id="x12", backbone="gru", recipe="separate_flow_own_price", parent_id="x11",
            channels=("price",),
        ),
        CandidateRef(
            candidate_id="x13", backbone="gru", recipe="separate_flow", parent_id="x11", channels=("flow",),
            implementation={"background_candidate_id": "x02"},
        ),
        CandidateRef(candidate_id="b_joint", backbone="gru", recipe="separate_flow_own_price", parent_id="x11",
                     channels=("flow", "price"), implementation={"background_candidate_id": "x02"}),
    )
    llm_registration = {
        "model": "offline-structured-proposal", "model_version": "offline-v1",
        "prompt_template_id": "stochopia-p0-v1", "llm_budget": {"max_calls": 8.0, "max_tokens": 48000.0},
    }
    groups = (
        GroupSpec(group_id="evidence", kind="evidence_llm", allowed_evidence=_EVIDENCE, **llm_registration),
        GroupSpec(group_id="metrics", kind="metrics_llm", **llm_registration),
        GroupSpec(group_id="fixed", kind="no_llm", selection_rule="fixed-order"),
    )
    analysis = []
    common = {}
    for scene in scenes:
        analysis.extend(_analysis(scene.scenario_id, n_dev_days, len(train_seeds), n_resamples))
        common[scene.scenario_id] = tuple(key for key in scene.candidate_ids if key != scene.parent_id)
    dev = _split("dev", n_dev_days, namespace)
    dev = replace(dev, source={**dev.source, "planned_test_days": n_test_days})
    manifest = ExperimentManifest(
        exp_id=exp_id or namespace, scenarios=scenes, candidates=candidates, groups=groups, analysis=tuple(analysis),
        splits={"train": _split("train", n_train_days, namespace), "dev": dev},
        train_seeds={f"repeat-{i:02d}": seed for i, seed in enumerate(train_seeds, start=1)},
        train_plan={
            "world": "informed", "world_overrides": price_environment, "hidden": hidden, "layers": 1,
            "steps": steps, "batch": 16, "window": 64, "device": "cpu", "lr": lr,
        },
        seeds={"rollout": 45721, "delta_ci": 34913}, fit_version="tier0-fit-v1", measurement_version="tier0-day-v1",
        budget={"search": {"trials": 4.0}, "final_check": {"trials": 4.0}}, prediction_candidate_set=common,
        unselected_sample_rule="evaluate all common prediction candidates off budget for search",
    )
    if search_seed is None:
        return manifest
    if formal_models is None:
        raise ValueError("formal blocks require exact model and version pairs for gpt and deepseek")
    rotation = (17, 29, 41).index(search_seed)
    ids = {"x02": "bg_gru2"}
    formal_scenes = []
    for offset, scene in enumerate(scenes):
        menu = tuple(f"c_{scene.scenario_id[-1]}{i}" for i in range(4))
        first = 1 + (rotation + offset) % 3
        assigned = (menu[0], *menu[first:], *menu[1:first])
        ids.update(zip(scene.candidate_ids, assigned, strict=True))
        formal_scenes.append(replace(scene, parent_id=menu[0], candidate_ids=menu))

    own_price = "Adds an own-trade indicator to the price model and a separate own-trade price output."
    detached = "Draws background signs from a separate replay with no injected trades."
    swap_background = "Replaces the detached background model with an independently fitted two-layer GRU."
    descriptions = {
        "x01": f"{detached} Keeps the baseline price weights.",
        "a_self": f"Keeps joint generation. {own_price}", "a_joint": f"{detached} {own_price}",
        "x12": f"Keeps the baseline detached background weights. {own_price}",
        "x13": f"{swap_background} Keeps the baseline price weights.", "b_joint": f"{swap_background} {own_price}",
    }
    formal_candidates = []
    for candidate in candidates:
        implementation = dict(candidate.implementation)
        if candidate.candidate_id in descriptions:
            implementation["prompt_description"] = descriptions[candidate.candidate_id]
        if "background_candidate_id" in implementation:
            implementation["background_candidate_id"] = ids[implementation["background_candidate_id"]]
        formal_candidates.append(replace(candidate, candidate_id=ids[candidate.candidate_id],
                                         parent_id=ids[candidate.parent_id] if candidate.parent_id else None,
                                         implementation=implementation))
    formal_groups = []
    for provider in ("gpt", "deepseek"):
        model, version = formal_models[provider]
        policy = {**llm_registration, "model": model, "model_version": version}
        formal_groups.extend((GroupSpec(group_id=f"evidence_{provider}", kind="evidence_llm",
                                        allowed_evidence=("diagnostics",), **policy),
                              GroupSpec(group_id=f"metrics_{provider}", kind="metrics_llm", **policy)))
    formal_groups.append(GroupSpec(group_id="no_llm", kind="no_llm", selection_rule="seeded-random"))
    forecasts = tuple(replace(spec, acceptance={**spec.acceptance, "forecast": True}) for spec in analysis)
    return replace(manifest, exp_id=exp_id or f"p0-formal-s{search_seed}", scenarios=tuple(formal_scenes),
                   candidates=tuple(formal_candidates), groups=tuple(formal_groups), analysis=forecasts,
                   prediction_candidate_set={scene.scenario_id: scene.candidate_ids[1:] for scene in formal_scenes},
                   llm_ablation_dimensions=("model", "model_version"))


# --- Actual fitting and paired development observations ---


def _measure(
    manifest: ExperimentManifest, output: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, tuple[PairedEvalResult, ...]]],
           dict[str, list[PairedEvalResult]]]:
    """Measure the full dev menu and save actual price/background fits for Entry."""
    runner = Runner(manifest)
    store = DevelopmentStore(output, manifest.exp_id)
    store.register(manifest)
    per_seed = {}
    measurements = {}
    diagnostic_pairs = {}
    for scene in manifest.scenarios:
        scene_id = scene.scenario_id
        diagnoses = {}
        diagnostic_pairs[scene_id] = []
        results: dict[str, list[PairedEvalResult]] = {key: [] for key in manifest.prediction_candidate_set[scene_id]}
        probe_candidate = next(iter(results))
        for repeat, seed in manifest.train_seeds.items():
            parent = runner.fit_candidate(scene.parent_id, repeat, output / "weights")
            fits = [parent]
            diagnoses[repeat] = runner.diagnose(scene_id, parent)
            if scene_id == "case-b":
                ancestor = "bg_gru2" if "bg_gru2" in runner.candidates else "x02"
                fits.append(runner.fit_candidate(ancestor, repeat, output / "weights"))
            for candidate_id in results:
                candidate = runner.fit_candidate(candidate_id, repeat, output / "weights")
                fits.append(candidate)
                task = EvaluationTask(
                    task_id=f"{scene_id}-{repeat}-{candidate_id}", split="dev", scenario_id=scene_id,
                    parent_fit_id=parent.fit_id, candidate_fit_id=candidate.fit_id,
                    unit_ids=manifest.splits["dev"].unit_ids, action_ref=f"{scene_id}:action",
                    random_inputs_ref=f"{scene_id}:dev:rollout",
                )
                result = runner.run_paired_eval(task, parent, candidate)
                atomic_write_json(output / "measurements" / f"{task.task_id}.json", result.to_dict())
                results[candidate_id].append(result)
                if candidate_id == probe_candidate:
                    probe = replace(task, task_id=f"diagnosis-{task.task_id}", random_inputs_ref="diagnosis")
                    paired = runner.run_paired_eval(probe, parent, candidate)
                    diagnostic_pairs[scene_id].append(paired)
                    atomic_write_json(output / "diagnostic-pairs" / f"{probe.task_id}.json", paired.to_dict())
            with store.lock():
                for fit in fits:
                    store.save_fit(fit)
            print(f"{manifest.train_plan['steps']} {scene_id} seed={seed} measured dev menu", flush=True)
        per_seed[scene_id] = diagnoses
        measurements[scene_id] = {candidate: tuple(rows) for candidate, rows in results.items()}
    atomic_write_json(output / "diagnosis-by-repeat.json", per_seed)
    return per_seed, measurements, diagnostic_pairs


def _model_rows(
    results: Sequence[PairedEvalResult], readout_rows_for: str, repeat_ids: Sequence[str],
    unit_ids: Sequence[str] | None = None,
) -> list[UnitObservation]:
    rows = []
    for result in results:
        for row in result.observations:
            if (row.candidate_id == readout_rows_for and row.train_repeat_id in repeat_ids
                    and row.arm is not ArmKind.TRUTH and (unit_ids is None or row.unit_id in unit_ids)):
                rows.append(row)
    return rows


# --- Calibration estimates and data reuse across grid cells ---


def _paths(
    results: Sequence[PairedEvalResult], parent_id: str, repeats: Sequence[str], days: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent = _model_rows(results, parent_id, repeats, days)
    by_key = {(row.train_repeat_id, row.unit_id, row.readout): row for row in parent}
    free, clamped, truth = [], [], []
    for repeat in repeats:
        free_days, clamped_days, truth_days = [], [], []
        for day in days:
            free_row = by_key[(repeat, day, "free_rmse")]
            clamped_row = by_key[(repeat, day, "clamped_rmse")]
            free_days.append(free_row.response_path)
            clamped_days.append(clamped_row.response_path)
            truth_days.append(clamped_row.truth_path)
        free.append(free_days)
        clamped.append(clamped_days)
        truth.append(truth_days)
    return np.array(free), np.array(clamped), np.array(truth)


def _diagnostic_grid(
    scene: ScenarioSpec, results: Sequence[PairedEvalResult], direct: Mapping[str, Any],
    repeats: Sequence[str], days: Sequence[str], n_resamples: int,
) -> dict[str, Any]:
    """Bootstrap the difference of two pooled-path RMSEs over shared days.

    Paths are arrays [training repeat, day, event]. Sample days once per draw
    for free, clamped and truth and across all selected training repeats.
    Positive differences favor the preregistered channel in that scenario.
    Repeat noise records the sample standard deviation across seed margins.
    """
    expected = _CASES[scene.scenario_id]["channel"]
    free, clamped, truth = _paths(results, scene.parent_id, repeats, days)
    diagnoses = {}
    margins = []
    for index, repeat in enumerate(repeats):
        verdict = classify_diagnosis(
            free[index].mean(axis=0), clamped[index].mean(axis=0), truth[index].mean(axis=0),
            min_effect=_DIAGNOSTIC_EFFECT, dominance_margin=_DIAGNOSTIC_MARGIN,
        )
        diagnoses[repeat] = verdict
        sign = 1 if expected == "flow" else -1
        margins.append(sign * (float(verdict["flow_rmse"]) - float(verdict["price_rmse"])))

    random_seed = derive_seed("stochopia-tier0-calibration-channel", scene.scenario_id)
    rng = np.random.default_rng(random_seed)
    draws = rng.integers(len(days), size=(n_resamples, len(days)))
    flow = (free - clamped)[:, draws, :].mean(axis=2)
    price = (clamped - truth)[:, draws, :].mean(axis=2)
    margins_by_draw = np.sqrt((flow ** 2).mean(axis=2)) - np.sqrt((price ** 2).mean(axis=2))
    if expected == "price":
        margins_by_draw = -margins_by_draw
    interval = np.quantile(margins_by_draw.mean(axis=0), [0.025, 0.975]).tolist()
    row_noise = float(np.std(margins, ddof=1))
    margin = float(np.mean(margins))
    supported = interval[0] > _DIAGNOSTIC_MARGIN
    for repeat, diagnosis in diagnoses.items():
        if diagnosis["dominant_channel"] != expected or direct[repeat]["dominant_channel"] != expected:
            supported = False
    return {
        "expected_channel": expected, "status": ReportStatus.SUPPORTED if supported else ReportStatus.INCONCLUSIVE,
        "diagnoses": diagnoses, "margin": margin, "margin_interval": interval,
        "training_repeat_range": float(np.ptp(margins)), "training_repeat_sd": row_noise,
        "margin_to_repeat_noise": margin / row_noise if row_noise > 0 else None,
        "day_bootstrap_seed": random_seed,
    }


def _delta_summary(delta: PairedDelta) -> dict[str, Any]:
    values = tuple(delta.train_repeat_estimates.values())
    spread = max(values) - min(values) if len(values) >= 2 else None
    bounds = [min(values), max(values)] if values else None
    return delta.to_dict() | {"training_repeat_range": spread, "training_repeat_bounds": bounds,
                              "negative_training_repeats": sum(value < 0 for value in values)}


def _calibrate_case(
    scene: ScenarioSpec, analyses: Sequence[AnalysisSpec], direct: Mapping[str, Any],
    results: Mapping[str, tuple[PairedEvalResult, ...]], diagnostic_pairs: Sequence[PairedEvalResult],
    repeats: Sequence[str], days: Sequence[str], n_resamples: int,
) -> dict[str, Any]:
    case = _CASES[scene.scenario_id]
    diagnosis = _diagnostic_grid(scene, diagnostic_pairs, direct, repeats, days, n_resamples)
    specs = tuple(replace(spec, min_pairs=len(days), min_train_repeats=len(repeats))
                  for spec in analyses if spec.scenario_id == scene.scenario_id)
    estimates, statuses = {}, {}
    for candidate_id, paired in results.items():
        observations = _model_rows(paired, scene.parent_id, repeats, days)
        observations.extend(_model_rows(paired, candidate_id, repeats, days))
        deltas = {}
        for spec in specs:
            deltas[spec.analysis_id] = estimate_delta(
                spec, observations, parent_id=scene.parent_id, candidate_id=candidate_id, required_unit_ids=days,
            )
        estimates[candidate_id] = {key: _delta_summary(delta) for key, delta in deltas.items()}
        statuses[candidate_id] = assess_candidate(specs, deltas)
    primary = next(spec for spec in specs if spec.acceptance["role"] == "primary")
    supported = (
        diagnosis["status"] is ReportStatus.SUPPORTED and statuses[case["effective"]] is ReportStatus.SUPPORTED
        and statuses[case["joint"]] is ReportStatus.SUPPORTED and statuses[case["control"]] is ReportStatus.REFUTED
    )
    return {
        "status": ReportStatus.SUPPORTED if supported else ReportStatus.INCONCLUSIVE,
        "diagnosis": diagnosis, "primary_analysis_id": primary.analysis_id,
        "candidate_vs_parent": estimates, "candidate_status": statuses,
    }


# --- Development-only preproposal projection and command ---


def _projection(
    manifest: ExperimentManifest, measured: Mapping[str, Mapping[str, tuple[PairedEvalResult, ...]]],
    case_reports: Mapping[str, Any],
) -> dict[str, Any]:
    baseline: dict[str, dict[str, float]] = {}
    evidence: dict[str, dict[str, float]] = {field: {} for field in _EVIDENCE}
    for scene in manifest.scenarios:
        scene_id = scene.scenario_id
        rows = next(iter(measured[scene_id].values()))
        repeats = tuple(manifest.train_seeds)
        days = manifest.splits["dev"].unit_ids
        free, _, truth = _paths(rows, scene.parent_id, repeats, days)
        parent = _model_rows(rows, scene.parent_id, repeats)
        means = {}
        for readout in ("background_bias", "nll", "end_response"):
            values = [row.values["value"] for row in parent if row.readout == readout and row.window != "recovery"]
            means[readout] = float(np.mean(values))
        free_error = np.sqrt(((free - truth).mean(axis=1) ** 2).mean(axis=1))
        means["free_rmse"] = float(free_error.mean())
        baseline[scene_id] = means
        diagnosis = case_reports[scene_id]["diagnosis"]
        for key in ("flow_rmse", "price_rmse"):
            evidence[key][scene_id] = float(np.mean([row[key] for row in diagnosis["diagnoses"].values()]))
        evidence["dominance_margin"][scene_id] = float(diagnosis["margin"])
        evidence["background_bias"][scene_id] = means["background_bias"]
    return {"baseline_metrics": baseline, "evidence": evidence}


def calibrate(
    output: Path, *, exp_id: str | None, steps: Sequence[int], dev_days: Sequence[int], train_seeds: Sequence[int],
    train_days: int, n_resamples: int, hidden: int, windows_per_day: int, info_drift: float,
    informed_fraction: float, namespace: str, lr: float = _LR, test_days: int = _TEST_DAYS,
) -> dict[str, Any]:
    """Run a grid; write a formal manifest only when both causal contrasts qualify."""
    if len(train_seeds) < 2 or len(set(train_seeds)) != len(train_seeds):
        raise ValueError("calibration requires at least two distinct training seeds")
    if not steps or min(steps) < 1 or not dev_days or min(dev_days) < 2 or n_resamples < 1:
        raise ValueError("calibration requires positive steps and draws and at least two dev days")
    output.mkdir(parents=True, exist_ok=False)
    days_grid = sorted(set(dev_days))
    step_grid = sorted(set(steps))
    repeat_counts = (len(train_seeds),)
    maximum_days = max(days_grid)
    manifests = {}
    for step_count in step_grid:
        manifests[step_count] = build_manifest(
            exp_id, steps=step_count, n_train_days=train_days, n_dev_days=maximum_days, train_seeds=train_seeds,
            n_resamples=n_resamples, namespace=namespace, hidden=hidden, windows_per_day=windows_per_day,
            info_drift=info_drift, informed_fraction=informed_fraction, lr=lr, n_test_days=test_days,
        )
    atomic_write_json(output / "calibration-plan.json", {
        "candidate_roles": _CASES, "dev_day_counts": days_grid, "training_repeat_counts": repeat_counts,
        "configurations": {str(step): manifest.to_dict() for step, manifest in manifests.items()},
        "profile_selection": "smallest passing fit budget at full dev-day and repeat counts",
        "criteria": [
            "Channel diagnosis agrees across training repeats and pooled day margin clears the registered margin.",
            "Pooled day intervals support the targeted and joint candidates and refute the control.",
            "Per-repeat effect values and range are descriptive; the day interval is conditional on trained fits.",
        ],
    })
    summary: dict[str, Any] = {"status": ReportStatus.INCONCLUSIVE, "grid": [], "selected": None}
    for step_count, manifest in manifests.items():
        direct, measured, diagnostic_pairs = _measure(manifest, output / f"steps-{step_count}")
        for n_days in days_grid:
            days = manifest.splits["dev"].unit_ids[:n_days]
            for n_repeats in repeat_counts:
                repeats = tuple(manifest.train_seeds)[:n_repeats]
                reports = {}
                for scene in manifest.scenarios:
                    key = scene.scenario_id
                    reports[key] = _calibrate_case(
                        scene, manifest.analysis, direct[key], measured[key], diagnostic_pairs[key],
                        repeats, days, n_resamples,
                    )
                row = {
                    "steps": step_count, "n_dev_days": n_days, "train_seeds": [manifest.train_seeds[key] for key in repeats],
                    "cases": reports,
                }
                summary["grid"].append(row)
                passing = all(case["status"] is ReportStatus.SUPPORTED for case in reports.values())
                if passing and n_days == maximum_days and n_repeats == len(train_seeds) and summary["selected"] is None:
                    summary["status"] = ReportStatus.SUPPORTED
                    summary["selected"] = row
                    atomic_write_json(output / "experiment-manifest.json", manifest.to_dict())
                    atomic_write_json(output / "development-projection.json", _projection(manifest, measured, reports))
                atomic_write_json(output / "calibration-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exp-id")
    parser.add_argument("--steps", type=int, nargs="+", default=(_STEPS,))
    parser.add_argument("--dev-days", type=int, nargs="+", default=(_DEV_DAYS,))
    parser.add_argument("--train-seeds", type=int, nargs="+", default=_TRAIN_SEEDS)
    parser.add_argument("--train-days", type=int, default=_TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=_TEST_DAYS)
    parser.add_argument("--lr", type=float, default=_LR)
    parser.add_argument("--n-resamples", type=int, default=_N_RESAMPLES)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--windows-per-day", type=int, default=6)
    parser.add_argument("--info-drift", type=float, default=_INFO_DRIFT)
    parser.add_argument("--informed-fraction", type=float, default=_INFORMED_FRACTION)
    parser.add_argument("--namespace", default="p0-calibration-2026-11-23")
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = calibrate(**vars(args))
    print(f"Calibration: {summary['status'].value}; {args.output / 'calibration-summary.json'}", flush=True)


if __name__ == "__main__":
    main()
