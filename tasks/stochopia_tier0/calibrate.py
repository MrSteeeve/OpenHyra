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


_EVIDENCE = ("flow_rmse", "price_rmse", "dominance_margin", "background_bias")
_CASES = {
    "case-a": {"channel": "flow", "effective": "x01", "control": "x02", "primary": "background_bias"},
    "case-b": {"channel": "price", "effective": "x12", "control": "x13", "primary": "clamped_rmse"},
}
_DIAGNOSTIC_EFFECT = 0.20
_DIAGNOSTIC_MARGIN = 0.15
_WORLD_NAME = "p0-tier0-common-rng"


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
        ("background_bias", "execution", "signed_trade", "mean", 0.05, 0.025),
        ("clamped_rmse", "full", "tick", "mean_path_rmse", 0.20, 0.10),
        ("free_rmse", "full", "tick", "mean_path_rmse", 0.20, 0.10),
        ("end_response", "end", "tick", "mean", 0.20, 0.10),
        ("nll", "full", "nats", "mean", 0.10, 0.05),
    )
    configs = []
    for readout, window, unit, estimator, min_effect, equivalence in readouts:
        acceptance: dict[str, str | float] = {"role": "prediction"}
        if readout == _CASES[scene]["primary"]:
            acceptance = {"role": "primary", "max_delta": -min_effect}
        elif (scene, readout) == ("case-a", "clamped_rmse") or (scene, readout) == ("case-b", "background_bias"):
            acceptance = {"role": "guardrail", "min_delta": -equivalence, "max_delta": equivalence}
        configs.append(AnalysisSpec(
            analysis_id=f"{scene}:{readout}", scenario_id=scene, readout=readout, window=window, unit=unit,
            estimator=estimator, acceptance=acceptance, min_effect=min_effect,
            equivalence_band=(-equivalence, equivalence), min_pairs=min_days, min_train_repeats=min_repeats,
            pairing="day", alpha=0.05, n_resamples=n_resamples,
            seed=derive_seed("stochopia-tier0-calibration-bootstrap", scene, readout),
        ))
    return tuple(configs)


def build_manifest(
    exp_id: str, *, steps: int, n_train_days: int, n_dev_days: int, train_seeds: Sequence[int],
    n_resamples: int, namespace: str, hidden: int = 32, windows_per_day: int = 6,
    info_drift: float = 0.85, informed_fraction: float = 0.40,
) -> ExperimentManifest:
    """Register matching training for all GRU/SSM branches and two dev worlds.

    Both development mechanisms share the registered day seed namespace and
    trade-flow construction. Only the informed scene adds information drift.
    The common training mechanism is stated in the training plan and applied
    to all candidates, including the unseen parent of the separated price baseline.
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
            action_plan=action, window="full", parent_id="x00", candidate_ids=("x00", "x01", "x02"),
        ),
        ScenarioSpec(
            scenario_id="case-b", diagnostic_question="Which registered modification reduces the paired response error?",
            environment_kind="tier0_sim", environment={"world": "informed", "world_overrides": price_environment, "diagnosis": diagnosis},
            action_plan=action, window="full", parent_id="x11", candidate_ids=("x11", "x12", "x13"),
        ),
    )
    candidates = (
        CandidateRef(candidate_id="x00", backbone="gru", recipe="anonymous"),
        CandidateRef(candidate_id="x01", backbone="gru", recipe="separate_flow", parent_id="x00", channels=("flow",)),
        CandidateRef(candidate_id="x02", backbone="ssm", recipe="anonymous", parent_id="x00", channels=("flow", "price")),
        CandidateRef(candidate_id="x11", backbone="gru", recipe="separate_flow", parent_id="x00", channels=("flow",)),
        CandidateRef(
            candidate_id="x12", backbone="gru", recipe="separate_flow_own_price", parent_id="x11",
            channels=("price",),
        ),
        CandidateRef(
            candidate_id="x13", backbone="gru", recipe="separate_flow", parent_id="x11", channels=("flow",),
            implementation={"background_candidate_id": "x02"},
        ),
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
    return ExperimentManifest(
        exp_id=exp_id, scenarios=scenes, candidates=candidates, groups=groups, analysis=tuple(analysis),
        splits={"train": _split("train", n_train_days, namespace), "dev": _split("dev", n_dev_days, namespace)},
        train_seeds={f"repeat-{i:02d}": seed for i, seed in enumerate(train_seeds, start=1)},
        train_plan={
            "world": "informed", "world_overrides": price_environment, "hidden": hidden, "layers": 1,
            "steps": steps, "batch": 16, "window": 64, "device": "cpu", "lr": 0.003,
        },
        seeds={"rollout": 45721, "delta_ci": 34913}, fit_version="tier0-fit-v1", measurement_version="tier0-day-v1",
        budget={"search": {"trials": 4.0}, "final_check": {"trials": 4.0}}, prediction_candidate_set=common,
        unselected_sample_rule="evaluate all common prediction candidates off budget for search",
    )


# --- Actual fitting and paired development observations ---


def _measure(
    manifest: ExperimentManifest, output: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, tuple[PairedEvalResult, ...]]]]:
    runner = Runner(manifest)
    per_seed = {}
    measurements = {}
    for scene in manifest.scenarios:
        scene_id = scene.scenario_id
        case = _CASES[scene_id]
        diagnoses = {}
        results: dict[str, list[PairedEvalResult]] = {case["effective"]: [], case["control"]: []}
        for repeat, seed in manifest.train_seeds.items():
            parent = runner.fit_candidate(scene.parent_id, repeat, output / "weights")
            diagnoses[repeat] = runner.diagnose(scene_id, parent)
            for candidate_id in results:
                candidate = runner.fit_candidate(candidate_id, repeat, output / "weights")
                task = EvaluationTask(
                    task_id=f"{scene_id}-{repeat}-{candidate_id}", split="dev", scenario_id=scene_id,
                    parent_fit_id=parent.fit_id, candidate_fit_id=candidate.fit_id,
                    unit_ids=manifest.splits["dev"].unit_ids, action_ref=f"{scene_id}:action",
                    random_inputs_ref=f"{scene_id}:dev:rollout",
                )
                result = runner.run_paired_eval(task, parent, candidate)
                atomic_write_json(output / "measurements" / f"{task.task_id}.json", result.to_dict())
                results[candidate_id].append(result)
            margin = diagnoses[repeat]["flow_rmse"] - diagnoses[repeat]["price_rmse"]
            primary = f"{scene_id}:{case['primary']}"
            paired_estimates = []
            for rows in results.values():
                delta = rows[-1].paired_diffs.get(primary)
                paired_estimates.append(None if delta is None else delta.estimate)
            print(
                f"{manifest.train_plan['steps']} {scene_id} seed={seed} flow-minus-price={margin:.3f} "
                f"primary effective/control={paired_estimates}", flush=True,
            )
        per_seed[scene_id] = diagnoses
        measurements[scene_id] = {candidate: tuple(rows) for candidate, rows in results.items()}
    atomic_write_json(output / "diagnosis-by-repeat.json", per_seed)
    return per_seed, measurements


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
    supported = interval[0] >= _DIAGNOSTIC_MARGIN
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
    return delta.to_dict() | {"training_repeat_range": spread}


def _calibrate_case(
    scene: ScenarioSpec, analyses: Sequence[AnalysisSpec], direct: Mapping[str, Any],
    results: Mapping[str, tuple[PairedEvalResult, ...]], repeats: Sequence[str], days: Sequence[str],
    n_resamples: int,
) -> dict[str, Any]:
    case = _CASES[scene.scenario_id]
    effective, control = case["effective"], case["control"]
    diagnosis = _diagnostic_grid(scene, results[effective], direct, repeats, days, n_resamples)
    effective_obs = _model_rows(results[effective], effective, repeats, days)
    control_obs = _model_rows(results[control], control, repeats, days)
    parent_obs = _model_rows(results[effective], scene.parent_id, repeats, days)
    specs = []
    effective_deltas, control_deltas = {}, {}
    for analysis in analyses:
        if analysis.scenario_id != scene.scenario_id or analysis.acceptance["role"] == "prediction":
            continue
        spec = replace(analysis, min_pairs=len(days), min_train_repeats=len(repeats))
        specs.append(spec)
        effective_deltas[spec.analysis_id] = estimate_delta(
            spec, [*parent_obs, *effective_obs], parent_id=scene.parent_id, candidate_id=effective,
            required_unit_ids=days,
        )
        control_deltas[spec.analysis_id] = estimate_delta(
            spec, [*parent_obs, *control_obs], parent_id=scene.parent_id, candidate_id=control,
            required_unit_ids=days,
        )
    primary = next(spec for spec in specs if spec.acceptance["role"] == "primary")
    contrast = estimate_delta(
        primary, [*control_obs, *effective_obs], parent_id=control, candidate_id=effective, required_unit_ids=days,
    )
    positive_status = assess_candidate(specs, effective_deltas)
    negative_status = assess_candidate(specs, control_deltas)
    contrast_supported = contrast.interval is not None and contrast.interval[1] <= -primary.min_effect
    repeats_supported = all(value <= -primary.min_effect for value in contrast.train_repeat_estimates.values())
    supported = (
        diagnosis["status"] is ReportStatus.SUPPORTED and positive_status is ReportStatus.SUPPORTED
        and negative_status is ReportStatus.REFUTED and contrast_supported
    )
    return {
        "status": ReportStatus.SUPPORTED if supported else ReportStatus.INCONCLUSIVE,
        "diagnosis": diagnosis, "primary_analysis_id": primary.analysis_id,
        "candidate_vs_parent": {key: _delta_summary(delta) for key, delta in effective_deltas.items()},
        "control_vs_parent": {key: _delta_summary(delta) for key, delta in control_deltas.items()},
        "candidate_vs_control": _delta_summary(contrast),
        "effective_status": positive_status, "control_status": negative_status,
        "contrast_supported": contrast_supported, "all_repeat_contrasts_supported": repeats_supported,
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
        rows = measured[scene_id][_CASES[scene_id]["effective"]]
        repeats = tuple(manifest.train_seeds)
        days = manifest.splits["dev"].unit_ids
        free, clamped, truth = _paths(rows, scene.parent_id, repeats, days)
        parent = _model_rows(rows, scene.parent_id, repeats)
        means = {}
        for readout in ("background_bias", "nll", "end_response"):
            values = [row.values["value"] for row in parent if row.readout == readout]
            means[readout] = float(np.mean(values))
        free_error = np.sqrt(((free - truth).mean(axis=1) ** 2).mean(axis=1))
        price_error = np.sqrt(((clamped - truth).mean(axis=1) ** 2).mean(axis=1))
        flow_error = np.sqrt(((free - clamped).mean(axis=1) ** 2).mean(axis=1))
        means["free_rmse"] = float(free_error.mean())
        baseline[scene_id] = means
        evidence["flow_rmse"][scene_id] = float(flow_error.mean())
        evidence["price_rmse"][scene_id] = float(price_error.mean())
        evidence["dominance_margin"][scene_id] = float(case_reports[scene_id]["diagnosis"]["margin"])
        evidence["background_bias"][scene_id] = means["background_bias"]
    return {"baseline_metrics": baseline, "evidence": evidence}


def calibrate(
    output: Path, *, exp_id: str, steps: Sequence[int], dev_days: Sequence[int], train_seeds: Sequence[int],
    train_days: int, n_resamples: int, hidden: int, windows_per_day: int, info_drift: float,
    informed_fraction: float, namespace: str,
) -> dict[str, Any]:
    """Run a grid; write a formal manifest only when both causal contrasts qualify."""
    if len(train_seeds) < 2 or len(set(train_seeds)) != len(train_seeds):
        raise ValueError("calibration requires at least two distinct training seeds")
    if not steps or min(steps) < 1 or not dev_days or min(dev_days) < 2 or n_resamples < 1:
        raise ValueError("calibration requires positive steps and draws and at least two dev days")
    output.mkdir(parents=True, exist_ok=False)
    days_grid = sorted(set(dev_days))
    step_grid = sorted(set(steps))
    repeat_counts = sorted({2, len(train_seeds)})
    maximum_days = max(days_grid)
    manifests = {}
    for step_count in step_grid:
        manifests[step_count] = build_manifest(
            exp_id, steps=step_count, n_train_days=train_days, n_dev_days=maximum_days, train_seeds=train_seeds,
            n_resamples=n_resamples, namespace=namespace, hidden=hidden, windows_per_day=windows_per_day,
            info_drift=info_drift, informed_fraction=informed_fraction,
        )
    atomic_write_json(output / "calibration-plan.json", {
        "candidate_roles": _CASES, "dev_day_counts": days_grid, "training_repeat_counts": repeat_counts,
        "configurations": {str(step): manifest.to_dict() for step, manifest in manifests.items()},
        "profile_selection": "smallest passing fit budget at full dev-day and repeat counts",
        "criteria": [
            "Channel diagnosis agrees across training repeats and pooled day margin clears the registered margin.",
            "Pooled day intervals support the candidate and refute the control under the registered gates.",
            "Pooled direct candidate-minus-control interval clears the negative primary effect threshold.",
            "Per-repeat effect values and range are descriptive; the day interval is conditional on trained fits.",
        ],
    })
    summary: dict[str, Any] = {"status": ReportStatus.INCONCLUSIVE, "grid": [], "selected": None}
    for step_count, manifest in manifests.items():
        direct, measured = _measure(manifest, output / f"steps-{step_count}")
        for n_days in days_grid:
            days = manifest.splits["dev"].unit_ids[:n_days]
            for n_repeats in repeat_counts:
                repeats = tuple(manifest.train_seeds)[:n_repeats]
                reports = {}
                for scene in manifest.scenarios:
                    key = scene.scenario_id
                    reports[key] = _calibrate_case(
                        scene, manifest.analysis, direct[key], measured[key], repeats, days, n_resamples,
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
    parser.add_argument("--exp-id", default="p0-tier0-offline")
    parser.add_argument("--steps", type=int, nargs="+", default=(120, 400, 850))
    parser.add_argument("--dev-days", type=int, nargs="+", default=(8, 16))
    parser.add_argument("--train-seeds", type=int, nargs="+", default=(11, 41, 79))
    parser.add_argument("--train-days", type=int, default=16)
    parser.add_argument("--n-resamples", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--windows-per-day", type=int, default=6)
    parser.add_argument("--info-drift", type=float, default=0.85)
    parser.add_argument("--informed-fraction", type=float, default=0.40)
    parser.add_argument("--namespace", default="p0-calibration-2026-11-23")
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = calibrate(**vars(args))
    print(f"Calibration: {summary['status'].value}; {args.output / 'calibration-summary.json'}", flush=True)


if __name__ == "__main__":
    main()
