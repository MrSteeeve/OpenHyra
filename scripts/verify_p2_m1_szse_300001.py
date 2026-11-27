"""Re-evaluate four frozen P1 models with the P2-M1 measurement protocol.

No model proposal, provider calls, or new market acquisition. Full derived
profiles and copied checkpoints stay in an external work directory; public
evidence contains only aggregate measurements and provenance.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from stochopia.common import atomic_write_json
from tasks.stochopia_p1.formal_verify import load_frozen_study
from tasks.stochopia_p1.gate import CandidateGate
from tasks.stochopia_p1.holdout_data import validate_m5_receipt
from tasks.stochopia_p1.market_eval import MarketEvaluator, _run_worker, _stage_candidate
from tasks.stochopia_p1.store import EvaluationRequest, ModelRef, PairedEvaluation

Json = dict[str, Any]
SYMBOL = "300001.SZ"
SCHEMA = "p2-m1"
TRAIN_SEEDS = (103, 137, 211)
MODEL_SLOTS = {
    "M0": ("seed-gru", "fad85b95ceff23f1cab4957248a0d41ec700b866c1fd195bf0a280708f657771"),
    "M1": ("300001.SZ-29-evidence_ds-r01", "390a0534d967842aeff3422a643afc17a044ed5da1a79a4b5ecc57a6c308e7d8"),
    "M2": ("300001.SZ-29-evidence_ds-r02", "37f4cd961b83240e6dff4815d22ab50625754c23c35c8833417fcf5d399a7c28"),
    "M3": ("300001.SZ-41-evidence_ds-r03", "204fb2d926da7c83a194a33eca421e52b2588f2c45417ad975a9fda65177b958"),
}
SCHEDULE = ((1, ("M0", "M3")), (1, ("M1", "M2")), (4, ("M0", "M3")), (8, ("M0", "M3")))
CONTINUOUS = ("impact_curve_mae", "post_stop_relaxation_delta", "visible_minus_blind_auc")
METRICS = (*CONTINUOUS, "bounded_relaxation_ratio", "c2_pass_prob", "lob_impact_score")
FROZEN_DIRS = ("artifacts/p1-formal-20261124", "artifacts/p1-m5-formal-20261124")


def _json(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _frozen_trees(root: Path) -> Json:
    seal = _git(root, "rev-parse", "p1-szse-final^{}")
    if seal != "93e43f28354126e150adfb96e7da99f21f05025f":
        raise ValueError("unexpected P1 evidence tag")
    if _git(root, "diff", "--name-only", seal, "--", *FROZEN_DIRS):
        raise ValueError("P1 frozen evidence must be unchanged")
    return {name: _git(root, "rev-parse", f"{seal}:{name}") for name in FROZEN_DIRS}


def _source_models(manifest: Json, archive: Path) -> dict[str, ModelRef]:
    trials = {}
    for trajectory in manifest["trajectories"].values():
        for trial in trajectory["trials"]:
            trials[trial["trial_id"]] = trial
    models = {}
    for label, (trial_id, digest) in MODEL_SLOTS.items():
        model = manifest["baselines"][SYMBOL]["model"] if label == "M0" else trials[trial_id]["candidate"]
        if model["source_sha256"] != digest:
            raise ValueError(f"{label} differs from the preselected frozen source")
        source = archive / "sources" / f"{digest}.py"
        if _digest(source) != digest:
            raise ValueError(f"{label} source hash mismatch")
        models[label] = ModelRef(model["model_id"], source.read_text(encoding="utf-8"))
    return models


def _weights(cache_root: Path, digest: str) -> tuple[Path, Json]:
    """Check original read-only freeze-associated checkpoints, never fit on test."""
    folder = cache_root / SYMBOL / "train" / f"{digest}-103-137-211"
    fit = _json(folder / "fit.json")
    if tuple(row["seed"] for row in fit["fits"]) != TRAIN_SEEDS:
        raise ValueError("cached weights differ from the registered training seed panel")
    return folder, {str(seed): _digest(folder / f"{seed}.pt") for seed in TRAIN_SEEDS}


def _registration(
    openhyra: Path, stochopia: Path, frozen: Path, cache: Path, market: Path,
    batch_windows: int, jobs: int, timeout_s: float,
) -> tuple[Json, dict[str, ModelRef]]:
    manifest = load_frozen_study(frozen)
    receipt = _json(frozen / "verification/test-data-reconciliation.json")
    dates = validate_m5_receipt(manifest, receipt)
    if len(dates) != 20 or tuple(manifest["train_seeds"]) != TRAIN_SEEDS:
        raise ValueError("expected complete P1-M5 seed and holdout calendars")
    market_plan = _json(market / "manifest.json")
    if market_plan["train_dates"] != manifest["data"]["train_dates"] or market_plan["source_revision"] != manifest["data"]["source_revision"]:
        raise ValueError("market registration differs from the frozen training calendar/source revision")
    cache_hashes = {}
    for day in (*manifest["data"]["train_dates"], *dates):
        cache_hashes[str(day)] = _digest(market / "cache" / str(day) / f"{SYMBOL}.npz")
    volume_hashes = {}
    for day in manifest["data"]["train_dates"]:
        volume_hashes[str(day)] = _digest(market / "raw" / str(day) / "行情.parquet")
    models = _source_models(manifest, frozen)
    refs = {}
    for label, (trial_id, digest) in MODEL_SLOTS.items():
        _, weights = _weights(cache, digest)
        refs[label] = {"trial_id": trial_id, "source_sha256": digest, "checkpoint_sha256": weights}
    if _git(stochopia, "rev-parse", "p1-szse-final^{}") != "4b840e34d84308283d920aee7a5a6a071dde9c34":
        raise ValueError("unexpected Stochopia P1 tag")
    code = {"openhyra": _git(openhyra, "rev-parse", "HEAD"), "stochopia": _git(stochopia, "rev-parse", "HEAD")}
    return {
        "schema": SCHEMA, "symbol": SYMBOL, "models": refs, "device": "cpu",
        "rollout_protocol": "rollout_protocol_v2", "max_state_dim": 512,
        "continuous_quote_aggregation": "available finite marks with support; C2 retains strict path support",
        "train_dates": manifest["data"]["train_dates"], "test_dates": list(dates),
        "train_seeds": list(TRAIN_SEEDS), "rollout_seed": manifest["test_rollout_seed"],
        "source_revision": manifest["data"]["source_revision"], "market_cache_sha256": cache_hashes,
        "train_volume_source_sha256": volume_hashes,
        "frozen_trees": _frozen_trees(openhyra),
        "frozen_manifest_sha256": _digest(frozen / "study_manifest.json"),
        "code": code, "num_replicates": [1, 4, 8], "batch_windows": batch_windows,
        "jobs": jobs, "timeout_s": timeout_s, "bootstrap_draws": 2000,
        "variance_pair": ["M0", "M3"], "model_comparison_k": 1, "api_calls": 0,
        "calendar_role": "previously opened P1 holdout; P2 retrospective measurement validation only",
    }, models


def _seal(path: Path, registration: Json) -> None:
    if path.exists():
        if _json(path) != registration:
            raise ValueError("existing M1 registration differs; use a new private work root")
    else:
        atomic_write_json(path, registration)


def _stage_inputs(
    work: Path, evaluator: MarketEvaluator, registration: Json, models: Mapping[str, ModelRef], cache: Path,
) -> None:
    """Copy existing data/checkpoints into an experiment-local input tree once."""
    common = work / "common"
    marker = common / "ready.json"
    if not marker.is_file():
        common.mkdir(parents=True, exist_ok=True)
        evaluator._stage_caches(common, tuple(registration["train_dates"] + registration["test_dates"]))
        for day, digest in registration["market_cache_sha256"].items():
            if _digest(common / "cache" / day / f"{SYMBOL}.npz") != digest:
                raise ValueError("staged market cache digest mismatch")
        evaluator._stage_volume(common)
        atomic_write_json(common / "plan.json", {
            "symbol": SYMBOL, "train_dates": registration["train_dates"],
            "eval_dates": registration["test_dates"], "rollout_seed": registration["rollout_seed"],
        })
        atomic_write_json(marker, {"registration": registration})
    elif _json(marker)["registration"] != registration:
        raise ValueError("prepared inputs belong to a different study")
    for label, model in models.items():
        model_root = work / "models" / label
        source = model_root / "source"
        input_dir = model_root / "input"
        _stage_candidate(source, model)
        if not (input_dir / "input-ready.json").is_file():
            shutil.copytree(common, input_dir, dirs_exist_ok=True)
            atomic_write_json(input_dir / "input-ready.json", {"registered": True})
        for day, digest in registration["market_cache_sha256"].items():
            if _digest(input_dir / "cache" / day / f"{SYMBOL}.npz") != digest:
                raise ValueError("staged per-model cache digest mismatch")
        folder, hashes = _weights(cache, registration["models"][label]["source_sha256"])
        if hashes != registration["models"][label]["checkpoint_sha256"]:
            raise ValueError("original checkpoint changed during staging")
        weights = input_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        for seed in TRAIN_SEEDS:
            destination = weights / f"{seed}.pt"
            if not destination.exists():
                shutil.copyfile(folder / destination.name, destination)
            if _digest(destination) != hashes[str(seed)]:
                raise ValueError("copied checkpoint mismatch")
        diagnostic = CandidateGate(max_state_dim=512)(model.source)
        if diagnostic is not None:
            raise RuntimeError(f"{label} gate failure: {diagnostic}")


def _validate_profile(profile: Json, registration: Json, k: int) -> None:
    expected = {str(day) for day in registration["test_dates"]}
    if set(profile["day_metrics"]) != expected:
        raise ValueError("profile did not cover the frozen market calendar")
    market = profile["market_bootstrap"]
    if market["metric_schema_version"] != SCHEMA or market["num_replicates"] != k:
        raise ValueError("different metric/replicate protocol in profile")
    if len(market["visible_mid"]) != len(expected):
        raise ValueError("Monte Carlo replicas must not be additional day clusters")


def _profile(work: Path, label: str, k: int, registration: Json) -> Json:
    run = work / "profiles" / label / f"k{k}"
    ready = run / "complete.json"
    if ready.exists():
        record = _json(ready)
        if record["profile_sha256"] != _digest(run / "profile.json"):
            raise ValueError("completed private profile digest does not match")
        _validate_profile(_json(run / "profile.json"), registration, k)
        return record
    run.mkdir(parents=True, exist_ok=True)
    model_root = work / "models" / label
    started = perf_counter()
    print(f"START {label} K={k}", flush=True)
    attempt_number = len(list(run.glob("worker-*"))) + 1
    output = _run_worker(
        run / f"worker-{attempt_number:03d}", model_root / "source", model_root / "input", "evaluate",
        registration["timeout_s"], device="cpu", num_replicates=k,
        metric_schema_version=SCHEMA, rollout_batch_windows=registration["batch_windows"],
        max_state_dim=registration["max_state_dim"],
    )
    wall = perf_counter() - started
    profile = _json(output / "result.json")
    _validate_profile(profile, registration, k)
    atomic_write_json(run / "profile.json", profile)
    record = {"label": label, "k": k, "wall_seconds": wall,
              "evaluation_seconds": profile.get("evaluation_seconds"),
              "profile_sha256": _digest(run / "profile.json")}
    atomic_write_json(ready, record)
    print(f"DONE {label} K={k} wall={wall:.2f}s", flush=True)
    return record


def _pair(
    parent: Json, candidate: Json, models: Mapping[str, ModelRef], label: str,
    k: int, registration: Json,
) -> Json:
    from tasks.stochopia_p1.paired_stats import paired_readouts

    request = EvaluationRequest(
        f"p2-m1-{label}-k{k}", "shared", models["M0"], models[label],
        tuple(registration["train_dates"]), tuple(registration["test_dates"]),
        TRAIN_SEEDS, registration["rollout_seed"], "test", "m5",
    )
    pair = PairedEvaluation(
        {int(day): row for day, row in parent["day_metrics"].items()},
        {int(day): row for day, row in candidate["day_metrics"].items()},
        parent["evidence"], candidate["evidence"],
        parent["market_bootstrap"], candidate["market_bootstrap"],
    )
    pair.validate(request)
    return paired_readouts(pair, request, METRICS)


def _summarize(work: Path, registration: Json, models: Mapping[str, ModelRef]) -> Json:
    """Aggregate only fixed-model evidence and paired estimates, never raw L2."""
    measurements = {}
    for k, labels in SCHEDULE:
        for label in labels:
            run = work / "profiles" / label / f"k{k}"
            profile = _json(run / "profile.json")
            _validate_profile(profile, registration, k)
            measurements[f"{label}-k{k}"] = {
                "evidence": profile["evidence"], "timing": _json(run / "complete.json"),
                "n_days": len(profile["day_metrics"]),
            }
    pairs = {}
    for k in (1, 4, 8):
        baseline = _json(work / "profiles" / "M0" / f"k{k}" / "profile.json")
        for label in MODEL_SLOTS if k == 1 else ("M3",):
            candidate = _json(work / "profiles" / label / f"k{k}" / "profile.json")
            pairs[f"{label}-k{k}"] = _pair(baseline, candidate, models, label, k, registration)
    return {"registration": registration, "measurements": measurements, "pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--p1-cache-root", type=Path, required=True,
                        help="Original M5 markets/ (read-only)")
    parser.add_argument("--work-root", type=Path, required=True, help="External private scratch/cache")
    parser.add_argument("--artifact-root", type=Path, required=True, help="New M1 aggregate evidence")
    parser.add_argument("--stochopia-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2, choices=(1, 2))
    parser.add_argument("--batch-windows", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=7200)
    args = parser.parse_args()
    if args.batch_windows < 1 or args.timeout_s <= 0:
        parser.error("batch windows and timeout must be positive")
    openhyra = Path(__file__).resolve().parents[1]
    frozen = openhyra / FROZEN_DIRS[1]
    work = args.work_root.resolve()
    artifact = args.artifact_root.resolve()
    if any(work.is_relative_to(path) for path in (openhyra, args.stochopia_root.resolve(), args.market_root.resolve())):
        raise ValueError("private caches must reside outside experiment and market repositories")
    if artifact.is_relative_to(frozen.parent / "p1-formal-20261124") or artifact.is_relative_to(frozen):
        raise ValueError("cannot write to frozen P1 evidence")
    registration, models = _registration(openhyra, args.stochopia_root, frozen,
                                         args.p1_cache_root, args.market_root, args.batch_windows,
                                         args.jobs, args.timeout_s)
    work.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    _seal(work / "registration.json", registration)
    _seal(artifact / "registration.json", registration)
    evaluator = MarketEvaluator(
        args.market_root, SYMBOL, expected_dev_days=24, expected_test_days=20,
        authorized_test_dates=tuple(registration["test_dates"]), device="cpu",
    )
    _stage_inputs(work, evaluator, registration, models, args.p1_cache_root)
    timeline = work / "timing"
    timeline.mkdir(exist_ok=True)
    for k, labels in SCHEDULE:
        timing_path = timeline / f"{labels[0]}-{labels[1]}-k{k}.json"
        both_fresh = all(not (work / "profiles" / label / f"k{k}" / "complete.json").exists() for label in labels)
        started = perf_counter()
        started_utc = datetime.now(timezone.utc).isoformat()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_profile, work, label, k, registration) for label in labels]
            for future in futures:
                future.result()
        if both_fresh:
            atomic_write_json(timing_path, {"started_utc": started_utc, "wall_seconds": perf_counter() - started})
    result = _summarize(work, registration, models)
    result["schedule_timing"] = {path.stem: _json(path) for path in sorted(timeline.glob("*.json"))}
    if _frozen_trees(openhyra) != registration["frozen_trees"]:
        raise ValueError("P1 frozen evidence changed")
    atomic_write_json(artifact / "report.json", result)
    print("P2-M1 complete: 8 profiles, 20 original day clusters, zero API calls", flush=True)


if __name__ == "__main__":
    main()
