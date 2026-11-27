"""Run the frozen P1-M4 development roster, then verify a separate holdout batch."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from threading import Event

from stochopia.neural.primitives import seed_lob_source

from .entry import CandidateGate as Gate, DevelopmentLoop
from .formal_manifest import (
    ARM_MODELS, DEVELOPMENT_ROLLOUT_SEED, M5_SCHEMA, MANIFEST, ROUNDS, SYMBOLS, TRAIN_SEEDS,
    Trajectory, data_registration, freeze_study, roster, study_registration,
)
from .gate import CandidateGate
from .protocol import Messages, TwoStepProposer
from .prompt import M5_MIN_EFFECTS
from .store import Evaluator, Json, Ledger, MetricRule, ModelRef, StudyPlan


ProposerFactory = Callable[[Trajectory], TwoStepProposer]
SeedMetrics = Callable[[str, ModelRef], Mapping[str, object]]
Progress = Callable[[Trajectory, int, str], None]
TokenCounter = Callable[[Messages], int]
TokenCounterFactory = Callable[[str], TokenCounter]

OPENHYRA_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY = Path("config/p1-lob-models.yaml")
OPENHYRA_CODE = (
    "tasks/stochopia_p1", "sandbox.py", "worker_subprocess.py",
    "sandbox_artifacts.py", "island_scheduler.py", "scripts/verify_p2_m1_szse_300001.py",
)
STOCHOPIA_CODE = (
    "stochopia/__init__.py", "stochopia/common.py", "stochopia/data", "stochopia/llm.py",
    "stochopia/harness/lob_rollout.py", "stochopia/harness/lob_rollout_batch.py",
    "stochopia/harness/lob_batch_matching.py", "stochopia/measure/impact.py",
    "stochopia/measure/continuous_impact.py", "stochopia/neural",
    "experiments/p1_*", "experiments/p2_m1_*", "config/p1-lob-models.yaml", "pyproject.toml", "uv.lock",
)
SEED_MODEL_ID = "seed-gru"
V2_RESOURCE_RETRY_LOCK = {
    "stochopia_commit": "400443b5039ea1b073bbc9a1156cac079ad15e3f",
    "stochopia_tree": "047fbe62feeb895503c95316c290b6d6e0945d3e",
    "openhyra_commit": "66533dae928e0a64eda8232147c92fda4326e32e",
    "openhyra_tree": "36b043f47904b868aa9aaefd44d589e92b10ae8a",
}
V2_RESOURCE_RETRY_PATHS = {
    "stochopia": {"stochopia/neural/lob_training.py"},
    "openhyra": {
        "sandbox.py", "tasks/stochopia_p1/gate_worker.py", "tasks/stochopia_p1/market_worker.py",
        "tasks/stochopia_p1/market_eval.py", "tasks/stochopia_p1/formal.py",
        "tasks/stochopia_p1/formal_verify.py", "tasks/stochopia_p1/formal_archive.py",
        "tasks/stochopia_p1/verifier.py",
    },
}
M5_CLOSING_PATHS = {
    "stochopia": {"experiments/p1_m5_report.py", "experiments/p1_m5_closing_report.py"},
    "openhyra": {
        "tasks/stochopia_p1/formal.py",
        "tasks/stochopia_p1/formal_verify.py",
        "tasks/stochopia_p1/formal_archive.py",
        "tasks/stochopia_p1/closing_verify.py",
        "tasks/stochopia_p1/market_eval.py",
    },
}


def _metric_rules(analysis_version: str = "m4") -> tuple[MetricRule, ...]:
    """Register all eligible primary and supplemental readout names."""
    improve_up = {"lob_impact_score", "visible_over_blind_ratio"}
    metrics = (
        "lob_impact_score", "end_over_peak_ratio", "free_abs_drift_ticks",
        "trade_sign_acf_exp_gap", "hurst_gap", "visible_over_blind_ratio",
        "impact_exponent_gap", "event_count_rel_gap", "l1_depth_mae",
    )
    rules = []
    for name in metrics:
        minimum = M5_MIN_EFFECTS[name] if analysis_version == "m5" else 0.0
        if name == "lob_impact_score":
            rule = MetricRule(name, "increase", 0.50, (0.50, 100.0))
        elif name in improve_up:
            rule = MetricRule(name, "increase", minimum, (max(minimum, 1e-9), 100.0))
        else:
            rule = MetricRule(name, "decrease", minimum, (-100.0, -max(minimum, 1e-9)))
        rules.append(rule)
    return tuple(rules)


def _git_value(root: Path, argument: str) -> str:
    return subprocess.check_output(["git", "rev-parse", argument], cwd=root, text=True).strip()


def git_code_lock(stochopia_root: Path) -> Json:
    """Register both git trees and reject modifications to experiment source."""
    result = {}
    for repository, root, paths in (
        ("stochopia", stochopia_root, STOCHOPIA_CODE),
        ("openhyra", OPENHYRA_ROOT, OPENHYRA_CODE),
    ):
        state = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
            cwd=root, check=True, capture_output=True, text=True,
        )
        if state.stdout:
            raise RuntimeError(f"commit the {repository} experiment code before running M4")
        result[f"{repository}_commit"] = _git_value(root, "HEAD")
        result[f"{repository}_tree"] = _git_value(root, "HEAD^{tree}")
    return result


def _require_v2_resource_retry_code(manifest: Json, code: Json, stochopia_root: Path) -> None:
    """Check the one approved v2 migration and exclude changes to score or data logic."""
    if manifest["schema"] != "stochopia-p1-m4-v2" or manifest["code"] != V2_RESOURCE_RETRY_LOCK:
        raise ValueError("the retry is specific to the registered P1-M4 v2 code")
    for repository, root, paths in (
        ("stochopia", stochopia_root, STOCHOPIA_CODE),
        ("openhyra", OPENHYRA_ROOT, OPENHYRA_CODE),
    ):
        original = manifest["code"][f"{repository}_commit"]
        current = code[f"{repository}_commit"]
        if _git_value(root, f"{original}^{{tree}}") != manifest["code"][f"{repository}_tree"]:
            raise ValueError("the registered v2 source commit no longer matches its tree")
        if subprocess.run(["git", "merge-base", "--is-ancestor", original, current], cwd=root).returncode != 0:
            raise ValueError("the resource retry code must descend from the frozen v2 commits")
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", original, current, "--", *paths], cwd=root, text=True,
        ).splitlines()
        if set(changed) - V2_RESOURCE_RETRY_PATHS[repository]:
            raise ValueError("the resource retry contains changes outside the registered repair files")


def _require_m5_closing_code(manifest: Json, code: Json, stochopia_root: Path) -> None:
    """Permit only the closing verifier and report additions after the original M5 seal."""
    if manifest["schema"] != M5_SCHEMA:
        raise ValueError("closing verification requires the frozen M5 study")
    for repository, root, paths in (
        ("stochopia", stochopia_root, STOCHOPIA_CODE),
        ("openhyra", OPENHYRA_ROOT, OPENHYRA_CODE),
    ):
        original = manifest["code"][f"{repository}_commit"]
        current = code[f"{repository}_commit"]
        if _git_value(root, f"{original}^{{tree}}") != manifest["code"][f"{repository}_tree"]:
            raise ValueError("the original development commit no longer matches its tree")
        result = subprocess.run(["git", "merge-base", "--is-ancestor", original, current], cwd=root)
        if result.returncode != 0:
            raise ValueError("closing code must descend from the frozen development commit")
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", original, current, "--", *paths], cwd=root, text=True,
        ).splitlines()
        if set(changed) - M5_CLOSING_PATHS[repository]:
            raise ValueError("closing code includes changes outside the verification and report files")


def _development_plan(registration: Json, baseline: Json, item: Trajectory) -> StudyPlan:
    model = ModelRef(**baseline["model"])
    analysis_version = registration.get("analysis_version", "m4")
    return StudyPlan(
        seed_model=model, seed_evidence=baseline["metrics"],
        train_dates=tuple(registration["data"]["train_dates"]), dev_dates=tuple(registration["data"]["dev_dates"]),
        metric_rules=_metric_rules(analysis_version), rounds=ROUNDS, train_seeds=TRAIN_SEEDS,
        rollout_seed=DEVELOPMENT_ROLLOUT_SEED, mutation_seed=item.search_seed, nll_guardrail=0.05,
        groups=(item.arm,), trial_prefix=f"{item.stock}-{item.search_seed}",
        analysis_version=analysis_version,
    )


def _advance_trajectory(
    root: Path, registration: Json, item: Trajectory, baseline: Json,
    make_proposer: ProposerFactory, gate: Gate, evaluator: Evaluator, stop: Event,
    progress: Progress | None,
) -> None:
    plan = _development_plan(registration, baseline, item)
    trial_root = item.path(root)
    ledger = Ledger(trial_root)
    proposers = {} if ARM_MODELS[item.arm] is None else {item.arm: make_proposer(item)}
    loop = DevelopmentLoop(
        trial_root, plan, proposers=proposers, gate=gate, evaluate=evaluator, stop_requested=stop.is_set,
    )
    if all(ledger.read(*item.round_path(index), "result.json") is not None for index in range(1, ROUNDS + 1)):
        return
    remaining = sum(
        ledger.read(*item.round_path(index), "result.json") is None for index in range(1, ROUNDS + 1)
    )
    for _ in range(remaining):
        if stop.is_set():
            return
        result = loop.run_round()[item.arm]
        if progress is not None:
            progress(item, result["round"], result["verdict"])


def _run_concurrent(advance: Callable[[Trajectory], None], jobs: int, stop: Event) -> None:
    remaining = roster()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = {pool.submit(advance, item) for item in islice(remaining, jobs)}
        try:
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for task in completed:
                    task.result()
                if not stop.is_set():
                    pending.update(pool.submit(advance, item) for item in islice(remaining, len(completed)))
        except BaseException:
            stop.set()
            raise


def run_development(
    root: Path, market_root: Path, *, code_lock: Mapping[str, str], registry_text: str,
    make_proposer: ProposerFactory, seed_metrics: SeedMetrics,
    evaluators: Mapping[str, Evaluator], gate: Gate, jobs: int = 1, development_root: Path | None = None,
    progress: Progress | None = None,
) -> Json:
    """Run only the fixed train/dev roster. Transport and data IO are injected."""
    if jobs < 1 or set(evaluators) != set(SYMBOLS):
        raise ValueError("a positive job count and both stock evaluators are required")
    data = data_registration(market_root)
    registration = study_registration(data, code_lock, registry_text)
    ledger = Ledger(root)
    work_root = root if development_root is None else development_root

    with ledger.lock():
        if ledger.read(MANIFEST) is not None:
            raise ValueError("development is closed by the frozen study manifest")
        ledger.seal(registration, "development_plan.json")
        baselines = {}
        source = seed_lob_source()
        model = ModelRef(SEED_MODEL_ID, source)
        for symbol in SYMBOLS:
            name = f"{symbol}.json"
            prior = ledger.read("baselines", name)
            if prior is None:
                prior = ledger.seal(
                    {"model": asdict(model), "metrics": dict(seed_metrics(symbol, model))},
                    "baselines", name,
                )
            baselines[symbol] = prior

        stop = Event()

        def advance(item: Trajectory) -> None:
            try:
                _advance_trajectory(
                    work_root, registration, item, baselines[item.stock], make_proposer,
                    gate, evaluators[item.stock], stop, progress,
                )
            except BaseException:
                stop.set()
                raise

        if jobs == 1:
            for item in roster():
                advance(item)
        else:
            _run_concurrent(advance, jobs, stop)

    return {"trajectories": len(tuple(roster())), "trials": ROUNDS * len(tuple(roster()))}


# --- Production adapter setup and commands ---

def _factory(spec: str) -> TokenCounterFactory:
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("pass the tokenizer as module:factory")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise ValueError("the tokenizer factory must be callable")
    return factory


def _require_smoke_gates(smoke_root: Path) -> Json:
    """Verify original and specifically authorized supplemental smoke records."""
    api_smoke = json.loads((smoke_root / "api_report.json").read_text(encoding="utf-8"))
    market_smoke = json.loads((smoke_root / "no_llm_report.json").read_text(encoding="utf-8"))
    arms = {arm for arm, alias in ARM_MODELS.items() if alias is not None}
    if set(api_smoke["arms"]) != arms or api_smoke["tokens"] + api_smoke["unknown_token_reservation"] >= 15_000:
        raise ValueError("the original API smoke must record four arms within its budget")
    if not market_smoke["passed"] or market_smoke["paired_windows"] != 24:
        raise ValueError("the registered market smoke must pass all 24 windows")
    failed = {arm for arm, row in api_smoke["arms"].items() if not row["passed"]}
    report = {"original_api": api_smoke, "market": market_smoke}
    if failed:
        if failed != {"metrics_ds", "evidence_gpt"}:
            raise ValueError("the supplemental approval covers exactly evidence_gpt and metrics_ds")
        supplement = json.loads((smoke_root / "supplement" / "api_report.json").read_text(encoding="utf-8"))
        if (set(supplement["arms"]) != failed or not all(row["passed"] for row in supplement["arms"].values())
                or supplement["tokens"] + supplement["unknown_token_reservation"] > 10_000):
            raise ValueError("both supplemental API smoke arms must pass within the additional budget")
        report["supplement_api"] = supplement
    return report


def _provider_handshakes(ledger: Ledger, registry: Path) -> None:
    """Record one metered connectivity probe per provider before development.

    Runs under the development archive's lock, after the exact study plan is
    sealed. Successful probes are replayed from their durable records on resume.
    Transport failures pause this invocation without consuming a trial.
    """
    from .provider import ProviderAttemptError, ProviderOutageError, provider_handshake

    for alias in ("gpt-6.2-sol", "deepseek-flash"):
        prefix = ("handshakes", alias)
        if ledger.read(*prefix, "passed.json") is not None:
            continue
        work = ledger.file(*prefix)
        reservations = sorted(work.glob("attempt-*.json")) if work.exists() else []
        reservations = [path for path in reservations if not path.stem.endswith("-result")]
        for reservation in reservations:
            result = (*prefix, f"{reservation.stem}-result.json")
            prior = ledger.read(*result)
            if prior is None:
                ledger.seal({
                    "status": "interrupted", "input_tokens": None, "output_tokens": None,
                    "retryable": True,
                }, *result)
            elif prior["status"] == "passed":
                ledger.seal(prior, *prefix, "passed.json")
                break
        if ledger.read(*prefix, "passed.json") is not None:
            continue
        attempt = len(reservations) + 1
        stem = f"attempt-{attempt:03}"
        ledger.seal({"model": alias, "attempt": attempt}, *prefix, f"{stem}.json")
        try:
            reply = provider_handshake(registry, alias)
        except ProviderAttemptError as exc:
            ledger.seal({
                "status": "error", "error": str(exc), "retryable": exc.retryable,
                "input_tokens": exc.input_tokens, "output_tokens": exc.output_tokens,
            }, *prefix, f"{stem}-result.json")
            if exc.retryable:
                raise ProviderOutageError(f"{alias} connectivity check failed") from exc
            raise
        result = {
            "status": "passed", "model": alias, "attempt": attempt,
            "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
        }
        ledger.seal(result, *prefix, f"{stem}-result.json")
        ledger.seal(result, *prefix, "passed.json")


def _production_development(args: argparse.Namespace, code: Json) -> Json:
    from .market_eval import MarketEvaluator
    from .provider import registry_transport

    registry = args.models or args.stochopia_root / MODEL_REGISTRY
    canonical = (args.stochopia_root / MODEL_REGISTRY).read_text(encoding="utf-8")
    if registry.read_text(encoding="utf-8") != canonical:
        raise ValueError("the study must use the locked Stochopia model registry")
    registration = study_registration(data_registration(args.market_root), code, canonical)
    is_m5 = registration["schema"] == M5_SCHEMA
    ledger = Ledger(args.artifact_root)
    with ledger.lock():
        if ledger.read(MANIFEST) is not None:
            raise ValueError("development is already frozen")
        ledger.seal(registration, "development_plan.json")
        if is_m5:
            _provider_handshakes(ledger, registry)
        else:
            ledger.seal(_require_smoke_gates(Path("/tmp/p1-m4-v2-smoke")), "smoke_gate.json")
    aliases = tuple(model for model in dict.fromkeys(ARM_MODELS.values()) if model is not None)
    if args.tokenizer is not None:
        factory = _factory(args.tokenizer)
        counters = {model: factory(model) for model in aliases}
    else:
        from .token_count import load_token_counters

        counters = load_token_counters(args.tokenizers_dir)
    gate = CandidateGate()
    evaluators = {
        stock: MarketEvaluator(
            args.market_root, stock, cache_root=args.cache_root / "markets",
            expected_dev_days=24 if is_m5 else 12, expected_test_days=20 if is_m5 else 12,
        )
        for stock in SYMBOLS
    }

    def seed_metrics(stock: str, model: ModelRef) -> Mapping[str, object]:
        return evaluators[stock].seed_profile(model, rollout_seed=DEVELOPMENT_ROLLOUT_SEED)["evidence"]

    def proposer(item: Trajectory) -> TwoStepProposer:
        alias = ARM_MODELS[item.arm]
        if alias is None:
            raise ValueError("no_llm has no provider transport")
        return TwoStepProposer(
            registry_transport(registry, alias), gate, lambda _usage: None,
            count_tokens=counters[alias], analysis_version="m5" if is_m5 else "m4",
        )

    def progress(item: Trajectory, index: int, verdict: str) -> None:
        print(f"{item.stock} {item.arm} seed={item.search_seed} round={index} {verdict}", flush=True)

    return run_development(
        args.artifact_root, args.market_root, code_lock=code, registry_text=canonical,
        make_proposer=proposer, seed_metrics=seed_metrics, evaluators=evaluators, gate=gate, jobs=args.jobs,
        development_root=args.cache_root / "ledger",
        progress=progress,
    )


def _production_verify(args: argparse.Namespace, code: Json) -> Json:
    from .formal_verify import load_frozen_study, verify_frozen_study
    from .market_eval import MarketEvaluator

    manifest = load_frozen_study(args.artifact_root)
    if args.retry_failed:
        _require_v2_resource_retry_code(manifest, code, args.stochopia_root)
        from stochopia.neural.lob_training import P1_FIT_PROTOCOL

        if P1_FIT_PROTOCOL.max_train_seconds != 180.0:
            raise ValueError("resource retry requires the registered 180 second training budget")
    elif args.closing:
        _require_m5_closing_code(manifest, code, args.stochopia_root)
        registration = Ledger(args.artifact_root).read("verification", "closing-registration.json")
        if registration is None or registration["closing_code"] != code:
            raise ValueError("register the exact closing code before acquiring holdout data")
    elif manifest["code"] != code:
        raise ValueError("the current code differs from the frozen experiment")
    if manifest["data"] != data_registration(args.market_root):
        raise ValueError("market metadata differs from the frozen experiment")
    receipt = Ledger(args.artifact_root).read("verification", "test-data-reconciliation.json")
    if receipt is None:
        raise ValueError("finish the registered holdout reconstruction and reconciliation first")
    is_m5 = manifest["schema"] == M5_SCHEMA
    if is_m5:
        from .holdout_data import validate_m5_receipt

        dates = validate_m5_receipt(manifest, receipt)
    else:
        dates = tuple(manifest["test_dates"])
        if (receipt["test_dates"] != list(dates)
                or receipt["source_revision"] != manifest["data"]["source_revision"]
                or len(receipt["reconciliation"]) != 24 or len(receipt["anchors"]) != 48):
            raise ValueError("finish the registered holdout reconstruction and reconciliation first")
        expected_days = {(stock, day) for stock in SYMBOLS for day in dates}
        rows = {(row["symbol"], row["date"]) for row in receipt["reconciliation"]}
        anchors = {(row["symbol"], row["date"], row["session"]) for row in receipt["anchors"]
                   if row["spread_ticks"] > 0}
        expected_anchors = {(stock, day, session) for stock, day in expected_days for session in (0, 1)}
        if rows != expected_days or anchors != expected_anchors:
            raise ValueError("reconciliation must cover both positive-spread anchors on each registered stock-day")
    evaluators = {
        stock: MarketEvaluator(
            args.market_root, stock, cache_root=args.cache_root / "markets",
            authorized_test_dates=dates,
            expected_dev_days=24 if is_m5 else 12, expected_test_days=20 if is_m5 else 12,
        )
        for stock in SYMBOLS
    }
    return verify_frozen_study(
        args.artifact_root, evaluators=evaluators, retry_failed=args.retry_failed,
        retry_code=code if args.retry_failed else None, jobs=args.jobs, closing=args.closing,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("dev", "freeze", "register-closing", "verify"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--stochopia-root", type=Path, default=OPENHYRA_ROOT.parent / "Stochopia")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--models", type=Path)
    parsers = parser.add_mutually_exclusive_group()
    parsers.add_argument("--tokenizer", help="trusted module:factory supplying each provider token counter")
    parsers.add_argument("--tokenizers-dir", type=Path, help="directory for locally installed provider tokenizers")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true", help="repair only the registered v2 failed tasks")
    parser.add_argument("--closing", action="store_true", help="evaluate the registered M5 closing roster")
    args = parser.parse_args()
    if args.cache_root is None:
        args.cache_root = args.market_root / "m4-model-cache" / args.artifact_root.name
    if args.cache_root.resolve().is_relative_to(args.artifact_root.resolve()):
        parser.error("working files and model caches must stay outside the evidence archive")
    if args.phase == "dev" and args.tokenizer is None and args.tokenizers_dir is None:
        parser.error("dev requires --tokenizers-dir or --tokenizer")
    if args.retry_failed and args.phase != "verify":
        parser.error("--retry-failed is available only in the verifier phase")
    if args.closing and (args.phase != "verify" or args.retry_failed):
        parser.error("--closing is only available for verification without --retry-failed")
    return args


def main() -> None:
    args = _arguments()
    code = git_code_lock(args.stochopia_root)
    if args.phase == "dev":
        result = _production_development(args, code)
    elif args.phase == "freeze":
        from .formal_archive import write_archive_tables

        plan = Ledger(args.artifact_root).read("development_plan.json")
        if plan is None or plan["code"] != code:
            raise ValueError("the current code differs from the development registration")
        manifest = freeze_study(args.artifact_root, args.market_root, development_root=args.cache_root / "ledger")
        write_archive_tables(args.artifact_root)
        result = {"manifest": str(args.artifact_root / MANIFEST), "tasks": len(manifest["tasks"])}
    elif args.phase == "register-closing":
        from .formal_verify import load_frozen_study, register_closing_verification

        manifest = load_frozen_study(args.artifact_root)
        _require_m5_closing_code(manifest, code, args.stochopia_root)
        if manifest["data"] != data_registration(args.market_root):
            raise ValueError("market metadata differs from the frozen experiment")
        registration = register_closing_verification(args.artifact_root, code=code)
        result = {"profiles": len(registration["profiles"]), "comparisons": len(registration["tasks"])}
    else:
        from .formal_archive import write_archive_tables

        report = _production_verify(args, code)
        write_archive_tables(args.artifact_root)
        result = {"report": str(args.artifact_root / "verification" / "report.json"), "tasks": report["measured_tasks"]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
