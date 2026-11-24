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
    ARM_MODELS, DEVELOPMENT_ROLLOUT_SEED, MANIFEST, ROUNDS, SYMBOLS, TRAIN_SEEDS,
    Trajectory, data_registration, freeze_study, roster, study_registration,
)
from .gate import CandidateGate
from .protocol import Messages, TwoStepProposer
from .store import Evaluator, Json, Ledger, MetricRule, ModelRef, StudyPlan


ProposerFactory = Callable[[Trajectory], TwoStepProposer]
SeedMetrics = Callable[[str, ModelRef], Mapping[str, object]]
Progress = Callable[[Trajectory, int, str], None]
TokenCounter = Callable[[Messages], int]
TokenCounterFactory = Callable[[str], TokenCounter]

OPENHYRA_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY = Path("config/p1-lob-models.yaml")
OPENHYRA_CODE = ("tasks/stochopia_p1", "sandbox.py", "island_scheduler.py")
STOCHOPIA_CODE = (
    "stochopia/__init__.py", "stochopia/common.py", "stochopia/data", "stochopia/llm.py",
    "stochopia/harness/lob_rollout.py", "stochopia/measure/impact.py", "stochopia/neural",
    "experiments/p1_*", "config/p1-lob-models.yaml", "pyproject.toml", "uv.lock",
)
SEED_MODEL_ID = "seed-gru"


def _metric_rules() -> tuple[MetricRule, ...]:
    """Register all eligible primary and supplemental readout names."""
    improve_up = {"lob_impact_score", "visible_over_blind_ratio"}
    metrics = (
        "lob_impact_score", "end_over_peak_ratio", "free_abs_drift_ticks",
        "trade_sign_acf_exp_gap", "hurst_gap", "visible_over_blind_ratio",
        "impact_exponent_gap", "event_count_rel_gap", "l1_depth_mae",
    )
    rules = []
    for name in metrics:
        if name == "lob_impact_score":
            rule = MetricRule(name, "increase", 0.50, (0.50, 100.0))
        elif name in improve_up:
            rule = MetricRule(name, "increase", 0.0, (1e-9, 100.0))
        else:
            rule = MetricRule(name, "decrease", 0.0, (-100.0, -1e-9))
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


def _development_plan(registration: Json, baseline: Json, item: Trajectory) -> StudyPlan:
    model = ModelRef(**baseline["model"])
    return StudyPlan(
        seed_model=model, seed_evidence=baseline["metrics"],
        train_dates=tuple(registration["data"]["train_dates"]), dev_dates=tuple(registration["data"]["dev_dates"]),
        metric_rules=_metric_rules(), rounds=ROUNDS, train_seeds=TRAIN_SEEDS,
        rollout_seed=DEVELOPMENT_ROLLOUT_SEED, mutation_seed=item.search_seed, nll_guardrail=0.05,
        groups=(item.arm,), trial_prefix=f"{item.stock}-{item.search_seed}",
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
    loop = DevelopmentLoop(trial_root, plan, proposers=proposers, gate=gate, evaluate=evaluator)
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


def _production_development(args: argparse.Namespace, code: Json) -> Json:
    from .market_eval import MarketEvaluator
    from .provider import registry_transport

    smoke = _require_smoke_gates(Path("/tmp/p1-m4-v2-smoke"))

    registry = args.models or args.stochopia_root / MODEL_REGISTRY
    canonical = (args.stochopia_root / MODEL_REGISTRY).read_text(encoding="utf-8")
    if registry.read_text(encoding="utf-8") != canonical:
        raise ValueError("the study must use the locked Stochopia model registry")
    ledger = Ledger(args.artifact_root)
    with ledger.lock():
        ledger.seal(smoke, "smoke_gate.json")
    aliases = tuple(model for model in dict.fromkeys(ARM_MODELS.values()) if model is not None)
    if args.tokenizer is not None:
        factory = _factory(args.tokenizer)
        counters = {model: factory(model) for model in aliases}
    else:
        from .token_count import load_token_counters

        counters = load_token_counters(args.tokenizers_dir)
    gate = CandidateGate()
    evaluators = {
        stock: MarketEvaluator(args.market_root, stock, cache_root=args.cache_root / "markets")
        for stock in SYMBOLS
    }

    def seed_metrics(stock: str, model: ModelRef) -> Mapping[str, object]:
        return evaluators[stock].seed_profile(model, rollout_seed=DEVELOPMENT_ROLLOUT_SEED)["evidence"]

    def proposer(item: Trajectory) -> TwoStepProposer:
        alias = ARM_MODELS[item.arm]
        if alias is None:
            raise ValueError("no_llm has no provider transport")
        return TwoStepProposer(registry_transport(registry, alias), gate, lambda _usage: None, count_tokens=counters[alias])

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
    if manifest["code"] != code:
        raise ValueError("the current code differs from the frozen experiment")
    if manifest["data"] != data_registration(args.market_root):
        raise ValueError("market metadata differs from the frozen experiment")
    receipt = Ledger(args.artifact_root).read("verification", "test-data-reconciliation.json")
    if (
        receipt is None or receipt["test_dates"] != manifest["test_dates"]
        or receipt["source_revision"] != manifest["data"]["source_revision"]
        or len(receipt["reconciliation"]) != 24 or len(receipt["anchors"]) != 48
    ):
        raise ValueError("finish the registered holdout reconstruction and reconciliation first")
    expected_days = {(stock, day) for stock in SYMBOLS for day in manifest["test_dates"]}
    rows = {(row["symbol"], row["date"]) for row in receipt["reconciliation"]}
    anchors = {(row["symbol"], row["date"], row["session"]) for row in receipt["anchors"]
               if row["spread_ticks"] > 0}
    if rows != expected_days or anchors != {(stock, day, session)
                                             for stock, day in expected_days for session in (0, 1)}:
        raise ValueError("reconciliation must cover both positive-spread anchors on each registered stock-day")
    evaluators = {
        stock: MarketEvaluator(
            args.market_root, stock, cache_root=args.cache_root / "markets",
            authorized_test_dates=tuple(manifest["test_dates"]),
        )
        for stock in SYMBOLS
    }
    return verify_frozen_study(args.artifact_root, evaluators=evaluators)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("dev", "freeze", "verify"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--stochopia-root", type=Path, default=OPENHYRA_ROOT.parent / "Stochopia")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--models", type=Path)
    parsers = parser.add_mutually_exclusive_group()
    parsers.add_argument("--tokenizer", help="trusted module:factory supplying each provider token counter")
    parsers.add_argument("--tokenizers-dir", type=Path, help="directory for locally installed provider tokenizers")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.cache_root is None:
        args.cache_root = args.market_root / "m4-model-cache" / args.artifact_root.name
    if args.cache_root.resolve().is_relative_to(args.artifact_root.resolve()):
        parser.error("working files and model caches must stay outside the evidence archive")
    if args.phase == "dev" and args.tokenizer is None and args.tokenizers_dir is None:
        parser.error("dev requires --tokenizers-dir or --tokenizer")
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
    else:
        from .formal_archive import write_archive_tables

        report = _production_verify(args, code)
        write_archive_tables(args.artifact_root)
        result = {"report": str(args.artifact_root / "verification" / "report.json"), "tasks": report["measured_tasks"]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
