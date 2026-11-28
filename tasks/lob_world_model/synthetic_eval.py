"""Subprocess bridge for matched open-candidate evaluations on synthetic worlds."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any

from experiments.synthetic_tau_z.design import SEEDS, summarize_days
from stochopia.common import atomic_write_json
from worker_subprocess import run_worker_subprocess

from .gate import CandidateGate, _runtime_pythonpath
from .market_eval import _exclusive
from .market_worker import MODEL_FAILURE_EXIT, NONFINITE_LOSS_EXIT
from .synthetic_data import (
    DATA_PROTOCOL, ENV_IDS, SINGLE_FACTOR_PAIRS, WORLD_IDS, Z_TO_SINGLE_FACTOR_PAIR,
    materialize_holdout, prepare_development,
)
from .synthetic_stats import METRICS, paired_summary

Json = dict[str, Any]
_WORKER = Path(__file__).resolve().with_name("synthetic_worker.py")


class _CandidateFailure(RuntimeError):
    def __init__(self, message: str, *, status: str = "EVALUATION_FAILED") -> None:
        super().__init__(message)
        self.status = status


def _json(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _run(work: Path, mode: str, source: str, files: dict[str, Path], timeout_s: float) -> Path:
    """Only copy authorized phase inputs to the isolated worker directory."""
    work.mkdir()
    candidate = work / "candidate.py"
    candidate.write_text(source, encoding="utf-8")
    inputs = work / "inputs"
    inputs.mkdir()
    for name, original in files.items():
        shutil.copyfile(original, inputs / name)
    output = work / "output"
    result = run_worker_subprocess(
        [sys.executable, "-s", "-B", str(_WORKER), mode, str(candidate), str(inputs), str(output)],
        work_dir=work, timeout_s=timeout_s, extra_env={"PYTHONPATH": _runtime_pythonpath()},
    )
    if result["returncode"] != 0 or result["timed_out"] or result["cancelled"]:
        if result["cancelled"]:
            raise RuntimeError("synthetic worker cancelled")
        trace = (result["stdout"] + "\n" + result["stderr"]).replace(str(work), "<worker>")
        detail = f"{mode} failed: {trace[-1800:]}"
        if result["returncode"] == NONFINITE_LOSS_EXIT and not result["timed_out"]:
            raise _CandidateFailure(detail, status="NONFINITE_LOSS")
        if result["timed_out"] or result["returncode"] == MODEL_FAILURE_EXIT:
            raise _CandidateFailure(detail)
        raise RuntimeError(f"unexpected synthetic worker failure ({result['returncode']}): {detail}")
    return output


class SyntheticEvaluator:
    """Three fixed-seed fits, one shared baseline cache, and a separately authorized holdout."""

    def __init__(self, root: Path, env_id: str, *, timeout_s: float = 420.0) -> None:
        if env_id not in WORLD_IDS or timeout_s <= 0:
            raise ValueError("unknown synthetic environment or invalid worker timeout")
        self.root = root.resolve() / env_id
        self.env_id = env_id
        self.timeout_s = timeout_s
        self.gate = CandidateGate(max_state_dim=60)
        self.data = self.root / "dataset"
        self.authorization = self.root / "holdout.authorization.json"

    def prepare_dev(self) -> Json:
        with _exclusive(self.data):
            if self.authorization.exists() and not (self.data / "development.json").exists():
                raise ValueError("cannot create development history after holdout authorization")
            self.development = prepare_development(self.data, self.env_id)
        return self.development["z"]

    def _weights(self, cache: Path, work: Path, source: str) -> Json:
        receipt = cache / "fit.json"
        if not receipt.exists():
            output = _run(work, "fit", source, {"train.pkl": self.data / "train.pkl"}, self.timeout_s)
            fit = _json(output / "fit.json")
            cache.mkdir(exist_ok=True)
            hashes = {}
            for seed in SEEDS:
                name = f"{seed}.pt"
                shutil.copyfile(output / name, cache / name)
                hashes[str(seed)] = _digest(cache / name)
            fit["training_checkpoint_hashes"] = hashes
            atomic_write_json(receipt, fit)
        fit = _json(receipt)
        for seed in SEEDS:
            if _digest(cache / f"{seed}.pt") != fit["training_checkpoint_hashes"][str(seed)]:
                raise ValueError("registered training checkpoint fingerprint mismatch")
        return fit

    def evaluate(self, source: str, *, split: str = "dev") -> Json:
        digest = sha256(source.encode("utf-8")).hexdigest()
        cache = self.root / "models" / digest
        base = {"env_id": self.env_id, "source_sha256": digest, "split": split}
        if split == "dev":
            if self.authorization.exists():
                raise ValueError("development evaluation is closed after the freeze")
            if not hasattr(self, "development"):
                self.prepare_dev()
        elif split == "holdout_test":
            if not self.authorization.exists() or not (self.data / "holdout.json").exists():
                raise ValueError("holdout is not materialized by the frozen verifier")
            approved = _json(self.authorization).get("approved")
            if not isinstance(approved, list) or digest not in approved:
                raise ValueError("candidate was not frozen for holdout evaluation")
        else:
            raise ValueError("unknown split")
        path = cache / f"{split}.json"
        with _exclusive(cache):
            if path.exists():
                return _json(path)
            if split == "dev":
                failure = self.gate(source)
                if failure is not None:
                    result = {**base, "status": "GATE_REJECTED", "error": failure}
                    atomic_write_json(path, result)
                    return result
            if split != "dev" and not (cache / "fit.json").exists():
                raise ValueError("holdout must reuse already fitted checkpoints")
            try:
                with TemporaryDirectory(prefix="synthetic-worker-") as directory:
                    work = Path(directory)
                    fit = self._weights(cache, work / "fit", source)
                    files = {f"{seed}.pt": cache / f"{seed}.pt" for seed in SEEDS}
                    files["panel.pkl"] = self.data / f"{split}.pkl"
                    output = _run(work / "measure", "measure", source, files, self.timeout_s)
                    measured = _json(output / "result.json")
                    if (fit["params"], fit["state"]) != (measured["params"], measured["state"]):
                        raise _CandidateFailure("checkpoint model capacity mismatch")
                    result = {**base, **fit, **measured, "status": "VALID"}
            except _CandidateFailure as exc:
                result = {**base, "status": exc.status, "error": str(exc)}
            atomic_write_json(path, result)
            return result

    def open_holdout(self, freeze_path: Path) -> Json:
        """Seal the approved candidate set before any independent holdout generation."""
        freeze_sha256 = _digest(freeze_path)
        approved = _json(freeze_path).get("approved")
        if (not isinstance(approved, Mapping) or len(approved) < 2 or self.env_id not in approved
                or not set(approved).issubset(WORLD_IDS) or not all(approved.values())):
            raise ValueError("both environments require a frozen digest set")
        for environment, digests in approved.items():
            if not isinstance(digests, list):
                raise ValueError("approved digest set must be a JSON list")
            for digest in digests:
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("expected frozen hexadecimal source digest")
                model = self.root.parent / environment / "models" / digest
                if not (model / "dev.json").exists() or _json(model / "dev.json").get("status") != "VALID":
                    raise ValueError("holdout requires a successful development evaluation in both environments")
        authorization = {"freeze_sha256": freeze_sha256, "approved": sorted(set(approved[self.env_id]))}
        with _exclusive(self.data):
            if self.authorization.exists() and _json(self.authorization) != authorization:
                raise ValueError("holdout approved candidates were changed after the freeze")
            atomic_write_json(self.authorization, authorization)
            return materialize_holdout(self.data, self.env_id, freeze_sha256=freeze_sha256)


def single_factor_pair_for_z(z_metric: str) -> tuple[str, tuple[str, str]]:
    """Map one Z profile metric to its P2-M2 single-factor counterfactual world pair."""
    pair = Z_TO_SINGLE_FACTOR_PAIR.get(z_metric)
    if pair is None:
        raise ValueError(f"unknown Z profile metric for single-factor transplant: {z_metric!r}")
    return pair, SINGLE_FACTOR_PAIRS[pair]


def evaluate_boundary_claim(entry: Mapping[str, Any], world_z: Mapping[str, Any],
                            weaker_when_z_metric: str, weaker_when_z_direction: str,
                            target_metric: str) -> Json:
    """Compare declared (weaker_when_z_metric, weaker_when_z_direction, target_metric) against transplant tau.

    target_metric is always a lower-is-better error readout, so stronger positive
    improvement corresponds to a more negative tau_X = err(X) - err(B0), i.e.
    delta_tau_strong_minus_weak = tau_X(z_strong) - tau_X(z_weak) < 0.
    """
    envs = tuple(world_z)
    if len(envs) != 2:
        raise ValueError("boundary evaluation requires exactly two paired worlds")
    low, high = envs
    z_low = world_z.get(low, {}).get(weaker_when_z_metric, {}).get("median")
    z_high = world_z.get(high, {}).get(weaker_when_z_metric, {}).get("median")
    if z_low is None or z_high is None or z_low == z_high:
        return {"verdict": "INCONCLUSIVE", "reason": "Z metric does not separate the paired worlds",
                "z_weak_env": None, "z_strong_env": None, "delta_strong_minus_weak": None}
    lower_env, higher_env = (low, high) if z_low < z_high else (high, low)
    if weaker_when_z_direction == "lower":
        weak_env, strong_env = lower_env, higher_env
    elif weaker_when_z_direction == "higher":
        weak_env, strong_env = higher_env, lower_env
    else:
        raise ValueError(f"invalid weaker_when_z_direction: {weaker_when_z_direction!r}")
    if entry.get("status") != "VALID":
        return {"verdict": "INCONCLUSIVE", "reason": "candidate unavailable in one or both worlds",
                "z_weak_env": weak_env, "z_strong_env": strong_env,
                "z_weak_median": world_z[weak_env][weaker_when_z_metric]["median"],
                "z_strong_median": world_z[strong_env][weaker_when_z_metric]["median"],
                "delta_strong_minus_weak": None}
    tau = entry.get("tau") or {}
    strong_metric = tau.get(strong_env, {}).get(target_metric)
    weak_metric = tau.get(weak_env, {}).get(target_metric)
    if not isinstance(strong_metric, Mapping) or not isinstance(weak_metric, Mapping):
        raise ValueError(f"target_metric {target_metric!r} missing from transplant tau summary")
    daily = [None if a is None or b is None else a - b
             for a, b in zip(strong_metric["daily"], weak_metric["daily"], strict=True)]
    delta = {**summarize_days(daily), "daily": daily}
    interval = delta.get("interval")
    strong_est = strong_metric.get("estimate")
    if interval is None:
        verdict = "INCONCLUSIVE"
    elif interval[1] < 0 and strong_est is not None and strong_est < 0:
        verdict = "BOUNDARY_CONFIRMED"
    elif interval[0] > 0:
        verdict = "BOUNDARY_REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "z_weak_env": weak_env,
        "z_strong_env": strong_env,
        "z_weak_median": world_z[weak_env][weaker_when_z_metric]["median"],
        "z_strong_median": world_z[strong_env][weaker_when_z_metric]["median"],
        "tau_strong": strong_metric,
        "tau_weak": weak_metric,
        "delta_strong_minus_weak": delta,
    }


def transplant_contrast(root: Path, candidates: dict[str, str], reference: str, *,
                        environments: tuple[str, str] = ENV_IDS, timeout_s: float = 420.0) -> Json:
    """Second loop, zero API: refit frozen operators unchanged in two paired worlds.

    tau_X(z) is candidate minus reference on world z's independent holdout,
    each fitted on that world's own training days. The paired worlds share
    registered day indices, so tau_X(z_high) - tau_X(z_low) is resampled over
    common day indices with the registered draws. Use a fresh work root per pair.
    """
    if len(environments) != 2 or environments[0] == environments[1] or any(env not in WORLD_IDS for env in environments):
        raise ValueError("transplant_contrast requires two distinct registered synthetic worlds")
    low, high = environments
    evaluators = {env: SyntheticEvaluator(root, env, timeout_s=timeout_s) for env in environments}
    world_z = {env: evaluator.prepare_dev() for env, evaluator in evaluators.items()}
    digests = {name: sha256(source.encode("utf-8")).hexdigest() for name, source in candidates.items()}
    ref = sha256(reference.encode("utf-8")).hexdigest()
    sources = {ref: reference, **{digests[name]: source for name, source in candidates.items()}}
    freeze = root.resolve() / "transplant-freeze.json"
    if not freeze.exists():
        dev = {env: {digest: evaluator.evaluate(source)["status"] for digest, source in sources.items()}
               for env, evaluator in evaluators.items()}
        valid = sorted(d for d in dev[low] if dev[low].get(d) == dev[high].get(d) == "VALID")
        if ref not in valid:
            raise ValueError("reference model is not valid in both worlds")
        atomic_write_json(freeze, {"environments": [low, high], "approved": {env: valid for env in environments},
                                  "dev_status": dev})
    sealed = _json(freeze)
    approved_map = sealed.get("approved")
    dev_map = sealed.get("dev_status")
    if (not isinstance(approved_map, Mapping) or set(approved_map) != {low, high}
            or any(not isinstance(approved_map.get(env), list) for env in environments)):
        raise ValueError("transplant-freeze.json has missing or mismatched approved environments")
    if (not isinstance(dev_map, Mapping)
            or any(not isinstance(dev_map.get(env), Mapping) for env in environments)):
        raise ValueError("transplant-freeze.json is missing dev_status for requested environments")
    for digest in sources:
        if any(digest not in dev_map[env] for env in environments):
            raise ValueError(f"digest {digest} is absent from sealed transplant-freeze.json")
    if any(ref not in approved_map[env] for env in environments):
        raise ValueError("reference digest is absent from sealed approved set")
    holdout: dict[str, dict[str, Json]] = {}
    for env, evaluator in evaluators.items():
        evaluator.open_holdout(freeze)
        env_runs: dict[str, Json] = {}
        for digest in approved_map[env]:
            source = sources.get(digest)
            if source is None:
                raise ValueError(f"approved digest {digest} missing from supplied candidate sources")
            env_runs[digest] = evaluator.evaluate(source, split="holdout_test")
        holdout[env] = env_runs
    dates = {}
    for env in environments:
        ref_run = holdout.get(env, {}).get(ref)
        if not isinstance(ref_run, Mapping) or ref_run.get("status") != "VALID" or not isinstance(ref_run.get("daily"), list):
            raise ValueError(f"holdout reference evaluation unavailable in {env}")
        dates[env] = [row["trading_date"] for row in ref_run["daily"]]
    if dates[low] != dates[high]:
        raise ValueError("paired worlds have different registered holdout day indexes")
    result: Json = {
        "environments": [low, high], "world_z": world_z, "reference_sha256": ref,
        "freeze_sha256": _digest(freeze),
        "estimand": "tau_X(high) - tau_X(low); tau = candidate minus reference on holdout",
        "candidates": {},
    }
    for name, digest in digests.items():
        runs = {env: holdout.get(env, {}).get(digest) for env in environments}
        if any(run is None or run.get("status") != "VALID" for run in runs.values()):
            result["candidates"][name] = {
                "source_sha256": digest, "status": "UNAVAILABLE",
                "dev_status": {env: dev_map[env].get(digest, "MISSING") for env in environments},
            }
            continue
        tau = {env: paired_summary(runs[env], holdout[env][ref]) for env in environments}
        difference = {}
        for metric in METRICS:
            if metric == "c2_bootstrap_pass_rate":
                continue  # nonlinear day-panel admission; no per-day effect to difference
            daily = [None if a is None or b is None else b - a
                     for a, b in zip(tau[low][metric]["daily"], tau[high][metric]["daily"], strict=True)]
            difference[metric] = {**summarize_days(daily), "daily": daily}
        result["candidates"][name] = {"source_sha256": digest, "status": "VALID", "tau": tau,
                                      "tau_difference": difference}
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=transplant_contrast.__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="frozen reference candidate.py (e.g. B0)")
    parser.add_argument("--pair", choices=tuple(SINGLE_FACTOR_PAIRS), default=None,
                        help="named single-factor world pair (memory, thinness, clustering)")
    parser.add_argument("--low-env", default=ENV_IDS[0], choices=WORLD_IDS)
    parser.add_argument("--high-env", default=ENV_IDS[1], choices=WORLD_IDS)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=PATH",
                        help="frozen discovery candidate.py, repeatable")
    args = parser.parse_args()
    envs = SINGLE_FACTOR_PAIRS[args.pair] if args.pair else (args.low_env, args.high_env)
    candidates = dict(item.split("=", 1) for item in args.candidate)
    sources = {name: Path(path).read_text(encoding="utf-8") for name, path in candidates.items()}
    result = transplant_contrast(args.work_root, sources, args.reference.read_text(encoding="utf-8"),
                                 environments=envs)
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
