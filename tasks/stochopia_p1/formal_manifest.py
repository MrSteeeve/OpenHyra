"""Preregister the fixed M4 roster and freeze model sources and verifier tasks."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

from stochopia.common import atomic_write_bytes
from stochopia.neural.lob_interface import MAX_WORLD_MODEL_PARAMS, MAX_WORLD_MODEL_STATE
from stochopia.neural.lob_training import P1_FIT_PROTOCOL

from .store import IMPACT, Json, Ledger, ModelRef, validate_dates


SYMBOLS = ("000001.SZ", "300001.SZ")
ARM_MODELS = {
    "evidence_gpt": "gpt-6.2-sol",
    "metrics_gpt": "gpt-6.2-sol",
    "evidence_ds": "deepseek-flash",
    "metrics_ds": "deepseek-flash",
    "no_llm": None,
}
SEARCH_SEEDS = (17, 29, 41)
TRAIN_SEEDS = (103, 137, 211)
ROUNDS = 3
DEVELOPMENT_ROLLOUT_SEED = 12_300
TEST_ROLLOUT_SEED = 9_127
TEST_DATES = (
    20170324, 20170327, 20170328, 20170329, 20170330, 20170331,
    20170405, 20170406, 20170407, 20170410, 20170411, 20170412,
)
EMBARGO_DATES = (20170306, 20170323)
FORMAL_SCHEMA = "stochopia-p1-m4-v2"
MANIFEST = "study_manifest.json"


@dataclass(frozen=True)
class Trajectory:
    stock: str
    arm: str
    search_seed: int

    @property
    def key(self) -> str:
        return f"{self.stock}-{self.arm}-{self.search_seed}"

    def path(self, root: Path) -> Path:
        return root.joinpath("development", self.stock, self.arm, f"seed-{self.search_seed}")

    def round_path(self, index: int) -> tuple[str, str]:
        return self.arm, f"round-{index:03}"


def roster() -> Iterator[Trajectory]:
    for stock in SYMBOLS:
        for arm in ARM_MODELS:
            for search_seed in SEARCH_SEEDS:
                yield Trajectory(stock, arm, search_seed)


def data_registration(market_root: Path) -> Json:
    """Read the registered calendar and dataset revision, with no data file IO."""
    record = json.loads((market_root / "manifest.json").read_text(encoding="utf-8"))
    required_counts = {"train_dates": 36, "dev_dates": 12, "test_dates": 12}
    for field, size in required_counts.items():
        dates = tuple(record[field])
        if len(dates) != size or any(type(day) is not int for day in dates):
            raise ValueError(f"M4 requires {size} registered {field}")
        validate_dates(dates, field)
    if tuple(record["symbols"]) != SYMBOLS or tuple(record["test_dates"]) != TEST_DATES:
        raise ValueError("M4 requires the two registered stocks and exact holdout calendar")
    if tuple(record["embargo_dates"]) != EMBARGO_DATES:
        raise ValueError("M4 requires both registered embargo dates")
    if not (
        record["train_dates"][-1] < EMBARGO_DATES[0] < record["dev_dates"][0]
        and record["dev_dates"][-1] < EMBARGO_DATES[1] < TEST_DATES[0]
    ):
        raise ValueError("M4 requires chronological train, embargo, dev and test periods")
    return {
        key: record[key]
        for key in ("source_repo", "source_revision", "symbols", "train_dates", "dev_dates", "embargo_dates")
    }


def study_registration(data: Json, code: Mapping[str, str], model_registry: str) -> Json:
    return {
        "schema": FORMAL_SCHEMA,
        "code": dict(code),
        "model_registry_sha256": sha256(model_registry.encode("utf-8")).hexdigest(),
        "data": data,
        "arms": ARM_MODELS,
        "rounds": ROUNDS,
        "search_seeds": list(SEARCH_SEEDS),
        "train_seeds": list(TRAIN_SEEDS),
        "development_rollout_seed": DEVELOPMENT_ROLLOUT_SEED,
        "fit_protocol": asdict(P1_FIT_PROTOCOL),
        "limits": {"parameters": MAX_WORLD_MODEL_PARAMS, "real_state_scalars": MAX_WORLD_MODEL_STATE},
    }


class SourceArchive:
    """Publish functional content addresses for the frozen executable sources."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    def add(self, source: str) -> str:
        payload = source.encode("utf-8")
        digest = sha256(payload).hexdigest()
        path = self.ledger.file("sources", f"{digest}.py")
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"frozen candidate source changed: {digest}")
        else:
            atomic_write_bytes(path, payload)
        return digest

    def ref(self, model: Json) -> Json:
        return {"model_id": model["model_id"], "source_sha256": self.add(model["source"])}

    def load(self, model: Json) -> ModelRef:
        digest = model["source_sha256"]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("frozen source must have a SHA-256 address")
        payload = self.ledger.file("sources", f"{digest}.py").read_bytes()
        if sha256(payload).hexdigest() != digest:
            raise ValueError(f"frozen candidate source changed: {digest}")
        return ModelRef(model["model_id"], payload.decode("utf-8"))


# --- Development evidence serialization ---

def _read_trial(ledger: Ledger, item: Trajectory, index: int, archive: SourceArchive) -> Json:
    path = item.round_path(index)
    result = ledger.read(*path, "result.json")
    if result is None:
        raise ValueError(f"incomplete formal trial {item.key} round {index}")
    if result["round"] != index:
        raise ValueError(f"unexpected round in {item.key}")
    program = ledger.read(*path, "program.json")
    if program is None:
        raise ValueError(f"missing candidate compilation record in {item.key} round {index}")
    reference = {
        key: value for key, value in result.items()
        if key not in ("parent", "candidate", "evaluation")
    }
    reference["parent"] = archive.ref(result["parent"])
    candidate = result["candidate"]
    if candidate is not None and candidate["source"] != program.get("source"):
        raise ValueError(f"candidate source differs from the compiled program in {item.key} round {index}")
    reference["candidate"] = None if candidate is None else archive.ref(candidate)
    reference["program"] = {key: value for key, value in program.items() if key != "source"}
    if program.get("source") is not None:
        reference["program"]["source_sha256"] = archive.add(program["source"])
    for name in ("usage", "requests", "compiler"):
        directory = ledger.file(*path, name)
        records = []
        for file in sorted(directory.glob("*.json")):
            record = ledger.read(*path, name, file.name)
            if record is None:
                raise ValueError(f"missing trial artifact {file.name}")
            source = record.pop("source", None)
            record.pop("reply", None)
            record.pop("messages", None)  # Verbatim messages stay in the external run ledger.
            if source is not None:
                record["source_sha256"] = archive.add(source)
            records.append({"record_id": file.stem, **record})
        reference[name] = records
    if "evaluation" in result:
        pair = result["evaluation"]
        reference["parent_metrics"] = pair["parent_evidence"]
        reference["candidate_metrics"] = pair["candidate_evidence"]
    return reference


def _rank_score(trial: Json) -> float | None:
    score = trial.get("candidate_metrics", {}).get(IMPACT)
    if type(score) not in (int, float) or not isfinite(score):
        return None
    return float(score)


def _choose_best(candidates: list[Json]) -> Json | None:
    scored = [row for row in candidates if _rank_score(row) is not None]
    if not scored:
        return None
    return max(scored, key=_rank_score)


def _batch_task(
    task_id: str, stock: str, group: str, parent: Json, candidate: Json,
    predictions: list[Json], roles: list[str],
) -> Json:
    return {
        "task_id": task_id, "stock": stock, "group": group,
        "parent": parent, "candidate": candidate,
        "predictions": predictions, "roles": roles,
    }


# --- Frozen tasks and cross-regime assignments ---

def _schedule(study: Json, baselines: Json, trajectories: Json) -> Json:
    trials = {}
    initial_tasks = {}
    pair_tasks = {}
    for stock, seed in baselines.items():
        model = seed["model"]
        task_id = f"{stock}-seed"
        initial_tasks[stock] = _batch_task(task_id, stock, "shared", model, model, [], ["initial"])

    selections = {}
    for trajectory in roster():
        rows = trajectories[trajectory.key]["trials"]
        initial = initial_tasks[trajectory.stock]["task_id"]
        first = None if rows[0]["candidate"] is None else rows[0]["trial_id"]
        promoted = [row for row in rows if row["verdict"] == "VERIFIED_GAIN"]
        final = promoted[-1]["trial_id"] if promoted else initial
        audit = [row["trial_id"] for row in rows if row["verdict"] == "UNVERIFIED_GAIN"]
        selections[trajectory.key] = {
            **asdict(trajectory), "initial": initial, "first": first, "final": final,
            "audit_only_rejected_unverified": audit,
        }
        for row in rows:
            task_id = row["trial_id"]
            if task_id in trials:
                raise ValueError(f"duplicate formal trial ID {task_id}")
            trials[task_id] = row
            roles = []
            if task_id == first:
                roles.append("first")
            if task_id == final:
                roles.append("final")
            if task_id in audit:
                roles.append("audit_only_rejected_unverified")
            if roles:
                pair_tasks[task_id] = _batch_task(
                    task_id, trajectory.stock, trajectory.arm, row["parent"], row["candidate"],
                    row["predictions"], roles,
                )

    cross_regime = {}
    for arm in ARM_MODELS:
        designs = {}
        for stock in SYMBOLS:
            target = next(symbol for symbol in SYMBOLS if symbol != stock)
            island_rows = [value for value in selections.values() if value["stock"] == stock and value["arm"] == arm]
            winners = [trials[row["final"]] for row in island_rows if row["final"] in trials]
            best = _choose_best(winners)
            fallback = None
            if best is None:
                legal = []
                for item in roster():
                    if item.arm == arm and item.stock == stock:
                        legal.extend(trajectories[item.key]["trials"])
                fallback = _choose_best(legal)
            representative = best if best is not None else fallback
            choice_id = representative["trial_id"] if representative is not None else None
            cross_id = None
            if representative is not None:
                cross_id = f"cross-{arm}-{stock}-to-{target}"
                pair_tasks[cross_id] = _batch_task(
                    cross_id, target, arm, baselines[target]["model"], representative["candidate"], [],
                    ["cross_regime" if best is not None else "cross_regime_auxiliary"],
                )
                if choice_id not in pair_tasks:
                    pair_tasks[choice_id] = _batch_task(
                        choice_id, stock, arm, representative["parent"], representative["candidate"],
                        representative["predictions"], ["cross_regime_auxiliary_diagonal"],
                    )
            designs[stock] = {
                "target_stock": target,
                "diagonal": choice_id if best is not None else initial_tasks[stock]["task_id"],
                "off_diagonal": cross_id if best is not None else initial_tasks[target]["task_id"],
                "representative": choice_id if best is not None else initial_tasks[stock]["task_id"],
                "dev_score": _rank_score(best) if best is not None else baselines[stock]["metrics"].get(IMPACT),
                "retains_seed": best is None,
                "auxiliary": (
                    {"diagonal": choice_id, "off_diagonal": cross_id, "dev_score": _rank_score(fallback)}
                    if fallback is not None else None
                ),
            }
        cross_regime[arm] = designs
    return {
        **study, "test_dates": list(TEST_DATES), "test_rollout_seed": TEST_ROLLOUT_SEED,
        "baselines": baselines, "trajectories": trajectories, "selections": selections,
        "tasks": [*initial_tasks.values(), *pair_tasks.values()],
        "cross_regime": cross_regime,
    }


def freeze_study(root: Path, market_root: Path, *, development_root: Path | None = None) -> Json:
    """Atomically seal all 90 trials and the holdout schedule before unblinding."""
    parent = Ledger(root)
    with parent.lock():
        study = parent.read("development_plan.json")
        if study is None:
            raise ValueError("initialize the development study before freezing")
        if study["data"] != data_registration(market_root):
            raise ValueError("market metadata differs from the original registration")
        archive = SourceArchive(parent)
        previous = parent.read(MANIFEST)
        if previous is not None:
            validate_frozen(previous, archive)
            return previous
        work_root = root if development_root is None else development_root
        trajectory_ledgers = {item: Ledger(item.path(work_root)) for item in roster()}
        for item, ledger in trajectory_ledgers.items():
            for index in range(1, ROUNDS + 1):
                if ledger.read(*item.round_path(index), "result.json") is None:
                    raise ValueError(f"incomplete formal trial {item.key} round {index}")
        baselines = {}
        for stock in SYMBOLS:
            seed = parent.read("baselines", f"{stock}.json")
            if seed is None:
                raise ValueError(f"missing shared seed development baseline for {stock}")
            baselines[stock] = {"model": archive.ref(seed["model"]), "metrics": seed["metrics"]}
        trajectories = {}
        for item, ledger in trajectory_ledgers.items():
            rows = [_read_trial(ledger, item, index, archive) for index in range(1, ROUNDS + 1)]
            trajectories[item.key] = {**asdict(item), "trials": rows}
        manifest = _schedule(study, baselines, trajectories)
        validate_frozen(manifest, archive)
        return parent.seal(manifest, MANIFEST)


def validate_frozen(manifest: Json, archive: SourceArchive) -> None:
    """Check the complete roster and every executable source before data access."""
    if manifest.get("schema") != FORMAL_SCHEMA or tuple(manifest.get("test_dates", ())) != TEST_DATES:
        raise ValueError("unexpected formal manifest schema or test calendar")
    if set(manifest["trajectories"]) != {item.key for item in roster()}:
        raise ValueError("formal roster is incomplete")
    expected = _schedule({}, manifest["baselines"], manifest["trajectories"])
    for key in ("tasks", "selections", "cross_regime"):
        if manifest[key] != expected[key]:
            raise ValueError("holdout schedule differs from the sealed development decisions")
    sources: set[str] = set()
    for trajectory in manifest["trajectories"].values():
        if len(trajectory["trials"]) != ROUNDS:
            raise ValueError("formal roster requires three completed trials per trajectory")
        for trial in trajectory["trials"]:
            for key in ("parent", "candidate", "program"):
                item = trial.get(key)
                if item and item.get("source_sha256"):
                    sources.add(item["source_sha256"])
            for record in trial["compiler"]:
                if record.get("source_sha256"):
                    sources.add(record["source_sha256"])
    for row in manifest["tasks"]:
        sources.add(row["parent"]["source_sha256"])
        sources.add(row["candidate"]["source_sha256"])
    for digest in sources:
        archive.load({"model_id": digest, "source_sha256": digest})
