"""Anonymous, preregistered synthetic train/dev materialization and delayed holdout."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import pickle
from typing import Any

from experiments.synthetic_tau_z.design import (
    BATCH_WINDOWS, BOOTSTRAP_SEED, CONTEXT, DATA_SEED, EVAL_DAYS, EVENTS_PER_SESSION, FIT_PROTOCOL, K,
    N_BOOTSTRAP, ROLLOUT_SEED, SEEDS, TRAIN_DAYS, TRUTH_K, TRUTH_SEED,
)
from experiments.synthetic_tau_z.discovery_metrics import MeasurementPanel, reference_calibration
from stochopia.common import atomic_write_json
from stochopia.measure.data_properties import measure_data_properties
from stochopia.sim.queue_reactive_data import QueueDataset, generate_dataset, generate_evaluation_dataset, counterfactual_truth
from stochopia.sim.queue_reactive_world import standard_worlds

ENV_IDS = ("env_alpha", "env_beta")
# Only the materializer sees generator names. The worker gets anonymous day arrays.
_WORLD_KEYS = {"env_alpha": "W6_thin_hawkes", "env_beta": "W2_thick_long"}
Z_KEYS = (
    "sign_acf_sum_1_32", "flow_vr_32_4", "trade_to_top1_depth", "midprice_shift_rate",
    "cancel_to_limit_ratio", "zero_wait_prob", "interarrival_cv2", "book_imbalance_bias", "flow_sign_bias",
)

DATA_PROTOCOL = {
    "data_seed": DATA_SEED, "truth_seed": TRUTH_SEED, "train_days": TRAIN_DAYS,
    "dev_days": EVAL_DAYS, "holdout_days": EVAL_DAYS, "holdout_first_day": TRAIN_DAYS + EVAL_DAYS,
    "events_per_session": EVENTS_PER_SESSION, "context": CONTEXT, "sessions_per_day": 2,
    "training_seeds": list(SEEDS), "fit": asdict(FIT_PROTOCOL), "model_k": K, "truth_k": TRUTH_K,
    "rollout_seed": ROLLOUT_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": N_BOOTSTRAP,
    "batch_windows": BATCH_WINDOWS,
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_pickle(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        pickle.dump(value, output, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _panel(dataset: QueueDataset, scale: float, participation: float) -> MeasurementPanel:
    truth = counterfactual_truth(dataset, num_replicates=TRUTH_K, seed=TRUTH_SEED)
    return MeasurementPanel(dataset.evaluation, dataset.windows, truth, scale, participation)


def prepare_development(root: Path, env_id: str) -> dict[str, Any]:
    """Prepare only the first 28 days; no holdout tapes or windows are generated here."""
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "development.json"
    if receipt.exists():
        saved = _json(receipt)
        if saved["protocol"] != DATA_PROTOCOL or saved["env_id"] != env_id:
            raise ValueError("development cache differs from the registered protocol")
        for filename, expected in saved["file_hashes"].items():
            if _digest(root / filename) != expected:
                raise ValueError("development cache fingerprint mismatch")
        return saved
    spec = standard_worlds()[_WORLD_KEYS[env_id]]
    dataset = generate_dataset(spec, seed=DATA_SEED, train_days=TRAIN_DAYS, eval_days=EVAL_DAYS,
                               events_per_session=EVENTS_PER_SESSION, context_len=CONTEXT)
    measured = measure_data_properties(dataset.train)
    flattened = {name: value for dimension in measured.values() for name, value in dimension.items()}
    z = {name: flattened[name] for name in Z_KEYS}
    scale, participation = reference_calibration(dataset.train)
    panel = _panel(dataset, scale, participation)
    _write_pickle(root / "train.pkl", dataset.train)
    _write_pickle(root / "dev.pkl", panel)
    saved = {
        "env_id": env_id, "protocol": DATA_PROTOCOL, "z": z,
        "train_dates": [day.trading_date for day in dataset.train],
        "dev_dates": [day.trading_date for day in dataset.evaluation],
        "reference": {"scale_ticks": scale, "participation": participation},
        "file_hashes": {name: _digest(root / name) for name in ("train.pkl", "dev.pkl")},
    }
    atomic_write_json(receipt, saved)
    return saved


def materialize_holdout(root: Path, env_id: str, *, freeze_sha256: str) -> dict[str, Any]:
    """Caller has already durably sealed source digests; generate disjoint days only now."""
    development = _json(root / "development.json")
    receipt = root / "holdout.json"
    if receipt.exists():
        saved = _json(receipt)
        if saved["freeze_sha256"] != freeze_sha256 or saved["data_sha256"] != _digest(root / "holdout_test.pkl"):
            raise ValueError("holdout cache no longer matches the frozen study")
        return saved
    spec = standard_worlds()[_WORLD_KEYS[env_id]]
    dataset = generate_evaluation_dataset(
        spec, seed=DATA_SEED, first_day=TRAIN_DAYS + EVAL_DAYS, eval_days=EVAL_DAYS,
        events_per_session=EVENTS_PER_SESSION, context_len=CONTEXT,
    )
    dates = [day.trading_date for day in dataset.evaluation]
    if set(dates).intersection((*development["train_dates"], *development["dev_dates"])):
        raise ValueError("holdout dates overlap development or training")
    panel = _panel(dataset, development["reference"]["scale_ticks"], development["reference"]["participation"])
    _write_pickle(root / "holdout_test.pkl", panel)
    saved = {"env_id": env_id, "dates": dates, "freeze_sha256": freeze_sha256,
             "data_sha256": _digest(root / "holdout_test.pkl")}
    atomic_write_json(receipt, saved)
    return saved
