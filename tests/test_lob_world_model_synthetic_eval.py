"""Offline split isolation, cache integrity and statistical pairing for discovery."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from tasks.lob_world_model import synthetic_eval as bridge
from tasks.lob_world_model.synthetic_stats import SCALAR_METRICS, pool_training_seeds, paired_summary, summarize_levels


def _panel(offset: float = 0.0, missing: int = 0) -> list[dict]:
    rows = []
    for day in range(12):
        values = {metric: float(day + offset) for metric in SCALAR_METRICS}
        if day < missing:
            values["counterfactual_impact_mae"] = None
        values.update(trading_date=20300117 + day, c2_child_ticks=[1., 2., 3., 4.])
        rows.append(values)
    return rows


def test_seed_pool_and_aligned_bootstrap_disallow_pseudoreplication() -> None:
    base = pool_training_seeds([_panel(-1), _panel(0), _panel(1)])
    candidate = pool_training_seeds([_panel(0, 6), _panel(1), _panel(2)])
    first = {"env_id": "env_alpha", "split": "dev", "daily": candidate}
    second = {"env_id": "env_alpha", "split": "dev", "daily": base}
    paired = paired_summary(first, second)
    assert paired["dt_nll"]["estimate"] == 1
    assert paired["dt_nll"]["interval"] == [1., 1.]
    assert paired["dt_nll"]["paired_days"] == 12
    assert paired["counterfactual_impact_mae"]["paired_days"] == 6
    assert paired["counterfactual_impact_mae"]["estimate"] == 1
    assert paired["c2_bootstrap_pass_rate"]["estimate"] == 0
    assert paired["c2_bootstrap_pass_rate"]["bootstrap_total"] == 2000
    assert summarize_levels(candidate)[0]["dt_nll"] == 6.5
    with pytest.raises(ValueError, match="split"):
        paired_summary({**first, "split": "holdout_test"}, second)
    with pytest.raises(ValueError, match="environment"):
        paired_summary({**first, "env_id": "env_beta"}, second)
    with pytest.raises(ValueError, match="calendars"):
        paired_summary({**first, "daily": candidate[1:]}, second)


def test_holdout_is_unmaterialized_before_global_candidate_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def dev(root: Path, env_id: str) -> dict:
        root.mkdir(exist_ok=True, parents=True)
        bridge.atomic_write_json(root / "development.json", {})
        for name in ("train.pkl", "dev.pkl"):
            (root / name).write_bytes(name.encode())
        return {"z": {"rate": {"median": 1.0, "intraday_iqr": 0.2}}}

    def holdout(root: Path, env_id: str, *, freeze_sha256: str) -> dict:
        assert (root.parent / "holdout.authorization.json").is_file()
        (root / "holdout_test.pkl").write_bytes(b"future")
        bridge.atomic_write_json(root / "holdout.json", {"freeze_sha256": freeze_sha256})
        return {"freeze_sha256": freeze_sha256}

    def run(work: Path, mode: str, source: str, files: dict[str, Path], timeout_s: float) -> Path:
        calls.append((mode, set(files), [path.name for path in files.values()]))
        output = work / "output"
        output.mkdir(parents=True)
        if mode == "fit":
            for seed in bridge.SEEDS:
                (output / f"{seed}.pt").write_text(f"weight-{seed}")
            bridge.atomic_write_json(output / "fit.json", {"params": 17, "state": 6, "fits": []})
        else:
            bridge.atomic_write_json(output / "result.json", {"params": 17, "state": 6,
                                                             "aggregate": {}, "daily": []})
        return output

    monkeypatch.setattr(bridge, "prepare_development", dev)
    monkeypatch.setattr(bridge, "materialize_holdout", holdout)
    monkeypatch.setattr(bridge, "_run", run)
    a = bridge.SyntheticEvaluator(tmp_path, "env_alpha")
    b = bridge.SyntheticEvaluator(tmp_path, "env_beta")
    a.gate = b.gate = lambda source: None
    a.prepare_dev()
    b.prepare_dev()
    assert not list(tmp_path.rglob("*holdout*"))
    source = "candidate"
    digest = sha256(source.encode()).hexdigest()
    with pytest.raises(ValueError, match="not materialized"):
        a.evaluate(source, split="holdout_test")
    first = a.evaluate(source)
    assert first == a.evaluate(source)
    assert len(calls) == 2
    assert calls[0] == ("fit", {"train.pkl"}, ["train.pkl"])
    assert calls[1][0:2] == ("measure", {"panel.pkl", "103.pt", "137.pt", "211.pt"})
    assert calls[1][2][-1] == "dev.pkl"
    freeze = tmp_path / "frozen.json"
    freeze.write_text(json.dumps({"approved": {"env_alpha": [digest], "env_beta": [digest]}}))
    with pytest.raises(ValueError, match="both environments"):
        a.open_holdout(freeze)
    assert not list(tmp_path.rglob("*holdout*"))
    b.evaluate(source)
    a.open_holdout(freeze)
    with pytest.raises(ValueError, match="closed"):
        a.evaluate(source)
    with pytest.raises(ValueError, match="not frozen"):
        a.evaluate("different", split="holdout_test")
    tested = a.evaluate(source, split="holdout_test")
    assert calls[-1][0] == "measure" and calls[-1][2][-1] == "holdout_test.pkl"
    assert tested["training_checkpoint_hashes"] == first["training_checkpoint_hashes"]
    assert len([mode for mode, _, _ in calls if mode == "fit"]) == 2
    assert tested == a.evaluate(source, split="holdout_test")
    assert len(calls) == 5


def test_insufficient_support_preserved_as_unavailable() -> None:
    rows = _panel(1., 8)
    for row in rows:
        row["c2_child_ticks"] = None
    summary, support = summarize_levels(rows)
    assert summary["c2_bootstrap_pass_rate"] is None
    assert support["counterfactual_impact_mae"]["paired_days"] == 4
    assert support["counterfactual_impact_mae"]["interval"] is None
    pair = paired_summary({"split": "dev", "daily": rows}, {"split": "dev", "daily": _panel()})
    assert pair["counterfactual_impact_mae"]["interval"] is None
    assert pair["c2_bootstrap_pass_rate"]["estimate"] is None
