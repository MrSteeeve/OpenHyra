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


def test_transplant_contrast_refits_frozen_operator_in_both_worlds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """tau_X(beta) - tau_X(alpha) on shared holdout day indices, holdout opened only after the freeze."""
    effects = {"reference": {"env_alpha": 0., "env_beta": 0.}, "x": {"env_alpha": -1., "env_beta": -3.}}
    calls = []

    def dev(root: Path, env_id: str) -> dict:
        root.mkdir(exist_ok=True, parents=True)
        bridge.atomic_write_json(root / "development.json", {})
        return {"z": {}}

    def holdout(root: Path, env_id: str, *, freeze_sha256: str) -> dict:
        (root / "holdout_test.pkl").write_bytes(b"future")
        bridge.atomic_write_json(root / "holdout.json", {"freeze_sha256": freeze_sha256})
        return {"freeze_sha256": freeze_sha256}

    def run(work: Path, mode: str, source: str, files: dict[str, Path], timeout_s: float) -> Path:
        env = files.get("panel.pkl", files.get("train.pkl")).parent.parent.name
        calls.append((mode, env, source))
        if source == "bad" and env == "env_beta":
            raise bridge._CandidateFailure("candidate tensor failure")
        output = work / "output"
        output.mkdir(parents=True)
        if mode == "fit":
            for seed in bridge.SEEDS:
                (output / f"{seed}.pt").write_text(f"{source}-{seed}")
            bridge.atomic_write_json(output / "fit.json", {"params": 17, "state": 6, "fits": []})
        else:
            daily = _panel(effects.get(source, effects["reference"])[env])
            bridge.atomic_write_json(output / "result.json", {"params": 17, "state": 6, "aggregate": {}, "daily": daily})
        return output

    monkeypatch.setattr(bridge, "prepare_development", dev)
    monkeypatch.setattr(bridge, "materialize_holdout", holdout)
    monkeypatch.setattr(bridge, "_run", run)
    monkeypatch.setattr(bridge, "CandidateGate", lambda max_state_dim: (lambda source: None))
    result = bridge.transplant_contrast(tmp_path, {"x": "x", "bad": "bad"}, "reference")
    x = result["candidates"]["x"]
    assert result["environments"] == ["env_alpha", "env_beta"] and x["status"] == "VALID"
    assert x["tau"]["env_alpha"]["dt_nll"]["estimate"] == -1 and x["tau"]["env_beta"]["dt_nll"]["estimate"] == -3
    assert x["tau_difference"]["counterfactual_impact_mae"]["estimate"] == -2
    assert x["tau_difference"]["counterfactual_impact_mae"]["interval"] == [-2., -2.]
    assert x["tau_difference"]["dt_nll"]["paired_days"] == 12 and "c2_bootstrap_pass_rate" not in x["tau_difference"]
    assert result["candidates"]["bad"]["status"] == "UNAVAILABLE"
    assert result["candidates"]["bad"]["dev_status"] == {"env_alpha": "VALID", "env_beta": "EVALUATION_FAILED"}
    measured = [(env, source) for mode, env, source in calls if mode == "measure"]
    assert measured.count(("env_alpha", "bad")) == 1  # dev only; not approved for holdout
    # Resuming reuses the sealed freeze and cached fits without new worker runs.
    count = len(calls)
    assert bridge.transplant_contrast(tmp_path, {"x": "x", "bad": "bad"}, "reference") == result
    assert len(calls) == count
    # Passing an unregistered candidate against an already-sealed transplant-freeze raises ValueError, never KeyError.
    with pytest.raises(ValueError, match="absent from sealed"):
        bridge.transplant_contrast(tmp_path, {"x": "x", "unseen": "unseen"}, "reference")


def test_single_factor_pairs_and_boundary_falsification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert bridge.single_factor_pair_for_z("sign_acf_sum_1_32") == ("memory", ("w1_thick_short", "w2_thick_long"))
    assert bridge.single_factor_pair_for_z("midprice_shift_rate") == ("thinness", ("w1_thick_short", "w3_thin_short"))
    assert bridge.single_factor_pair_for_z("interarrival_cv2") == ("clustering", ("w3_thin_short", "w6_thin_hawkes"))

    effects = {"reference": {"w3_thin_short": 0., "w6_thin_hawkes": 0.},
               "hawkes_op": {"w3_thin_short": .02, "w6_thin_hawkes": -.20}}
    z_values = {"w3_thin_short": {"interarrival_cv2": {"median": .94, "intraday_iqr": .12}},
                "w6_thin_hawkes": {"interarrival_cv2": {"median": 5.03, "intraday_iqr": .91}}}

    def dev(root: Path, env_id: str) -> dict:
        root.mkdir(exist_ok=True, parents=True)
        bridge.atomic_write_json(root / "development.json", {})
        return {"z": z_values[env_id]}

    def holdout(root: Path, env_id: str, *, freeze_sha256: str) -> dict:
        (root / "holdout_test.pkl").write_bytes(b"future")
        bridge.atomic_write_json(root / "holdout.json", {"freeze_sha256": freeze_sha256})
        return {"freeze_sha256": freeze_sha256}

    def run(work: Path, mode: str, source: str, files: dict[str, Path], timeout_s: float) -> Path:
        env = files.get("panel.pkl", files.get("train.pkl")).parent.parent.name
        output = work / "output"
        output.mkdir(parents=True)
        if mode == "fit":
            for seed in bridge.SEEDS:
                (output / f"{seed}.pt").write_text(f"{source}-{seed}")
            bridge.atomic_write_json(output / "fit.json", {"params": 17, "state": 6, "fits": []})
        else:
            bridge.atomic_write_json(output / "result.json", {
                "params": 17, "state": 6, "aggregate": {}, "daily": _panel(effects[source][env]),
            })
        return output

    monkeypatch.setattr(bridge, "prepare_development", dev)
    monkeypatch.setattr(bridge, "materialize_holdout", holdout)
    monkeypatch.setattr(bridge, "_run", run)
    monkeypatch.setattr(bridge, "CandidateGate", lambda max_state_dim: (lambda source: None))
    res = bridge.transplant_contrast(tmp_path / "pair", {"hawkes_op": "hawkes_op"}, "reference",
                                     environments=("w3_thin_short", "w6_thin_hawkes"))
    entry = res["candidates"]["hawkes_op"]
    confirmed = bridge.evaluate_boundary_claim(entry, res["world_z"], "interarrival_cv2", "lower", "dt_nll")
    assert confirmed["verdict"] == "BOUNDARY_CONFIRMED"
    assert confirmed["z_weak_env"] == "w3_thin_short" and confirmed["z_strong_env"] == "w6_thin_hawkes"
    assert confirmed["delta_strong_minus_weak"]["estimate"] == pytest.approx(-.22)
    refuted = bridge.evaluate_boundary_claim(entry, res["world_z"], "interarrival_cv2", "higher", "dt_nll")
    assert refuted["verdict"] == "BOUNDARY_REFUTED"

    # Malformed transplant-freeze.json raises ValueError cleanly.
    bad_root = tmp_path / "bad_freeze"
    bad_root.mkdir()
    (bad_root / "transplant-freeze.json").write_text(json.dumps({"approved": {"w3_thin_short": []}}))
    with pytest.raises(ValueError, match="approved"):
        bridge.transplant_contrast(bad_root, {"hawkes_op": "hawkes_op"}, "reference",
                                   environments=("w3_thin_short", "w6_thin_hawkes"))
