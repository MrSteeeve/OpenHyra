"""Research protocol checks: independent P, continuous draws and broad demands."""
from copy import deepcopy
import json
import math

import pytest

from examples.structuring_roundtrip import synthetic_task
from tasks.stochopia_structuring.equity_demands import DEMAND_IDS, demand_spec
from tasks.stochopia_structuring.equity_protocol_v2 import (
    protocol_config, scenario_paths, demand_spec_v2, interval_check_paths,
    path_pair_diagnostics,
)


def test_continuous_sampling_is_reproducible_and_splits_are_independent():
    config = protocol_config()
    dev, held = scenario_paths("development", config), scenario_paths("held_out", config)
    assert dev == scenario_paths("development", config)
    assert len(dev) == len(held) == 3 * 64
    assert not {tuple(p["ratios"]) for p in dev} & {tuple(p["ratios"]) for p in held}
    assert len({p["ratios"][-1] for p in dev}) == len(dev)
    assert all(p["ratios"][0] == 1. and len(p["ratios"]) == 4 for p in dev)
    assert all(value > 0. for p in dev for value in p["ratios"])
    assert len({p["ratios"][-1] for p in dev}) > 7
    assert math.isclose(sum(p["weight"] for p in dev), 1.)
    for regime in config["distributions"]:
        group = [p for p in dev if p["distribution_id"] == regime["id"]]
        assert len(group) == 64
        assert math.isclose(sum(p["weight"] for p in group), regime["probability"])
        # Opposite normal increments cancel; drift remains at each step.
        for first, opposite in zip(group[:32], group[32:]):
            expected = math.exp(2 * (regime["annual_drift"] - .5 * regime["annual_vol"] ** 2) / 4)
            assert first["ratios"][-1] * opposite["ratios"][-1] == pytest.approx(expected)


def test_p_knobs_do_not_change_the_q_snapshot():
    base = synthetic_task()
    snapshot = deepcopy(base.snapshots[0])
    config = protocol_config()
    changed = deepcopy(config)
    changed["distributions"][0]["annual_drift"] = .5
    changed["distributions"][0]["annual_vol"] = .5
    assert scenario_paths("development", config)[0]["ratios"] != scenario_paths("development", changed)[0]["ratios"]
    assert base.snapshots[0] == snapshot
    assert base.snapshots[0].risk_free_rate == .02
    assert base.snapshots[0].realized_vol_20d == .20
    assert "risk_free_rate" not in config and "pricing_volatility" not in config
    assert demand_spec_v2("upside-protected", changed)["p_distribution"]["q_independent"]


@pytest.mark.parametrize("demand_id", DEMAND_IDS)
def test_v2_preserves_approved_budget_loss_and_anchors_but_broadens_requirements(demand_id):
    old, new = demand_spec(demand_id), demand_spec_v2(demand_id)
    for key in ("investment_budget_cny", "horizon_months", "max_terminal_principal_loss", "anchor_requirements"):
        assert old[key] == new[key]
    assert new["schema"].endswith("v2")
    assert len(new["interval_requirements"]) == 2
    checks = interval_check_paths(new)
    assert len(checks) == 22
    for interval in new["interval_requirements"]:
        group = [c for c in checks if c["interval_name"] == interval["name"]]
        assert group[0]["spot_ratios"][-1] == pytest.approx(interval["terminal_ratio_min"])
        assert group[-1]["spot_ratios"][-1] == pytest.approx(interval["terminal_ratio_max"])
        assert all(c["min_net_return"] == interval["min_net_return"] for c in group)
    assert not new["path_dependence_required"]
    assert new["p_distribution"]["market_calibrated"] is False
    assert json.loads(json.dumps(new)) == new


def test_both_scoring_decisions_are_explicitly_supported():
    robust = demand_spec_v2("two-way-protected", protocol_config({"score_mode": "worst_distribution"}))
    mean = demand_spec_v2("two-way-protected")
    assert robust["score_mode"] == "worst_distribution"
    assert robust["objective"] == "equity-demand-robust-return.v2"
    assert mean["score_mode"] == "mean"
    assert mean["objective"] == "equity-demand-expected-return.v2"
    assert robust["anchor_requirements"] == mean["anchor_requirements"]
    assert robust["interval_requirements"] == mean["interval_requirements"]


def test_fixture_parameters_are_configurable_and_fresh():
    config = protocol_config({"paths_per_distribution": 32, "interval_grid_points": 7})
    assert len(scenario_paths("development", config)) == 96
    assert len(interval_check_paths(demand_spec_v2("two-way-risk20", config))) == 14
    config["distributions"][0]["annual_drift"] = 1.
    assert protocol_config()["distributions"][0]["annual_drift"] == .06
    for pair in path_pair_diagnostics(config):
        left, right = pair["paths"]
        assert left != right and left[-1] == right[-1] == pair["terminal_ratio"]
