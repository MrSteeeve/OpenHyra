"""Disclosed research fixtures for continuous P scenarios and interval demands.

These GBM parameters are synthetic, not estimates of an equity market. This
module never edits the pricing snapshot: Q valuation stays owned by the existing
evaluator. Scenario weights describe P only.
"""
from __future__ import annotations

from copy import deepcopy
import math
import random
from statistics import NormalDist

from .equity_demands import demand_spec


OBJECTIVE_V2 = "equity-demand-robust-return.v2"
EXPECTED_OBJECTIVE_V2 = "equity-demand-expected-return.v2"


def protocol_config(overrides=None):
    """Return a fresh, JSON-serializable protocol, with explicit fixture knobs."""
    config = {
        "schema": "stochopia-equity-protocol.v2",
        "horizon_months": 3,
        "paths_per_distribution": 64,
        "sampling": "stratified-antithetic-normal",
        "score_mode": "mean",
        "split_seeds": {"public": 31001, "development": 41001, "held_out": 51001},
        "distributions": [
            {"id": "central", "annual_drift": .06, "annual_vol": .20, "probability": 1 / 3},
            {"id": "bullish", "annual_drift": .18, "annual_vol": .24, "probability": 1 / 3},
            {"id": "bearish", "annual_drift": -.12, "annual_vol": .28, "probability": 1 / 3},
        ],
        "interval_grid_points": 11,
        "intervals": {
            "upside": [
                {"name": "mild-drawdown", "terminal_ratio_min": .90, "terminal_ratio_max": 1., "min_net_return": 0.},
                {"name": "participate-upside", "terminal_ratio_min": 1.10, "terminal_ratio_max": 1.20, "min_net_return": .01},
            ],
            "two-way": [
                {"name": "moderate-downside", "terminal_ratio_min": .80, "terminal_ratio_max": .90, "min_net_return": .01},
                {"name": "moderate-upside", "terminal_ratio_min": 1.10, "terminal_ratio_max": 1.20, "min_net_return": .01},
            ],
        },
        "market_calibrated": False,
        "q_pricing": "existing pricing snapshot and evaluator; independent of P configuration",
        "interval_check_scope": "finite inclusive grid, not a proof over every real-valued terminal ratio",
        "path_dependence_required": False,
    }
    if overrides:
        config.update(deepcopy(overrides))
    return config


def _config(config):
    return protocol_config() if config is None else config


def scenario_paths(split, config=None):
    """Sample independent-in-time GBM increments on the existing monthly grid.

    Each step uses randomized stratified standard normals and their antithetic
    partners. Weights sum to each regime's probability and to one globally.
    The same split/config is reproducible; different split seeds use new draws.
    """
    config = _config(config)
    count = config["paths_per_distribution"]
    if count < 2 or count % 2:
        raise ValueError("paths_per_distribution must be a positive even count")
    if config["horizon_months"] != 3:
        raise ValueError("equity protocol v2 uses the existing three-month grid")
    if config["sampling"] != "stratified-antithetic-normal":
        raise ValueError("unknown P sampling method")
    regimes = config["distributions"]
    if not math.isclose(sum(r["probability"] for r in regimes), 1., abs_tol=1e-12):
        raise ValueError("P regime probabilities must sum to one")
    seed = config["split_seeds"][split]
    half, dt = count // 2, 1 / 12
    normal = NormalDist()
    output = []
    for index, regime in enumerate(regimes):
        rng = random.Random(seed + 1009 * index)
        columns = []
        for _ in range(3):
            values = [normal.inv_cdf((j + rng.random()) / half) for j in range(half)]
            rng.shuffle(values)
            columns.append(values)
        mu, sigma = regime["annual_drift"], regime["annual_vol"]
        for i in range(count):
            sign = 1 if i < half else -1
            row = i % half
            ratios = [1.]
            for column in columns:
                ratios.append(ratios[-1] * math.exp((mu - .5 * sigma ** 2) * dt
                    + sigma * math.sqrt(dt) * sign * column[row]))
            output.append({"name": f"{split}-{regime['id']}-{i:03d}", "ratios": ratios,
                "distribution_id": regime["id"], "weight": regime["probability"] / count})
    return output


def demand_spec_v2(demand_id, config=None):
    """Keep the approved budget/loss/anchors, add separately reported intervals."""
    config = _config(config)
    mode = config["score_mode"]
    if mode not in ("worst_distribution", "mean"):
        raise ValueError("score_mode must be worst_distribution or mean")
    spec = demand_spec(demand_id)
    intervals = deepcopy(config["intervals"][spec["family"]])
    for interval in intervals:
        interval["grid_points"] = config["interval_grid_points"]
    spec.update({
        "schema": "stochopia-equity-demand.v2",
        "objective": OBJECTIVE_V2 if mode == "worst_distribution" else EXPECTED_OBJECTIVE_V2,
        "score_mode": mode,
        "interval_requirements": intervals,
        "interval_check_scope": config["interval_check_scope"],
        "path_dependence_required": False,
        "p_distribution": {
            "kind": "continuous-synthetic-gbm-mixture",
            "distributions": deepcopy(config["distributions"]),
            "paths_per_distribution": config["paths_per_distribution"],
            "sampling": config["sampling"], "split_seeds": deepcopy(config["split_seeds"]),
            "market_calibrated": False,
            "q_independent": True,
        },
        "protocol_config": deepcopy(config),
    })
    return spec


def interval_check_paths(spec):
    """Inclusive terminal grids on geometric paths; keep interval group labels."""
    checks = []
    for interval in spec["interval_requirements"]:
        count = interval["grid_points"]
        if count < 2:
            raise ValueError("interval_grid_points must include at least two endpoints")
        low, high = interval["terminal_ratio_min"], interval["terminal_ratio_max"]
        for i in range(count):
            terminal = low + (high - low) * i / (count - 1)
            checks.append({"name": f"{interval['name']}-{i:03d}",
                "interval_name": interval["name"], "terminal_ratio": terminal,
                "spot_ratios": [terminal ** (j / 3) for j in range(4)],
                "min_net_return": interval["min_net_return"]})
    return checks


def path_pair_diagnostics(config=None):
    """Optional equal-terminal/different-history probes, without extra rewards."""
    return [
        {"name": "up-same-terminal", "terminal_ratio": 1.10,
         "paths": [[1., 1.03, 1.06, 1.10], [1., .75, .92, 1.10]]},
        {"name": "down-same-terminal", "terminal_ratio": .90,
         "paths": [[1., .97, .94, .90], [1., 1.25, 1.08, .90]]},
    ]
