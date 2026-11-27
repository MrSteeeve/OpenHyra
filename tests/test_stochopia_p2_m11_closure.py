"""M1.1 regression: explicit scalar bootstrap semantics and worker numerics."""

from pathlib import Path
import json
import sys

import pytest

from tasks.stochopia_p1 import gate_worker, market_worker
from tasks.stochopia_p1.market_resampling import paired_readouts
from .test_stochopia_p1_m1_impact import (
    CANDIDATE_MULTIPLIERS, DAYS, PARENT_MULTIPLIERS, PairedEvaluation, _payload, _request,
)


def test_c2_bootstrap_rate_scalar_aliases() -> None:
    rows = {day: {} for day in DAYS}
    pair = PairedEvaluation(rows, rows, {}, {}, _payload(PARENT_MULTIPLIERS), _payload(CANDIDATE_MULTIPLIERS))
    result = paired_readouts(pair, _request(), ("c2_bootstrap_pass_rate", "delta_c2_bootstrap_pass_rate"))
    legacy = result["delta_c2_pass_prob"]
    assert result["c2_bootstrap_pass_rate"] == legacy["candidate_bootstrap_mean"] == 1.0
    assert result["parent_c2_bootstrap_pass_rate"] == legacy["parent_bootstrap_mean"]
    assert result["delta_c2_bootstrap_pass_rate"] == legacy["mean"]
    assert result["delta_c2_bootstrap_pass_rate"] < legacy["estimate"]


def test_worker_rollout_numerical_failure_exit_66(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments import p1_m2_baseline
    from stochopia.harness.lob_rollout_batch import NonfiniteRolloutProbability
    import torch

    # The isolated process configures threads once. Tests already configured
    # PyTorch's interop pool, so suppress only that one-time setup side effect.
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(gate_worker, "load_candidate", lambda _path: object())
    def fail(*args: object, **kwargs: object) -> None:
        raise NonfiniteRolloutProbability("degenerate categorical head")
    monkeypatch.setattr(p1_m2_baseline, "evaluate_symbol", fail)
    (tmp_path / "plan.json").write_text(json.dumps(
        {"symbol": "fixture", "train_dates": [1], "eval_dates": [2], "rollout_seed": 4}
    ))
    monkeypatch.setattr(sys, "argv", ["market_worker", "evaluate", str(tmp_path / "candidate.py"),
                                      str(tmp_path), str(tmp_path / "output")])
    with pytest.raises(SystemExit) as error:
        market_worker.cli()
    assert error.value.code == market_worker.NONFINITE_LOSS_EXIT == 66
