"""Real tiny synthetic dataset and isolated measure smoke, without formal model fits."""

from pathlib import Path
import pickle

import numpy as np
import torch

from experiments.synthetic_tau_z.discovery_metrics import MeasurementPanel, reference_calibration
from experiments.synthetic_tau_z.design import SEEDS
from stochopia.neural.probes import GRUProbe
from stochopia.sim.queue_reactive_data import generate_dataset, generate_evaluation_dataset, counterfactual_truth
from stochopia.sim.queue_reactive_world import standard_worlds
from tasks.lob_world_model.synthetic_eval import _json, _run
from tasks.lob_world_model.synthetic_stats import METRICS


def test_disjoint_calendar_and_real_measurement_worker(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    recipe = standard_worlds()["W2_thick_long"]
    data = generate_dataset(recipe, seed=173, train_days=2, eval_days=1)
    future = generate_evaluation_dataset(recipe, seed=173, first_day=3, eval_days=1)
    assert [d.trading_date for d in data.train] == [20300101, 20300102]
    assert [d.trading_date for d in data.evaluation] == [20300103]
    assert [d.trading_date for d in future.evaluation] == [20300104]
    assert future.train == ()
    assert len(future.windows) == 2
    scale, participation = reference_calibration(data.train)
    truth = counterfactual_truth(data, num_replicates=16, seed=17)
    panel = MeasurementPanel(data.evaluation, data.windows, truth, scale, participation)
    inputs = tmp_path / "panel.pkl"
    with inputs.open("wb") as stream:
        pickle.dump(panel, stream)
    files = {"panel.pkl": inputs}
    source = ('from stochopia.neural.probes import GRUProbe\n\n'
              'class CandidateLOBWorldModel(GRUProbe):\n'
              '    def __init__(self):\n'
              '        super().__init__("base")\n')
    for seed in SEEDS:
        torch.manual_seed(seed)
        name = f"{seed}.pt"
        torch.save(GRUProbe("base").state_dict(), tmp_path / name)
        files[name] = tmp_path / name
    output = _run(tmp_path / "worker", "measure", source, files, timeout_s=90)
    result = _json(output / "result.json")
    assert result["state"] == 60 and result["params"] <= 28_000
    assert set(result["aggregate"]) == set(METRICS)
    assert set(result["seed_daily"]) == set(map(str, SEEDS))
    assert len(result["daily"]) == 1
    assert result["daily"][0]["trading_date"] == 20300103
    assert np.isfinite(result["aggregate"]["val_nll_total"])
    assert np.isfinite(result["aggregate"]["exhausted_replicate_rate"])
    assert result["support"]["val_nll_total"]["interval"] is None
    assert result["support"]["c2_bootstrap_pass_rate"]["interval"] is None
