"""Check the retrospective roster and replica cluster boundary without running L2."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.verify_p2_m1_szse_300001 import MODEL_SLOTS, _seal, _source_models, _validate_profile, _weights

ROOT = Path(__file__).resolve().parents[1]


def test_four_frozen_source_slots() -> None:
    archive = ROOT / "artifacts" / "p1-m5-formal-20261124"
    manifest = json.loads((archive / "study_manifest.json").read_text())
    models = _source_models(manifest, archive)
    assert tuple(models) == ("M0", "M1", "M2", "M3")
    for label, model in models.items():
        assert sha256(model.source.encode()).hexdigest() == MODEL_SLOTS[label][1]


@pytest.mark.parametrize("k", (1, 4, 8))
def test_profile_keeps_real_days(k: int) -> None:
    dates = [20170414, 20170417]
    profile = {
        "day_metrics": {str(day): {} for day in dates},
        "market_bootstrap": {"metric_schema_version": "p2-m1", "num_replicates": k,
                             "visible_mid": [[], []]},
    }
    registration = {"test_dates": dates}
    _validate_profile(profile, registration, k)
    profile["market_bootstrap"]["visible_mid"] += [[]]
    with pytest.raises(ValueError, match="additional day clusters"):
        _validate_profile(profile, registration, k)


def test_registration_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "registration.json"
    _seal(path, {"k": 4})
    _seal(path, {"k": 4})
    original = path.read_bytes()
    with pytest.raises(ValueError, match="registration differs"):
        _seal(path, {"k": 8})
    assert path.read_bytes() == original


def test_weights_match_exact_training_seed_panel(tmp_path: Path) -> None:
    digest = MODEL_SLOTS["M0"][1]
    root = tmp_path / "300001.SZ" / f"unused"
    folder = root.parent / "train" / f"{digest}-103-137-211"
    folder.mkdir(parents=True)
    for seed in (103, 137, 211):
        (folder / f"{seed}.pt").write_bytes(str(seed).encode())
    (folder / "fit.json").write_text(json.dumps({"fits": [{"seed": seed} for seed in (103, 137, 211)]}))
    chosen, hashes = _weights(tmp_path, digest)
    assert chosen == folder
    assert hashes == {str(seed): sha256(str(seed).encode()).hexdigest() for seed in (103, 137, 211)}
