"""Integrity and registered numerical roster checks on retained public evidence."""

from hashlib import sha256
import csv
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "artifacts"


# The validation bundle predates SHA256SUMS. These are the original bytes from
# Git tree 9e3a28dc846f90fe6ce53638ac1cdfe5b918b923, without altering that tree.
VALIDATION_SHA256 = {
    "README.md": "dd8c11df34224da519ba4e11c76279324728d2223a2853cbb7d762539d8863fa",
    "legacy-compatibility.json": "6e5f4da3eab8aa9dd8b971f2efaf25aea8b1b0568e667f419ade49bc656b300c",
    "registration.json": "53d5320215897304e6e7580fba7f77724e165c7401f495c42d034572929abe9b",
    "regression.json": "b1ea07bb0b72eeb23a68f4f81076d79b4a59e4d1d9bd102a819afc97ec19255c",
    "report.json": "942558aff9fd76aaffe0b370ec3ac739ff5293d0ced047b1cf6f84f625cad352",
}


def test_frozen_validation_files_match_original_tree_bytes() -> None:
    folder = ROOT / "p2-m1-validation"
    assert {path.name: sha256(path.read_bytes()).hexdigest() for path in folder.iterdir()} == VALIDATION_SHA256


def test_frozen_synthetic_receipts_match_every_file() -> None:
    for name in ("p2-m2-synthetic", "p2-m3-isomorphic-control", "p2-m31-calibrated-control", "p2-m32-open-catalog-control"):
        folder = ROOT / name
        for line in (folder / "SHA256SUMS").read_text().splitlines():
            digest, path = line.split(maxsplit=1)
            assert sha256((folder / path.strip()).read_bytes()).hexdigest() == digest


def test_synthetic_roster_and_hypothesis_boundaries_are_retained() -> None:
    folder = ROOT / "p2-m2-synthetic"
    with (folder / "experiment_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert Counter(row["experiment"] for row in rows) == {"E1": 36, "E2": 24, "E3": 18}
    assert len({row["job_id"] for row in rows}) == 78
    assert all(row["state_scalars"] == "60" for row in rows)
    laws = json.loads((folder / "frozen_synthetic_laws.json").read_text())
    assert laws["H1"]["status"] == laws["H2"]["status"] == "INCONCLUSIVE"
    assert laws["H3"]["status"] == "SUPPORTED_WITH_BOUNDARY"
