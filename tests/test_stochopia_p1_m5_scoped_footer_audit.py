"""Scope-limited stock metadata inventory using only mocked file footers."""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.p1_m5_closing_data import scoped_footer_audit as scoped
from experiments.p1_m5_closing_data.reconcile import SYMBOLS, TABLE_FILES, metadata_inventory
from experiments.p1_m5_closing_data.footer_inventory import PRESENT, ABSENT, UNKNOWN

Json = dict[str, Any]


def _dates(start: date, count: int) -> list[int]:
    return [int((start + timedelta(days=i)).strftime("%Y%m%d")) for i in range(count)]


def _read(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Json) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def sealed_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Json]:
    """Seal a synthetic 90-slot manifest and valid 20/20 receipt; no data/cache access."""
    artifact = tmp_path / "artifact"
    verification = artifact / "verification"
    verification.mkdir(parents=True)
    receipt_path = verification / "test-data-reconciliation.json"
    test_dates = _dates(date(2017, 4, 14), 20)
    replacements = _dates(date(2017, 5, 14), 10)
    data = {"test_dates": test_dates, "replacement_test_dates": replacements,
            "source_repo": "audit/sample", "source_revision": "frozen-revision"}
    manifest = {"schema": scoped.M5_SCHEMA, "data": data,
                "test_dates": test_dates, "replacement_test_dates": replacements,
                "trajectories": {"trial-roster": {"trials": [0] * 90}}}
    monkeypatch.setattr(scoped, "load_frozen_study", lambda _: manifest)
    rows = []
    anchors = []
    for symbol in SYMBOLS:
        for day in test_dates:
            rows.append({"symbol": symbol, "date": day, "n_events": 100})
            for session in (0, 1):
                anchors.append({"symbol": symbol, "date": day, "session": session, "spread_ticks": 1})
    paths = [f"{day}/{name}" for day in (20170601, 20171201, 20180102) for name in TABLE_FILES]
    inventory = metadata_inventory(paths, data["source_repo"], data["source_revision"])
    receipt = {"test_dates": test_dates, "planned_test_dates": test_dates,
               "replacement_test_dates": replacements, "source_revision": data["source_revision"],
               "excluded_test_dates": [], "reconciliation": rows, "anchors": anchors,
               "remote_availability": inventory, "split_summary": [{"scope": "immutable test evidence"}]}
    _write(receipt_path, receipt)
    return artifact, receipt_path, receipt


def test_scoped_footer_audit_preserves_receipt_and_bounds_uninspected_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sealed_receipt: tuple[Path, Path, Json],
) -> None:
    artifact, receipt_path, original = sealed_receipt
    called: set[tuple[int, str]] = set()

    def fake_presence(repo: str, revision: str, day: int, table: str) -> tuple[int, str, dict[str, str], str | None]:
        assert (repo, revision) == ("audit/sample", "frozen-revision")
        assert day in (20170601, 20171201)
        called.add((day, table))
        statuses = {symbol: PRESENT for symbol in SYMBOLS}
        if day == 20171201 and table == "行情.parquet":
            statuses = {"000001.SZ": ABSENT, "300001.SZ": UNKNOWN}
        return day, table, statuses, None

    monkeypatch.setattr(scoped, "file_presence", fake_presence)
    progress = tmp_path / "progress.json"
    scoped.run(artifact, progress, through=20171231, workers=1)
    updated = _read(receipt_path)
    assert called == {(day, name) for day in (20170601, 20171201) for name in TABLE_FILES}
    assert len(_read(progress)["completed"]) == 6
    for key, expected in original.items():
        if key != "remote_availability":
            assert updated[key] == expected

    inventory = updated["remote_availability"]
    scope = inventory["footer_audit"]
    assert scope["complete_date_bundles_attempted"] == 2
    assert scope["table_footers_attempted"] == 6
    assert scope["table_footers_failed"] == 0
    assert scope["uninspected_after_scope"] == 1
    assert scope["by_stock"]["000001.SZ"]["trade_day_files"] == 1
    assert scope["by_stock"]["000001.SZ"]["absent_dates"] == [20171201]
    assert scope["by_stock"]["300001.SZ"]["unresolved_dates"] == [20171201]
    assert inventory["000001.SZ"]["possible_trade_day_files"] == 2
    assert inventory["300001.SZ"]["possible_trade_day_files"] == 3
    assert inventory["000001.SZ"]["unresolved_dates"] == [20180102]
    assert inventory["300001.SZ"]["unresolved_dates"] == [20171201, 20180102]


def test_scoped_footer_audit_refuses_concurrent_receipt_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sealed_receipt: tuple[Path, Path, Json],
) -> None:
    artifact, receipt_path, original = sealed_receipt

    def fake_presence(repo: str, revision: str, day: int, table: str) -> tuple[int, str, dict[str, str], str | None]:
        changed = _read(receipt_path)
        changed["reconciliation"][0]["n_events"] = 101
        _write(receipt_path, changed)
        return day, table, {symbol: PRESENT for symbol in SYMBOLS}, None

    monkeypatch.setattr(scoped, "file_presence", fake_presence)
    with pytest.raises(ValueError, match="Underlying receipt changed"):
        scoped.run(artifact, tmp_path / "progress.json", through=20171231, workers=1)
    updated = _read(receipt_path)
    assert updated["reconciliation"][0]["n_events"] == 101
    assert updated["remote_availability"] == original["remote_availability"]


def test_scoped_footer_audit_rejects_unregistered_scope_without_remote_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_call(*args: object) -> None:
        pytest.fail("invalid scope reached a remote fetch or frozen-manifest read")

    monkeypatch.setattr(scoped, "file_presence", must_not_call)
    monkeypatch.setattr(scoped, "load_frozen_study", must_not_call)
    with pytest.raises(ValueError, match="limited to June–December 2017"):
        scoped.run(tmp_path, tmp_path / "progress.json", through=20261130, workers=1)
