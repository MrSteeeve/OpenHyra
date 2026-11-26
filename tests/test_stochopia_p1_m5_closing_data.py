from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.p1_m5_closing_data import reconcile as module


def test_metadata_inventory_only_complete_dates(tmp_path: Path) -> None:
    complete = [f"{day}/{name}" for day in ("20170531", "20170601", "20170602", "20170703") for name in module.TABLE_FILES]
    paths = [*complete, complete[-1], "20170603/行情.parquet", "20170699/行情.parquet", "README.md"]
    inventory = module.metadata_inventory(iter(paths), "repo", "hash")
    assert inventory["market_wide_trade_day_files"] == 3
    assert inventory["market_wide_parquet_files"] == 9
    assert inventory["market_wide_month_counts"] == {"2017-06": 2, "2017-07": 1}
    assert inventory["market_wide_partial_dates"] == [20170603]
    assert inventory["market_wide_partial_parquet_files"] == 1
    assert inventory["by_stock"]["000001.SZ"]["trade_day_files"] is None
    assert inventory["by_stock"]["000001.SZ"]["status"] == "UNKNOWN_FROM_FILE_TREE"
    assert inventory["by_stock"]["300001.SZ"]["possible_market_dates"] == 3


def test_panel_weighting_and_window_count() -> None:
    rows = []
    anchors = []
    for symbol in module.SYMBOLS:
        for day, weight, value in [(20170103, 1, 0.5), (20170104, 3, 0.9)]:
            rows.append({"split": "train", "symbol": symbol, "date": day, "n_events": weight * 10,
                         "n_snapshots": weight, "touch_match_rate": value, "top10_log_vol_mae": value / 10})
            for session in (0, 1):
                anchors.append({"split": "train", "symbol": symbol, "date": day, "session": session,
                                "valid": day != 20170103 or session != 0})
    panel = module.summarize(rows, anchors, {"train": (20170103, 20170104)})
    for row in panel:
        assert row["n_events"] == 40
        assert row["n_snapshots"] == 4
        assert row["valid_bilateral_windows"] == 3
        assert row["planned_windows"] == 4
        assert row["touch_match_rate"] == pytest.approx(0.8)
        assert row["top10_log_vol_mae"] == pytest.approx(0.08)
    with pytest.raises(ValueError, match="Incomplete"):
        module.summarize(rows[:-1], anchors, {"train": (20170103, 20170104)})


def test_development_allowlist_refuses_future() -> None:
    registration = {"symbols": module.SYMBOLS, "train_dates": list(range(20170101, 20170137)),
                    "dev_dates": list(range(20170301, 20170324)) + [20170414]}
    with pytest.raises(ValueError, match="pre-holdout"):
        module.registered_development_dates(registration)


def test_remote_inventory_cannot_start_before_freeze(tmp_path: Path) -> None:
    def must_not_call(*args: object) -> None:
        raise AssertionError("Network list attempted before freeze")
    with pytest.raises(ValueError, match="freeze all development trials"):
        module.enrich_receipt(tmp_path, tmp_path / "development.json", list_paths=must_not_call)


def _day_panel(split: str, dates: tuple[int, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    anchors = []
    for symbol in module.SYMBOLS:
        for day in dates:
            rows.append({"split": split, "symbol": symbol, "date": day, "n_events": 100,
                         "n_snapshots": 10, "touch_match_rate": 0.9, "top10_log_vol_mae": 0.01})
            for session in (0, 1):
                anchors.append({"split": split, "symbol": symbol, "date": day, "session": session,
                                "valid": True, "spread_ticks": 1})
    return rows, anchors


def _synthetic_dates(start: date, count: int) -> tuple[int, ...]:
    return tuple(int((start + timedelta(days=j)).strftime("%Y%m%d")) for j in range(count))


def test_valid_receipt_is_enriched_idempotently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks.stochopia_p1 import formal_verify, holdout_data

    train_dates = _synthetic_dates(date(2017, 1, 1), 36)
    dev_dates = _synthetic_dates(date(2017, 3, 1), 24)
    test_dates = _synthetic_dates(date(2017, 4, 14), 20)
    replacements = _synthetic_dates(date(2017, 5, 14), 10)
    registration = {"source_repo": "audit/sample", "source_revision": "frozen-revision",
                    "symbols": list(module.SYMBOLS), "train_dates": list(train_dates),
                    "dev_dates": list(dev_dates), "test_dates": list(test_dates),
                    "replacement_test_dates": list(replacements)}
    manifest = {"schema": holdout_data.M5_SCHEMA, "data": registration, "test_dates": list(test_dates),
                "replacement_test_dates": list(replacements), "trajectories": {"synthetic": {"trials": [0] * 90}}}
    monkeypatch.setattr(formal_verify, "load_frozen_study", lambda _: manifest)

    train_rows, train_anchors = _day_panel("train", train_dates)
    dev_rows, dev_anchors = _day_panel("dev", dev_dates)
    development = {"source_revision": registration["source_revision"], "reconciliation": train_rows + dev_rows,
                   "anchors": train_anchors + dev_anchors, "method": "synthetic",
                   "split_summary": module.summarize(train_rows + dev_rows, train_anchors + dev_anchors,
                                                     {"train": train_dates, "dev": dev_dates})}
    development_file = tmp_path / "development.json"
    development_file.write_text(json.dumps(development))
    test_rows, test_anchors = _day_panel("test", test_dates)
    receipt = {"planned_test_dates": list(test_dates), "replacement_test_dates": list(replacements),
               "source_revision": registration["source_revision"], "test_dates": list(test_dates),
               "excluded_test_dates": [], "reconciliation": test_rows, "anchors": test_anchors}
    verification = tmp_path / "artifact" / "verification"
    receipt_file = verification / "test-data-reconciliation.json"
    verification.mkdir(parents=True)
    calls = []

    def list_paths(repo: str, revision: str) -> tuple[str, ...]:
        calls.append((repo, revision))
        return tuple(f"20170601/{table}" for table in module.TABLE_FILES)

    # The API must not be called for an incomplete holdout receipt.
    incomplete = {**receipt, "anchors": receipt["anchors"][:-1]}
    receipt_file.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="cover both"):
        module.enrich_receipt(verification.parent, development_file, list_paths=list_paths)
    assert not calls
    receipt_file.write_text(json.dumps(receipt))
    module.enrich_receipt(verification.parent, development_file, list_paths=list_paths)
    enriched = json.loads(receipt_file.read_text())
    assert calls == [("audit/sample", "frozen-revision")]
    for key, value in receipt.items():
        assert enriched[key] == value
    assert len(enriched["split_summary"]) == 6
    assert len(enriched["development_reconciliation"]) == 120
    assert len(enriched["development_anchors"]) == 240
    assert enriched["remote_availability"]["market_wide_trade_day_files"] == 1
    module.enrich_receipt(verification.parent, development_file, list_paths=list_paths)
    assert len(calls) == 1
    assert json.loads(receipt_file.read_text()) == enriched
