"""Exercise the Registered holdout frozen holdout calendar using synthetic data and transport."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
from pyarrow import ArrowInvalid
import pytest

from stochopia.data import BookReconciliation, SZSEDaySlice
from tasks.lob_world_model import holdout_data


PRIMARY = (
    20170414, 20170417, 20170418, 20170419, 20170420, 20170421, 20170424,
    20170425, 20170426, 20170427, 20170428, 20170502, 20170503, 20170504,
    20170505, 20170508, 20170509, 20170510, 20170511, 20170512,
)
REPLACEMENTS = (
    20170515, 20170516, 20170517, 20170518, 20170519, 20170522, 20170523,
    20170524, 20170525, 20170526,
)
REVISION = "frozen-dataset-revision"
SYMBOLS = ("000001.SZ", "300001.SZ")


def _manifest() -> dict[str, Any]:
    return {
        "schema": holdout_data.REGISTERED_HOLDOUT_SCHEMA,
        "test_dates": list(PRIMARY), "replacement_test_dates": list(REPLACEMENTS),
        "data": {
            "test_dates": list(PRIMARY), "replacement_test_dates": list(REPLACEMENTS),
            "train_dates": [20170103], "dev_dates": [20170307, 20170412],
            "embargo_dates": [20170306, 20170323, 20170413],
            "symbols": list(SYMBOLS), "source_repo": "synthetic/l2", "source_revision": REVISION,
        },
    }


def _panel(date: int) -> dict[str, Any]:
    return {
        "reconciliation": [{"date": date, "symbol": stock, "n_events": 280} for stock in SYMBOLS],
        "anchors": [{"date": date, "symbol": stock, "session": session, "spread_ticks": 1}
                    for stock in SYMBOLS for session in (0, 1)],
    }


@pytest.fixture
def registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict[str, Any]]:
    artifact = tmp_path / "artifact"
    market = tmp_path / "market"
    market.mkdir()
    frozen = _manifest()
    (market / "manifest.json").write_text(json.dumps(frozen["data"]), encoding="utf-8")
    monkeypatch.setattr(holdout_data, "_registered_horizons", lambda root, train: {stock: 50 for stock in SYMBOLS})
    monkeypatch.setattr(holdout_data, "_build_registered_day", lambda root, date, horizons: _panel(date))
    return artifact, market, frozen


def test_holdout_download_requires_a_formal_seal(tmp_path: Path) -> None:
    def unexpected_download(root: Path, dates: tuple[int, ...], revision: str) -> None:
        pytest.fail("acquisition attempted before the study was frozen")

    def unsealed(_artifact: Path) -> dict[str, Any]:
        raise ValueError("complete and freeze before acquisition")

    with pytest.raises(ValueError, match="complete and freeze"):
        holdout_data.prepare_holdout_data(
            tmp_path / "unfrozen", tmp_path / "no-market",
            load_registration=unsealed, download=unexpected_download,
        )


def test_holdout_rejects_a_different_registered_replacement_calendar(
    registered: tuple[Path, Path, dict[str, Any]],
) -> None:
    artifact, market, frozen = registered
    changed = deepcopy(frozen["data"])
    changed["replacement_test_dates"][-1] = 20170531
    (market / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")

    def unexpected_download(root: Path, dates: tuple[int, ...], revision: str) -> None:
        pytest.fail("acquisition attempted with a different data registration")

    with pytest.raises(ValueError, match="registration differs"):
        holdout_data.prepare_holdout_data(artifact, market, load_registration=lambda _root: frozen, download=unexpected_download)


def test_holdout_replaces_only_excluded_days_in_pool_order(
    registered: tuple[Path, Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, market, frozen = registered
    downloaded = []
    failed = {PRIMARY[1]: "source_missing:table", REPLACEMENTS[0]: "source_missing:table"}

    def download(root: Path, dates: tuple[int, ...], revision: str) -> None:
        assert root == market and revision == REVISION
        assert len(dates) == 1
        date = dates[0]
        downloaded.append(date)
        if date in failed:
            raise holdout_data.HoldoutDayUnavailable(failed[date])

    def build(root: Path, date: int, horizons: dict[str, int]) -> dict[str, Any]:
        if date == PRIMARY[4]:
            raise holdout_data.HoldoutDayUnavailable("invalid_anchor:300001.SZ:session-1")
        return _panel(date)

    monkeypatch.setattr(holdout_data, "_build_registered_day", build)
    receipt_path = holdout_data.prepare_holdout_data(artifact, market, load_registration=lambda _root: frozen, download=download)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    excluded_dates = [PRIMARY[1], PRIMARY[4], REPLACEMENTS[0]]
    expected = tuple(date for date in (*PRIMARY, *REPLACEMENTS[:3]) if date not in excluded_dates)
    assert downloaded == list((*PRIMARY, *REPLACEMENTS[:3]))
    assert holdout_data.validate_holdout_receipt(frozen, receipt) == expected
    assert len(receipt["reconciliation"]) == 40 and len(receipt["anchors"]) == 80
    assert receipt["excluded_test_dates"] == [
        {"date": PRIMARY[1], "reason": failed[PRIMARY[1]]},
        {"date": PRIMARY[4], "reason": "invalid_anchor:300001.SZ:session-1"},
        {"date": REPLACEMENTS[0], "reason": failed[REPLACEMENTS[0]]},
    ]
    before = receipt_path.read_bytes()
    assert holdout_data.prepare_holdout_data(artifact, market, load_registration=lambda _root: frozen, download=download) == receipt_path
    assert downloaded == list((*PRIMARY, *REPLACEMENTS[:3]))
    assert receipt_path.read_bytes() == before


def test_transport_interruption_resumes_same_date_without_replacements(
    registered: tuple[Path, Path, dict[str, Any]],
) -> None:
    artifact, market, frozen = registered
    downloaded = []
    interrupted = False

    def download(root: Path, dates: tuple[int, ...], revision: str) -> None:
        nonlocal interrupted
        date = dates[0]
        downloaded.append(date)
        if date == PRIMARY[2] and not interrupted:
            interrupted = True
            raise TimeoutError("synthetic transport timeout")

    receipt_path = artifact / "verification" / "test-data-reconciliation.json"
    with pytest.raises(TimeoutError, match="transport timeout"):
        holdout_data.prepare_holdout_data(artifact, market, load_registration=lambda _root: frozen, download=download)
    assert not receipt_path.exists()
    progress = artifact / "verification" / "acquisition" / "progress.json"
    recorded = json.loads(progress.read_text(encoding="utf-8"))
    assert [row["date"] for row in recorded["attempts"]] == list(PRIMARY[:2])
    holdout_data.prepare_holdout_data(artifact, market, load_registration=lambda _root: frozen, download=download)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert downloaded == list((*PRIMARY[:3], *PRIMARY[2:]))
    assert holdout_data.validate_holdout_receipt(frozen, receipt) == PRIMARY
    assert receipt["excluded_test_dates"] == []


def test_receipt_validation_rejects_skipped_pool_dates_and_invalid_anchors(
    registered: tuple[Path, Path, dict[str, Any]],
) -> None:
    artifact, market, frozen = registered
    receipt_path = holdout_data.prepare_holdout_data(artifact, market, load_registration=lambda _root: frozen, download=lambda *args: None)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    invalid = deepcopy(receipt)
    invalid["test_dates"][-1] = REPLACEMENTS[1]
    invalid["excluded_test_dates"] = [{"date": PRIMARY[-1], "reason": "no_quotes:000001.SZ"}]
    with pytest.raises(ValueError, match="chronological replacement order"):
        holdout_data.validate_holdout_receipt(frozen, invalid)
    invalid = deepcopy(receipt)
    invalid["anchors"][0]["spread_ticks"] = 0
    with pytest.raises(ValueError, match="positive-spread anchors"):
        holdout_data.validate_holdout_receipt(frozen, invalid)
    invalid = deepcopy(receipt)
    invalid["anchors"][0] = invalid["anchors"][1]
    with pytest.raises(ValueError, match="positive-spread anchors"):
        holdout_data.validate_holdout_receipt(frozen, invalid)


def test_replacement_pool_exhaustion_never_opens_an_extra_date(
    registered: tuple[Path, Path, dict[str, Any]],
) -> None:
    artifact, market, frozen = registered
    downloaded = []

    def download(root: Path, dates: tuple[int, ...], revision: str) -> None:
        downloaded.extend(dates)
        if dates[0] in PRIMARY[:11]:
            raise holdout_data.HoldoutDayUnavailable("no_quotes:300001.SZ")

    with pytest.raises(ValueError, match="exhausted with 19/20"):
        holdout_data.prepare_holdout_data(artifact, market, load_registration=lambda _root: frozen, download=download)
    assert downloaded == list((*PRIMARY, *REPLACEMENTS))
    assert not (artifact / "verification" / "test-data-reconciliation.json").exists()


def _synthetic_day(stock: str, date: int, *, invalid_opening: bool = False) -> SZSEDaySlice:
    events = np.arange(280, dtype=np.int64)
    times = np.concatenate([37_800_000 - 1280 + events[:140] * 10,
                            50_310_000 - 1280 + events[:140] * 10])
    ticks = np.zeros((len(events), 2, 10), dtype=np.int64)
    ticks[:, 0, 0] = 99
    ticks[:, 1, 0] = 101
    shares = np.where(ticks > 0, 100, 0)
    if invalid_opening:
        ticks[268, 1, 0] = 99
    return SZSEDaySlice(
        wind_code=stock, trading_date=date, prev_close_ticks=100, session_bounds=((0, 140), (140, 280)),
        time_ms=times, seq_id=events, dt_ms=np.full(len(events), 10),
        action_type=np.zeros_like(events), rel_ticks=np.zeros_like(events),
        volume_shares=np.full(len(events), 100), cancel_age_ms=np.full(len(events), -1),
        pre_book_ticks=ticks, pre_book_shares=shares, mid_ticks_x2=ticks[:, :, 0].sum(axis=1),
        spread_ticks=ticks[:, 1, 0] - ticks[:, 0, 0],
    )


def test_day_build_uses_both_stocks_and_never_caches_a_single_valid_stock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = []
    valid = False

    def reconstruct(root: Path, stock: str) -> tuple[SZSEDaySlice, BookReconciliation]:
        day = _synthetic_day(stock, PRIMARY[0], invalid_opening=not valid and stock == SYMBOLS[1])
        return day, BookReconciliation(10, 1.0, 0.0, 0.0)

    def save(path: Path, day: SZSEDaySlice) -> None:
        cached.append((path, day.wind_code))

    monkeypatch.setattr(holdout_data, "reconstruct_szse_day", reconstruct)
    monkeypatch.setattr(holdout_data, "save_day_slice", save)
    horizons = {stock: 50 for stock in SYMBOLS}
    with pytest.raises(holdout_data.HoldoutDayUnavailable, match="invalid_anchor:300001.SZ:session-1"):
        holdout_data._build_registered_day(tmp_path, PRIMARY[0], horizons)
    assert cached == []
    valid = True
    evidence = holdout_data._build_registered_day(tmp_path, PRIMARY[0], horizons)
    assert len(evidence["reconciliation"]) == 2 and len(evidence["anchors"]) == 4
    assert [stock for _, stock in cached] == list(SYMBOLS)
    assert all(path.parent.name == str(PRIMARY[0]) for path, _ in cached)


def test_download_classifies_only_explicit_remote_entry_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EntryNotFoundError(Exception):
        def __init__(self, status: int | None) -> None:
            self.response = None if status is None else SimpleNamespace(status_code=status)

    hf = ModuleType("huggingface_hub")
    hf.__path__ = []
    errors = ModuleType("huggingface_hub.errors")
    errors.EntryNotFoundError = EntryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    fetcher = ModuleType("fetch_fast_m2")
    fetcher.__file__ = str(tmp_path / "fetch_fast_m2.py")
    fetcher.ROOT = tmp_path
    fetcher.SYMBOLS = SYMBOLS
    fetcher.REPO = "synthetic/l2"
    fetcher.COLUMNS = ("quotes",)
    failure: Exception = EntryNotFoundError(404)

    def fetch(day: str, table: str, revision: str, ranges: object) -> tuple[str, str, int, int, int]:
        assert day == str(PRIMARY[0]) and revision == REVISION
        raise failure

    fetcher.fetch = fetch
    monkeypatch.setitem(sys.modules, "fetch_fast_m2", fetcher)
    registered = _manifest()["data"]
    (tmp_path / "manifest.json").write_text(json.dumps(registered), encoding="utf-8")
    with pytest.raises(ValueError, match="registered dates and revision"):
        holdout_data._download(tmp_path, (20170531,), REVISION)
    with pytest.raises(holdout_data.HoldoutDayUnavailable, match="source_missing:20170414:quotes"):
        holdout_data._download(tmp_path, (PRIMARY[0],), REVISION)
    failure = ArrowInvalid("Must pass at least one table")
    with pytest.raises(holdout_data.HoldoutDayUnavailable, match="source_missing:20170414:quotes"):
        holdout_data._download(tmp_path, (PRIMARY[0],), REVISION)
    failure = ArrowInvalid("invalid parquet schema")
    with pytest.raises(ArrowInvalid, match="invalid parquet schema"):
        holdout_data._download(tmp_path, (PRIMARY[0],), REVISION)
    failure = EntryNotFoundError(None)
    with pytest.raises(EntryNotFoundError):
        holdout_data._download(tmp_path, (PRIMARY[0],), REVISION)
    failure = TimeoutError("transport")
    with pytest.raises(TimeoutError, match="transport"):
        holdout_data._download(tmp_path, (PRIMARY[0],), REVISION)
