"""Acquire and reconcile registered SZSE test slices after the formal seal."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Any

from stochopia.common import atomic_write_json
from stochopia.data import SZSEDaySlice, reconstruct_szse_day, save_day_slice
from stochopia.neural.lob_training import encode_day

from .market_eval import _exclusive
from .store import validate_dates


SYMBOLS = ("000001.SZ", "300001.SZ")
REGISTERED_HOLDOUT_SCHEMA = "stochopia-p1-m5-v1"  # Immutable identifier of the registered slice schema.
REGISTERED_M4_HOLDOUT_SCHEMA = "stochopia-p2-m4-v1"
_REPLACEMENT_SCHEMAS = {
    REGISTERED_HOLDOUT_SCHEMA: 10,
    REGISTERED_M4_HOLDOUT_SCHEMA: 8,
}


def _manifest_symbols(record: dict[str, Any]) -> tuple[str, ...]:
    source = record["data"] if "data" in record and isinstance(record["data"], dict) else record
    raw = source.get("symbols", SYMBOLS)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("registered symbols must be a non-empty list of .SZ stock codes")
    symbols = tuple(raw)
    if any(not isinstance(s, str) or not s.endswith(".SZ") for s in symbols) or len(set(symbols)) != len(symbols):
        raise ValueError("registered symbols must be distinct .SZ stock codes")
    return symbols


_UNAVAILABLE_REASONS = frozenset({
    "source_missing", "no_quotes", "incomplete_day", "incomplete_window", "invalid_anchor",
})


class HoldoutDayUnavailable(ValueError):
    """A deterministic day-level data exclusion under the registered protocol."""


def _download(cache_root: Path, dates: tuple[int, ...], revision: str) -> None:
    """Reuse the M2 projected range reader with the exact sealed date allowlist."""
    registered = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    expected_symbols = _manifest_symbols(registered)
    allowed = (*registered["test_dates"], *registered.get("replacement_test_dates", ()))
    if revision != registered["source_revision"] or not dates or not set(dates).issubset(allowed):
        raise ValueError("projected holdout acquisition must use only registered dates and revision")
    sys.path.insert(0, str(cache_root))
    try:
        fetcher = importlib.import_module("fetch_fast_m2")
    finally:
        sys.path.pop(0)
    if (Path(fetcher.__file__).resolve() != cache_root / "fetch_fast_m2.py"
            or Path(fetcher.ROOT).resolve() != cache_root
            or tuple(fetcher.SYMBOLS) != expected_symbols or fetcher.REPO != registered["source_repo"]):
        raise ValueError("the projected downloader must match the registered market-data root and symbols")

    from huggingface_hub.errors import EntryNotFoundError
    from pyarrow import ArrowInvalid

    missing = []
    with ThreadPoolExecutor(max_workers=24) as ranges, ThreadPoolExecutor(max_workers=6) as files:
        jobs = {files.submit(fetcher.fetch, str(day), table, revision, ranges): (day, table)
                for day in dates for table in fetcher.COLUMNS}
        for job in as_completed(jobs):
            try:
                day, table, rows, _, _ = job.result()
            except EntryNotFoundError as exc:
                response = getattr(exc, "response", None)
                # A missing remote file is deterministic. Connection failures,
                # absent local caches, and repository/revision errors are not.
                if response is None or response.status_code != 404:
                    raise
                missing.append(jobs[job])
            except ArrowInvalid as exc:
                # The projected reader concatenates its selected row groups.
                # An empty group list means neither registered stock is present.
                if str(exc) != "Must pass at least one table":
                    raise
                missing.append(jobs[job])
            else:
                if rows == 0:
                    missing.append(jobs[job])
                print(f"Holdout {day} {table}: {rows}", flush=True)
    if missing:
        first_date, first_table = min(missing)
        raise HoldoutDayUnavailable(f"source_missing:{first_date}:{first_table}")


def _build(market: Path, symbols: list[str], train_dates: list[int], test_dates: tuple[int, ...]) -> dict[str, Any]:
    """Reconstruct the two symbols and check both formal anchors on every test day."""
    from stochopia.market.windows import CHILDREN, calibrate_market, load_market_days, select_windows

    rows = []
    anchors = []
    for symbol in symbols:
        fitted = load_market_days(market, symbol, train_dates)
        horizon = 2 * CHILDREN * int(calibrate_market(market, fitted, symbol)["spacing_ms"])
        del fitted
        for date in test_dates:
            day, report = reconstruct_szse_day(market / "raw" / str(date), symbol)
            if report is None or day.trading_date != date or len(day.session_bounds) != 2:
                raise ValueError(f"Incomplete registered SZSE day {date} for {symbol}")
            windows = select_windows([encode_day(day)], horizon)
            for window in windows:
                bid, ask = map(int, day.pre_book_ticks[window.start, :, 0])
                bid_shares, ask_shares = map(int, day.pre_book_shares[window.start, :, 0])
                if not (0 < bid < ask and bid_shares > 0 and ask_shares > 0):
                    raise ValueError(f"Invalid registered opening book {date} {symbol} session {window.session}")
                anchors.append({"symbol": symbol, "date": date, "session": window.session,
                                "seq_id": int(day.seq_id[window.start]), "spread_ticks": ask - bid})
            save_day_slice(market / "cache" / str(date) / f"{symbol}.npz", day)
            readings = {name: None if isinstance(value, float) and not isfinite(value) else value
                        for name, value in asdict(report).items()}
            rows.append({"date": date, "symbol": symbol, "n_events": len(day.seq_id), **readings})
    return {"reconciliation": rows, "anchors": anchors}


# --- Registered holdout registered calendar and day evidence ---

def _replacement_calendars(manifest: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    primary = tuple(manifest["test_dates"])
    replacements = tuple(manifest["replacement_test_dates"])
    expected_replacements = _REPLACEMENT_SCHEMAS.get(manifest.get("schema"))
    if expected_replacements is None or len(primary) != 20 or len(replacements) != expected_replacements:
        raise ValueError("Registered holdout requires its registered twenty-day and replacement calendars")
    if any(type(day) is not int for day in (*primary, *replacements)):
        raise ValueError("Registered holdout calendar dates must be integers")
    validate_dates((*primary, *replacements), "Registered holdout test and replacement dates")
    if (primary != tuple(manifest["data"]["test_dates"])
            or replacements != tuple(manifest["data"]["replacement_test_dates"])):
        raise ValueError("Registered holdout holdout calendars differ from the frozen data registration")
    return primary, replacements


def _registered_horizons(
    market: Path, train_dates: list[int], symbols: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Calibrate the registered injection horizon on training days once per stock."""
    from stochopia.market.windows import CHILDREN, calibrate_market, load_market_days

    if symbols is None:
        manifest_path = market / "manifest.json"
        symbols = (_manifest_symbols(json.loads(manifest_path.read_text(encoding="utf-8")))
                   if manifest_path.exists() else SYMBOLS)
    horizons = {}
    for symbol in symbols:
        fitted = load_market_days(market, symbol, train_dates)
        horizons[symbol] = 2 * CHILDREN * int(calibrate_market(market, fitted, symbol)["spacing_ms"])
        del fitted
    return horizons


def _build_registered_day(market: Path, date: int, horizons: dict[str, int]) -> dict[str, Any]:
    """Accept a date only when all registered stocks have both positive-spread anchors."""
    from stochopia.market.windows import select_windows

    if not horizons:
        raise ValueError("registered horizons must be non-empty")
    rows = []
    anchors = []
    days: list[SZSEDaySlice] = []
    for symbol in horizons:
        try:
            day, report = reconstruct_szse_day(market / "raw" / str(date), symbol)
        except ValueError as exc:
            if str(exc).startswith(f"No quotes for {symbol} "):
                raise HoldoutDayUnavailable(f"no_quotes:{symbol}") from exc
            raise
        if (report is None or report.n_snapshots == 0 or day.wind_code != symbol
                or day.trading_date != date or len(day.session_bounds) != 2):
            raise HoldoutDayUnavailable(f"incomplete_day:{symbol}")
        try:
            windows = select_windows([encode_day(day)], horizons[symbol])
        except ValueError as exc:
            if str(exc).startswith(f"Incomplete dev window for {symbol} {date} "):
                raise HoldoutDayUnavailable(f"incomplete_window:{symbol}") from exc
            raise
        if len(windows) != 2:
            raise HoldoutDayUnavailable(f"incomplete_window:{symbol}")
        for window in windows:
            bid, ask = map(int, day.pre_book_ticks[window.start, :, 0])
            bid_shares, ask_shares = map(int, day.pre_book_shares[window.start, :, 0])
            if not (0 < bid < ask and bid_shares > 0 and ask_shares > 0):
                raise HoldoutDayUnavailable(f"invalid_anchor:{symbol}:session-{window.session}")
            anchors.append({"symbol": symbol, "date": date, "session": window.session,
                            "seq_id": int(day.seq_id[window.start]), "spread_ticks": ask - bid})
        readings = {name: None if isinstance(value, float) and not isfinite(value) else value
                    for name, value in asdict(report).items()}
        rows.append({"date": date, "symbol": symbol, "n_events": len(day.seq_id), **readings})
        days.append(day)
    for day in days:
        save_day_slice(market / "cache" / str(date) / f"{day.wind_code}.npz", day)
    return {"reconciliation": rows, "anchors": anchors}


def _validate_anchor_panel(
    report: dict[str, Any], dates: tuple[int, ...], symbols: tuple[str, ...] = SYMBOLS,
) -> None:
    """Check each date contributes once per stock and session, with positive spreads."""
    rows = report["reconciliation"]
    anchors = report["anchors"]
    expected_rows = {(symbol, date) for symbol in symbols for date in dates}
    expected_anchors = {(symbol, date, session) for symbol, date in expected_rows for session in (0, 1)}
    if (len(rows) != len(expected_rows) or len(anchors) != len(expected_anchors)
            or {(row["symbol"], row["date"]) for row in rows} != expected_rows
            or {(row["symbol"], row["date"], row["session"]) for row in anchors
                if row["spread_ticks"] > 0} != expected_anchors):
        raise ValueError("Registered holdout reconciliation must cover both positive-spread anchors on each effective stock-day")


def _validate_exclusion_reason(reason: str) -> None:
    if not isinstance(reason, str):
        raise ValueError("a replacement date requires a deterministic day-exclusion reason")
    code, _, detail = reason.partition(":")
    if code not in _UNAVAILABLE_REASONS or not detail:
        raise ValueError("a replacement date requires a deterministic day-exclusion reason")


def validate_holdout_receipt(manifest: dict[str, Any], receipt: dict[str, Any]) -> tuple[int, ...]:
    """Bind the effective receipt to the frozen calendar and chronological replacement rule.

    Twenty accepted dates and every excluded date must account for a prefix of
    the primary calendar plus its replacement pool. The evidence panel has
    N x 20 stock-day rows and N x 20 x 2 session anchor rows.
    """
    primary, replacements = _replacement_calendars(manifest)
    symbols = _manifest_symbols(manifest)
    if (tuple(receipt["planned_test_dates"]) != primary
            or tuple(receipt["replacement_test_dates"]) != replacements
            or receipt["source_revision"] != manifest["data"]["source_revision"]):
        raise ValueError("Registered holdout holdout receipt differs from the frozen data registration")
    dates = tuple(receipt["test_dates"])
    excluded = receipt["excluded_test_dates"]
    excluded_dates = tuple(row["date"] for row in excluded)
    if len(dates) != 20 or len(excluded_dates) > len(replacements):
        raise ValueError("Registered holdout requires twenty accepted dates from the registered replacement pool")
    if any(type(day) is not int for day in (*dates, *excluded_dates)):
        raise ValueError("Registered holdout receipt dates must be integers")
    consumed = (*primary, *replacements)[:len(dates) + len(excluded_dates)]
    if (dates != tuple(day for day in consumed if day not in excluded_dates)
            or excluded_dates != tuple(day for day in consumed if day not in dates)):
        raise ValueError("Registered holdout holdout receipt violates chronological replacement order")
    for row in excluded:
        _validate_exclusion_reason(row["reason"])
    _validate_anchor_panel(receipt, dates, symbols)
    return dates


# --- Registered holdout resumable acquisition ---

def _acquisition_progress(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    primary, replacements = _replacement_calendars(manifest)
    symbols = _manifest_symbols(manifest)
    registration = {
        "planned_test_dates": list(primary), "replacement_test_dates": list(replacements),
        "source_revision": manifest["data"]["source_revision"],
    }
    if not path.exists():
        return {**registration, "attempts": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    if any(state[key] != value for key, value in registration.items()):
        raise ValueError("Registered holdout acquisition progress differs from the frozen data registration")
    attempts = state["attempts"]
    if [row["date"] for row in attempts] != list((*primary, *replacements)[:len(attempts)]):
        raise ValueError("Registered holdout acquisition progress must be a chronological prefix")
    accepted = 0
    for row in attempts:
        if accepted == 20:
            raise ValueError("Registered holdout acquisition must stop at the twentieth valid date")
        if row["status"] == "accepted":
            _validate_anchor_panel(row, (row["date"],), symbols)
            accepted += 1
        elif row["status"] == "excluded":
            _validate_exclusion_reason(row["reason"])
        else:
            raise ValueError("unknown Registered holdout acquisition status")
    return state


def _prepare_with_replacements(
    market: Path, manifest: dict[str, Any], acquisition: Path, receipt: Path,
    download: Callable[[Path, tuple[int, ...], str], None],
) -> None:
    """Download one sealed calendar day at a time and persist only completed decisions."""
    primary, replacements = _replacement_calendars(manifest)
    symbols = _manifest_symbols(manifest)
    progress_path = acquisition / "progress.json"
    progress = _acquisition_progress(progress_path, manifest)
    attempts = progress["attempts"]
    accepted = sum(row["status"] == "accepted" for row in attempts)
    revision = manifest["data"]["source_revision"]
    horizons: dict[str, int] | None = None
    for date in (*primary, *replacements)[len(attempts):]:
        if accepted == 20:
            break
        try:
            download(market, (date,), revision)
            if horizons is None:
                horizons = _registered_horizons(market, manifest["data"]["train_dates"])
            report = _build_registered_day(market, date, horizons)
        except HoldoutDayUnavailable as exc:
            reason = str(exc)
            _validate_exclusion_reason(reason)
            row = {"date": date, "status": "excluded", "reason": reason}
        else:
            _validate_anchor_panel(report, (date,), symbols)
            row = {"date": date, "status": "accepted", **report}
            accepted += 1
        attempts.append(row)
        atomic_write_json(progress_path, progress)

    if accepted != 20:
        raise ValueError(f"Registered holdout replacement pool exhausted with {accepted}/20 common valid dates")
    effective = []
    excluded = []
    rows = []
    anchors = []
    for attempt in attempts:
        if attempt["status"] == "accepted":
            effective.append(attempt["date"])
            rows.extend(attempt["reconciliation"])
            anchors.extend(attempt["anchors"])
        else:
            excluded.append({"date": attempt["date"], "reason": attempt["reason"]})
    completed = {
        "test_dates": effective, "planned_test_dates": primary, "replacement_test_dates": replacements,
        "excluded_test_dates": excluded, "source_revision": revision,
        "reconciliation": rows, "anchors": anchors,
    }
    validate_holdout_receipt(manifest, completed)
    atomic_write_json(receipt, completed)


def prepare_holdout_data(
    artifact_root: Path, market_root: Path, *,
    load_registration: Callable[[Path], dict[str, Any]],
    download: Callable[[Path, tuple[int, ...], str], None] = _download,
) -> Path:
    """Acquire only the sealed calendar returned by the caller's registration verifier.

    The caller owns the study seal and code-lock check. This function consumes
    that verified registration before opening the market data or invoking the
    downloader, and checks the local data manifest against it.
    """
    manifest = load_registration(artifact_root)
    market = market_root.resolve()
    registered = json.loads((market / "manifest.json").read_text(encoding="utf-8"))
    dates = tuple(manifest["test_dates"])
    revision = manifest["data"]["source_revision"]
    train_dates = manifest["data"]["train_dates"]
    expected_symbols = _manifest_symbols(manifest)
    registered_symbols = _manifest_symbols(registered)
    if (dates != tuple(registered["test_dates"]) or revision != registered["source_revision"]
            or train_dates != registered["train_dates"] or registered_symbols != expected_symbols):
        raise ValueError("market-data registration differs from the frozen study")
    has_replacement_pool = manifest.get("schema") in _REPLACEMENT_SCHEMAS
    if has_replacement_pool:
        _, replacements = _replacement_calendars(manifest)
        if (replacements != tuple(registered["replacement_test_dates"])
                or manifest["data"]["dev_dates"] != registered["dev_dates"]
                or manifest["data"].get("embargo_dates", []) != registered.get("embargo_dates", [])
                or manifest["data"]["source_repo"] != registered["source_repo"]):
            raise ValueError("Registered holdout market-data registration differs from the frozen study")
    verification = artifact_root.resolve() / "verification"
    receipt = verification / "test-data-reconciliation.json"
    acquisition = verification / "acquisition"
    with _exclusive(acquisition):
        if receipt.exists():
            if has_replacement_pool:
                validate_holdout_receipt(manifest, json.loads(receipt.read_text(encoding="utf-8")))
            return receipt
        if has_replacement_pool:
            _prepare_with_replacements(market, manifest, acquisition, receipt, download)
        else:
            download(market, dates, revision)
            report = _build(market, list(registered_symbols), train_dates, dates)
            atomic_write_json(receipt, {"test_dates": dates, "source_revision": revision, **report})
    return receipt
