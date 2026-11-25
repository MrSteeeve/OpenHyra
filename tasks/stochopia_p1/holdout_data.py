"""Acquire and reconcile registered SZSE test slices after the formal seal."""

from __future__ import annotations

import argparse
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

from .formal_manifest import M5_SCHEMA, SYMBOLS
from .market_eval import _exclusive
from .store import validate_dates


_UNAVAILABLE_REASONS = frozenset({
    "source_missing", "no_quotes", "incomplete_day", "incomplete_window", "invalid_anchor",
})


class HoldoutDayUnavailable(ValueError):
    """A deterministic day-level data exclusion under the registered protocol."""


def _download(cache_root: Path, dates: tuple[int, ...], revision: str) -> None:
    """Reuse the M2 projected range reader with the exact sealed date allowlist."""
    registered = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
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
            or tuple(fetcher.SYMBOLS) != SYMBOLS or fetcher.REPO != registered["source_repo"]):
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
    from experiments.p1_m2_baseline import CHILDREN, _calibrate, _load_days, _windows

    rows = []
    anchors = []
    for symbol in symbols:
        fitted = _load_days(market, symbol, train_dates)
        horizon = 2 * CHILDREN * int(_calibrate(market, fitted, symbol)["spacing_ms"])
        del fitted
        for date in test_dates:
            day, report = reconstruct_szse_day(market / "raw" / str(date), symbol)
            if report is None or day.trading_date != date or len(day.session_bounds) != 2:
                raise ValueError(f"Incomplete registered SZSE day {date} for {symbol}")
            windows = _windows([encode_day(day)], horizon)
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


# --- M5 registered calendar and day evidence ---

def _m5_calendars(manifest: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    primary = tuple(manifest["test_dates"])
    replacements = tuple(manifest["replacement_test_dates"])
    if manifest["schema"] != M5_SCHEMA or len(primary) != 20 or len(replacements) != 10:
        raise ValueError("M5 requires its registered twenty-day and ten-day replacement calendars")
    if any(type(day) is not int for day in (*primary, *replacements)):
        raise ValueError("M5 calendar dates must be integers")
    validate_dates((*primary, *replacements), "M5 test and replacement dates")
    if (primary != tuple(manifest["data"]["test_dates"])
            or replacements != tuple(manifest["data"]["replacement_test_dates"])):
        raise ValueError("M5 holdout calendars differ from the frozen data registration")
    return primary, replacements


def _m5_horizons(market: Path, train_dates: list[int]) -> dict[str, int]:
    """Calibrate the registered injection horizon on training days once per stock."""
    from experiments.p1_m2_baseline import CHILDREN, _calibrate, _load_days

    horizons = {}
    for symbol in SYMBOLS:
        fitted = _load_days(market, symbol, train_dates)
        horizons[symbol] = 2 * CHILDREN * int(_calibrate(market, fitted, symbol)["spacing_ms"])
        del fitted
    return horizons


def _build_m5_day(market: Path, date: int, horizons: dict[str, int]) -> dict[str, Any]:
    """Accept a date only when both stocks have both registered positive-spread anchors."""
    from experiments.p1_m2_baseline import _windows

    rows = []
    anchors = []
    days: list[SZSEDaySlice] = []
    for symbol in SYMBOLS:
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
            windows = _windows([encode_day(day)], horizons[symbol])
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


def _validate_m5_panel(report: dict[str, Any], dates: tuple[int, ...]) -> None:
    """Check each date contributes once per stock and session, with positive spreads."""
    rows = report["reconciliation"]
    anchors = report["anchors"]
    expected_rows = {(symbol, date) for symbol in SYMBOLS for date in dates}
    expected_anchors = {(symbol, date, session) for symbol, date in expected_rows for session in (0, 1)}
    if (len(rows) != len(expected_rows) or len(anchors) != len(expected_anchors)
            or {(row["symbol"], row["date"]) for row in rows} != expected_rows
            or {(row["symbol"], row["date"], row["session"]) for row in anchors
                if row["spread_ticks"] > 0} != expected_anchors):
        raise ValueError("M5 reconciliation must cover both positive-spread anchors on each effective stock-day")


def _validate_exclusion_reason(reason: str) -> None:
    if not isinstance(reason, str):
        raise ValueError("a replacement date requires a deterministic day-exclusion reason")
    code, _, detail = reason.partition(":")
    if code not in _UNAVAILABLE_REASONS or not detail:
        raise ValueError("a replacement date requires a deterministic day-exclusion reason")


def validate_m5_receipt(manifest: dict[str, Any], receipt: dict[str, Any]) -> tuple[int, ...]:
    """Bind the effective receipt to the frozen calendar and chronological replacement rule.

    Twenty accepted dates and every excluded date must account for a prefix of
    the primary calendar plus its replacement pool. The evidence panel has
    2 x 20 stock-day rows and 2 x 20 x 2 session anchor rows.
    """
    primary, replacements = _m5_calendars(manifest)
    if (tuple(receipt["planned_test_dates"]) != primary
            or tuple(receipt["replacement_test_dates"]) != replacements
            or receipt["source_revision"] != manifest["data"]["source_revision"]):
        raise ValueError("M5 holdout receipt differs from the frozen data registration")
    dates = tuple(receipt["test_dates"])
    excluded = receipt["excluded_test_dates"]
    excluded_dates = tuple(row["date"] for row in excluded)
    if len(dates) != 20 or len(excluded_dates) > len(replacements):
        raise ValueError("M5 requires twenty accepted dates from the registered replacement pool")
    if any(type(day) is not int for day in (*dates, *excluded_dates)):
        raise ValueError("M5 receipt dates must be integers")
    consumed = (*primary, *replacements)[:len(dates) + len(excluded_dates)]
    if (dates != tuple(day for day in consumed if day not in excluded_dates)
            or excluded_dates != tuple(day for day in consumed if day not in dates)):
        raise ValueError("M5 holdout receipt violates chronological replacement order")
    for row in excluded:
        _validate_exclusion_reason(row["reason"])
    _validate_m5_panel(receipt, dates)
    return dates


# --- M5 resumable acquisition ---

def _m5_progress(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    primary, replacements = _m5_calendars(manifest)
    registration = {
        "planned_test_dates": list(primary), "replacement_test_dates": list(replacements),
        "source_revision": manifest["data"]["source_revision"],
    }
    if not path.exists():
        return {**registration, "attempts": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    if any(state[key] != value for key, value in registration.items()):
        raise ValueError("M5 acquisition progress differs from the frozen data registration")
    attempts = state["attempts"]
    if [row["date"] for row in attempts] != list((*primary, *replacements)[:len(attempts)]):
        raise ValueError("M5 acquisition progress must be a chronological prefix")
    accepted = 0
    for row in attempts:
        if accepted == 20:
            raise ValueError("M5 acquisition must stop at the twentieth valid date")
        if row["status"] == "accepted":
            _validate_m5_panel(row, (row["date"],))
            accepted += 1
        elif row["status"] == "excluded":
            _validate_exclusion_reason(row["reason"])
        else:
            raise ValueError("unknown M5 acquisition status")
    return state


def _prepare_m5(
    market: Path, manifest: dict[str, Any], acquisition: Path, receipt: Path,
    download: Callable[[Path, tuple[int, ...], str], None],
) -> None:
    """Download one sealed calendar day at a time and persist only completed decisions."""
    primary, replacements = _m5_calendars(manifest)
    progress_path = acquisition / "progress.json"
    progress = _m5_progress(progress_path, manifest)
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
                horizons = _m5_horizons(market, manifest["data"]["train_dates"])
            report = _build_m5_day(market, date, horizons)
        except HoldoutDayUnavailable as exc:
            reason = str(exc)
            _validate_exclusion_reason(reason)
            row = {"date": date, "status": "excluded", "reason": reason}
        else:
            _validate_m5_panel(report, (date,))
            row = {"date": date, "status": "accepted", **report}
            accepted += 1
        attempts.append(row)
        atomic_write_json(progress_path, progress)

    if accepted != 20:
        raise ValueError(f"M5 replacement pool exhausted with {accepted}/20 common valid dates")
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
    validate_m5_receipt(manifest, completed)
    atomic_write_json(receipt, completed)


def prepare_holdout_data(
    artifact_root: Path, market_root: Path, *,
    download: Callable[[Path, tuple[int, ...], str], None] = _download,
) -> Path:
    """Publish a receipt only after the sealed dataset's complete anchor audit."""
    from .formal_verify import load_frozen_study

    manifest = load_frozen_study(artifact_root)
    market = market_root.resolve()
    registered = json.loads((market / "manifest.json").read_text(encoding="utf-8"))
    dates = tuple(manifest["test_dates"])
    revision = manifest["data"]["source_revision"]
    train_dates = manifest["data"]["train_dates"]
    if (dates != tuple(registered["test_dates"]) or revision != registered["source_revision"]
            or train_dates != registered["train_dates"] or registered["symbols"] != ["000001.SZ", "300001.SZ"]):
        raise ValueError("market-data registration differs from the frozen study")
    m5 = manifest["schema"] == M5_SCHEMA
    if m5:
        _, replacements = _m5_calendars(manifest)
        if (replacements != tuple(registered["replacement_test_dates"])
                or manifest["data"]["dev_dates"] != registered["dev_dates"]
                or manifest["data"]["embargo_dates"] != registered["embargo_dates"]
                or manifest["data"]["source_repo"] != registered["source_repo"]):
            raise ValueError("M5 market-data registration differs from the frozen study")
    verification = artifact_root.resolve() / "verification"
    receipt = verification / "test-data-reconciliation.json"
    acquisition = verification / "acquisition"
    with _exclusive(acquisition):
        if receipt.exists():
            if m5:
                validate_m5_receipt(manifest, json.loads(receipt.read_text(encoding="utf-8")))
            return receipt
        if m5:
            _prepare_m5(market, manifest, acquisition, receipt, download)
        else:
            download(market, dates, revision)
            report = _build(market, registered["symbols"], train_dates, dates)
            atomic_write_json(receipt, {"test_dates": dates, "source_revision": revision, **report})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--stochopia-root", type=Path, default=Path(__file__).resolve().parents[3] / "Stochopia")
    args = parser.parse_args()
    from .formal import git_code_lock
    from .formal_verify import load_frozen_study

    manifest = load_frozen_study(args.artifact_root)
    if manifest["schema"] == M5_SCHEMA and manifest["code"] != git_code_lock(args.stochopia_root):
        raise ValueError("holdout acquisition requires the frozen M5 code")
    print(prepare_holdout_data(args.artifact_root, args.market_root))


if __name__ == "__main__":
    main()
