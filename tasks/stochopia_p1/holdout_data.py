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
from stochopia.data import reconstruct_szse_day, save_day_slice
from stochopia.neural.lob_training import encode_day

from .formal_verify import load_frozen_study
from .market_eval import _exclusive


def _download(cache_root: Path, dates: tuple[int, ...], revision: str) -> None:
    """Reuse the M2 projected range reader with the exact sealed date allowlist."""
    sys.path.insert(0, str(cache_root))
    try:
        fetcher = importlib.import_module("fetch_fast_m2")
    finally:
        sys.path.pop(0)
    if Path(fetcher.__file__).resolve() != cache_root / "fetch_fast_m2.py":
        raise ValueError("the projected downloader must be the one in the market-data root")
    with ThreadPoolExecutor(max_workers=24) as ranges, ThreadPoolExecutor(max_workers=6) as files:
        jobs = [files.submit(fetcher.fetch, str(day), table, revision, ranges)
                for day in dates for table in fetcher.COLUMNS]
        for job in as_completed(jobs):
            day, table, rows, _, _ = job.result()
            print(f"Holdout {day} {table}: {rows}", flush=True)


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


def prepare_holdout_data(
    artifact_root: Path, market_root: Path, *,
    download: Callable[[Path, tuple[int, ...], str], None] = _download,
) -> Path:
    """Publish a receipt only after the sealed dataset's complete anchor audit."""
    manifest = load_frozen_study(artifact_root)
    market = market_root.resolve()
    registered = json.loads((market / "manifest.json").read_text(encoding="utf-8"))
    dates = tuple(manifest["test_dates"])
    revision = manifest["data"]["source_revision"]
    train_dates = manifest["data"]["train_dates"]
    if (dates != tuple(registered["test_dates"]) or revision != registered["source_revision"]
            or train_dates != registered["train_dates"] or registered["symbols"] != ["000001.SZ", "300001.SZ"]):
        raise ValueError("market-data registration differs from the frozen study")
    verification = artifact_root.resolve() / "verification"
    receipt = verification / "test-data-reconciliation.json"
    with _exclusive(verification / "acquisition"):
        if receipt.exists():
            return receipt
        download(market, dates, revision)
        report = _build(market, registered["symbols"], train_dates, dates)
        atomic_write_json(receipt, {"test_dates": dates, "source_revision": revision, **report})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_holdout_data(args.artifact_root, args.market_root))


if __name__ == "__main__":
    main()
