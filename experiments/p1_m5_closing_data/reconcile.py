"""M5 closing reconciliation: development-only audit and post-freeze repository inventory.

The development subcommand reads only the 120 registered train/dev stock-days. The
inventory subcommand refuses to call the Hub before the validated 90-trial M5 seal
and receipt exist. Inventory lists *file metadata*, never reads a parquet data page.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
from math import isfinite
from pathlib import Path, PurePosixPath
from collections.abc import Callable, Iterable
from typing import Any
import json

Json = dict[str, Any]
SYMBOLS = ("000001.SZ", "300001.SZ")
TABLE_FILES = frozenset(("行情.parquet", "逐笔委托.parquet", "逐笔成交.parquet"))
LAST_DEV_DAY = 20170413
REMOTE_AFTER = 20170601


# --- Train/dev read-only supplementary audit ---

def registered_development_dates(registration: Json) -> dict[str, tuple[int, ...]]:
    """Return the exact M5 train/dev allowlist; reject all new holdout paths."""
    dates = {split: tuple(registration[f"{split}_dates"]) for split in ("train", "dev")}
    if tuple(registration["symbols"]) != SYMBOLS or tuple(map(len, dates.values())) != (36, 24):
        raise ValueError("Expected the registered 36/24 two-stock M5 development split")
    combined = (*dates["train"], *dates["dev"])
    if any(type(day) is not int or day > LAST_DEV_DAY for day in combined):
        raise ValueError("This command only reads distinct ordered pre-holdout stock-days")
    if tuple(sorted(set(combined))) != combined:
        raise ValueError("This command only reads distinct ordered pre-holdout stock-days")
    return dates


def _previous_rows(previous: Path, market: Path, dates: dict[str, tuple[int, ...]]) -> dict[tuple[str, int], Json]:
    """Reuse prior reconciliation only for the identical underlying raw files."""
    rows = json.loads(previous.read_text(encoding="utf-8"))
    allowed = set((*dates["train"], *dates["dev"]))
    result = {}
    for row in rows:
        symbol, day = row["symbol"], row["date"]
        if symbol not in SYMBOLS or day not in allowed:
            continue
        same = all(
            (previous.parent / "raw" / str(day) / name).samefile(market / "raw" / str(day) / name)
            for name in TABLE_FILES
        )
        if not same:
            continue
        key = (symbol, day)
        if key in result:
            raise ValueError(f"Duplicate prior reconciliation for {key}")
        result[key] = row
    return result


def _anchor_rows(day: Any, split: str, horizon: int) -> list[Json]:
    """Evaluate both registered midpoint/context/horizon windows without rollout.

    The start/end calculations exactly mirror experiments.p1_m2_baseline._windows.
    A usable anchor additionally needs 0 < bid < ask and both touch depths > 0.
    Train anchors are descriptive; no formal training rollout is implied.
    """
    import numpy as np
    from experiments.p1_m2_baseline import CONTEXT, WINDOW_MIDPOINTS

    rows = []
    if len(day.session_bounds) != 2:
        raise ValueError(f"Unexpected session count on {day.wind_code} {day.trading_date}")
    for session, (start, end) in enumerate(day.session_bounds):
        position = max(start + CONTEXT, int(np.searchsorted(day.time_ms, WINDOW_MIDPOINTS[session])))
        cutoff = int(np.searchsorted(day.time_ms, day.time_ms[position] + horizon, side="right")) if position < end else end
        valid = False
        spread = None
        reason = "incomplete_window"
        if position < cutoff < end:
            bid, ask = map(int, day.pre_book_ticks[position, :, 0])
            bid_shares, ask_shares = map(int, day.pre_book_shares[position, :, 0])
            spread = ask - bid
            valid = 0 < bid < ask and min(bid_shares, ask_shares) > 0
            reason = None if valid else "invalid_anchor"
        rows.append({"symbol": day.wind_code, "date": day.trading_date, "split": split,
                     "session": session, "valid": valid, "spread_ticks": spread, "reason": reason})
    return rows


def _weighted(rows: list[Json], metric: str) -> float | None:
    """Snapshot-weighted mean of stock-day reconciliation readouts.

    Touch match is the mean of bid&ask matches per quote snapshot. Top-10 log-depth
    MAE is the mean of 20 absolute log1p depth errors per snapshot. Both have equal
    per-snapshot weight; unavailable days with zero snapshots contribute no weight.
    """
    n = 0
    total = 0.0
    for row in rows:
        count = row["n_snapshots"]
        value = row[metric]
        if count <= 0:
            continue
        if not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"Reconciliation metric missing despite {count} snapshots: {metric}")
        n += count
        total += count * value
    return total / n if n else None


def summarize(rows: list[Json], anchors: list[Json], dates: dict[str, tuple[int, ...]]) -> list[Json]:
    """Aggregate exact split/stock panels and reject missing/duplicate evidence."""
    expected_rows = sum(len(split_dates) * len(SYMBOLS) for split_dates in dates.values())
    if len(rows) != expected_rows or len(anchors) != 2 * expected_rows:
        raise ValueError("Incomplete or extraneous split reconciliation panel")
    result = []
    for split, split_dates in dates.items():
        for symbol in SYMBOLS:
            panel = [r for r in rows if r["split"] == split and r["symbol"] == symbol]
            windows = [r for r in anchors if r["split"] == split and r["symbol"] == symbol]
            if len(panel) != len(split_dates) or {r["date"] for r in panel} != set(split_dates):
                raise ValueError(f"Incomplete {split} {symbol} reconciliation panel")
            expected = {(day, session) for day in split_dates for session in (0, 1)}
            if len(windows) != len(expected) or {(a["date"], a["session"]) for a in windows} != expected:
                raise ValueError(f"Incomplete {split} {symbol} window panel")
            valid = sum(bool(r["valid"]) for r in windows)
            result.append({"split": split, "symbol": symbol, "trading_days": len(panel),
                           "n_events": sum(r["n_events"] for r in panel),
                           "n_snapshots": sum(r["n_snapshots"] for r in panel),
                           "valid_bilateral_windows": valid, "planned_windows": len(windows),
                           "touch_match_rate": _weighted(panel, "touch_match_rate"),
                           "top10_log_vol_mae": _weighted(panel, "top10_log_vol_mae")})
    return result


def development_audit(market: Path, previous: Path, output: Path) -> None:
    """Read registered local stock-days, reuse identical prior rows, and reconstruct missing rows."""
    import numpy as np
    import torch
    from experiments.p1_m2_baseline import CHILDREN, PARTICIPATION, SESSION_DURATION_MS
    from stochopia.common import atomic_write_json
    from stochopia.data import load_day_slice, reconstruct_szse_day

    torch.set_num_threads(1)
    registration = json.loads((market / "manifest.json").read_text(encoding="utf-8"))
    dates = registered_development_dates(registration)
    old = _previous_rows(previous, market, dates)
    rows = []
    anchors = []
    horizons = {}
    for symbol in SYMBOLS:
        # The spacing formula in _calibrate depends only on this aggressive-event count.
        aggressive = 0
        for day in dates["train"]:
            data = load_day_slice(market / "cache" / str(day) / f"{symbol}.npz")
            aggressive += int(np.count_nonzero((data.action_type == 2) | (data.action_type == 3)))
        frequency = aggressive / (len(dates["train"]) * SESSION_DURATION_MS)
        if not frequency:
            raise ValueError(f"No training aggressive events for {symbol}")
        spacing = 10 * max(1, round(1.0 / (10 * frequency * PARTICIPATION)))
        horizon = 2 * CHILDREN * spacing
        horizons[symbol] = horizon
        for split, split_dates in dates.items():
            for day in split_dates:
                cached = load_day_slice(market / "cache" / str(day) / f"{symbol}.npz")
                if (cached.wind_code, cached.trading_date) != (symbol, day):
                    raise ValueError(f"Cache metadata mismatch for {symbol} {day}")
                prior = old.get((symbol, day))
                if prior is None:
                    reconstructed, readings = reconstruct_szse_day(market / "raw" / str(day), symbol)
                    if readings is None or not np.array_equal(reconstructed.seq_id, cached.seq_id):
                        raise ValueError(f"Reconstruction differs from existing cache for {symbol} {day}")
                    prior = {"symbol": symbol, "date": day, "n_events": len(reconstructed.seq_id), **asdict(readings)}
                    print(f"Reconciled {split} {symbol} {day}", flush=True)
                if prior["n_events"] != len(cached.seq_id):
                    raise ValueError(f"Event count mismatch for {symbol} {day}")
                row = {k: (None if isinstance(value, float) and not isfinite(value) else value) for k, value in prior.items()}
                rows.append({**row, "split": split})
                anchors.extend(_anchor_rows(cached, split, horizon))
    summary = summarize(rows, anchors, dates)
    if any(r["valid_bilateral_windows"] != r["planned_windows"] for r in summary if r["split"] == "dev"):
        raise ValueError("Existing development calendar failed the formal anchor audit")
    atomic_write_json(output, {"source_revision": registration["source_revision"], "reconciliation": rows,
                               "anchors": anchors, "split_summary": summary, "horizon_ms": horizons,
                               "prior_rows_reused": len(old),
                               "method": "snapshot-weighted reconciliation; train anchors descriptive, formal evaluations use dev/test"})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


# --- Post-freeze remote repo tree inventory (no payload reads) ---

def _date_directory(value: str) -> int | None:
    if len(value) != 8 or not value.isascii() or not value.isdigit():
        return None
    try:
        date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError:
        return None
    day = int(value)
    return day if day >= REMOTE_AFTER else None


def metadata_inventory(paths: Iterable[str], repo_id: str, revision: str) -> Json:
    """List complete dates/months at pinned repository revision without stock row content.

    The source uses DATE/{行情,逐笔委托,逐笔成交}.parquet; day file bundles are
    multi-stock, so repository filenames cannot prove either stock is present.
    Return exact repository-date counts and null symbol counts with the explanation.
    """
    tables_by_day: dict[int, set[str]] = defaultdict(set)
    for path in paths:
        parts = PurePosixPath(path).parts
        if len(parts) != 2 or parts[1] not in TABLE_FILES:
            continue
        day = _date_directory(parts[0])
        if day is not None:
            tables_by_day[day].add(parts[1])
    complete = sorted(day for day, tables in tables_by_day.items() if tables == TABLE_FILES)
    partial = sorted(set(tables_by_day) - set(complete))
    months = Counter(str(day)[:4] + "-" + str(day)[4:6] for day in complete)
    per_stock = {symbol: {"months": None, "trade_day_files": None, "status": "UNKNOWN_FROM_FILE_TREE",
                          "possible_market_dates": len(complete), "possible_market_months": sorted(months)}
                 for symbol in SYMBOLS}
    return {"repo_id": repo_id, "source_revision": revision, "inclusive_from": REMOTE_AFTER,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(), "metadata_only": True,
            "market_wide_months": sorted(months), "market_wide_month_counts": dict(sorted(months.items())),
            "market_wide_trade_day_files": len(complete), "market_wide_parquet_files": 3 * len(complete),
            "market_wide_complete_dates": complete, "market_wide_partial_dates": partial,
            "market_wide_partial_parquet_files": sum(len(tables_by_day[day]) for day in partial),
            "by_stock": per_stock, **per_stock,
            "note": ("Daily files are not partitioned by symbol; these are exact market-wide complete-file dates. "
                     "Exact symbol-level availability, suspensions and valid anchors cannot be inferred from file names alone. "
                     "No parquet payload or row-group statistics were read.")}


def _repo_file_paths(repo_id: str, revision: str) -> Iterable[str]:
    """Query repository file metadata only (no parquet content download)."""
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(
        repo_id, repo_type="dataset", revision=revision,
        recursive=True, expand=False, token=True,
    )
    for entry in entries:
        # RepoFolder lacks size. Count actual table files only.
        if hasattr(entry, "size") and entry.size > 0:
            yield entry.path


def enrich_receipt(
    artifact: Path, development_file: Path, *,
    list_paths: Callable[[str, str], Iterable[str]] | None = None, footers: bool = False,
) -> None:
    """Add supplementary table evidence and metadata-only remote inventory after a valid seal."""
    from stochopia.common import atomic_write_json
    from tasks.stochopia_p1.formal_manifest import M5_SCHEMA
    from tasks.stochopia_p1.formal_verify import load_frozen_study
    from tasks.stochopia_p1.holdout_data import validate_m5_receipt
    from tasks.stochopia_p1.market_eval import _exclusive

    manifest = load_frozen_study(artifact)
    if manifest["schema"] != M5_SCHEMA or sum(len(t["trials"]) for t in manifest["trajectories"].values()) != 90:
        raise ValueError("Remote inventory requires all 90 sealed development slots")
    development = json.loads(development_file.read_text(encoding="utf-8"))
    if development["source_revision"] != manifest["data"]["source_revision"]:
        raise ValueError("Development reconciliation revision differs from frozen registration")
    registration = {**manifest["data"], "symbols": SYMBOLS}
    dates = registered_development_dates(registration)
    if development["split_summary"] != summarize(development["reconciliation"], development["anchors"], dates):
        raise ValueError("Development aggregate does not reproduce its per-day evidence")
    verification = artifact / "verification"
    receipt_path = verification / "test-data-reconciliation.json"
    with _exclusive(verification / "acquisition"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        test_dates = validate_m5_receipt(manifest, receipt)
        test_rows = [{**r, "split": "test"} for r in receipt["reconciliation"]]
        test_anchors = [{**a, "split": "test", "valid": True} for a in receipt["anchors"]]
        test_summary = summarize(test_rows, test_anchors, {"test": test_dates})
        repo_id = manifest["data"]["source_repo"]
        revision = manifest["data"]["source_revision"]
        if "remote_availability" not in receipt:
            if list_paths is None:
                list_paths = _repo_file_paths
            receipt["remote_availability"] = metadata_inventory(list_paths(repo_id, revision), repo_id, revision)
        inventory = receipt["remote_availability"]
        if (inventory["repo_id"] != repo_id or inventory["source_revision"] != revision
                or inventory["metadata_only"] is not True or inventory["inclusive_from"] != REMOTE_AFTER):
            raise ValueError("Remote inventory differs from frozen dataset registration or metadata-only scope")
        if footers and "footer_statistics_inspected" not in inventory:
            from .footer_inventory import inspect_stock_footers
            receipt["remote_availability"] = inspect_stock_footers(inventory)
        receipt["development_reconciliation"] = development["reconciliation"]
        receipt["development_anchors"] = development["anchors"]
        receipt["split_summary"] = [*development["split_summary"], *test_summary]
        receipt["summary_method"] = development["method"]
        validate_m5_receipt(manifest, receipt)
        atomic_write_json(receipt_path, receipt)
    print(f"Enriched {receipt_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    development = sub.add_parser("development")
    development.add_argument("--market", type=Path, required=True)
    development.add_argument("--previous", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    remote = sub.add_parser("enrich-receipt")
    remote.add_argument("--artifact", type=Path, required=True)
    remote.add_argument("--development-file", type=Path, required=True)
    remote.add_argument("--footers", action="store_true", help="Also certify stock membership using footer wind_code bounds only; no data pages")
    args = parser.parse_args()
    if args.command == "development":
        development_audit(args.market, args.previous, args.output)
    else:
        enrich_receipt(args.artifact, args.development_file, footers=args.footers)
