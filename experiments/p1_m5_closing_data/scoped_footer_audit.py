"""Post-freeze scope-limited remote metadata audit using committed M5 helpers."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from stochopia.common import atomic_write_json
from tasks.stochopia_p1.formal_manifest import M5_SCHEMA
from tasks.stochopia_p1.formal_verify import load_frozen_study
from tasks.stochopia_p1.holdout_data import validate_m5_receipt
from tasks.stochopia_p1.market_eval import _exclusive
from .reconcile import REMOTE_AFTER, SYMBOLS, TABLE_FILES
from .footer_inventory import ABSENT, PRESENT, UNKNOWN, combine_presence, file_presence

Json = dict[str, Any]


def _read(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def run(artifact: Path, progress_path: Path, *, through: int, workers: int) -> None:
    """Inspect only through registered 2017 H2, leaving later dates explicitly unknown."""
    if through != 20171231 or not 1 <= workers <= 6:
        raise ValueError("The scoped audit is limited to June–December 2017 and 1–6 workers")
    manifest = load_frozen_study(artifact)
    if manifest["schema"] != M5_SCHEMA or sum(len(entry["trials"]) for entry in manifest["trajectories"].values()) != 90:
        raise ValueError("No remote footer audit before the full 90-slot seal")
    verification = artifact / "verification"
    receipt_path = verification / "test-data-reconciliation.json"
    original = _read(receipt_path)
    validate_m5_receipt(manifest, original)
    inventory = original["remote_availability"]
    repo = manifest["data"]["source_repo"]
    revision = manifest["data"]["source_revision"]
    if (inventory["repo_id"] != repo or inventory["source_revision"] != revision
            or inventory["metadata_only"] is not True or inventory["inclusive_from"] != REMOTE_AFTER):
        raise ValueError("Remote inventory was not registered at the frozen dataset revision")
    dates = [day for day in inventory["market_wide_complete_dates"] if REMOTE_AFTER <= day <= through]
    job_keys = {(day, table) for day in dates for table in TABLE_FILES}
    registration = {"repo": repo, "revision": revision, "through": through, "dates": dates}
    if progress_path.exists():
        progress = _read(progress_path)
        if progress["registration"] != registration:
            raise ValueError("Metadata-only progress belongs to another audit")
    else:
        progress = {"registration": registration, "completed": []}
    completed = {(row["date"], row["table"]) for row in progress["completed"]}
    if not completed.issubset(job_keys) or len(completed) != len(progress["completed"]):
        raise ValueError("Metadata-only progress includes duplicate or unregistered files")
    # Use the exact committed range-only fetcher; no parquet data-page requests.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(file_presence, repo, revision, day, table)
                   for day, table in sorted(job_keys - completed)]
        for task in as_completed(pending):
            day, table, presence, error = task.result()
            progress["completed"].append({"date": day, "table": table, "presence": presence, "error": error})
            atomic_write_json(progress_path, progress)
            n = len(progress["completed"])
            if n % 30 == 0 or n == len(job_keys):
                print(f"Scoped footer metadata: {n}/{len(job_keys)}", flush=True)
    full_panels = {(day, table): {symbol: UNKNOWN for symbol in SYMBOLS}
                   for day in inventory["market_wide_complete_dates"] for table in TABLE_FILES}
    errors = Counter()
    for row in progress["completed"]:
        presence = row["presence"]
        if set(presence) != set(SYMBOLS) or not set(presence.values()).issubset({PRESENT, ABSENT, UNKNOWN}):
            raise ValueError("Unknown metadata stock-presence verdict")
        full_panels[(row["date"], row["table"])] = presence
        if row["error"] is not None:
            errors[row["error"]] += 1
    full_inventory = combine_presence(deepcopy(inventory), full_panels)
    scoped_inventory = {"market_wide_complete_dates": dates, "by_stock": {}}
    scoped_inventory = combine_presence(scoped_inventory, full_panels)
    full_inventory["footer_statistics_inspected"] = ["wind_code.min", "wind_code.max", "wind_code.null_count", "row_group.num_rows"]
    full_inventory["footer_metadata_errors"] = dict(sorted(errors.items()))
    full_inventory["footer_audit"] = {
        "scope_from": REMOTE_AFTER, "scope_through": through,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete_date_bundles_attempted": len(dates), "table_footers_attempted": len(job_keys),
        "table_footers_failed": sum(errors.values()),
        "uninspected_after_scope": len(inventory["market_wide_complete_dates"]) - len(dates),
        "by_stock": scoped_inventory["by_stock"],
        "method": "strict exact HTTP 206 Range reads of Parquet metadata and 8-byte trailer only; no data pages; all 3 wind_code tables required",
    }
    full_inventory["note"] = (
        "Date files bundle multiple stocks. The pinned tree enumerates market-wide dates through 2026. "
        "Only 2017-06-01 through 2017-12-31 stock membership was audited via wind_code footer metadata; "
        "remaining dates are uninspected and unknown, not absent. Matching exact min/max endpoints in all three tables certify presence. "
        "Interior bounds and footer failures remain unknown. No data pages or market-data row content was requested; "
        "these results do not establish positive-spread anchors or absence of suspension."
    )
    with _exclusive(verification / "acquisition"):
        latest = _read(receipt_path)
        validate_m5_receipt(manifest, latest)
        if any(latest[key] != value for key, value in original.items() if key != "remote_availability"):
            raise ValueError("Underlying receipt changed during stock-membership inventory")
        if latest["remote_availability"] != inventory:
            raise ValueError("Another process changed remote inventory")
        latest["remote_availability"] = full_inventory
        validate_m5_receipt(manifest, latest)
        atomic_write_json(receipt_path, latest)
    summary = {symbol: {key: value for key, value in scoped_inventory[symbol].items()
                        if key not in {"confirmed_dates", "unresolved_dates", "absent_dates"}}
               for symbol in SYMBOLS}
    print(json.dumps({"scoped": summary, "table_errors": dict(errors)}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--through", type=int, default=20171231)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run(args.artifact, args.progress, through=args.through, workers=args.workers)


if __name__ == "__main__":
    main()
