"""Optional conservative stock-level inventory using Parquet footer statistics ONLY.

This helper is invoked explicitly after the 90-trial seal and holdout receipt have
already been validated by closing_data.enrich_receipt. No parquet data page is
read, saved or requested. Only the file footer and its 8-byte trailer are fetched
with strict HTTP 206 exact-range checks; only wind_code bounds are inspected.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse
import struct

from .reconcile import SYMBOLS, TABLE_FILES

PRESENT = "PRESENT"
ABSENT = "ABSENT"
UNKNOWN = "UNKNOWN"
MAX_FOOTER_BYTES = 16 * 2**20


def parse_footer_metadata(footer: bytes, trailer: bytes) -> Any:
    """Given metadata bytes and the exact 8-byte footer trailer, read no data pages."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if len(trailer) != 8 or trailer[4:] != b"PAR1" or struct.unpack("<I", trailer[:4])[0] != len(footer):
        raise ValueError("Malformed Parquet footer trailer")
    return pq.read_metadata(pa.BufferReader(b"PAR1" + footer + trailer))


def stock_presence(metadata: Any, symbol: str) -> str:
    """Use row-group wind_code min/max endpoints only, not any market-data values.

    A matching nonempty min or max endpoint is sufficient evidence of a stock
    row for the source's fixed-length Wind security identifiers. A symbol strictly between min and max is possible, not proven; null or
    missing bounds mean unknown. Outside all exact bounds is certified absent.
    """
    uncertain = False
    for index in range(metadata.num_row_groups):
        group = metadata.row_group(index)
        if not group.num_rows:
            continue
        code = None
        for column_index in range(group.num_columns):
            column = group.column(column_index)
            if column.path_in_schema == "wind_code":
                code = column
                break
        if code is None:
            uncertain = True
            continue
        statistics = code.statistics
        if statistics is None:
            uncertain = True
            continue
        if statistics.has_null_count and statistics.null_count == group.num_rows:
            continue
        if not statistics.has_min_max:
            uncertain = True
            continue
        lo, hi = statistics.min, statistics.max
        if not isinstance(lo, str) or not isinstance(hi, str) or lo > hi:
            uncertain = True
            continue
        if symbol in (lo, hi):
            return PRESENT
        if lo < symbol < hi:
            uncertain = True
    return UNKNOWN if uncertain else ABSENT


def _range(url: str, token: str | None, size: int, start: int, end: int) -> bytes:
    """Fetch exactly an inclusive footer byte range or fail without downloading body."""
    import requests

    if not 0 <= start <= end < size or end - start + 1 > MAX_FOOTER_BYTES:
        raise ValueError("Invalid Parquet footer byte span")
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    if token and urlparse(url).hostname == "huggingface.co":
        headers["Authorization"] = f"Bearer {token}"
    with requests.get(url, headers=headers, timeout=(10, 30), stream=True) as response:
        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
            raise ValueError(f"Metadata range not honored: HTTP {response.status_code}")
        expected = end - start + 1
        body = response.raw.read(expected + 1)
        if len(body) != expected:
            raise ValueError("Parquet footer range length differs from request")
    return body


def file_presence(repo: str, revision: str, day: int, table: str) -> tuple[int, str, dict[str, str], str | None]:
    """Query remote file HEAD and footer ranges; any failure is UNKNOWN, never absence."""
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url

    try:
        url = hf_hub_url(repo, f"{day}/{table}", repo_type="dataset", revision=revision)
        meta = get_hf_file_metadata(url, token=True, timeout=30)
        size = meta.size
        if not isinstance(size, int) or size < 12:
            raise ValueError("Missing remote parquet length")
        token = get_token()
        trailer = _range(meta.location, token, size, size - 8, size - 1)
        length = struct.unpack("<I", trailer[:4])[0]
        if trailer[4:] != b"PAR1" or not 0 < length <= min(MAX_FOOTER_BYTES, size - 12):
            raise ValueError("Missing or oversized remote parquet metadata")
        footer = _range(meta.location, token, size, size - 8 - length, size - 9)
        metadata = parse_footer_metadata(footer, trailer)
        return day, table, {symbol: stock_presence(metadata, symbol) for symbol in SYMBOLS}, None
    except Exception as exc:
        # Do not print exception text: it may identify a signed CDN URL.
        return day, table, {symbol: UNKNOWN for symbol in SYMBOLS}, type(exc).__name__


def combine_presence(inventory: dict[str, Any], panels: dict[tuple[int, str], dict[str, str]]) -> dict[str, Any]:
    """Aggregate per-stock exact or bounded counts across three registered day tables."""
    for symbol in SYMBOLS:
        confirmed = []
        possible = []
        absent = []
        for day in inventory["market_wide_complete_dates"]:
            statuses = [panels[(day, table)][symbol] for table in TABLE_FILES]
            if ABSENT in statuses:
                absent.append(day)
            elif all(status == PRESENT for status in statuses):
                confirmed.append(day)
                possible.append(day)
            else:
                possible.append(day)
        unknown = sorted(set(possible) - set(confirmed))
        confirmed_months = sorted({str(day)[:4] + "-" + str(day)[4:6] for day in confirmed})
        possible_months = sorted({str(day)[:4] + "-" + str(day)[4:6] for day in possible})
        row = {"status": "BOUNDED_FOOTER_METADATA" if unknown else "EXACT_FOOTER_METADATA",
               "trade_day_files": None if unknown else len(confirmed),
               "months": confirmed_months if confirmed_months == possible_months else None,
               "confirmed_trade_day_files": len(confirmed), "possible_trade_day_files": len(possible),
               "confirmed_months": confirmed_months, "possible_market_months": possible_months,
               "confirmed_dates": confirmed, "unresolved_dates": unknown, "absent_dates": absent}
        inventory["by_stock"][symbol] = inventory[symbol] = row
    return inventory


def inspect_stock_footers(inventory: dict[str, Any], *, workers: int = 12) -> dict[str, Any]:
    """Convert file-tree availability into stock-wise counts using only footer metadata."""
    repo = inventory["repo_id"]
    revision = inventory["source_revision"]
    panels = {}
    errors = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future = [pool.submit(file_presence, repo, revision, day, table)
                  for day in inventory["market_wide_complete_dates"] for table in TABLE_FILES]
        for index, task in enumerate(as_completed(future), 1):
            day, table, presence, error = task.result()
            panels[(day, table)] = presence
            if error:
                errors[error] += 1
            if index % 50 == 0 or index == len(future):
                print(f"Metadata-only footer progress: {index}/{len(future)}", flush=True)
    combine_presence(inventory, panels)
    inventory["footer_statistics_inspected"] = ["wind_code.min", "wind_code.max", "wind_code.null_count", "row_group.num_rows"]
    inventory["footer_metadata_errors"] = dict(sorted(errors.items()))
    inventory["note"] = ("Date files contain a mix of stocks. Only exact wind_code row-group min/max endpoint matches "
                         "certify symbol presence in a registered date/table; interior bounds and metadata errors remain unknown. "
                         "Every complete date needs presence in all three tables. No market-data row or Parquet data page read; "
                         "these counts do not establish positive-spread anchor quality or absence of suspension.")
    return inventory
