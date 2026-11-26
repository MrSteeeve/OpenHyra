import struct

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.p1_m5_closing_data.reconcile import metadata_inventory, TABLE_FILES
from experiments.p1_m5_closing_data.footer_inventory import (
    ABSENT, PRESENT, UNKNOWN, combine_presence, parse_footer_metadata, stock_presence,
)


def metadata(codes: list[str], row_group_size: int) -> object:
    buffer = pa.BufferOutputStream()
    pq.write_table(pa.table({"wind_code": codes, "depth": list(range(len(codes)))}), buffer, row_group_size=row_group_size)
    raw = buffer.getvalue().to_pybytes()
    trailer = raw[-8:]
    length = struct.unpack("<I", trailer[:4])[0]
    return parse_footer_metadata(raw[-length-8:-8], trailer)


def test_exact_endpoint_is_presence() -> None:
    data = metadata(["000001.SZ", "300001.SZ", "666666.SZ"], 1)
    assert stock_presence(data, "000001.SZ") == PRESENT
    assert stock_presence(data, "300001.SZ") == PRESENT
    assert stock_presence(data, "300002.SZ") == ABSENT


def test_interior_is_not_presence() -> None:
    data = metadata(["000001.SZ", "666666.SZ"], 2)
    assert stock_presence(data, "000001.SZ") == PRESENT
    assert stock_presence(data, "300001.SZ") == UNKNOWN
    assert stock_presence(data, "000000.SZ") == ABSENT


def test_three_table_count_bounds() -> None:
    dates = [20170601, 20170602, 20170603]
    inv = metadata_inventory([f"{day}/{name}" for day in dates for name in TABLE_FILES], "repo", "hash")
    data = {(day, table): {"000001.SZ": PRESENT, "300001.SZ": PRESENT} for day in dates for table in TABLE_FILES}
    data[(dates[1], "逐笔成交.parquet")]["300001.SZ"] = UNKNOWN
    data[(dates[2], "行情.parquet")]["300001.SZ"] = ABSENT
    combine_presence(inv, data)
    assert inv["000001.SZ"]["trade_day_files"] == 3
    assert inv["300001.SZ"]["trade_day_files"] is None
    assert inv["300001.SZ"]["confirmed_trade_day_files"] == 1
    assert inv["300001.SZ"]["possible_trade_day_files"] == 2
    assert inv["300001.SZ"]["months"] == ["2017-06"]
