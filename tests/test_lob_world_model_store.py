"""Durable ledger write-once and process/thread ownership regression."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from tasks.lob_world_model.store import Ledger


def test_ledger_seal_requires_lock_and_never_rewrites_existing_record(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    with pytest.raises(RuntimeError, match="own its lock"):
        ledger.seal({"value": 1}, "record.json")
    with ledger.lock():
        original = ledger.seal({"value": 1}, "record.json")
        path = ledger.file("record.json")
        before = path.read_bytes()
        assert ledger.seal({"value": 1}, "record.json") == original
        with pytest.raises(ValueError, match="cannot change"):
            ledger.seal({"value": 2}, "record.json")
        assert path.read_bytes() == before
        with ThreadPoolExecutor(max_workers=1) as worker:
            result = worker.submit(ledger.seal, {"value": 2}, "other.json")
            with pytest.raises(RuntimeError, match="own its lock"):
                result.result()
        assert ledger.read("other.json") is None
    assert ledger.read("record.json") == original
