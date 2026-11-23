"""Lock ownership and publication checks for the P0 store."""

from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import pytest

STOCHOPIA = Path(os.environ.get("STOCHOPIA_REPO", Path(__file__).resolve().parents[2] / "Stochopia"))
if str(STOCHOPIA) not in sys.path:
    sys.path.append(str(STOCHOPIA))
pytest.importorskip("stochopia.harness.contract", reason="P0 store tests require the companion Stochopia checkout")

from stochopia.harness.contract import DataSplit, FitRecord
from tasks.stochopia_tier0.store import DevelopmentStore, RunBusyError, SelectorUsageStore, VerifierStore
from tasks.stochopia_tier0.verifier import Verifier, VerifierConfig


def test_lock_reenters_same_instance_after_nested_exception(tmp_path: Path) -> None:
    store = DevelopmentStore(tmp_path, "experiment")
    challenger = DevelopmentStore(tmp_path, "experiment")
    with store.lock():
        with pytest.raises(ValueError, match="interrupted"):
            with store.lock():
                store.commit_measurement(1, "trial", {"completed": True})
                raise ValueError("interrupted inside nested lock")
        with pytest.raises(RunBusyError), challenger.lock():
            pytest.fail("another instance acquired an active run lock")
        store.publish_feedback(1, {"complete": True})
    with challenger.lock():
        assert challenger.measurement(1, "trial") == {"completed": True}
        assert challenger.feedback(1) == {"complete": True}


def test_store_writes_require_a_lock(tmp_path: Path) -> None:
    store = DevelopmentStore(tmp_path, "experiment")
    verifier = VerifierStore(tmp_path / "verify")
    with pytest.raises(AssertionError, match="lock"):
        store.preregister(1, {"complete": True})
    with pytest.raises(AssertionError, match="lock"):
        store.commit_measurement(1, "trial", {"complete": True})
    with pytest.raises(AssertionError, match="lock"):
        store.publish_feedback(1, {"complete": True})
    with pytest.raises(AssertionError, match="lock"):
        verifier.publish_report({"complete": True})
    assert not store.path.exists()
    assert not verifier.path.exists()


def test_other_thread_cannot_reenter_or_publish_with_same_instance(tmp_path: Path) -> None:
    store = DevelopmentStore(tmp_path, "experiment")

    def attempt() -> None:
        with pytest.raises(RunBusyError), store.lock():
            pytest.fail("a different thread acquired an active run lock")
        with pytest.raises(AssertionError, match="lock"):
            store.publish_feedback(1, {"complete": True})

    with store.lock(), ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(attempt).result(timeout=10)
    assert store.feedback(1) is None


def _child_attempt(store: DevelopmentStore, sender: Connection) -> None:
    with sender:
        could_write = True
        try:
            store.publish_feedback(1, {"complete": True})
        except AssertionError:
            could_write = False
        could_lock = True
        try:
            with store.lock():
                pass
        except RunBusyError:
            could_lock = False
        sender.send((could_write, could_lock))


def test_fork_does_not_inherit_reentrant_ownership(tmp_path: Path) -> None:
    store = DevelopmentStore(tmp_path, "experiment")
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    with store.lock(), receiver, sender:
        child = context.Process(target=_child_attempt, args=(store, sender))
        child.start()
        try:
            assert receiver.poll(10), "the child never completed its lock attempt"
            assert receiver.recv() == (False, False)
            child.join(timeout=10)
            assert child.exitcode == 0
        finally:
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
    assert store.feedback(1) is None


def _wait_in_child(receiver: Connection) -> None:
    with receiver:
        receiver.recv()


def test_outer_exit_releases_lock_while_forked_descriptor_is_open(tmp_path: Path) -> None:
    store = DevelopmentStore(tmp_path, "experiment")
    challenger = DevelopmentStore(tmp_path, "experiment")
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    with receiver, sender:
        child = context.Process(target=_wait_in_child, args=(receiver,))
        with store.lock():
            child.start()
        try:
            with challenger.lock():
                assert child.is_alive()
        finally:
            sender.send(None)
            child.join(timeout=10)
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
        assert child.exitcode == 0


def test_public_fit_api_preserves_layout_and_sealed_payload(tmp_path: Path) -> None:
    store = DevelopmentStore(tmp_path, "experiment")
    record = FitRecord(
        fit_id="fit", candidate_id="base", train_repeat_id="first", train_data_ref="registered-train",
        config={"model": "gru"}, seed=7, model_ref="saved-model",
    )
    assert store.model_directory() == store.path / "models"
    assert store.fit("base", "first") is None
    with store.lock():
        store.save_fit(record)
    assert (store.path / "fits" / "base" / "first.json").is_file()
    assert store.fit("base", "first") == record


def test_selector_usage_store_is_group_scoped_and_requires_ownership(tmp_path: Path) -> None:
    evidence = SelectorUsageStore(tmp_path, "experiment", "evidence")
    metrics = SelectorUsageStore(tmp_path, "experiment", "metrics")
    with pytest.raises(AssertionError, match="lock"):
        evidence.save_usage({"attempts": []})
    with evidence.lock():
        evidence.save_usage({"attempts": []})
        evidence.save_usage({"attempts": ["reserved"]})
    assert evidence.usage() == {"attempts": ["reserved"]}
    assert metrics.usage() is None
    assert (tmp_path / "experiment" / "evidence" / "usage.json").is_file()


def test_verifier_config_accepts_integral_scalars_but_rejects_bool(tmp_path: Path) -> None:
    split = DataSplit(source={"kind": "simulated", "world": "plain", "namespace": "test"}, unit_ids=("test-day",))
    verifier = Verifier(tmp_path / "verify")
    with pytest.raises(FileNotFoundError) as missing:
        verifier.config()
    assert missing.value.filename == str(verifier.store.path / "test.json")
    config = VerifierConfig("experiment", "batch", split, {"rollout": np.int64(7), "delta_ci": np.int32(17)})
    verifier.register_config(config)
    assert verifier.config().seeds == {"rollout": 7, "delta_ci": 17}
    with pytest.raises(ValueError, match="integer rollout"):
        VerifierConfig("experiment", "batch", split, {"rollout": True, "delta_ci": 17})
