"""Integration coverage for V5 harness P0 fixes."""

import json

import pytest

from harness_v5 import (
    V5Bridge,
    _adapt_common_fields,
    adapt_bermudan_metrics,
    get_metrics_adapter,
)
from provenance import SOURCE_FILES


class TestAdapterRouting:
    def test_bermudan_protocol_returns_bermudan_adapter(self):
        assert (
            get_metrics_adapter("bermudan-lsmc-feature-ir.v1")
            is adapt_bermudan_metrics
        )

    def test_unknown_protocol_returns_common_adapter(self):
        assert get_metrics_adapter("sums-diffs") is _adapt_common_fields

    def test_common_adapter_preserves_all_fields(self):
        raw = {
            "artifact_sha256": "abc123",
            "set_hash": "def456",
            "total_seconds": 3.5,
            "metric": "C(A)",
        }

        adapted = _adapt_common_fields(raw)

        assert adapted["artifact_sha256"] == "abc123"
        assert adapted["set_hash"] == "def456"
        assert adapted["total_seconds"] == 3.5
        assert adapted["score_metric"] == "C(A)"
        assert adapted["elapsed_s"] == 3.5

    def test_common_adapter_normalizes_runtime(self):
        adapted = _adapt_common_fields(
            {"runtime_seconds": 5.2, "metric": "score"}
        )

        assert adapted["elapsed_s"] == pytest.approx(5.2)
        assert adapted["score_metric"] == "score"


class TestBermudanAdapterArtifactPriority:
    def test_bermudan_adapter_prefers_sandbox_artifact_sha(self):
        adapted = adapt_bermudan_metrics(
            {
                "artifact_sha256": "sandbox_hash",
                "candidate_hash": "evaluator_hash",
            }
        )

        assert adapted["artifact_sha256"] == "sandbox_hash"

    def test_bermudan_adapter_falls_back_to_candidate_hash(self):
        adapted = adapt_bermudan_metrics(
            {"candidate_hash": "evaluator_hash"}
        )

        assert adapted["artifact_sha256"] == "evaluator_hash"


class TestReviewGracefulDegradation:
    @staticmethod
    def _epoch_id(epoch):
        return f"{epoch.island_id}_epoch_{epoch.epoch:02d}"

    @staticmethod
    def _score_candidate(bridge, record_id, epoch, score=0.5, status="ok"):
        bridge.on_candidate_evaluated(
            record_id=record_id,
            island_epoch_id=TestReviewGracefulDegradation._epoch_id(epoch),
            score=score,
            status=status,
            parent_ids=[],
            metrics={},
        )

    def test_review_defers_when_only_one_epoch_scored(self, tmp_path):
        bridge = V5Bridge(tmp_path / "run")
        epochs = bridge.initialize([f"seed-{index}" for index in range(4)])
        self._score_candidate(bridge, "candidate-one", epochs[0])

        assert bridge.on_context_complete(10) == {}

    def test_review_defers_when_no_epochs_scored(self, tmp_path):
        bridge = V5Bridge(tmp_path / "run")
        epochs = bridge.initialize([f"seed-{index}" for index in range(4)])
        self._score_candidate(
            bridge,
            "candidate-crash",
            epochs[0],
            score=None,
            status="crash",
        )

        assert bridge.on_context_complete(10) == {}

    def test_review_proceeds_when_multiple_epochs_scored(self, tmp_path):
        bridge = V5Bridge(tmp_path / "run")
        epochs = bridge.initialize([f"seed-{index}" for index in range(4)])
        for index, epoch in enumerate(epochs[:3]):
            self._score_candidate(
                bridge,
                f"candidate-{index}",
                epoch,
                score=0.5 + index * 0.01,
            )

        result = bridge.on_context_complete(10)

        assert isinstance(result, dict)


class TestSyncErrorLedger:
    def test_sync_error_logged_on_v5_failure(self, tmp_path):
        bridge = V5Bridge(tmp_path / "run")
        bridge._log_sync_error(
            "test-record", "on_candidate_evaluated", ValueError("test")
        )

        ledger = tmp_path / "run" / "v5" / "sync_errors.jsonl"
        assert ledger.exists()
        entry = json.loads(ledger.read_text(encoding="utf-8").strip())
        assert entry["record_id"] == "test-record"
        assert entry["operation"] == "on_candidate_evaluated"
        assert entry["error"] == "ValueError('test')"
        assert entry["legacy_eb_committed"] is True
        assert entry["timestamp"]

    def test_sync_error_count_in_diagnostics(self, tmp_path):
        bridge = V5Bridge(tmp_path / "run")
        bridge._log_sync_error("record-1", "operation-1", ValueError("one"))
        bridge._log_sync_error("record-2", "operation-2", RuntimeError("two"))

        assert bridge.get_island_diagnostics()["sync_error_count"] == 2

    def test_reconcile_detects_missing_v5_records(self, tmp_path):
        bridge = V5Bridge(tmp_path / "run")

        result = bridge._reconcile(
            legacy_record_ids=["legacy-1", "legacy-2"]
        )

        assert set(result["missing_in_v5"]) == {"legacy-1", "legacy-2"}
        assert result["v5_event_count"] == 0


class TestProvenanceV5SourceFiles:
    def test_v5_source_files_in_provenance(self):
        v5_source_files = (
            "analogy_graph.py",
            "behavior_index.py",
            "behavior_profiler.py",
            "context_retrieval.py",
            "experience_events.py",
            "harness_v5.py",
            "island_scheduler.py",
            "matched_control.py",
            "mechanism_cards.py",
            "object_store.py",
            "probe_suite.py",
            "schemas_v5.py",
        )

        assert len(v5_source_files) == 12
        assert set(v5_source_files).issubset(set(SOURCE_FILES))


class TestV5LifecycleEndToEnd:
    def test_full_lifecycle(self, tmp_path):
        from harness_v5 import V5Bridge

        run_dir = tmp_path / "run"
        bridge = V5Bridge(run_dir, num_islands=4)
        epochs = bridge.initialize(
            seed_record_ids=[f"seed-{index}" for index in range(4)],
            frozen_baseline_score=0.1,
        )
        assert len(epochs) == 4
        epoch_ids = [
            f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
            for epoch in epochs
        ]

        for index, (epoch_id, score) in enumerate(
            zip(epoch_ids, (0.1, 0.2, 0.3, 0.4))
        ):
            bridge.on_candidate_evaluated(
                record_id=f"ok-{index}",
                island_epoch_id=epoch_id,
                score=score,
                status="ok",
                parent_ids=[],
                metrics={},
            )
        for index in range(2):
            bridge.on_candidate_evaluated(
                record_id=f"crash-{index}",
                island_epoch_id=epoch_ids[index],
                score=None,
                status="crash",
                parent_ids=[],
                metrics={},
            )
        bridge.on_candidate_evaluated(
            record_id="rejected-0",
            island_epoch_id=epoch_ids[2],
            score=None,
            status="rejected",
            parent_ids=[],
            metrics={},
        )

        events = bridge.event_store.read_experiment_events()
        assert len(events) == 7
        status_by_record = {event.record_id: event.status for event in events}
        assert status_by_record["ok-0"] == "ok"
        assert status_by_record["ok-1"] == "ok"
        assert status_by_record["ok-2"] == "ok"
        assert status_by_record["ok-3"] == "ok"
        assert status_by_record["crash-0"] == "runtime_error"
        assert status_by_record["crash-1"] == "runtime_error"
        assert status_by_record["rejected-0"] == "static_rejected"

        replacements = bridge.on_context_complete(10)
        assert isinstance(replacements, dict)
        assert len(bridge._cards) >= 4

        diagnostics = bridge.get_island_diagnostics()
        assert isinstance(diagnostics, dict)
        bridge.save_state()

        resumed = V5Bridge(run_dir, num_islands=4)
        resumed_diagnostics = resumed.get_island_diagnostics()
        assert resumed_diagnostics["cards_cached"] == diagnostics["cards_cached"]

        event_count_before_replay = len(
            resumed.event_store.read_experiment_events()
        )
        resumed.on_candidate_evaluated(
            record_id="ok-0",
            island_epoch_id=epoch_ids[0],
            score=0.1,
            status="ok",
            parent_ids=[],
            metrics={},
        )
        event_count_after_replay = len(
            resumed.event_store.read_experiment_events()
        )
        assert event_count_after_replay == event_count_before_replay

        all_record_ids = [event.record_id for event in events]
        reconciliation = resumed._reconcile(
            legacy_record_ids=all_record_ids
        )
        assert reconciliation["missing_in_v5"] == []
        assert reconciliation["v5_event_count"] == 7

    def test_degraded_lifecycle(self, tmp_path):
        from harness_v5 import V5Bridge

        bridge = V5Bridge(tmp_path / "run")
        bridge._log_sync_error(
            "sync-record", "on_candidate_evaluated", RuntimeError("sync failed")
        )

        diagnostics = bridge.get_island_diagnostics()
        assert diagnostics["sync_error_count"] == 1

        reconciliation = bridge._reconcile(
            legacy_record_ids=["sync-record"]
        )
        assert "sync-record" in reconciliation["missing_in_v5"]

    def test_resume_with_gaps(self, tmp_path):
        from harness_v5 import V5Bridge

        run_dir = tmp_path / "run"
        bridge = V5Bridge(run_dir, num_islands=2)
        epochs = bridge.initialize(seed_record_ids=["seed-0", "seed-1"])
        epoch_id = f"{epochs[0].island_id}_epoch_{epochs[0].epoch:02d}"
        bridge.on_candidate_evaluated(
            record_id="record-0",
            island_epoch_id=epoch_id,
            score=0.5,
            status="ok",
            parent_ids=[],
            metrics={},
        )
        bridge.save_state()

        resumed = V5Bridge(run_dir, num_islands=2)
        reconciliation = resumed._reconcile(
            legacy_record_ids=["record-0", "legacy-1", "legacy-2"]
        )
        assert len(reconciliation["missing_in_v5"]) == 2


class TestRecordSeed:
    def test_record_seed_creates_event(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run')
        bridge.record_seed('seed-0', score=0.5, metrics={'artifact_sha256': 'abc'})
        events = bridge.event_store.read_experiment_events()
        assert len(events) == 1
        assert events[0].record_id == 'seed-0'
        assert events[0].experiment_plan_id == 'seed'
        assert events[0].status == 'ok'
        assert events[0].score == 0.5

    def test_record_seed_idempotent(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run')
        bridge.record_seed('seed-0', score=0.5, metrics={})
        bridge.record_seed('seed-0', score=0.5, metrics={})
        events = bridge.event_store.read_experiment_events()
        assert len(events) == 1

    def test_record_seed_creates_card(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run')
        bridge.record_seed('seed-0', score=0.5, metrics={})
        assert 'seed-0' in bridge._cards

    def test_record_seed_with_per_instance(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run')
        metrics = {
            'per_instance_results': {'inst_a': 0.4, 'inst_b': 0.6},
            'artifact_sha256': 'hash123',
        }
        bridge.record_seed('seed-0', score=0.5, metrics=metrics)
        events = bridge.event_store.read_experiment_events()
        assert events[0].per_instance_metrics_ref != ''


class TestResolveSyncError:
    def test_resolve_makes_healthy(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run')
        bridge._log_sync_error('rec-1', 'op', ValueError('x'))
        assert bridge.sync_status == 'degraded'
        bridge.resolve_sync_error('rec-1', 'repaired')
        assert bridge.sync_status == 'healthy'

    def test_partial_resolve_stays_degraded(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run')
        bridge._log_sync_error('rec-1', 'op', ValueError('x'))
        bridge._log_sync_error('rec-2', 'op', ValueError('y'))
        bridge.resolve_sync_error('rec-1', 'repaired')
        assert bridge.sync_status == 'degraded'

    def test_diagnostics_shows_unresolved_count(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run')
        bridge._log_sync_error('rec-1', 'op', ValueError('x'))
        bridge._log_sync_error('rec-2', 'op', ValueError('y'))
        bridge.resolve_sync_error('rec-1', 'repaired')
        diag = bridge.get_island_diagnostics()
        assert diag['unresolved_sync_errors'] == 1
        assert diag['sync_error_count'] == 2


class TestReconcileReferenceIntegrity:
    def test_reconcile_detects_missing_cards(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run')
        bridge.record_seed('seed-0', score=0.5, metrics={})
        bridge._cards.pop('seed-0', None)
        result = bridge._reconcile()
        assert 'seed-0' in result['missing_cards']

    def test_reconcile_auto_resolves_fixed_errors(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run')
        bridge._log_sync_error('seed-0', 'on_candidate_evaluated', ValueError('x'))
        assert bridge.sync_status == 'degraded'
        bridge.record_seed('seed-0', score=0.5, metrics={})
        bridge._reconcile(legacy_record_ids=['seed-0'])
        assert bridge.sync_status == 'healthy'


class TestRepairIfNeeded:
    @staticmethod
    def _epoch_id(epoch):
        return f'{epoch.island_id}_epoch_{epoch.epoch:02d}'

    def test_replay_repairs_missing_card(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run')
        epochs = bridge.initialize(['seed-0', 'seed-1'])
        epoch_id = self._epoch_id(epochs[0])
        bridge.on_candidate_evaluated(
            record_id='rec-0', island_epoch_id=epoch_id,
            score=0.5, status='ok', parent_ids=[], metrics={},
        )
        assert 'rec-0' in bridge._cards
        bridge._cards.pop('rec-0')
        bridge.on_candidate_evaluated(
            record_id='rec-0', island_epoch_id=epoch_id,
            score=0.5, status='ok', parent_ids=[], metrics={},
        )
        assert 'rec-0' in bridge._cards
        events = bridge.event_store.read_experiment_events()
        rec0_events = [e for e in events if e.record_id == 'rec-0']
        assert len(rec0_events) == 1


class TestAdapterErrorCapture:
    def test_adapter_exception_logged_to_sync_ledger(self, tmp_path):
        bridge = V5Bridge(tmp_path / 'run')
        bridge._log_sync_error('test-rec', 'metrics_adapter', TypeError('bad type'))
        assert bridge.sync_status == 'degraded'
        ledger = tmp_path / 'run' / 'v5' / 'sync_errors.jsonl'
        entry = json.loads(ledger.read_text(encoding='utf-8').strip())
        assert entry['operation'] == 'metrics_adapter'
