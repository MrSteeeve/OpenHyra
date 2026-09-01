from concurrent.futures import ThreadPoolExecutor

import pytest

from experience_events import ExperienceEventStore
from schemas_v5 import AnalogyResult, AnnotationEvent, ExperimentEvent, ExperimentPlan


SHA_A = "a" * 64
SHA_B = "b" * 64


def make_experiment_event(record_id: str = "sol_0001") -> ExperimentEvent:
    return ExperimentEvent(
        record_id=record_id,
        algorithm_bundle_sha256=SHA_A,
        experiment_plan_id="plan_0001",
        island_epoch_id="island_01_epoch_01",
        status="ok",
        score=0.25,
        score_metric="paired_lower_bound_lcb",
        per_instance_metrics_ref=f"sha256:{SHA_A}",
        behavior_profile_ref=f"sha256:{SHA_B}",
        runtime_metrics_ref=f"sha256:{SHA_A}",
        parent_ids=["sol_0000"],
        inspiration_ids=["sol_0002"],
        created_at="2026-09-01T00:00:00Z",
    )


def make_plan_event() -> ExperimentPlan:
    return ExperimentPlan(
        id="plan_0001",
        action="continue",
        target_island_epoch_id="island_01_epoch_01",
        generation_operator="analogy_transfer",
        parent_ids=["sol_0000"],
        inspiration_ids=["sol_0002"],
        analogy_hypothesis_id="analogy_0001",
        implementation_intent="add a normalized input feature",
        negative_constraints=["do_not_change_budget"],
        success_criterion="paired_lcb_gt_0",
        budget={
            "candidate_count": 2,
            "sandbox_seconds_per_cell": 60,
            "max_artifact_bytes": 8_388_608,
        },
    )


def make_annotation_event() -> AnnotationEvent:
    return AnnotationEvent(
        id="annotation_0001",
        annotation_type="mechanism_inference",
        target_record_ids=["sol_0001"],
        evidence_record_ids=["sol_0000", "sol_0001"],
        model="model-id",
        backend="backend-id",
        prompt_sha256=SHA_A,
        response_sha256=SHA_B,
        parser_schema="openhyra-mechanism-card.v1",
        created_at="2026-09-01T00:00:00Z",
    )


def make_analogy_result() -> AnalogyResult:
    return AnalogyResult(
        analogy_hypothesis_id="analogy_0001",
        guided_record_id="sol_0001",
        control_record_id="sol_0002",
        guided_delta=0.06,
        control_delta=0.01,
        transfer_gain=0.05,
        transfer_gain_standard_error=0.02,
        predicted_slice_effect=0.07,
        prediction_direction_correct=True,
        verdict="transfer_supported",
    )


def test_append_and_read_experiment_event(tmp_path):
    store = ExperienceEventStore(tmp_path / "eb")
    event = make_experiment_event()

    store.append_experiment_event(event)
    restored = store.read_experiment_events()

    assert len(restored) == 1
    assert restored[0].record_id == event.record_id
    assert restored[0].score == event.score
    assert restored[0].to_dict() == event.to_dict()


def test_append_and_read_plan_event(tmp_path):
    store = ExperienceEventStore(tmp_path / "eb")
    plan = make_plan_event()

    store.append_plan_event(plan)

    assert [item.to_dict() for item in store.read_plan_events()] == [plan.to_dict()]


def test_append_and_read_annotation_event(tmp_path):
    store = ExperienceEventStore(tmp_path / "eb")
    annotation = make_annotation_event()

    store.append_annotation_event(annotation)

    assert [item.to_dict() for item in store.read_annotation_events()] == [
        annotation.to_dict()
    ]


def test_append_and_read_analogy_result(tmp_path):
    store = ExperienceEventStore(tmp_path / "eb")
    result = make_analogy_result()

    store.append_analogy_result(result)

    assert [item.to_dict() for item in store.read_analogy_results()] == [
        result.to_dict()
    ]


def test_multiple_appends(tmp_path):
    store = ExperienceEventStore(tmp_path / "eb")
    events = [make_experiment_event(f"sol_{index:04d}") for index in range(3)]

    for event in events:
        store.append_experiment_event(event)

    assert [event.record_id for event in store.read_experiment_events()] == [
        "sol_0000",
        "sol_0001",
        "sol_0002",
    ]


def test_object_store_integration(tmp_path):
    store = ExperienceEventStore(tmp_path / "eb")
    event = make_experiment_event()

    digest = store.append_experiment_event(event)
    stored_path = store.object_store.get_path(digest, "experiment_event.json")

    assert store.object_store.exists(digest)
    assert stored_path is not None
    assert stored_path.read_text(encoding="utf-8")
    assert store.object_store.verify(digest)


def test_bridge_legacy_record(tmp_path):
    store = ExperienceEventStore(tmp_path / "eb")
    legacy = {
        "id": "sol_0042",
        "score": 1.5,
        "status": "crash",
        "parent": "sol_0041",
        "created": "2026-08-31 12:00:00",
    }

    event = store.bridge_legacy_record(legacy)

    assert event.record_id == "sol_0042"
    assert event.score == 1.5
    assert event.status == "runtime_error"
    assert event.algorithm_bundle_sha256 == ""
    assert event.experiment_plan_id == ""
    assert event.parent_ids == []
    assert event.created_at == ""
    assert store.read_experiment_events() == []


@pytest.mark.parametrize(
    ("legacy_status", "v5_status"),
    [
        ("crash", "runtime_error"),
        ("rejected", "static_rejected"),
        ("violation", "violation"),
        ("cancelled", "cancelled"),
        ("timeout", "timeout"),
        ("ok", "ok"),
    ],
)
def test_bridge_legacy_status_mapping(tmp_path, legacy_status, v5_status):
    store = ExperienceEventStore(tmp_path / "eb")

    event = store.bridge_legacy_record(
        {"id": "sol_0001", "score": None, "status": legacy_status}
    )

    assert event.status == v5_status


def test_validation_on_append(tmp_path):
    store = ExperienceEventStore(tmp_path / "eb")
    invalid = make_experiment_event()
    invalid.status = "unknown"

    with pytest.raises(ValueError, match="status"):
        store.append_experiment_event(invalid)

    assert store.read_experiment_events() == []
    assert store.object_store.list_objects() == []


def test_thread_safety(tmp_path):
    store = ExperienceEventStore(tmp_path / "eb")

    def append_batch(worker: int) -> None:
        for item in range(10):
            store.append_experiment_event(
                make_experiment_event(f"worker_{worker}_event_{item}")
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append_batch, range(4)))

    events = store.read_experiment_events()
    assert len(events) == 40
    assert {event.record_id for event in events} == {
        f"worker_{worker}_event_{item}"
        for worker in range(4)
        for item in range(10)
    }
