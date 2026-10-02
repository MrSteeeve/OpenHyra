import pytest

from island_scheduler import IslandScheduler


def make_scheduler(tmp_path, **kwargs) -> IslandScheduler:
    return IslandScheduler(tmp_path / "islands.json", **kwargs)


def initialize_with_scored_records(tmp_path):
    scheduler = make_scheduler(tmp_path)
    epochs = scheduler.initialize(["baseline"], context_round=0, base_proposal_seed=7)
    scores = {}
    for index, epoch in enumerate(epochs):
        epoch_id = f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
        record_id = f"sol_{index:04d}"
        scheduler.assign_candidate(epoch_id, record_id)
        scores[record_id] = float(index + 1)
    return scheduler, epochs, scores


def test_initialize(tmp_path):
    scheduler = make_scheduler(tmp_path)

    epochs = scheduler.initialize(
        ["sol_seed"], context_round=0, base_proposal_seed=23
    )

    assert len(epochs) == 4
    assert all(epoch.status == "active" for epoch in epochs)
    assert [epoch.island_id for epoch in epochs] == [
        "island_00",
        "island_01",
        "island_02",
        "island_03",
    ]
    assert [epoch.proposal_seed for epoch in epochs] == [23, 10_023, 20_023, 30_023]
    assert all(epoch.seed_record_ids == ["sol_seed"] for epoch in epochs)


def test_initialize_twice_raises(tmp_path):
    scheduler = make_scheduler(tmp_path)
    scheduler.initialize(["sol_seed"], context_round=0, base_proposal_seed=1)

    with pytest.raises(RuntimeError, match="already initialized"):
        scheduler.initialize(["other_seed"], context_round=1, base_proposal_seed=2)


def test_assign_and_get_records(tmp_path):
    scheduler = make_scheduler(tmp_path)
    scheduler.initialize(["baseline"], context_round=0, base_proposal_seed=1)

    scheduler.assign_candidate("island_00_epoch_00", "sol_0001")
    scheduler.assign_candidate("island_00_epoch_00", "sol_0002")

    assert scheduler.get_island_records("island_00_epoch_00") == [
        "baseline",
        "sol_0001",
        "sol_0002",
    ]


def test_sample_island(tmp_path):
    scheduler = make_scheduler(tmp_path)
    scheduler.initialize(["baseline"], context_round=0, base_proposal_seed=1)
    active_ids = {
        f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
        for epoch in scheduler.get_active_epochs()
    }

    sampled = scheduler.sample_island_for_exploration(rng_seed=42)

    assert sampled in active_ids
    assert sampled == scheduler.sample_island_for_exploration(rng_seed=42)


def test_should_review(tmp_path):
    scheduler = make_scheduler(tmp_path, review_interval=10)
    scheduler.initialize(["baseline"], context_round=0, base_proposal_seed=1)

    assert not scheduler.should_review(0)
    assert not scheduler.should_review(9)
    assert scheduler.should_review(10)
    assert not scheduler.should_review(11)

    for index in range(4):
        scheduler.assign_candidate(f"island_{index:02d}_epoch_00", f"sol_{index}")
    scheduler.run_review(10, {f"sol_{index}": float(index) for index in range(4)})

    assert not scheduler.should_review(10)
    assert not scheduler.should_review(19)
    assert scheduler.should_review(20)


def test_run_review_culls_worst(tmp_path):
    scheduler, _, scores = initialize_with_scored_records(tmp_path)

    replacements = scheduler.run_review(context_round=10, scores=scores)

    assert set(replacements) == {"island_00_epoch_00", "island_01_epoch_00"}
    assert scheduler.get_epoch("island_00_epoch_00").status == "culled"
    assert scheduler.get_epoch("island_01_epoch_00").status == "culled"
    assert scheduler.get_epoch("island_02_epoch_00").status == "active"
    assert scheduler.get_epoch("island_03_epoch_00").status == "active"


def test_run_review_tie_breaking(tmp_path):
    scheduler, _, scores = initialize_with_scored_records(tmp_path)
    replacements = scheduler.run_review(context_round=10, scores=scores)
    for new_epoch_id in replacements.values():
        record_id = f"new_{new_epoch_id}"
        scheduler.assign_candidate(new_epoch_id, record_id)

    active = scheduler.get_active_epochs()
    equal_scores = {
        record_id: 5.0
        for epoch in active
        for record_id in scheduler.get_island_records(
            f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
        )
    }
    second_replacements = scheduler.run_review(20, equal_scores)

    assert set(second_replacements) == set(replacements.values())
    assert scheduler.get_epoch("island_02_epoch_00").status == "active"
    assert scheduler.get_epoch("island_03_epoch_00").status == "active"


def test_run_review_creates_new_epochs(tmp_path):
    scheduler, _, scores = initialize_with_scored_records(tmp_path)

    replacements = scheduler.run_review(context_round=10, scores=scores)

    assert replacements == {
        "island_01_epoch_00": "island_01_epoch_01",
        "island_00_epoch_00": "island_00_epoch_01",
    }
    for old_epoch_id, new_epoch_id in replacements.items():
        new_epoch = scheduler.get_epoch(new_epoch_id)
        assert new_epoch is not None
        assert new_epoch.status == "active"
        assert new_epoch.epoch == 1
        assert new_epoch.started_after_context_round == 10
        assert new_epoch.seed_record_ids in (["sol_0002"], ["sol_0003"])
        assert new_epoch.proposal_seed != scheduler.get_epoch(old_epoch_id).proposal_seed
        assert scheduler.get_island_records(new_epoch_id) == new_epoch.seed_record_ids


def test_persistence_round_trip(tmp_path):
    state_path = tmp_path / "nested" / "islands.json"
    scheduler = IslandScheduler(state_path)
    scheduler.initialize(["baseline"], context_round=0, base_proposal_seed=11)
    scheduler.assign_candidate("island_00_epoch_00", "sol_0001")

    restored = IslandScheduler(state_path)

    assert [epoch.to_dict() for epoch in restored.get_all_epochs()] == [
        epoch.to_dict() for epoch in scheduler.get_all_epochs()
    ]
    assert restored.get_island_records("island_00_epoch_00") == [
        "baseline", "sol_0001"
    ]
    assert restored.should_review(10)


def test_get_epoch_by_id(tmp_path):
    scheduler = make_scheduler(tmp_path)
    scheduler.initialize(["baseline"], context_round=3, base_proposal_seed=5)

    epoch = scheduler.get_epoch("island_02_epoch_00")

    assert epoch is not None
    assert epoch.island_id == "island_02"
    assert epoch.epoch == 0
    assert scheduler.get_epoch("island_99_epoch_99") is None


def test_culled_epochs_not_active(tmp_path):
    scheduler, _, scores = initialize_with_scored_records(tmp_path)
    replacements = scheduler.run_review(context_round=10, scores=scores)

    active_ids = {
        f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
        for epoch in scheduler.get_active_epochs()
    }

    assert set(replacements).isdisjoint(active_ids)
    assert set(replacements.values()).issubset(active_ids)
    assert len(active_ids) == 4
