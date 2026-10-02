from pathlib import Path

from algorithm_discovery import (
    AcquisitionPolicy,
    AlgorithmDiscoveryLoop,
    AlgorithmSpec,
    DiscoveryLedger,
    EvaluationResult,
)
from feedback import DirectionalFeedback, FeedbackPacket, ProblemStateLog


def _candidate(candidate_id, mechanism):
    return AlgorithmSpec(
        candidate_id=candidate_id,
        family=mechanism,
        mechanism_id=mechanism,
        implementation={"kind": mechanism},
        prediction={"direction": "positive"},
        falsifier={"condition": "delta < 0"},
    )


def _result(candidate_id, value, mechanism="m"):
    packet = FeedbackPacket(
        packet_id=f"packet-{candidate_id}",
        candidate_id=candidate_id,
        mechanism_id=mechanism,
        directional=[
            DirectionalFeedback(
                id=f"obs-{candidate_id}",
                candidate_id=candidate_id,
                mechanism_id=mechanism,
                slice_key="global",
                direction="positive" if value > 0 else "negative",
                observed={"paired_delta": value},
                recommendation={"next": "tune"},
            )
        ],
    )
    return EvaluationResult(
        candidate_id=candidate_id,
        status="ok",
        score=value,
        feedback=packet,
        split="development",
        seed=7,
    )


def test_round_barrier_persists_feedback_and_state(tmp_path: Path):
    ledger = DiscoveryLedger(tmp_path / "events.jsonl")
    state_log = ProblemStateLog(tmp_path / "feedback.jsonl")
    loop = AlgorithmDiscoveryLoop(ledger=ledger, state_log=state_log)
    candidates = [_candidate("a", "m"), _candidate("b", "n")]
    events = loop.run_round(
        candidates,
        lambda candidate: _result(
            candidate.candidate_id,
            1.0 if candidate.candidate_id == "a" else -1.0,
            mechanism=candidate.family,
        ),
        round_index=3,
    )
    assert len(events) == 2
    assert loop.state.state_version == 2
    assert loop.state.cells["m::global"].status == "uncertain"
    assert loop.state.cells["n::global"].mean == -1.0
    assert len(ledger.read()) == 2
    assert len(state_log.read()) == 2


def test_acquisition_prioritises_untried_coverage_deterministically():
    loop = AlgorithmDiscoveryLoop(acquisition=AcquisitionPolicy())
    ranked = loop.rank([_candidate("z", "known"), _candidate("a", "new")])
    assert ranked[0].candidate_id == "a"
    assert ranked[0].coverage_bonus == 1.0


def test_algorithm_spec_round_trip():
    candidate = _candidate("x", "symbolic")
    assert AlgorithmSpec.from_dict(candidate.to_dict()) == candidate
