from pathlib import Path

from intervention_router import (
    AcquisitionRouter,
    PendingHypothesisQueue,
    route_hypotheses,
)
from stopping import ContextDecision


def _hypothesis(identifier, **extra):
    payload = {
        "id": identifier,
        "family": "representation",
        "mechanism": f"mechanism {identifier}",
        "prediction": "improves the boundary slice",
        "failure_condition": "no held-out gain",
        "matched_control": "same parent and budget",
    }
    payload.update(extra)
    return payload


def test_typed_context_decision_round_trip():
    decision = ContextDecision.from_payload({
        "action": "continue",
        "analysis": "The boundary remains uncertain.",
        "reason": "Run a targeted probe.",
        "next": "probe the boundary",
        "phase": "diagnose",
        "intervention": {
            "scope": "representation",
            "operator": "replace",
            "target_slice": "near-boundary",
            "prediction": "exercise error falls",
            "falsifier": "no change on held-out cells",
            "evidence_ids": ["sol_0001"],
            "next_probe": "finite-difference boundary margin",
            "state_version": "state-3",
        },
    })
    assert decision.phase == "diagnose"
    assert decision.intervention_scope == "representation"
    payload = decision.to_dict()
    assert payload["intervention"]["operator"] == "replace"
    assert ContextDecision.from_payload(payload) == decision


def test_router_persists_unselected_and_updates_status(tmp_path: Path):
    queue = PendingHypothesisQueue(tmp_path / "pending.json")
    router = AcquisitionRouter(queue)
    selected = router.select(
        [_hypothesis("a"), _hypothesis("b"), _hypothesis("c")],
        count=1,
        iteration=0,
    )
    assert len(selected) == 1
    assert len(queue.entries()) == 3
    chosen = selected[0]["id"]
    router.observe_result(chosen, improved=False, result_id="sol_bad", iteration=0)
    assert queue.get(chosen).status == "refuted"
    resumed = PendingHypothesisQueue(tmp_path / "pending.json")
    assert resumed.get(chosen).status == "refuted"
    assert any(item["hypothesis"]["id"] != chosen for item in resumed.pending())


def test_route_hypotheses_prefers_expected_gain_and_returns_pending(tmp_path: Path):
    queue = PendingHypothesisQueue(tmp_path / "pending.json")
    selected, pending = route_hypotheses(
        [
            _hypothesis("low", expected_gain=0.0),
            _hypothesis("high", expected_gain=2.0),
        ],
        count=1,
        queue=queue,
    )
    assert selected[0]["id"] == "high"
    assert pending
