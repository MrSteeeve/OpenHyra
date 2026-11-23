"""Project Tier 0 development measurements onto the native OpenHyra schemas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from urllib.parse import quote

from algorithm_discovery import AlgorithmSpec, DiscoveryEvent, DiscoveryLedger, EvaluationResult, FeedbackOracle
from feedback import DirectionalFeedback, FeedbackPacket, ProblemState, not_observed
from stochopia.harness.contract import (
    AnalysisSpec, CandidateRef, PairedDelta, PairedEvalResult, ReportStatus, SearchTrialRecord, StructuredPrediction,
)
from stochopia.harness.report import _gain, _primary, assess_candidate, judge_prediction, summarize_candidate

from .store import JsonObject


def score_development(
    analyses: tuple[AnalysisSpec, ...], results: tuple[PairedEvalResult, ...], unit_ids: tuple[str, ...],
) -> tuple[dict[str, PairedDelta], ReportStatus]:
    """Apply Stochopia's registered gates to the completed development measurements."""
    completed = tuple(result for result in results if result.status == "completed")
    if not completed:
        return {}, ReportStatus.UNAVAILABLE
    paired = summarize_candidate(analyses, completed, required_unit_ids=unit_ids)
    return paired, assess_candidate(analyses, paired)


def published_feedback(stored: JsonObject) -> JsonObject:
    """Keep the existing per-group JSON envelope for Context and disk readers."""
    return {
        "trial": stored["trial"], "paired_diffs": stored["paired_diffs"],
        "results": [
            {key: row[key] for key in ("result_id", "status", "failure_reason")}
            for row in stored["results"]
        ],
    }


def candidate_spec(
    candidate: CandidateRef, trial: SearchTrialRecord, predictions: tuple[StructuredPrediction, ...],
) -> AlgorithmSpec:
    """Keep the declared edit and the group's preregistered forecasts bound to the same candidate."""
    explanations = {row.explanation_id for row in predictions}
    return AlgorithmSpec(
        candidate_id=candidate.candidate_id, family=candidate.backbone,
        mechanism_id=next(iter(explanations)) if len(explanations) == 1 else candidate.recipe,
        operator=candidate.recipe,
        parent_ids=(candidate.parent_id,) if candidate.parent_id is not None else (),
        implementation=candidate.to_dict(),
        prediction=[row.to_dict() for row in predictions] if predictions else not_observed("no preregistered forecasts"),
        falsifier={row.prediction_id: row.falsifier for row in predictions if row.falsifier is not None}
        or not_observed("no preregistered falsifiers"),
        metadata={"exp_id": trial.exp_id, "group_id": trial.group_id, "scenario_id": trial.scenario_id,
                  "trial_id": trial.trial_id, "selection_id": trial.selection_id},
    )


def _feedback_direction(interval: list[float] | None) -> str:
    if interval is None:
        return "uncertain"
    low, high = interval
    if low > 0:
        return "increase"
    if high < 0:
        return "decrease"
    return "hold" if low == high == 0 else "uncertain"


def _packet_id(*parts: str) -> str:
    return "tier0:" + ":".join(quote(part, safe="") for part in parts)


class Tier0FeedbackOracle(FeedbackOracle):
    """Construct one group's native packet from its qualified development result."""

    def __init__(self, group_id: str) -> None:
        self.group_id = group_id

    def project(self, candidate: AlgorithmSpec, result: EvaluationResult) -> FeedbackPacket:
        if (result.split != "development" or result.metrics.get("group_id") != self.group_id
                or candidate.metadata.get("group_id") != self.group_id):
            raise ValueError("native feedback requires a development result from its registered group")
        if candidate.candidate_id != result.candidate_id:
            raise ValueError("native feedback requires matching candidate and result IDs")
        data = {key: result.metrics[key] for key in ("exp_id", "group_id", "scenario_id", "trial_id", "round_index")}
        data["split"] = result.split
        forecasts_by_analysis: dict[str, list[JsonObject]] = {}
        if isinstance(candidate.prediction, list):
            for forecast in candidate.prediction:
                forecasts_by_analysis.setdefault(forecast["analysis_id"], []).append(forecast)
        directions = []
        for analysis_id, delta in result.metrics["paired_diffs"].items():
            interval, estimate, reason = delta["interval"], delta["estimate"], delta["reason"]
            forecasts = forecasts_by_analysis.get(analysis_id, [])
            directions.append(DirectionalFeedback(
                id=analysis_id, candidate_id=candidate.candidate_id, mechanism_id=candidate.mechanism_id,
                slice_key=f"{data['scenario_id']}:{analysis_id}", direction=_feedback_direction(interval),
                prediction=forecasts or not_observed("no preregistered forecasts for this readout"),
                falsifier={row["prediction_id"]: row["falsifier"] for row in forecasts if row["falsifier"] is not None}
                or not_observed("no preregistered falsifiers for this readout"),
                observed={
                    "paired_delta": estimate if interval is not None else not_observed(reason or "no measured interval"),
                    "point_estimate": estimate, "interval": interval, "n_pairs": delta["n_pairs"], "unit": delta["unit"],
                },
                evidence={"result_ids": result.metrics["result_ids"]}, data={**data, "analysis_id": analysis_id},
            ))
        return FeedbackPacket(
            packet_id=_packet_id(data["exp_id"], self.group_id, data["trial_id"]), candidate_id=result.candidate_id,
            mechanism_id=candidate.mechanism_id, directional=directions,
            observed={"qualification": result.status, "primary_metric": result.metrics["primary_readout"],
                      "aggregate_effect": result.score,
                      "prediction_verdicts": result.metrics["prediction_verdicts"]},
            evidence={"result_ids": result.metrics["result_ids"]}, data=data,
        )


@dataclass(frozen=True)
class TrialProjection:
    candidate: AlgorithmSpec
    result: EvaluationResult


def project_trial(
    trial: SearchTrialRecord,
    candidate: CandidateRef,
    published: JsonObject,
    analyses: tuple[AnalysisSpec, ...],
    predictions: Iterable[StructuredPrediction],
    *,
    round_index: int,
    seed: int,
) -> TrialProjection:
    """Translate one published trial without reading held-out outcomes or another group's forecasts."""
    forecasts = tuple(
        row for row in predictions
        if (row.group_id, row.scenario_id, row.candidate_id) == (trial.group_id, trial.scenario_id, trial.candidate_id)
        and row.round_index <= round_index
    )
    algorithm = candidate_spec(candidate, trial, forecasts)
    deltas = {key: PairedDelta.from_dict(row) for key, row in published["paired_diffs"].items()}
    primary = _primary(analyses)
    score = _gain(primary, deltas.get(primary.analysis_id) if primary else None)
    result_ids = [row["result_id"] for row in published["results"]]
    measurement_id = result_ids[0] if len(result_ids) == 1 else f"aggregate:dev:{trial.trial_id}"
    analysis_by_id = {analysis.analysis_id: analysis for analysis in analyses}
    verdicts = [
        judge_prediction(
            row, analysis_by_id[row.analysis_id], deltas.get(row.analysis_id),
            result_id=measurement_id, model_score_delta=score,
        ).to_dict()
        for row in forecasts
    ]
    result = EvaluationResult(
        candidate_id=trial.candidate_id, status=trial.development_status.value, score=score,
        split="development", seed=seed, cost=trial.cost,
        metrics={
            "exp_id": trial.exp_id, "group_id": trial.group_id, "scenario_id": trial.scenario_id,
            "trial_id": trial.trial_id, "round_index": round_index, "purpose": trial.purpose,
            "result_ids": result_ids, "paired_diffs": published["paired_diffs"], "prediction_verdicts": verdicts,
            "primary_analysis_id": primary.analysis_id if primary is not None else None,
            "primary_readout": primary.readout if primary is not None else None,
        },
    )
    packet = Tier0FeedbackOracle(trial.group_id).project(algorithm, result)
    return TrialProjection(algorithm, replace(result, feedback=packet))


def append_discovery_events(
    ledger: DiscoveryLedger, *, exp_id: str, group_id: str,
    rounds: Iterable[tuple[int, tuple[TrialProjection, ...]]],
) -> None:
    """Append each trial once with a state rebuilt exclusively from this group's packets."""
    existing = {row["result"]["metrics"]["trial_id"] for row in ledger.read()}
    state = ProblemState(state_id=_packet_id(exp_id, group_id))
    for round_index, projections in rounds:
        for projection in projections:
            packet = projection.result.feedback
            if packet is not None:
                state = state.append(packet)
        state_hash = state.state_hash
        for projection in projections:
            trial_id = projection.result.metrics["trial_id"]
            if trial_id not in existing:
                ledger.append(DiscoveryEvent(
                    round_index=round_index, candidate=projection.candidate, result=projection.result,
                    state_version=state.state_version, state_hash=state_hash,
                ))
                existing.add(trial_id)
