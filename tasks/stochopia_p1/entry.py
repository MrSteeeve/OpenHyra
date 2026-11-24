"""Run registered paired trials and evolve each group's hypotheses independently."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from island_scheduler import IslandScheduler

from .no_llm_mutator import mutate
from .paired_stats import EVALUATION_FAILED, VERIFIED_GAIN, assess, failed_evaluation, predictions_from_payload
from .protocol import (
    MAX_REPAIRS, CallUsage, ContextLimitExceeded, DynamicHypothesis, Messages, ProposalValidationError,
    ProviderReply, StructuredPrediction, SynthesisResult, TwoStepProposer,
)
from .store import (
    GROUPS, CandidateEvaluationFailed, DevelopmentExport, EvaluationRequest, Evaluator, Json,
    Ledger, ModelRef, PairedEvaluation, StudyPlan,
)


INVALID_PLAN = "invalid_plan"
INVALID_PROGRAM = "invalid_program"
CandidateGate = Callable[[str], str | None]


class ProposalBudgetExhausted(RuntimeError):
    """The persistent request budget has already been spent for this stage."""


# --- Groups, compiler gate and longitudinal ledger ---


def _round_name(index: int) -> str:
    return f"round-{index:03d}"


def _feedback(trial: Json) -> Json:
    status = trial["verdict"]
    forecasts = trial.get("prediction_verdicts", [])
    if forecasts:
        status = f"{status}; core={forecasts[0]['status']}"
    return {"hypothesis": trial["hypothesis"], "verdict": status}


class DevelopmentLoop:
    """Advance each group once per round with no access to the verifier root."""

    def __init__(
        self, root: Path, plan: StudyPlan, *, proposers: Mapping[str, TwoStepProposer],
        gate: CandidateGate, evaluate: Evaluator,
    ) -> None:
        if set(proposers) != {"evidence", "metrics"}:
            raise ValueError("schedule one isolated proposer for each LLM group")
        self.ledger = Ledger(root)
        self.plan = plan
        self.proposers = proposers
        self.gate = gate
        self.evaluate = evaluate
        self.rules = {rule.metric: rule for rule in plan.metric_rules}

    def _history(self, group: str, before: int) -> tuple[Json, ...]:
        history: list[Json] = []
        for index in range(1, before):
            result = self.ledger.read(group, _round_name(index), "result.json")
            if result is None:
                break
            history.append(result)
        return tuple(history)

    def _frontier(self, group: str, history: Sequence[Json]) -> tuple[IslandScheduler, str, ModelRef, Mapping[str, object]]:
        scheduler = IslandScheduler(self.ledger.file(group, "islands.json"), num_islands=self.plan.rounds)
        if not scheduler.get_active_epochs():
            scheduler.initialize([self.plan.seed_model.model_id], context_round=0, base_proposal_seed=self.plan.mutation_seed)
        if history and history[-1]["verdict"] == VERIFIED_GAIN:
            prior = history[-1]
            return (scheduler, prior["island"], ModelRef(**prior["candidate"]), prior["evaluation"]["candidate_evidence"])
        occupied = {row["island"] for row in history}
        island = next(f"{e.island_id}_epoch_{e.epoch:02d}" for e in scheduler.get_active_epochs()
                      if f"{e.island_id}_epoch_{e.epoch:02d}" not in occupied)
        return scheduler, island, self.plan.seed_model, self.plan.seed_evidence

    def _proposer(self, group: str, path: tuple[str, ...]) -> TwoStepProposer:
        adapter = self.proposers[group]
        sequence = 1
        while self.ledger.read(*path, "usage", f"{sequence:02d}.json") is not None:
            sequence += 1

        def record(usage: CallUsage) -> None:
            nonlocal sequence
            self.ledger.seal(asdict(usage), *path, "usage", f"{sequence:02d}.json")
            sequence += 1
            adapter.record_usage(usage)

        def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
            phase = "hypothesis" if response_json else "synthesis"
            budget = 1 if response_json else MAX_REPAIRS + 1
            for attempt in range(1, budget + 1):
                reservation = (*path, "requests", f"{phase}-{attempt:02d}.json")
                if self.ledger.read(*reservation) is None:
                    self.ledger.seal({"phase": phase, "attempt": attempt}, *reservation)
                    return adapter.transport(messages, response_json=response_json)
            raise ProposalBudgetExhausted(f"{phase} request budget exhausted")

        return TwoStepProposer(transport, self.gate, record, count_tokens=adapter.count_tokens,
                               core_metrics=tuple(self.rules))

    def _synthesize(
        self, group: str, index: int, parent: ModelRef, evidence: Mapping[str, object], history: tuple[Json, ...],
    ) -> Json:
        path = (group, _round_name(index))
        compiled = self.ledger.read(*path, "program.json")
        if compiled is not None:
            return compiled
        attempted = [row for row in history if row["hypothesis"] is not None]

        def seal(hypothesis: DynamicHypothesis, predictions: tuple[StructuredPrediction, ...]) -> None:
            local_ids = {row["hypothesis"]["hypothesis_id"] for row in attempted}
            continuation = None
            if history and history[-1]["verdict"] == VERIFIED_GAIN:
                continuation = history[-1]["hypothesis"]["hypothesis_id"]
            if hypothesis.hypothesis_id in local_ids:
                raise ProposalValidationError("hypothesis IDs must be new within a group")
            if hypothesis.parent_hypothesis_id is not None and hypothesis.parent_hypothesis_id not in local_ids:
                raise ProposalValidationError("a hypothesis can reference only its own group's earlier hypotheses")
            if continuation is not None and group != "metrics" and hypothesis.parent_hypothesis_id != continuation:
                raise ProposalValidationError("an island's next hypothesis must cite its promoted parent hypothesis")
            if not predictions or predictions[0].metric not in self.rules:
                raise ProposalValidationError("the first forecast must cover a registered core metric")
            self.ledger.seal({"hypothesis": asdict(hypothesis), "predictions": [asdict(p) for p in predictions]},
                             *path, "preregistration.json")

        previous = self.ledger.read(*path, "preregistration.json")
        recent = [_feedback(row) for row in attempted[-3:]]
        if group == "no_llm":
            edited = mutate(self.plan.mutation_seed + index)
            rule = self.plan.metric_rules[0]
            parent_id = history[-1]["hypothesis"]["hypothesis_id"] if history else None
            hypothesis = DynamicHypothesis(f"baseline-{index}", parent_id, rule.metric, "Random architecture baseline",
                                           edited.family, "The registered improvement range does not contain the effect")
            predictions = (StructuredPrediction(rule.metric, rule.direction, rule.no_llm_interval),)
            seal(hypothesis, predictions)
            error = self.gate(edited.source)
            proposal = SynthesisResult(hypothesis, predictions, edited.source if error is None else None, error is not None, error)
        else:
            proposer = self._proposer(group, path)
            context_history = recent if group == "evidence" else []
            try:
                if previous is None:
                    proposal = proposer.propose(group, parent.source, evidence, context_history, seal)
                else:
                    hypothesis = DynamicHypothesis(**previous["hypothesis"])
                    proposal = proposer.synthesize(group, parent.source, evidence, context_history,
                                                  hypothesis, predictions_from_payload(previous))
            except (ContextLimitExceeded, ProposalValidationError, ProposalBudgetExhausted) as exc:
                registered = self.ledger.read(*path, "preregistration.json")
                if registered is None:
                    return self.ledger.seal({"status": INVALID_PLAN, "error": str(exc)}, *path, "program.json")
                proposal = SynthesisResult(DynamicHypothesis(**registered["hypothesis"]),
                                           predictions_from_payload(registered), None, True, str(exc))

        preregistered = self.ledger.read(*path, "preregistration.json")
        if preregistered is None:
            raise RuntimeError("the synthesis adapter did not preregister its hypothesis")
        seal(proposal.hypothesis, proposal.predictions)
        status = INVALID_PROGRAM if proposal.invalid_program else "ready"
        return self.ledger.seal({"source": proposal.source, "status": status,
                                 "error": proposal.error}, *path, "program.json")

    def _run_trial(self, group: str, index: int) -> Json:
        history = self._history(group, index)
        scheduler, island, parent, evidence = self._frontier(group, history)
        path = (group, _round_name(index))
        program = self._synthesize(group, index, parent, evidence, history)
        registered = self.ledger.read(*path, "preregistration.json")
        forecasts = registered or {"hypothesis": None, "predictions": []}
        trial_id = f"{group}-r{index:02d}"
        common = {**forecasts, "trial_id": trial_id, "round": index, "island": island, "parent": asdict(parent)}
        if history and history[-1]["verdict"] == VERIFIED_GAIN:
            common["promoted_parent_hypothesis_id"] = history[-1]["hypothesis"]["hypothesis_id"]
        if program["status"] != "ready":
            return self.ledger.seal({**common, "candidate": None, "verdict": program["status"],
                                     "error": program["error"], "training_budget_consumed": False}, *path, "result.json")
        candidate = ModelRef(trial_id, program["source"])
        request = EvaluationRequest(trial_id, group, parent, candidate, self.plan.train_dates, self.plan.dev_dates,
                                    self.plan.train_seeds, self.plan.rollout_seed, "dev")
        self.ledger.seal(asdict(request), *path, "evaluation-request.json")
        recorded = self.ledger.read(*path, "evaluation.json")
        predictions = predictions_from_payload(forecasts)
        if recorded is None:
            try:
                pair = self.evaluate(request)
            except CandidateEvaluationFailed as exc:
                result = failed_evaluation(predictions, str(exc))
            else:
                pair.validate(request)
                result = pair.to_dict()
            recorded = self.ledger.seal(result, *path, "evaluation.json")
        if recorded.get("verdict") == EVALUATION_FAILED:
            return self.ledger.seal({**common, **recorded, "candidate": asdict(candidate)}, *path, "result.json")
        pair = PairedEvaluation.from_dict(recorded)
        assessment = assess(pair, request, predictions, self.rules, self.plan.nll_guardrail)
        if assessment["verdict"] == VERIFIED_GAIN:
            scheduler.assign_candidate(island, candidate.model_id)
        return self.ledger.seal({**common, **assessment, "candidate": asdict(candidate), "training_budget_consumed": True,
                                 "evaluation": pair.to_dict()}, *path, "result.json")

    def run_round(self) -> dict[str, Json]:
        """Resume the first unfinished round and complete its three groups."""
        with self.ledger.lock():
            self.ledger.seal(asdict(self.plan), "study.json")
            for index in range(1, self.plan.rounds + 1):
                completed = {group: self.ledger.read(group, _round_name(index), "result.json") for group in GROUPS}
                if any(result is None for result in completed.values()):
                    return {group: completed[group] or self._run_trial(group, index) for group in GROUPS}
        raise RuntimeError("all registered development rounds are complete")

    def export(self) -> DevelopmentExport:
        """Detach terminal rows with no holdout path or holdout date visibility."""
        with self.ledger.lock():
            self.ledger.seal(asdict(self.plan), "study.json")
            groups = {group: self._history(group, self.plan.rounds + 1) for group in GROUPS}
            if any(len(rows) != self.plan.rounds for rows in groups.values()):
                raise ValueError("finish all registered rounds before exporting")
            return DevelopmentExport(self.ledger.path, self.plan, groups)
