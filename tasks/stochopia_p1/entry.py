"""Run registered paired trials and evolve each group's hypotheses independently."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from island_scheduler import IslandScheduler

from .no_llm_mutator import mutate, placeholder_forecasts
from .paired_stats import (
    EVALUATION_FAILED, GUARDRAIL_VIOLATION, REFUTED, VERIFIED_GAIN, assess,
    failed_evaluation, predictions_from_payload,
)
from .protocol import (
    CompileAttempt, ContextLimitExceeded, DynamicHypothesis, ProposalValidationError,
    StructuredPrediction, SynthesisResult, TwoStepProposer, validate_forecasts,
)
from .request_log import ProposalBudgetExhausted, TrialCalls
from .store import (
    CandidateEvaluationFailed, DevelopmentExport, EvaluationRequest, Evaluator, Json, Ledger,
    ModelRef, NonfiniteCandidateLoss, PairedEvaluation, StudyPlan, group_mode,
)


INVALID_PLAN = "invalid_plan"
INVALID_PROGRAM = "INVALID_PROGRAM"
CandidateGate = Callable[[str], str | None]


# --- Groups, compiler gate and longitudinal ledger ---


def _round_name(index: int) -> str:
    return f"round-{index:03d}"


def _feedback(trial: Json) -> Json:
    status = trial["verdict"]
    forecasts = trial.get("prediction_verdicts", [])
    if forecasts:
        status = f"{status}; core={forecasts[0]['status']}"
        status += f"; guardrail={forecasts[1]['status']}"
    reason = trial.get("reason") or trial.get("error")
    if reason is None and trial["verdict"] == GUARDRAIL_VIOLATION:
        guards = trial["guardrails"]
        failed = [name for name in ("nll", "exhaustion", "drift") if not guards[f"{name}_passed"]]
        reason = "failed physical guardrails: " + ", ".join(failed)
    if reason is None and trial["verdict"] == REFUTED:
        reason = "benchmark gain did not pass both thresholds"
    if reason:
        status += f"; reason={str(reason)[:250]}"
    return {"hypothesis": trial["hypothesis"], "verdict": status}


def _latest_verified(history: Sequence[Json]) -> Json | None:
    return next((trial for trial in reversed(history) if trial["verdict"] == VERIFIED_GAIN), None)


class DevelopmentLoop:
    """Advance each group once per round with no access to the verifier root."""

    def __init__(
        self, root: Path, plan: StudyPlan, *, proposers: Mapping[str, TwoStepProposer],
        gate: CandidateGate, evaluate: Evaluator,
    ) -> None:
        expected_proposers = {group for group in plan.groups if group_mode(group) != "no_llm"}
        if set(proposers) != expected_proposers:
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
        scheduler = IslandScheduler(self.ledger.file(group, "islands.json"), num_islands=1)
        if not scheduler.get_active_epochs():
            scheduler.initialize([self.plan.seed_model.model_id], context_round=0, base_proposal_seed=self.plan.mutation_seed)
        prior = _latest_verified(history)
        if prior is not None:
            return (scheduler, prior["island"], ModelRef(**prior["candidate"]), prior["evaluation"]["candidate_evidence"])
        epoch = scheduler.get_active_epochs()[0]
        island = f"{epoch.island_id}_epoch_{epoch.epoch:02d}"
        return scheduler, island, self.plan.seed_model, self.plan.seed_evidence

    def _proposer(self, group: str, path: tuple[str, ...], index: int) -> TwoStepProposer:
        adapter = self.proposers[group]
        ledger = TrialCalls(self.ledger, path, adapter)

        def record_compile(attempt: CompileAttempt) -> None:
            self.ledger.seal(asdict(attempt), *path, "compiler", f"{attempt.attempt:02d}.json")

        def read_compile(index: int) -> CompileAttempt | None:
            row = self.ledger.read(*path, "compiler", f"{index:02d}.json")
            return CompileAttempt(**row) if row is not None else None

        return TwoStepProposer(
            ledger.transport, self.gate, ledger.record, count_tokens=adapter.count_tokens,
            core_metrics=tuple(self.rules), record_compile=record_compile, read_compile=read_compile,
            search_seed=self.plan.mutation_seed, search_round=index,
        )

    def _synthesize(
        self, group: str, index: int, parent: ModelRef, evidence: Mapping[str, object], history: tuple[Json, ...],
    ) -> Json:
        path = (group, _round_name(index))
        compiled = self.ledger.read(*path, "program.json")
        if compiled is not None:
            return compiled
        attempted = [row for row in history if row["hypothesis"] is not None]
        mode = group_mode(group)
        prior = _latest_verified(history)
        continuation = prior["hypothesis"]["hypothesis_id"] if prior is not None else None

        def seal(hypothesis: DynamicHypothesis, predictions: tuple[StructuredPrediction, ...]) -> None:
            local_ids = {row["hypothesis"]["hypothesis_id"] for row in attempted}
            if hypothesis.hypothesis_id in local_ids:
                raise ProposalValidationError("hypothesis IDs must be new within a group")
            if hypothesis.parent_hypothesis_id is not None and hypothesis.parent_hypothesis_id not in local_ids:
                raise ProposalValidationError("a hypothesis can reference only its own group's earlier hypotheses")
            if continuation is not None and mode != "metrics" and hypothesis.parent_hypothesis_id != continuation:
                raise ProposalValidationError("an island's next hypothesis must cite its promoted parent hypothesis")
            validate_forecasts(mode, predictions, tuple(self.rules))
            self.ledger.seal({"hypothesis": asdict(hypothesis), "predictions": [asdict(p) for p in predictions]},
                             *path, "preregistration.json")

        previous = self.ledger.read(*path, "preregistration.json")
        recent = [_feedback(row) for row in attempted[-3:]]
        if mode == "no_llm":
            edited = mutate(self.plan.mutation_seed + index)
            hypothesis = DynamicHypothesis(f"baseline-{index}", continuation, "Benchmark score", "Random architecture baseline",
                                           edited.family, "The registered improvement range does not contain the effect")
            predictions = placeholder_forecasts()
            seal(hypothesis, predictions)
            compiled_path = (*path, "compiler", "01.json")
            checked = self.ledger.read(*compiled_path)
            if checked is None:
                error = self.gate(edited.source)
                checked = self.ledger.seal(asdict(CompileAttempt(1, "code", edited.source, error)), *compiled_path)
            error = checked["diagnostic"]
            proposal = SynthesisResult(hypothesis, predictions, edited.source if error is None else None, error is not None, error)
        else:
            proposer = self._proposer(group, path, index)
            context_history = recent if mode == "evidence" else []
            try:
                if previous is None:
                    proposal = proposer.propose(mode, parent.source, evidence, context_history, seal)
                else:
                    hypothesis = DynamicHypothesis(**previous["hypothesis"])
                    proposal = proposer.synthesize(mode, parent.source, evidence, context_history,
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
        if self.plan.trial_prefix:
            trial_id = f"{self.plan.trial_prefix}-{trial_id}"
        common = {**forecasts, "trial_id": trial_id, "round": index, "island": island, "parent": asdict(parent)}
        prior = _latest_verified(history)
        if prior is not None:
            common["promoted_parent_hypothesis_id"] = prior["hypothesis"]["hypothesis_id"]
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
            except NonfiniteCandidateLoss as exc:
                result = failed_evaluation(predictions, str(exc))
                result.update(verdict=GUARDRAIL_VIOLATION, reason=str(exc))
            except CandidateEvaluationFailed as exc:
                result = failed_evaluation(predictions, str(exc))
            else:
                pair.validate(request)
                result = pair.to_dict()
            recorded = self.ledger.seal(result, *path, "evaluation.json")
        if recorded.get("verdict") in (EVALUATION_FAILED, GUARDRAIL_VIOLATION):
            return self.ledger.seal({**common, **recorded, "candidate": asdict(candidate)}, *path, "result.json")
        pair = PairedEvaluation.from_dict(recorded)
        assessment = assess(pair, request, predictions)
        if assessment["verdict"] == VERIFIED_GAIN:
            scheduler.assign_candidate(island, candidate.model_id)
        return self.ledger.seal({**common, **assessment, "candidate": asdict(candidate), "training_budget_consumed": True,
                                 "evaluation": pair.to_dict()}, *path, "result.json")

    def run_round(self) -> dict[str, Json]:
        """Resume the first unfinished round and complete its registered arms."""
        with self.ledger.lock():
            self.ledger.seal(asdict(self.plan), "study.json")
            for index in range(1, self.plan.rounds + 1):
                completed = {
                    group: self.ledger.read(group, _round_name(index), "result.json") for group in self.plan.groups
                }
                if any(result is None for result in completed.values()):
                    return {group: completed[group] or self._run_trial(group, index) for group in self.plan.groups}
        raise RuntimeError("all registered development rounds are complete")

    def export(self) -> DevelopmentExport:
        """Detach terminal rows with no holdout path or holdout date visibility."""
        with self.ledger.lock():
            self.ledger.seal(asdict(self.plan), "study.json")
            groups = {group: self._history(group, self.plan.rounds + 1) for group in self.plan.groups}
            if any(len(rows) != self.plan.rounds for rows in groups.values()):
                raise ValueError("finish all registered rounds before exporting")
            return DevelopmentExport(self.ledger.path, self.plan, groups)
