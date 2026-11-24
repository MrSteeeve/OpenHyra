"""Exercise the four registered live proposals without fitting their models."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from stochopia.neural.primitives import seed_lob_source

from .formal import _metric_rules
from .formal_manifest import ARM_MODELS
from .gate import CandidateGate
from .protocol import (
    CompileAttempt, DynamicHypothesis, Messages, ProviderReply, ProposalValidationError,
    StructuredPrediction, TwoStepProposer,
)
from .provider import ProviderAttemptError, registry_transport
from .request_log import ProposalBudgetExhausted, TrialCalls
from .store import Json, Ledger
from .token_count import load_token_counters


ROOT = Path("/tmp/p1-m4-v2-smoke")
TOKEN_LIMIT = 15_000
SUPPLEMENT_ARMS = ("metrics_ds", "evidence_gpt")
SUPPLEMENT_LIMIT = 10_000
SMOKE_PARENT_SOURCE = """from stochopia.neural.primitives import SeedGRULOBModel

class CandidateLOBWorldModel(SeedGRULOBModel):
    # Unmodified 48-state GRU with a 10-level book encoder and four masked heads.
    # Inherited modules: token_embed, input_projection, gru, heads.
    # _features(observation) returns projected features with last dimension 48.
    # sequence_context and step_context both call _features; head_logits is shared.
    # Override _features with super()._features(observation) to reuse the encoder.
    pass
"""


class SmokeBudget:
    def __init__(self, ledger: Ledger, *, limit: int) -> None:
        self.ledger = ledger
        self.limit = limit
        self.requests = 0
        self.spent = 0
        self.estimated_unknown = 0

    def bind(self, registry: Path, alias: str, count: Callable[[Messages], int]) -> Callable[..., ProviderReply]:
        def send(messages: Messages, *, response_json: bool) -> ProviderReply:
            estimate = count(messages)
            available = self.limit - self.spent - self.estimated_unknown - estimate
            completion = min(600 if response_json else 1_050, available)
            if completion < 240:
                raise ProposalBudgetExhausted("smoke budget cannot safely cover another request")
            self.requests += 1
            try:
                reply = registry_transport(registry, alias, completion_limit=completion)(
                    messages, response_json=response_json,
                )
            except ProviderAttemptError as exc:
                actual_input = exc.input_tokens
                actual_output = exc.output_tokens
                if actual_input is None or actual_output is None:
                    self.estimated_unknown += estimate + completion
                else:
                    self.spent += actual_input + actual_output
                self.ledger.seal({"input_tokens": actual_input, "output_tokens": actual_output,
                                  "output_limit": completion, "input_estimate": estimate},
                                 "budget", f"{self.requests:02d}.json")
                raise
            self.spent += reply.input_tokens + reply.output_tokens
            self.ledger.seal({"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                              "output_limit": completion, "input_estimate": estimate},
                             "budget", f"{self.requests:02d}.json")
            if self.spent + self.estimated_unknown > self.limit:
                raise RuntimeError(f"the provider token count exceeded the {self.limit:,}-token smoke limit")
            return reply

        return send


def _probe(
    ledger: Ledger, budget: SmokeBudget, arm: str, alias: str, registry: Path, evidence: Json,
    count: Callable[[Messages], int], *, supplement: bool = False,
) -> Json:
    gate = CandidateGate()
    adapter = TwoStepProposer(budget.bind(registry, alias, count), gate, lambda _call: None, count_tokens=count)
    calls = TrialCalls(ledger, (arm,), adapter)

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        edited = [dict(message) for message in messages]
        if response_json:
            extra = "Zero-data smoke only. Use one short sentence per hypothesis field and a small implementable change."
        else:
            extra = ("Zero-data smoke only. Keep the runnable candidate concise. You may subclass the trusted "
                     "SeedGRULOBModel from stochopia.neural.primitives and override only the changed methods.")
        if supplement:
            extra += (" Use one genuine, compact feature change in the shared _features hook. "
                      "Start from super()._features(observation) and keep its [*,48] shape. "
                      "Use documented attributes and StepObservation fields. The inherited sequence and step already share the hook.")
        edited[0]["content"] += f"\n{extra}"
        return calls.transport(edited, response_json=response_json)

    def seal(hypothesis: DynamicHypothesis, predictions: tuple[StructuredPrediction, ...]) -> None:
        ledger.seal({"hypothesis": asdict(hypothesis), "predictions": [asdict(item) for item in predictions]},
                    arm, "preregistration.json")

    def compile_record(attempt: CompileAttempt) -> None:
        ledger.seal(asdict(attempt), arm, "compiler", f"{attempt.attempt:02d}.json")

    mode = arm.split("_", 1)[0]
    proposer = TwoStepProposer(
        transport, gate, calls.record, count_tokens=count,
        core_metrics=tuple(rule.metric for rule in _metric_rules()), record_compile=compile_record,
        search_seed=17, search_round=1,
    )
    source = seed_lob_source() if supplement and arm == "evidence_gpt" else SMOKE_PARENT_SOURCE
    try:
        trial = proposer.propose(mode, source, evidence, (), seal,
                                 trial_id=f"smoke-{arm}", active_parent_hypothesis_id=None)
    except (ProposalValidationError, ProposalBudgetExhausted, ProviderAttemptError) as exc:
        return {"passed": False, "error": str(exc)}
    primary = trial.predictions[0]
    direction_passed = mode != "metrics" or (primary.metric == "lob_impact_score" and primary.direction == "increase")
    return {"passed": trial.source is not None and direction_passed, "compiled": trial.source is not None,
            "primary_metric": primary.metric, "primary_direction": primary.direction, "error": trial.error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--tokenizers-dir", type=Path, required=True)
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()
    evidence = json.loads(args.baseline.read_text(encoding="utf-8"))["metrics"]
    counters = load_token_counters(args.tokenizers_dir)
    arm_names = tuple(arm for arm, alias in ARM_MODELS.items() if alias is not None)
    limit = TOKEN_LIMIT - 1
    root = ROOT
    if args.supplement:
        original = json.loads((ROOT / "api_report.json").read_text(encoding="utf-8"))
        if set(original["arms"]) != set(arm_names):
            raise ValueError("supplement requires the original four-arm smoke receipt")
        failed = {arm for arm, row in original["arms"].items() if not row["passed"]}
        if failed != set(SUPPLEMENT_ARMS):
            raise ValueError("supplement is authorized only for the two original failed arms")
        arm_names = SUPPLEMENT_ARMS
        limit = SUPPLEMENT_LIMIT
        root = ROOT / "supplement"
    ledger = Ledger(root)
    with ledger.lock():
        if ledger.read("api_started.json") is not None:
            raise SystemExit("smoke requests are already reserved in this directory")
        ledger.seal({"token_limit": limit, "arms": arm_names}, "api_started.json")
        budget = SmokeBudget(ledger, limit=limit)
        rows = {}
        for arm in arm_names:
            alias = ARM_MODELS[arm]
            rows[arm] = _probe(ledger, budget, arm, alias, args.models, evidence, counters[alias],
                               supplement=args.supplement)
            ledger.seal(rows[arm], arm, "result.json")
            if args.supplement and not rows[arm]["passed"]:
                break
        report = ledger.seal({"arms": rows, "requests": budget.requests, "tokens": budget.spent,
                              "unknown_token_reservation": budget.estimated_unknown}, "api_report.json")
    print(json.dumps(report, ensure_ascii=False))
    if set(rows) != set(arm_names) or not all(row["passed"] for row in rows.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
