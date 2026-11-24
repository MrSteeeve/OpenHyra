"""Persist provider attempts before charging a new development request."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter, sleep

from .protocol import MAX_REPAIRS, CallUsage, Messages, ProviderReply, TwoStepProposer
from .provider import ProviderAttemptError
from .store import Ledger


MAX_HTTP_RETRIES = 3


class ProposalBudgetExhausted(RuntimeError):
    """The registered logical request cannot be tried again."""


class TrialCalls:
    """Budget and replay metered proposal requests for one candidate only."""

    def __init__(self, ledger: Ledger, path: tuple[str, ...], adapter: TwoStepProposer) -> None:
        self.ledger = ledger
        self.path = path
        self.adapter = adapter
        self.synthesis = 0

    def record(self, usage: CallUsage) -> None:
        index = 1 if usage.stage == "hypothesis" else self.synthesis + 1
        path = (*self.path, "usage", f"{index:02d}.json")
        previous = self.ledger.read(*path)
        self.ledger.seal(asdict(usage), *path)
        if previous is None:
            self.adapter.record_usage(usage)

    def transport(self, messages: Messages, *, response_json: bool) -> ProviderReply:
        if response_json:
            phase = "hypothesis"
            stage = "hypothesis"
            logical_index = 1
        else:
            self.synthesis += 1
            phase = "synthesis"
            stage = "code" if self.synthesis == 1 else "repair"
            logical_index = self.synthesis
            if logical_index > MAX_REPAIRS + 1:
                raise ProposalBudgetExhausted("synthesis request budget exhausted")
        for http_index in range(1, MAX_HTTP_RETRIES + 2):
            stem = f"{phase}-{logical_index:02d}"
            if http_index > 1:
                stem += f"-http-{http_index:02d}"
            reservation_path = (*self.path, "requests", f"{stem}.json")
            result_path = (*self.path, "requests", f"{stem}-result.json")
            prior = self.ledger.read(*result_path)
            if prior is not None:
                reply = self._reply_or_retry(prior)
                if reply is not None:
                    return reply
                continue
            if self.ledger.read(*reservation_path) is not None:
                # The process may have stopped after spending the request.
                # This attempt has no response that can be replayed safely.
                raise ProposalBudgetExhausted(f"provider response unavailable after reserved {stage} request")
            if http_index > 1:
                sleep(2 ** (http_index - 2))
            self.ledger.seal({"phase": phase, "stage": stage, "attempt": logical_index, "http_attempt": http_index},
                             *reservation_path)
            started = perf_counter()
            try:
                reply = self.adapter.transport(messages, response_json=response_json)
            except ProviderAttemptError as exc:
                payload = {
                    "reply": None, "error": str(exc), "retryable": exc.retryable, "input_tokens": exc.input_tokens,
                    "output_tokens": exc.output_tokens, "seconds": perf_counter() - started,
                }
            else:
                payload = {
                    "reply": asdict(reply), "error": reply.error, "retryable": False, "input_tokens": reply.input_tokens,
                    "output_tokens": reply.output_tokens, "seconds": perf_counter() - started,
                }
            self.ledger.seal(payload, *result_path)
            replayed = self._reply_or_retry(payload)
            if replayed is not None:
                return replayed
        raise ProposalBudgetExhausted(f"{stage} request failed after {MAX_HTTP_RETRIES} transport retries")

    @staticmethod
    def _reply_or_retry(payload: dict[str, object]) -> ProviderReply | None:
        reply = payload["reply"]
        if isinstance(reply, dict):
            return ProviderReply(**reply)
        if payload["retryable"]:
            return None
        raise ProviderAttemptError(str(payload["error"]))
