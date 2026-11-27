"""Persist provider attempts before charging a new development request."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from time import perf_counter, sleep
from typing import Literal

from .contracts import MAX_HYPOTHESIS_REPAIRS, MAX_REPAIRS, CallUsage, Messages, ProviderReply, RequestAdapter
from .provider import ProviderAttemptError, ProviderOutageError
from .store import Ledger


HTTP_RETRY_DELAYS = (2.0, 5.0, 15.0, 30.0, 60.0)
MAX_HTTP_RETRIES = len(HTTP_RETRY_DELAYS)


class ProposalBudgetExhausted(RuntimeError):
    """The registered logical request cannot be tried again."""


class TrialCalls:
    """Budget and replay metered proposal requests for one candidate only."""

    def __init__(
        self, ledger: Ledger, path: tuple[str, ...], adapter: RequestAdapter,
        *, stop_requested: Callable[[], bool] | None = None,
        analysis_version: Literal["m4", "m5"] = "m4",
    ) -> None:
        self.ledger = ledger
        self.path = path
        self.adapter = adapter
        self.stop_requested = stop_requested
        self.analysis_version = analysis_version
        self.hypotheses = 0
        self.synthesis = 0

    def record(self, usage: CallUsage) -> None:
        if usage.stage in ("hypothesis", "hypothesis_repair"):
            index = self.hypotheses
        else:
            # Reserve plan slots independently of retries so synthesis can resume
            # from the write-once plan without replaying successful JSON turns.
            index = MAX_HYPOTHESIS_REPAIRS + 1 + self.synthesis
        path = (*self.path, "usage", f"{index:02d}.json")
        previous = self.ledger.read(*path)
        self.ledger.seal(asdict(usage), *path)
        if previous is None:
            self.adapter.record_usage(usage)

    def _ensure_running(self) -> None:
        if self.stop_requested is not None and self.stop_requested():
            raise ProviderOutageError("development paused before the next provider attempt")

    def transport(self, messages: Messages, *, response_json: bool) -> ProviderReply:
        """Replay durable attempts, then try one fresh batch of up to six HTTP requests.

        Each batch contains an initial attempt and five delayed retries. An outage
        leaves all responses and token metadata sealed in their own attempt files.
        A resumed logical request can acquire a new batch without resampling any
        recorded successful response or reusing the previous attempt identifiers.
        """
        if response_json:
            self.hypotheses += 1
            phase = "hypothesis"
            stage = "hypothesis" if self.hypotheses == 1 else "hypothesis_repair"
            logical_index = self.hypotheses
            if logical_index > MAX_HYPOTHESIS_REPAIRS + 1:
                raise ProposalBudgetExhausted("hypothesis request budget exhausted")
        else:
            self.synthesis += 1
            phase = "synthesis"
            stage = "code" if self.synthesis == 1 else "repair"
            logical_index = self.synthesis
            if logical_index > MAX_REPAIRS + 1:
                raise ProposalBudgetExhausted("synthesis request budget exhausted")
        http_index = 0
        fresh_attempts = 0
        while True:
            http_index += 1
            stem = f"{phase}-{logical_index:02d}"
            if http_index > 1:
                stem += f"-http-{http_index:02d}"
            reservation_path = (*self.path, "requests", f"{stem}.json")
            result_path = (*self.path, "requests", f"{stem}-result.json")
            reservation = {
                "phase": phase, "stage": stage, "attempt": logical_index, "http_attempt": http_index,
                "response_json": response_json, "messages": messages,
            }
            previous_reservation = self.ledger.read(*reservation_path)
            if previous_reservation is not None and previous_reservation != reservation:
                raise ValueError(f"the recorded {stage} request differs from the resumed request")
            prior = self.ledger.read(*result_path)
            if prior is not None:
                reply = self._reply_or_retry(prior)
                if reply is not None:
                    return reply
                continue
            if previous_reservation is not None:
                if self.analysis_version == "m5":
                    # Record the possibly spent request without inventing usage
                    # or reusing its physical attempt identifier.
                    self.ledger.seal({
                        "reply": None,
                        "error": f"Provider response unavailable after reserved {stage} request",
                        "retryable": True,
                        "input_tokens": None,
                        "output_tokens": None,
                        "seconds": None,
                        "interruption_without_result": True,
                    }, *result_path)
                    continue
                # The old study counted an unresolvable reservation as spent.
                raise ProposalBudgetExhausted(f"provider response unavailable after reserved {stage} request")
            self._ensure_running()
            if fresh_attempts >= MAX_HTTP_RETRIES + 1:
                raise ProviderOutageError(f"{stage} request failed after {MAX_HTTP_RETRIES} transport retries")
            if fresh_attempts:
                sleep(HTTP_RETRY_DELAYS[fresh_attempts - 1])
                self._ensure_running()
            self.ledger.seal(reservation, *reservation_path)
            fresh_attempts += 1
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
    @staticmethod
    def _reply_or_retry(payload: dict[str, object]) -> ProviderReply | None:
        reply = payload["reply"]
        if isinstance(reply, dict):
            return ProviderReply(**reply)
        if payload["retryable"]:
            return None
        raise ProviderAttemptError(
            str(payload["error"]),
            input_tokens=payload["input_tokens"],
            output_tokens=payload["output_tokens"],
        )
