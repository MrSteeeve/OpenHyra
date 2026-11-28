"""Durable, shared, no-retry reservation of formal proposal HTTP calls."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TextIO

from .contracts import Messages, ProviderReply, Transport
from .provider import ProviderAttemptError


class BudgetExceededError(RuntimeError):
    """A new physical request would exceed or reuse the registered call budget."""


class CallBudgetGuard:
    """Reserve exactly one call before invoking a caller-supplied one-shot transport.

    Every reservation is charged, even if the HTTP request fails or a process
    crashes. A lock spans only ledger reading/writing, so different trajectories
    may call independently without racing the global budget. There is no retry.
    """

    def __init__(self, path: Path, limit: int = 36) -> None:
        if type(limit) is not int or not 1 <= limit <= 36:
            raise ValueError("formal call limit must be an integer between 1 and 36")
        self.path = Path(path)
        self.limit = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self, stream: TextIO) -> list[dict[str, object]]:
        stream.seek(0)
        entries = [json.loads(line) for line in stream if line.strip()]
        if any(record["limit"] != self.limit for record in entries):
            raise ValueError("stored call budget differs from the registered limit")
        return entries

    def _append(self, stream: TextIO, record: dict[str, object]) -> None:
        stream.write(json.dumps({"limit": self.limit, **record}, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    def entries(self) -> list[dict[str, object]]:
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            return self._read(stream)

    @property
    def count(self) -> int:
        return sum(record["event"] == "reserved" for record in self.entries())

    def _reserve(self, call_id: str, messages: Messages, response_json: bool) -> None:
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("call_id must be nonempty text")
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            prior = self._read(stream)
            reservations = [record for record in prior if record["event"] == "reserved"]
            if len(reservations) >= self.limit:
                raise BudgetExceededError(f"formal call budget exhausted ({self.limit})")
            if any(record["call_id"] == call_id for record in reservations):
                raise BudgetExceededError(f"formal call slot already consumed: {call_id}")
            self._append(stream, {"event": "reserved", "call_id": call_id,
                                  "messages": messages, "response_json": response_json})

    def _complete(self, call_id: str, result: dict[str, object]) -> None:
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            self._append(stream, {"event": "completed", "call_id": call_id, **result})

    def call(
        self, transport: Transport, messages: Messages, *, call_id: str, response_json: bool = False,
    ) -> ProviderReply:
        """Charge and invoke once. Record provider-reported counts or explicit unknowns."""
        self._reserve(call_id, messages, response_json)
        started = perf_counter()
        try:
            reply = transport(messages, response_json=response_json)
        except ProviderAttemptError as exc:
            self._complete(call_id, {
                "reply": None, "error": str(exc), "raw_reply": exc.raw_reply,
                "input_tokens": exc.input_tokens, "output_tokens": exc.output_tokens,
                "retryable": exc.retryable, "status_code": exc.status_code,
                "seconds": perf_counter() - started,
            })
            raise
        except BaseException as exc:
            self._complete(call_id, {
                "reply": None, "error": type(exc).__name__, "raw_reply": None,
                "input_tokens": None, "output_tokens": None, "retryable": False,
                "status_code": None, "seconds": perf_counter() - started,
            })
            raise
        self._complete(call_id, {"reply": asdict(reply), "error": reply.error, "raw_reply": reply.text,
                                 "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                                 "retryable": False, "status_code": None, "seconds": perf_counter() - started})
        return reply

    def result(
        self, call_id: str, *, messages: Messages | None = None, response_json: bool = False,
    ) -> ProviderReply | None:
        """Replay a receipt only for the same request; unreserved slots return None."""
        entries = [record for record in self.entries() if record["call_id"] == call_id]
        if not entries:
            return None
        reservation = entries[0]
        if reservation["event"] != "reserved":
            raise ValueError("invalid formal call reservation")
        if messages is not None and (reservation["messages"] != messages or reservation["response_json"] != response_json):
            raise ValueError("resumed formal request differs from the recorded reservation")
        if len(entries) == 1:
            raise BudgetExceededError(f"formal call slot consumed but result unavailable: {call_id}")
        record = entries[-1]
        if record["event"] != "completed":
            raise ValueError("invalid formal call receipt")
        if record["reply"] is not None:
            return ProviderReply(**record["reply"])
        raise ProviderAttemptError(str(record["error"]), retryable=bool(record["retryable"]),
                                   raw_reply=record["raw_reply"], status_code=record["status_code"],
                                   input_tokens=record["input_tokens"], output_tokens=record["output_tokens"])
