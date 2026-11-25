"""Bind the synchronous P1 protocol to Stochopia's registered chat clients."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from stochopia.llm import (
    BaseLLMClient,
    LLMError,
    LLMRequestError,
    LLMTruncationError,
    ModelConfig,
    OpenAICompatClient,
    create_client,
    load_model_registry,
)

from .protocol import MAX_OUTPUT_TOKENS, Messages, ProviderReply, Transport


ClientFactory = Callable[[str, dict[str, ModelConfig]], BaseLLMClient]


class ProviderAttemptError(RuntimeError):
    """One incomplete provider request, including any reported token usage."""

    def __init__(
        self, message: str, *, retryable: bool = False,
        input_tokens: int | None = None, output_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class ProviderOutageError(ProviderAttemptError):
    """Pause development after transient provider failures exhaust their retries."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


def registry_transport(
    models_path: Path, model_name: str, *, client_factory: ClientFactory = create_client,
    completion_limit: int | None = None,
) -> Transport:
    """Bind one registered model for a synchronous study, without calling it at setup."""
    registry = load_model_registry(models_path)
    if model_name not in registry:
        raise ValueError(f"P1 model {model_name!r} is absent from the model registry")
    config = registry[model_name]
    if config.provider != "openai-compatible":
        raise ValueError("P1 requires the registered OpenAI-compatible JSON/text protocol")
    if not 1 <= config.max_tokens <= MAX_OUTPUT_TOKENS:
        raise ValueError(f"P1 completion cap must be between 1 and {MAX_OUTPUT_TOKENS}")
    if completion_limit is not None:
        if not 1 <= completion_limit <= config.max_tokens:
            raise ValueError("completion_limit must fit the registered output cap")
        config = replace(config, max_tokens=completion_limit)

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        request_config = replace(config, response_json=response_json)
        client = client_factory(model_name, {model_name: request_config})
        if isinstance(client, OpenAICompatClient):
            # The protocol owns repair attempts. The transport sends one request.
            client.RETRY_DELAYS = ()
            client.TRUNCATION_RETRIES = 0
        error = None
        try:
            text = asyncio.run(client.chat(messages, max_tokens=request_config.max_tokens))
        except LLMTruncationError:
            text = ""
            error = f"Registered provider stopped at its {request_config.max_tokens}-token output cap"
        except LLMRequestError as exc:
            if exc.status_code is not None:
                message = f"Provider HTTP {exc.status_code}"
            else:
                message = "Provider transport failure"
            status = exc.status_code
            retryable = exc.transport or status == 429 or (status is not None and 500 <= status < 600)
            usage = client.total_usage
            raise ProviderAttemptError(
                message,
                retryable=retryable,
                input_tokens=usage["prompt_tokens"] if usage["calls"] else None,
                output_tokens=usage["completion_tokens"] if usage["calls"] else None,
            ) from exc
        except LLMError as exc:
            usage = client.total_usage
            raise ProviderAttemptError(
                "Provider rejected the request or returned an invalid response",
                input_tokens=usage["prompt_tokens"] if usage["calls"] else None,
                output_tokens=usage["completion_tokens"] if usage["calls"] else None,
            ) from exc
        usage = client.total_usage
        return ProviderReply(text, usage["prompt_tokens"], usage["completion_tokens"], error)

    return transport


def provider_handshake(
    models_path: Path, model_name: str, *, client_factory: ClientFactory = create_client,
) -> ProviderReply:
    """Send one data-free JSON probe on the registered transport and return its metered reply."""
    messages: Messages = [
        {"role": "system", "content": 'Return only the JSON object {"ok": true}.'},
        {"role": "user", "content": "Connectivity check."},
    ]
    reply = registry_transport(models_path, model_name, client_factory=client_factory)(
        messages, response_json=True,
    )
    try:
        payload = json.loads(reply.text)
        if reply.error is not None or payload != {"ok": True} or payload["ok"] is not True:
            raise ValueError("unexpected handshake output")
    except ValueError as exc:
        raise ProviderAttemptError(
            "Provider handshake did not return the expected JSON object",
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
        ) from exc
    return reply
