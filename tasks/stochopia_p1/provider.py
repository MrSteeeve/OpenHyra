"""Bind the synchronous P1 protocol to Stochopia's registered chat clients."""

from __future__ import annotations

import asyncio
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


def registry_transport(
    models_path: Path, model_name: str, *, client_factory: ClientFactory = create_client,
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
            retryable = exc.transport or (exc.status_code is not None and 500 <= exc.status_code < 600)
            raise ProviderAttemptError(message, retryable=retryable) from exc
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
