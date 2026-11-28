"""Offline request preflight counters for the registered provider transports."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .contracts import Messages


def openai_token_counter() -> Callable[[Messages], int]:
    """Conservative local count across two encodings plus message framing reserve."""
    import tiktoken

    encodings = (tiktoken.get_encoding("o200k_base"), tiktoken.get_encoding("cl100k_base"))

    def count(messages: Messages) -> int:
        content = [packet["content"] for packet in messages]
        return max(sum(len(enc.encode_ordinary(value)) for value in content) for enc in encodings) + 96

    return count


def load_token_counters(directory: Path) -> dict[str, Callable[[Messages], int]]:
    """Load provider tokenizers and reserve room for server message framing.

    GPT-6.2 Sol is unmapped in tiktoken 0.14. Use the larger of the two
    recent OpenAI text encodings as a conservative estimate. The transport
    separately records and enforces the provider's actual token count.
    """
    from tokenizers import Tokenizer

    deepseek = Tokenizer.from_file(str(directory / "deepseek-flash.json"))

    def ds(messages: Messages) -> int:
        content = [packet["content"] for packet in messages]
        return sum(len(deepseek.encode(value, add_special_tokens=False).ids) for value in content) + 64

    return {"gpt-6.2-sol": openai_token_counter(), "deepseek-flash": ds}
