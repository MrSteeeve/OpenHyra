"""Registered, metered chat providers for the research harness."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import yaml


class LLMError(Exception):
    """Provider error; optional assistant text is distinct from transport diagnostics."""

    def __init__(self, message: str, *, raw_reply: str | None = None) -> None:
        super().__init__(message)
        self.raw_reply = raw_reply


class LLMTruncationError(LLMError):
    """输出因 max_tokens 上限被截断；chat 内部会自动升额重试，重试耗尽才抛出。"""


class LLMRequestError(LLMError):
    """Request failure whose HTTP or transport cause is available to a caller."""

    def __init__(self, message: str, *, status_code: int | None = None, transport: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.transport = transport


@dataclass
class ModelConfig:
    """模型注册表中的一条配置。"""

    name: str
    provider: str
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    temperature: float | None = 0.7
    max_tokens: int = 2000
    timeout: float = 60.0
    response_json: bool = False
    use_max_completion_tokens: bool = False
    extra_body: Mapping[str, Any] | None = None
    strict_model_match: bool = False


class ModelRegistry(dict):
    """模型注册表：dict[模型名 -> ModelConfig]。"""


def load_model_registry(path: str | Path) -> ModelRegistry:
    """Read the adjacent models.yaml registry or an explicitly supplied registry file."""
    p = Path(path)
    if not p.is_file():
        raise LLMError(f"模型注册表文件不存在：{p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LLMError(f"模型注册表 YAML 解析失败（{p}）：{exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise LLMError(f"模型注册表格式错误（{p}）：顶层需要 models 映射")
    registry: dict[str, ModelConfig] = {}
    for name, cfg in data["models"].items():
        if not isinstance(cfg, dict) or "provider" not in cfg:
            raise LLMError(f"模型 {name} 配置格式错误（{p}）：需要 provider 字段")
        allowed = {"provider", "base_url", "model", "api_key_env", "temperature", "max_tokens", "timeout",
                   "response_json", "use_max_completion_tokens", "extra_body", "strict_model_match"}
        unknown = set(cfg) - allowed
        if unknown:
            raise LLMError(f"模型 {name} 配置含未知字段（{p}）：{sorted(unknown)}")
        registry[name] = ModelConfig(name=name, **cfg)
    return ModelRegistry(registry)


class BaseLLMClient:
    """LLM 客户端基类，统计 token 用量。"""

    def __init__(self) -> None:
        self.total_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        raise NotImplementedError

    def _record_provider_usage(
        self, data: object, input_key: str, output_key: str,
    ) -> bool:
        """Record known usage before checking text; missing/malformed counts stay unknown."""
        if not isinstance(data, dict) or not isinstance(data.get("usage"), dict):
            return False
        usage = data["usage"]
        complete = True
        for source, target in ((input_key, "prompt_tokens"), (output_key, "completion_tokens")):
            value = usage.get(source)
            if type(value) is int and value >= 0:
                self.total_usage[target] += value
                self.total_usage[f"{target}_reported"] = True
            else:
                complete = False
        if complete:
            self.total_usage["calls"] += 1
        return complete

    def _record_error_usage(self, response: httpx.Response, input_key: str, output_key: str) -> None:
        try:
            self._record_provider_usage(response.json(), input_key, output_key)
        except (json.JSONDecodeError, ValueError):
            pass

    def _record_max_tokens_escalation(self, requested: int, effective: int) -> None:
        """截断升额重试后成功：把实际生效的 max_tokens 记入 usage（随 trace 序列化）。"""
        self.total_usage.setdefault("max_tokens_escalations", []).append(
            {"requested_max_tokens": requested, "effective_max_tokens": effective}
        )


class OpenAICompatClient(BaseLLMClient):
    """OpenAI 兼容 chat/completions 客户端，429/5xx/超时指数退避重试。

    输出被 max_tokens 截断（finish_reason=length）时不直接报错，而是把
    max_tokens 翻倍后立即重试（独立计数，最多 TRUNCATION_RETRIES 次，
    封顶 TRUNCATION_MAX_TOKENS_CEILING），仍截断才抛 LLMTruncationError；升额后成功时
    实际生效的 max_tokens 记入 total_usage["max_tokens_escalations"]。
    """

    RETRY_DELAYS = (1.0, 2.0, 4.0)
    TRUNCATION_RETRIES = 2
    TRUNCATION_MAX_TOKENS_CEILING = 16000

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.unexpected_model: str | None = None

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        cfg = self.config
        if self.unexpected_model is not None:
            raise LLMError(f"已停止模型 {cfg.name} 的请求：供应商返回了不匹配的 model")
        api_key = os.environ.get(cfg.api_key_env, "").strip()
        if not api_key:
            raise LLMError(
                f"模型 {cfg.name} 缺少 API key：请设置环境变量 {cfg.api_key_env}"
                "。"
            )
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        token_cap = "max_completion_tokens" if cfg.use_max_completion_tokens else "max_tokens"
        payload = {
            **(cfg.extra_body or {}),
            "model": cfg.model,
            "messages": messages,
            token_cap: cfg.max_tokens if max_tokens is None else max_tokens,
        }
        temperature = cfg.temperature if temperature is None else temperature
        if temperature is not None:
            payload["temperature"] = temperature
        if cfg.response_json:
            payload["response_format"] = {"type": "json_object"}
        if seed is not None:
            # OpenAI 兼容端点的尽力复现参数；不支持该字段的供应商会静默忽略。
            payload["seed"] = seed
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        requested_max_tokens = payload[token_cap]
        last_error = ""
        net_attempt = 0
        trunc_attempt = 0
        while True:
            started = perf_counter()
            status_code = None
            failed_transport = False
            self.total_usage["api_attempts"] = self.total_usage.get("api_attempts", 0) + 1
            try:
                async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                # 超时：可重试
                failed_transport = True
                last_error = "请求超时"
            except httpx.TransportError as exc:
                # 瞬时传输层故障（连接被掐断、协议错误、代理抖动等）：可重试
                failed_transport = True
                last_error = "网络传输错误"
            except httpx.HTTPError as exc:
                # 其他请求错误（非法 URL、重定向过多等）：不重试，直接抛出
                raise LLMError(f"模型 {cfg.name} 请求失败（{url}）：{exc!r}") from exc
            else:
                status_code = resp.status_code
                if resp.status_code >= 400:
                    self._record_error_usage(resp, "prompt_tokens", "completion_tokens")
                if resp.status_code == 429 or resp.status_code >= 500:
                    # 限流或服务端错误：可重试
                    last_error = f"HTTP {resp.status_code}"
                elif resp.status_code >= 400:
                    # 其余 4xx：不重试，带响应体摘要直接抛出
                    raise LLMRequestError(
                        f"模型 {cfg.name} 调用失败（HTTP {resp.status_code}，不可重试）",
                        status_code=status_code,
                    )
                else:
                    try:
                        text = self._parse_response(resp)
                    except LLMTruncationError:
                        # 截断：升额重试（独立计数，不占用网络重试次数、不退避等待）
                        if (
                            trunc_attempt >= self.TRUNCATION_RETRIES
                            or payload[token_cap] >= self.TRUNCATION_MAX_TOKENS_CEILING
                        ):
                            raise
                        trunc_attempt += 1
                        payload[token_cap] = min(
                            payload[token_cap] * 2, self.TRUNCATION_MAX_TOKENS_CEILING
                        )
                        continue
                    if trunc_attempt:
                        self._record_max_tokens_escalation(requested_max_tokens, payload[token_cap])
                    return text
            finally:
                self.total_usage["api_seconds"] = self.total_usage.get("api_seconds", 0) + perf_counter() - started
            if net_attempt < len(self.RETRY_DELAYS):
                await asyncio.sleep(self.RETRY_DELAYS[net_attempt])
                net_attempt += 1
                continue
            raise LLMRequestError(
                f"模型 {cfg.name} 重试 {len(self.RETRY_DELAYS)} 次后仍失败：{last_error}",
                status_code=status_code, transport=failed_transport,
            )

    def _parse_response(self, resp: httpx.Response) -> str:
        """Meter usage before inspecting content, including truncated or malformed replies."""
        cfg = self.config
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"Model {cfg.name} returned a malformed JSON response") from exc
        metered = self._record_provider_usage(data, "prompt_tokens", "completion_tokens")
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Model {cfg.name} returned a malformed assistant response") from exc
        raw_reply = content if isinstance(content, str) else None
        if cfg.strict_model_match and data.get("model") != cfg.model:
            self.unexpected_model = str(data.get("model") or "")
            raise LLMError(f"Model {cfg.name} returned an unregistered model ID", raw_reply=raw_reply)
        if not metered:
            raise LLMError(f"Model {cfg.name} did not report complete token usage", raw_reply=raw_reply)
        if choice.get("finish_reason") == "length":
            raise LLMTruncationError(f"Model {cfg.name} exceeded its output cap", raw_reply=raw_reply)
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"Model {cfg.name} returned no assistant text")
        return content


class AnthropicClient(BaseLLMClient):
    """Messages protocol with one physical HTTP request and explicit metering."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    async def chat(
        self, messages: list[dict], temperature: float | None = None,
        max_tokens: int | None = None, seed: int | None = None,
    ) -> str:
        cfg = self.config
        api_key = os.environ.get(cfg.api_key_env, "").strip()
        if not api_key:
            raise LLMError(f"Model {cfg.name} requires environment variable {cfg.api_key_env}")
        if seed is not None:
            raise LLMError("The Anthropic Messages protocol does not accept a seed")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conversation = [m for m in messages if m["role"] != "system"]
        payload = {
            **(cfg.extra_body or {}),
            "model": cfg.model,
            "messages": conversation,
            "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
        }
        if system:
            payload["system"] = system
        effective_temperature = cfg.temperature if temperature is None else temperature
        if effective_temperature is not None:
            payload["temperature"] = effective_temperature
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        self.total_usage["api_attempts"] = self.total_usage.get("api_attempts", 0) + 1
        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                response = await client.post(cfg.base_url.rstrip("/") + "/messages", json=payload, headers=headers)
        except httpx.TransportError as exc:
            raise LLMRequestError("Provider transport failure", transport=True) from exc
        except httpx.HTTPError as exc:
            raise LLMRequestError("Provider request failed") from exc
        finally:
            self.total_usage["api_seconds"] = self.total_usage.get("api_seconds", 0) + perf_counter() - started
        if response.status_code >= 400:
            self._record_error_usage(response, "input_tokens", "output_tokens")
            raise LLMRequestError(f"Provider HTTP {response.status_code}", status_code=response.status_code)
        return self._parse_response(response)

    def _parse_response(self, response: httpx.Response) -> str:
        cfg = self.config
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"Model {cfg.name} returned a malformed JSON response") from exc
        metered = self._record_provider_usage(data, "input_tokens", "output_tokens")
        if not isinstance(data, dict) or not isinstance(data.get("content"), list):
            raise LLMError(f"Model {cfg.name} returned a malformed assistant response")
        text_blocks = [block.get("text") for block in data["content"]
                       if isinstance(block, dict) and block.get("type") == "text"]
        if any(not isinstance(text, str) for text in text_blocks):
            raise LLMError(f"Model {cfg.name} returned a malformed text block")
        raw_reply = "\n".join(text_blocks)
        if cfg.strict_model_match and data.get("model") != cfg.model:
            raise LLMError(f"Model {cfg.name} returned an unregistered model ID", raw_reply=raw_reply)
        if not metered:
            raise LLMError(f"Model {cfg.name} did not report complete token usage", raw_reply=raw_reply)
        if data.get("stop_reason") == "max_tokens":
            raise LLMTruncationError(f"Model {cfg.name} exceeded its output cap", raw_reply=raw_reply)
        if not raw_reply.strip():
            raise LLMError(f"Model {cfg.name} returned no assistant text")
        return raw_reply


def create_client(model_name: str, registry: dict[str, ModelConfig]) -> BaseLLMClient:
    """Create the supported registered transport without performing a network request."""
    if model_name not in registry:
        raise LLMError(f"Unknown registered model {model_name!r}")
    config = registry[model_name]
    if config.provider == "openai-compatible":
        return OpenAICompatClient(config)
    if config.provider == "anthropic":
        return AnthropicClient(config)
    raise LLMError(f"Unsupported provider {config.provider!r}")
