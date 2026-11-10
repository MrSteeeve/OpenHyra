"""Program generators for the world-model task: offline fixtures and API models.

A generator receives a JSON request (parent program, development feedback,
experience log, assigned direction) and returns a complete ``world_model.py``
plus a declared mechanism, the targets it predicts will improve, and the
observation that would falsify it.  One HTTP call per proposal, no tools, no
retries; failures consume budget and are recorded as experience.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

GENERATION_SCHEMA = "openhyra-stochopia-world-model-generation.v1"
API_MODELS = Path(__file__).resolve().parents[1] / "stochopia_structuring" / "api_models.json"

SYSTEM_PROMPT = """You are revising a probabilistic world model of the Chinese equity index market.

The world model is a complete Python 3.12 program `world_model.py` (numpy and scipy only; no network,
no file access other than its --input/--output directories, no subprocesses). The evaluator runs it once
per walk-forward origin in an isolated sandbox with a 20 s time limit:

    python world_model.py --input IN --output OUT

IN/history.npz holds fixed-length trailing windows ending at the origin session (last row = origin):
  returns (504, 2)  daily log returns of [CSI500, CSI1000]
  rate_3m (504,)    91-day continuous SHIBOR zero rate
  carry   (126, 2)  91-day futures-implied carry for [CSI500 (IC), CSI1000 (IM)]; NaN when not quoted
  iv_1m   (126,)    CSI1000 30-day ATM implied volatility from MO options; NaN when not quoted
There are no calendar dates and no index levels. IN/query.json holds horizon (21), n_paths (256), seed.

Write OUT/forecast.npz with exactly two arrays:
  returns (n_paths, horizon, 2)  joint sample paths of the next 21 daily log returns, |r| <= 0.5
  q       (n_paths, 3)           samples of [CSI1000 iv_1m, CSI500 carry, CSI1000 carry] observed 21 sessions ahead
Use numpy.random.default_rng(query["seed"]) so the program is deterministic.

The external environment is the real market. Each sample set is scored by CRPS (Brier for binary targets)
against what actually happened, as skill relative to an EWMA Gaussian GBM + random-walk reference:
  return: 21-day log return; volatility: 21-day realised volatility;
  payoff_probe: payoffs of derivative structures on each index path (up digital, 0.95 down-touch,
  1.03 up-touch, ATM arithmetic Asian call, 0.97-1.03 range accrual, 21-day mini snowball KO 1.02 / KI 0.93);
  q_quote: the three Q-side quotes. Score = mean over families of mean target skill (0 = reference, higher is better).
Only development origins are ever shown to you; a later held-out period is scored once at the end.

You maintain a theory of this market. Use the feedback (per-target skill, calibration, trailing-volatility
slices) and the experience log to decide what structural change to make. State a mechanism, which targets
you predict will improve, and what result would falsify it.

Reply with exactly two fenced blocks and nothing else:
```json
{"mechanism": "...", "predicted_improvements": ["target or family names"], "falsifier": "..."}
```
```python
# complete world_model.py
```"""

DIRECTIONS = (
    {"id": "volatility_dynamics", "instruction": "Revise how volatility evolves over the horizon (clustering, mean reversion, leverage)."},
    {"id": "tail_and_dependence", "instruction": "Revise the shock distribution: fat tails, skew, cross-index dependence that varies with stress."},
    {"id": "q_side_dynamics", "instruction": "Revise the Q-side forecasts (IV and carry dynamics) and their link to the P-side state."},
    {"id": "joint_state", "instruction": "Introduce a shared latent state linking P-side paths and Q-side quotes, or restart from a different principle."},
)

# Fences must start a line; an inline ``` inside a docstring does not end a block.
_BLOCK = re.compile(r"^```[ \t]*(json|python|py)[ \t]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE | re.IGNORECASE)


def parse_program_response(text: str) -> dict[str, Any]:
    blocks: dict[str, str] = {}
    for kind, body in _BLOCK.findall(text):
        kind = "python" if kind.lower() in {"python", "py"} else kind.lower()
        blocks.setdefault(kind, body)
    if "python" not in blocks:
        raise ValueError("response has no ```python block")
    if "json" not in blocks:
        raise ValueError("response has no ```json block")
    meta = json.loads(blocks["json"])
    if not isinstance(meta, Mapping):
        raise ValueError("json block must be an object")
    predicted = meta.get("predicted_improvements", [])
    if not isinstance(predicted, list) or not all(isinstance(item, str) for item in predicted):
        raise ValueError("predicted_improvements must be a list of strings")
    return {
        "program": blocks["python"],
        "mechanism": str(meta.get("mechanism", "")),
        "predicted_improvements": predicted,
        "falsifier": str(meta.get("falsifier", "")),
    }


@dataclass
class Generation:
    proposal: dict[str, Any] | None
    raw_output: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None


class OfflineProgramGenerator:
    """Handwritten fixture programs, cycled in order (no model, no adaptation)."""

    FIXTURES = (
        ("garch_t", "GARCH(1,1) volatility with Student-t shocks; AR(1) mean reversion for IV and carry",
         ["volatility", "payoff_probe"]),
        ("filtered_bootstrap", "Filtered historical simulation with variance mean reversion; empirical Q changes",
         ["return", "volatility"]),
    )

    def __init__(self, directory: Path | None = None):
        self.directory = directory or Path(__file__).with_name("offline_variants")
        self.calls = 0

    def metadata(self) -> dict[str, Any]:
        return {"kind": "offline-fixture", "adaptive": False, "fixtures": [item[0] for item in self.FIXTURES]}

    def generate(self, request: Mapping[str, Any]) -> Generation:
        name, mechanism, predicted = self.FIXTURES[self.calls % len(self.FIXTURES)]
        self.calls += 1
        program = (self.directory / f"{name}.py").read_text(encoding="utf-8")
        proposal = {"program": program, "mechanism": mechanism, "predicted_improvements": predicted,
                    "falsifier": "the predicted families do not beat the parent on development origins"}
        return Generation(proposal, raw_output=f"fixture:{name}", usage={"api_calls": 0})


class APIProgramGenerator:
    """Anthropic Messages or OpenAI-compatible Chat Completions, one call per proposal."""

    def __init__(self, *, base_url: str, model: str, api_key_env: str, api_format: str = "chat-completions",
                 max_calls: int = 8, max_tokens: int | None = None, required_max_tokens: int | None = None,
                 request_options: Mapping[str, Any] | None = None, group_id: str | None = None,
                 timeout: float = 120.0, read_timeout: float = 1800.0):
        if api_format not in {"chat-completions", "anthropic-messages"}:
            raise ValueError(f"unsupported API format: {api_format}")
        self.base_url, self.model, self.api_key_env, self.api_format = base_url, model, api_key_env, api_format
        self.max_calls = int(max_calls)
        self.max_tokens = max_tokens if max_tokens is not None else required_max_tokens
        if api_format == "anthropic-messages" and self.max_tokens is None:
            raise ValueError("Anthropic Messages requires max_tokens")
        self.request_options = dict(request_options or {})
        if set(self.request_options) & {"model", "messages", "system", "max_tokens", "stream", "tools"}:
            raise ValueError("provider options cannot override model, prompt, token cap or tools")
        self.group_id = group_id or model
        self.timeout = float(timeout)
        self.read_timeout = float(read_timeout)
        self.calls = 0

    @classmethod
    def from_preset(cls, name: str, *, config_path: Path = API_MODELS, **overrides: Any) -> "APIProgramGenerator":
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        preset = config["models"][name]
        kwargs = {key: preset[key] for key in ("base_url", "model", "api_key_env", "api_format")}
        kwargs.update(required_max_tokens=preset.get("required_max_tokens"),
                      request_options=preset.get("request_options", {}), group_id=name)
        kwargs.update(overrides)
        return cls(**kwargs)

    def metadata(self) -> dict[str, Any]:
        return {"kind": "api", "base_url": self.base_url, "model": self.model, "api_format": self.api_format,
                "api_key_env": self.api_key_env, "group_id": self.group_id, "max_calls": self.max_calls,
                "max_tokens": self.max_tokens, "request_options": self.request_options,
                "retries": 0, "tools": [], "read_timeout_seconds": self.read_timeout,
                "system_prompt": SYSTEM_PROMPT}

    def request_body(self, request: Mapping[str, Any]) -> dict[str, Any]:
        content = json.dumps(request, ensure_ascii=False)
        body: dict[str, Any] = {**self.request_options, "model": self.model}
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.api_format == "anthropic-messages":
            body.update(system=SYSTEM_PROMPT, messages=[{"role": "user", "content": content}])
        else:
            body["messages"] = [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": content}]
        return body

    def _text(self, data: Mapping[str, Any], usage: dict[str, Any]) -> tuple[str, str | None]:
        provider = data.get("usage") or {}
        usage["provider_usage"] = provider
        usage["response_model"] = data.get("model")
        if self.api_format == "anthropic-messages":
            text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
            return text, data.get("stop_reason")
        choice = data["choices"][0]
        return choice["message"].get("content") or "", choice.get("finish_reason")

    def credential_present(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def generate(self, request: Mapping[str, Any]) -> Generation:
        import httpx

        if self.calls >= self.max_calls:
            raise RuntimeError("API proposal call budget exhausted")
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f"missing API credential in {self.api_key_env}")
        self.calls += 1
        started = time.monotonic()
        usage: dict[str, Any] = {"api_calls": 1, "call_index": self.calls, "group_id": self.group_id,
                                 "started_at_utc": datetime.now(timezone.utc).isoformat()}
        headers = {"Content-Type": "application/json"}
        if self.api_format == "anthropic-messages":
            headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
            endpoint = "/messages"
        else:
            headers["Authorization"] = f"Bearer {key}"
            endpoint = "/chat/completions"
        raw = ""
        try:
            response = httpx.post(self.base_url.rstrip("/") + endpoint, json=self.request_body(request),
                                  headers=headers, timeout=httpx.Timeout(self.timeout, read=self.read_timeout))
            usage["http_status"] = response.status_code
            response.raise_for_status()
            raw, finish = self._text(response.json(), usage)
            usage["finish_reason"] = finish
            if finish not in {"stop", "end_turn", "stop_sequence"}:
                raise ValueError(f"API did not finish a text answer: {finish}")
            return Generation(parse_program_response(raw), raw, usage, time.monotonic() - started)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
            return Generation(None, raw, usage, time.monotonic() - started, f"{type(exc).__name__}: {exc}")
