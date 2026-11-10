"""Swappable public conditions -> DSL generator; one API request per proposal."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import httpx

from tasks.stochopia_structuring.evaluator import PROTOCOL
from tasks.stochopia_structuring.equity_demands import demand_spec, equity_fixture
from examples.dsl_roundtrip import fixture_payload
from stochopia.adapters.structurer_dsl import DSL_CONTRACT_VERSION
from structurerdsl.program import Program


SYSTEM_PROMPT = """You design new equity research contracts in a monthly cashflow DSL.
Your task is payoff design for the supplied demand, not investor personas,
negotiation, or multi-agent games. Product names in examples are references;
you may freely compose different supported cashflow structures and parameters.
If a demand is provided, its public payoff anchors, budget and terminal loss
limit are hard constraints. The evaluator normalizes notional to the fixed
actual purchase budget using Q fair value, so changing notional cannot improve
the score by increasing investment. Principal protection, when required, must
cover actual purchase cost at redemption. A funded root alone does not prove
this: its payoff floor must cover its Q price. When protection is optional,
protected designs remain allowed. The disclosed finite P scenario distribution
is for expected net return ranking; it is separate from Q-model pricing.
Return exactly one JSON object with contract_sexpr, params (object), notional
(CNY), and rationale. Generate a single fixed contract for all supplied public
conditions. You may change structure and continuous parameters. Do not set a
purchase price: the evaluator trades at Q-model future-cashflow fair value.
No markup. S=(spot u) is spot / issuance fixing, time is years. Integer grid
indices are months. Fixed horizon is 3 months; schedules only in 1,2,3.
Root is (premium body) or (funded 1 $issue_price body); the latter repays
principal at maturity (or termination) automatically. Reserved $issue_price
is environment-bound and must not be used in payoff formulas.
Use (pay expr maturity), (both body body), (scale number body), (give body),
(when condition maturity body), or automatic (terminate_if condition obs expr).
Termination payoff expr adds to funded principal; remaining payments cease.
Expressions include (terminal (spot u)), (avg_arith (spot u) obs),
(avg_geo (spot u) obs), (running_max (spot u)), (running_min (spot u)),
(count_within (spot u) low high obs), (count_above (spot u) level obs),
(lin slope expr intercept), (add expr expr), (mul_const c expr),
(max expr expr), (min expr expr), (clip expr low high), time and numbers.
Conditions include (gt expr expr), (lt expr expr), (and cond cond), (or cond cond).
Other supported primitives are listed in action_schema; stick to known syntax.
Payments only at maturity or automatic termination, one underlying, no interim
coupons or discretionary exercise. Aggregate client payoff floor >=0.
Example: (contract (meta (underlyings u) (maturity 3)
(schedules (obs 1 2 3))) (premium (pay
(clip (lin 1 (terminal (spot u)) -0.95) 0 0.1) maturity)))
The objective and feedback are evaluator-owned. Use previous development
feedback when available; propose a complete artifact, not a diff or actions.
"""

SYSTEM_PROMPT_V2 = SYSTEM_PROMPT.replace(
    "The disclosed finite P scenario distribution\nis for expected net return ranking; it is separate from Q-model pricing.",
    "The disclosed continuous synthetic P distributions determine return ranking;\nthey are separate from Q-model pricing.",
).replace("params (object)", "params (object containing numeric scalar values only)") + """
For demand schema v2, public interval_requirements supplement the original
anchors. All sampled interval checks and principal-loss limits must pass.
The score_mode is fixed by the experiment: mean ranks probability-weighted
expected returns; worst_distribution ranks the lowest expected return among
the disclosed P distributions. These are research assumptions, not market
forecasts. No bonus is awarded for program length, extra operations or novelty.
Constraint margins and independent Q-price estimates diagnose failed or
fragile designs; neither a rationale nor a reported score overrides execution.
Put explanations in rationale, not params; params must not contain strings,
lists, dates or descriptions. Use an empty params object for numeric literals.
"""


def system_prompt_for(request):
    if request.get("research_goal") == "equity-liquidity-mechanism-discovery.v1":
        from tasks.stochopia_structuring.liquidity_research import liquidity_prompt
        return liquidity_prompt(request)
    if request.get("research_goal") == "path-demand-cost-discovery.v2":
        horizon = request["horizon_months"]
        return f"""Design one useful equity research cashflow contract with the existing DSL.
The horizon is {horizon} months with monthly grid indices 1..{horizon}.
Read the supplied path demand, not an investor persona. Market P and Q share
the same volatility input; Q sets the actual purchase price, P measures demand.
You choose payments and notional, never the issue price. Service budget is
an upper limit; notional is NOT rescaled to spend it all. The cost is contract
Q price plus the minimum operational cash reserve needed for this demand.
Unallocated budget stays outside the mandate, not as free in-account wealth.
Return one JSON with contract_sexpr, params (numeric scalars), notional (CNY),
and rationale. Root (premium body) or (funded 1 $issue_price body) is allowed;
funded adds principal at maturity or automatic termination. Do not put
$issue_price in payments or params. One underlying u, (spot u) is issue-relative.
Use meta (underlyings u) (maturity {horizon}) (schedules (obs 1 ... {horizon})),
writing every integer rather than an ellipsis. Existing expressions include
spot, terminal, running_min, running_max, avg_arith, avg_geo, count_within,
count_above, ret, lin, pl, add, mul_const, max, min, clip, step, time.
Contracts use pay, both, scale, give, when, until, activate_if, terminate_if.
Dated payments use (pay amount (at obs i)), where i is a zero-based schedule
index; e.g. (at obs 0) is month 1. Conditions must be knowable at their date.
Future terminal or future average information cannot condition early payments.
All date cashflows must be nonnegative and respect the supplied payment cap.
Customer external cash events are NOT contract observables. Their cash needs,
exit discount, withdrawal order and sale permissions come from the demand.
Do not claim funded principal means protecting the actual purchase price.
The main research question is feedback-supported discovery of useful payment
mechanisms with economic value. Lower total funded service cost is evaluated at fixed
joint demand satisfaction, separately from program shape or product names.
Known vanilla/barrier/Asian/autocall/range and portfolio-insurance families
are controls. A catalog nonmatch is not novelty proof. No bonus for longer DSL.
Use development feedback and parent ONLY when supplied. Independent requests
have no history. Return a complete contract, not a patch or invented primitive.
"""
    prompt = SYSTEM_PROMPT_V2 if request.get("demand", {}).get("schema") == "stochopia-equity-demand.v2" else SYSTEM_PROMPT
    if request.get("research_goal") == "feasible-mechanism-discovery.v1":
        prompt = prompt.replace("The disclosed continuous synthetic P distributions determine return ranking;",
                                "The disclosed continuous synthetic P distributions measure economic usefulness;")
        prompt = prompt.replace("The score_mode is fixed by the experiment: mean ranks probability-weighted\nexpected returns; worst_distribution ranks the lowest expected return among\nthe disclosed P distributions.",
            "Economic evaluation reports mean or worst-distribution returns as configured.\nCross-mechanism discovery is not ranked by this economic score.")
        prompt += "\nThe PRIMARY research goal is discovering feasible, distinct cashflow mechanisms. " \
            "Return is a secondary economic measure and selects representatives only within a mechanism. " \
            "Lower-return mechanisms remain in the discovery portfolio. Do not abandon a distinct " \
            "feasible mechanism merely because another has higher mean return. Terminal-only designs " \
            "can be innovative; adding a path statistic, a name, a strike or leverage does not prove novelty. " \
            "When supplied, use the archive and executed contrasts to avoid rewriting the same mechanism. " \
            "A missing reference match is only a research hypothesis, not a market-first claim.\n"
    if request.get("design_brief"):
        prompt += "\nFollow the experiment's design_brief and research_goal for this proposal. " \
            "Keep the supplied demand and DSL grammar. Discuss its " \
            "economic benefit and trade-off in rationale; return the same single-contract JSON format.\n"
    return prompt


@dataclass
class Generation:
    artifact: dict
    raw_output: str = ""
    usage: dict = field(default_factory=dict)
    seconds: float = 0.
    error: str | None = None
    raw_response: dict = field(default_factory=dict)


class ContractGenerator(Protocol):
    """Future learned generators only need to replace this interface."""
    def generate(self, request: dict) -> Generation: ...
    def metadata(self) -> dict: ...


def parse_generation(text):
    # A JSON code fence is harmless formatting; no second model call to repair.
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    raw = json.loads(stripped)
    payload = {"version": DSL_CONTRACT_VERSION,
               "program": Program.from_sexpr(raw["contract_sexpr"]).to_json(),
               "params": raw.get("params", {}), "notional": raw["notional"],
               "pitch": raw.get("rationale", "")}
    # Semantic checks belong to the evaluator so failures become experience.
    artifact = {"protocol": PROTOCOL, "contract": payload}
    # Generation records and AlgorithmSpec require portable finite JSON.
    json.dumps(artifact, allow_nan=False)
    return artifact


class FixtureGenerator:
    """Deterministic fixture proposals for offline pipeline verification."""
    cases = ("vanilla", "asian", "range", "autocall")

    def metadata(self):
        return {"kind": "offline-fixture", "cases": list(self.cases)}

    def generate(self, request):
        case = self.cases[(request["round_index"] * 2 + request["slot"]) % len(self.cases)]
        return Generation({"protocol": PROTOCOL, "contract": fixture_payload(case)},
                          raw_output=case, usage={"api_calls": 0})


class EquityFixtureGenerator:
    """Fixed handwritten controls for checking the new demand pipeline."""
    def metadata(self):
        return {"kind": "offline-equity-fixture", "adaptive": False, "api_calls": 0}

    def generate(self, request):
        spec = demand_spec(request["demand"]["demand_id"])
        variant = (request["round_index"] * 2 + request["slot"]) % 4
        return Generation({"protocol": PROTOCOL, "contract": equity_fixture(spec["family"], variant)},
                          raw_output=f"handwritten-{spec['family']}-{variant}", usage={"api_calls": 0})


class APIContractGenerator:
    """Native Messages or Chat Completions; one HTTP request per proposal."""
    def __init__(self, *, base_url, model, api_key_env, max_calls=4,
                 max_tokens=None, temperature=0., timeout=120.,
                 api_format="chat-completions", group_id=None,
                 request_options=None, model_info=None, required_max_tokens=None):
        if api_format not in {"chat-completions", "anthropic-messages"}:
            raise ValueError(f"unsupported API format: {api_format}")
        self.base_url, self.model, self.api_key_env = base_url, model, api_key_env
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.effective_max_tokens = max_tokens if max_tokens is not None else required_max_tokens
        if api_format == "anthropic-messages" and self.effective_max_tokens is None:
            raise ValueError("Anthropic requires its model's maximum output token value")
        self.temperature, self.timeout = temperature, timeout
        self.api_format, self.group_id = api_format, group_id or model
        self.request_options = request_options or {}
        self.model_info = model_info or {}
        # Experiment-owned fields cannot be overridden by provider options.
        if set(self.request_options) & {"model", "models", "messages", "system", "max_tokens", "stream", "tools"}:
            raise ValueError("provider options cannot override model, prompt, token cap or tools")
        self.calls = 0

    def metadata(self):
        return {"kind": "api", "base_url": self.base_url,
                "model": self.model, "api_key_env": self.api_key_env,
                "group_id": self.group_id, "api_format": self.api_format,
                "request_options": self.request_options, "model_info": self.model_info,
                "max_calls": self.max_calls, "max_tokens": self.max_tokens,
                "effective_max_tokens": self.effective_max_tokens,
                "output_budget_policy": "explicit" if self.max_tokens is not None else "provider-default-or-required-maximum",
                "temperature": self.temperature, "connect_timeout_seconds": self.timeout,
                "read_timeout_seconds": None,
                "retries": 0, "tools": [], "system_prompt": SYSTEM_PROMPT,
                "system_prompt_v2": SYSTEM_PROMPT_V2,
                "system_prompt_discovery": system_prompt_for({
                    "demand": {"schema": "stochopia-equity-demand.v2"},
                    "research_goal": "feasible-mechanism-discovery.v1", "design_brief": True})}

    def request_body(self, request):
        content = json.dumps(request, ensure_ascii=False)
        prompt = system_prompt_for(request)
        payload = {**self.request_options, "model": self.model}
        if self.effective_max_tokens is not None:
            payload["max_tokens"] = self.effective_max_tokens
        if self.api_format == "anthropic-messages":
            payload.update(system=prompt, messages=[{"role": "user", "content": content}])
            # Haiku 5.5 rejects non-default sampling parameters. Omit them.
        else:
            payload["messages"] = [{"role": "system", "content": prompt},
                                   {"role": "user", "content": content}]
            if self.temperature is not None:
                payload["temperature"] = self.temperature
        return payload

    def parse_response(self, data, usage):
        provider_usage = data.get("usage") or {}
        usage["provider_usage"] = provider_usage
        usage.update(response_model=data.get("model"), response_id=data.get("id"),
                     response_provider=data.get("provider"), system_fingerprint=data.get("system_fingerprint"))
        if self.api_format == "anthropic-messages":
            raw = "".join(block["text"] for block in data["content"] if block.get("type") == "text")
            finish = data.get("stop_reason")
            # Anthropic's input_tokens excludes cache reads/creation.
            if "input_tokens" in provider_usage:
                usage["prompt_tokens"] = sum(provider_usage.get(k, 0) for k in (
                    "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            if "output_tokens" in provider_usage:
                usage["completion_tokens"] = provider_usage["output_tokens"]
            reasoning = provider_usage.get("output_tokens_details", {}).get("thinking_tokens")
        else:
            choice = data["choices"][0]
            raw = choice["message"].get("content") or ""
            finish = choice.get("finish_reason")
            for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if field_name in provider_usage:
                    usage[field_name] = provider_usage[field_name]
            reasoning = provider_usage.get("completion_tokens_details", {}).get("reasoning_tokens")
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            usage.setdefault("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"])
        usage["reasoning_tokens"] = reasoning  # Missing is unknown, not zero.
        usage["finish_reason"] = finish
        return raw, finish

    def generate(self, request):
        if self.calls >= self.max_calls:
            raise RuntimeError("API proposal call budget exhausted")
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f"missing API credential in {self.api_key_env}")
        self.calls += 1
        started = time.monotonic()
        raw, data = "", {}
        usage = {"api_calls": 1, "call_index": self.calls, "group_id": self.group_id,
                 "requested_model": self.model, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        headers = {"Content-Type": "application/json"}
        if self.api_format == "anthropic-messages":
            headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
            endpoint = "/messages"
        else:
            headers["Authorization"] = f"Bearer {key}"
            endpoint = "/chat/completions"
        try:
            response = httpx.post(self.base_url.rstrip("/") + endpoint,
                json=self.request_body(request), headers=headers,
                timeout=httpx.Timeout(self.timeout, read=None))
            usage["http_status"] = response.status_code
            usage["provider_request_id"] = response.headers.get("request-id") or response.headers.get("x-request-id")
            data = {"http_status": response.status_code, "body": response.text}
            response.raise_for_status()
            data = response.json()
            raw, finish = self.parse_response(data, usage)
            if finish in {"length", "max_tokens", "model_context_window_exceeded"}:
                raise ValueError("API output truncated at request or provider token limit")
            if finish not in {"stop", "end_turn", "stop_sequence"}:
                raise ValueError(f"API did not finish a text answer: {finish}")
            if request.get("research_goal") == "equity-liquidity-mechanism-discovery.v1":
                from tasks.stochopia_structuring.liquidity_research import parse_liquidity_generation
                artifact = parse_liquidity_generation(raw)
            else:
                artifact = parse_generation(raw)
            return Generation(artifact, raw, usage, time.monotonic() - started, raw_response=data)
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError, IndexError) as exc:
            # Fixed-budget failures are kept, never retried or silently replaced.
            if isinstance(exc, httpx.HTTPStatusError):
                data = {"http_status": exc.response.status_code, "body": exc.response.text}
            error = f"{type(exc).__name__}: {exc}"
            return Generation({"protocol": PROTOCOL, "generation_error": error},
                              raw, usage, time.monotonic() - started, error, data)
