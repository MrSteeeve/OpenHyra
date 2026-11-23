"""One anonymous proposal protocol for the registered P0 LLM groups."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote

from stochopia.harness.contract import (
    AnalysisSpec,
    Direction,
    ExperimentManifest,
    GroupSpec,
    ReportStatus,
    ScenarioSpec,
    SelectionRecord,
    StructuredPrediction,
)

from .store import JsonObject, SelectorUsageStore


class ChatClient(Protocol):
    """A single-attempt client with cumulative Stochopia usage counters."""

    total_usage: Mapping[str, Any]

    async def chat(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        ...


ClientFactory = Callable[[str, str], ChatClient]
OfflineResponder = Callable[[JsonObject], JsonObject | str]
PromptCounter = Callable[[list[dict[str, str]]], int]
Decision = tuple[SelectionRecord, tuple[StructuredPrediction, ...]]

_SYSTEM = (
    'Select one development option or retain the baseline. Forecast candidate minus baseline for every '
    'listed target before its result is seen. Output one JSON object with action (select, keep_parent, abstain), '
    'option (option ID if selecting, null otherwise), reason (brief text), and predictions (array). Each '
    'prediction has option, target, direction (increase, decrease, unchanged, abstain), and effect_interval '
    '(finite [low,high] of the stated sign for increase or decrease, null otherwise). If action is abstain, '
    'abstain on all targets. Treat all supplied values as development data.'
)
_OPTIONS = {
    "separate_flow": "Uses an additional recurrent stream that receives a different event subset.",
    "separate_flow_own_price": "Adds one binary event input to a recurrent output branch.",
    "self": "Adds one binary event input to a recurrent output branch.",
    "self-exogenous": "Adds one binary event input to a recurrent output branch.",
    "anonymous": "Changes the recurrent update while retaining the input fields.",
}
_METRICS = frozenset({
    "background_bias", "induced_background_sign", "clamped_rmse", "free_rmse", "nll", "heldout_loss",
    "peak_response", "free_peak_response", "end_response", "free_end_response", "no_injection_drift",
    "filled_size", "fill_size", "best_ask_depth",
})
_EVIDENCE = frozenset({"flow_rmse", "price_rmse", "dominance_margin", "background_bias", "min_effect"})
_BUNDLES = frozenset({"day-level-flow", "diagnostics"})
_UNIT = frozenset({"tick", "ticks", "shares", "nats", "bits", "sign", "fraction", "ratio", "probability", "unitless"})
_WINDOW = frozenset({"execution", "end", "recovery", "full", "peak", "post_execution", "no_injection"})
_PRED_STATES = frozenset({"increase", "decrease", "unchanged", "abstain"})
_STATUSES = frozenset(status.value for status in ReportStatus)
_SAFE_CONTRACT_ERRORS = frozenset({
    "directional predictions require an effect_interval",
    "effect_interval must agree with the prediction direction",
})


# --- Budget reservation and accounting ---


@dataclass(frozen=True)
class _Reservation:
    attempt: int
    prompt_bound: int
    completion_cap: int


class _UsageLedger:
    """Reserve budget before transport and retain unresolved charges across restarts."""

    def __init__(self, group: GroupSpec, exp_id: str, usage_root: Path | None, *,
                 retry_limit: int, output_cap: int, system_prompt: str) -> None:
        """Bind usage to a manifest group below usage_root/exp_id/group_id when provided.

        The IDs are literal path components. A redirected child directory or an
        existing ledger with changed budgets or policy settings is rejected.
        """
        self.max_calls = _budget_limit(group, "max_calls")
        self.max_tokens = _budget_limit(group, "max_tokens")
        self.settings: JsonObject = {
            "model": group.model, "version": group.model_version, "template": group.prompt_template_id,
            "max_calls": self.max_calls, "max_tokens": self.max_tokens,
            "retry_limit": retry_limit, "output_cap": output_cap,
        }
        if system_prompt != _SYSTEM:
            self.settings["system_prompt"] = system_prompt
        self._memory: JsonObject = {"settings": self.settings, "attempts": []}
        self._store = SelectorUsageStore(usage_root, exp_id, group.group_id) if usage_root is not None else None

    @contextmanager
    def _opened(self, *, write: bool = True) -> Iterator[JsonObject]:
        """Hold an optional group-scoped process lock while reading or editing usage."""
        if self._store is None:
            yield self._memory
            return
        with self._store.lock():
            disk = self._store.usage()
            if disk is None:
                disk = {"settings": self.settings, "attempts": []}
            if disk["settings"] != self.settings:
                raise ValueError("existing selector budget settings differ from this manifest")
            yield disk
            if write:
                self._store.save_usage(disk)

    def reserve(self, prompt_bound: int, output_limit: int) -> tuple[_Reservation | None, str | None]:
        with self._opened() as state:
            summary = _usage_summary(state)
            if state.get("provider_bound_violated"):
                return None, "Group inference reservation bound was violated."
            if summary["calls"] >= self.max_calls:
                return None, "Group inference call budget exhausted."
            remaining = self.max_tokens - summary["charged_tokens"] - prompt_bound
            if remaining < 1:
                return None, "Group inference token budget exhausted."
            proposal = _Reservation(summary["calls"], prompt_bound, min(output_limit, remaining))
            state["attempts"].append({
                "prompt_bound": prompt_bound, "completion_cap": proposal.completion_cap, "usage": None,
            })
            return proposal, None

    def settle(self, reservation: _Reservation, usage: tuple[int, int] | None) -> bool:
        """Return whether provider usage fits its reservation. Missing usage retains the reservation."""
        if usage is None:
            return True
        prompt, completion = usage
        within_bound = prompt <= reservation.prompt_bound and completion <= reservation.completion_cap
        with self._opened() as state:
            state["attempts"][reservation.attempt]["usage"] = [prompt, completion]
            if not within_bound:
                state["provider_bound_violated"] = True
        return within_bound

    def summary(self) -> dict[str, int]:
        with self._opened(write=False) as state:
            return _usage_summary(state)


def _usage_summary(state: JsonObject) -> dict[str, int]:
    prompt = 0
    completion = 0
    reserved = 0
    for entry in state["attempts"]:
        if entry["usage"] is None:
            reserved += entry["prompt_bound"] + entry["completion_cap"]
        else:
            prompt += entry["usage"][0]
            completion += entry["usage"][1]
    return {
        "calls": len(state["attempts"]), "prompt_tokens": prompt, "completion_tokens": completion,
        "reserved_tokens": reserved, "charged_tokens": prompt + completion + reserved,
    }


def _integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _budget_limit(group: GroupSpec, name: str) -> int:
    value = group.llm_budget.get(name)
    if _integer(value) and value >= 0:
        return int(value)
    number = _numeric(value)
    if number is not None and value >= 0 and int(value) == value:
        return int(value)
    raise ValueError(f"LLM budget requires a nonnegative integer {name}")


def _default_prompt_bound(messages: list[dict[str, str]]) -> int:
    """Use message bytes plus framing as the conservative default prompt allowance."""
    return sum(len(message["content"].encode("utf-8")) + 16 for message in messages) + 32


def _reported_usage(before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[int, int] | None:
    values = []
    for field in ("calls", "prompt_tokens", "completion_tokens"):
        left, right = before.get(field, 0), after.get(field)
        if not _integer(left) or not _integer(right) or int(right) < int(left):
            return None
        values.append(int(right) - int(left))
    calls, prompt, completion = values
    if calls != 1 or prompt + completion == 0:
        return None
    return prompt, completion


# --- Anonymous development prompt ---


@dataclass(frozen=True)
class _Prompt:
    payload: JsonObject
    candidates: Mapping[str, str]
    targets: Mapping[tuple[str, str], AnalysisSpec]


def _scope(snapshot: Any, scenario_id: str) -> Mapping[str, Any]:
    if not isinstance(snapshot, Mapping):
        return {}
    value = snapshot.get(scenario_id, snapshot)
    return value if isinstance(value, Mapping) else {}


def _numeric(value: Any) -> float | None:
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if isfinite(number) else None


def _numeric_mapping(value: Mapping[str, Any], allowed: frozenset[str]) -> dict[str, float]:
    result = {}
    for key in sorted(allowed):
        number = _numeric(value.get(key))
        if number is not None:
            result[key] = number
    return result


def _baseline(snapshot: JsonObject, scenario_id: str) -> dict[str, float]:
    return _numeric_mapping(_scope(snapshot.get("baseline_metrics", {}), scenario_id), _METRICS)


def _evidence(snapshot: JsonObject, group: GroupSpec, scenario_id: str) -> dict[str, float]:
    """Expose only named development measurements present in the evidence group's allowlist."""
    if group.kind != "evidence_llm":
        return {}
    raw = snapshot.get("evidence", {})
    if not isinstance(raw, Mapping):
        return {}
    approved = set(group.allowed_evidence)
    values = {}
    for key in sorted(approved & _EVIDENCE):
        value = raw.get(key)
        if isinstance(value, Mapping):
            value = value.get(scenario_id)
        number = _numeric(value)
        if number is not None:
            values[key] = number
    for key in sorted(approved & _BUNDLES):
        value = raw.get(key)
        if isinstance(value, Mapping):
            values.update(_numeric_mapping(_scope(value, scenario_id), _EVIDENCE))
    return values


def _interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    low, high = (_numeric(value[0]), _numeric(value[1]))
    if low is None or high is None or low > high:
        return None
    return low, high


def _history(snapshot: JsonObject, group_id: str, scenario_id: str, aliases: Mapping[str, str],
             analysis_aliases: Mapping[str, str]) -> list[JsonObject]:
    """Project a group's own published candidate/readout summary without IDs or error details."""
    projected = []
    history = snapshot.get("feedback", ())
    if not isinstance(history, (tuple, list)):
        return projected
    for item in history:
        if not isinstance(item, Mapping):
            continue
        trial = item.get("trial", {})
        if (not isinstance(trial, Mapping) or trial.get("group_id") != group_id
                or trial.get("scenario_id") != scenario_id or trial.get("status") not in ("completed", "unavailable")):
            continue
        candidate = trial.get("candidate_id")
        option = aliases.get(candidate) if isinstance(candidate, str) else None
        if option is None:
            continue
        status = trial.get("development_status")
        status = status if isinstance(status, str) and status in _STATUSES else "unavailable"
        effects = []
        paired = item.get("paired_diffs", {})
        if isinstance(paired, Mapping):
            for analysis_id, target in analysis_aliases.items():
                result = paired.get(analysis_id)
                if isinstance(result, Mapping):
                    effects.append({
                        "target": target, "estimate": _numeric(result.get("estimate")),
                        "interval": _interval(result.get("interval")),
                    })
        projected.append({"option": option, "status": status, "effects": effects})
    return projected


def _target(option: str, alias: str, analysis: AnalysisSpec) -> JsonObject:
    return {
        "option": option, "target": alias,
        "readout": analysis.readout if analysis.readout in _METRICS else "registered_readout",
        "window": analysis.window if analysis.window in _WINDOW else "registered_window",
        "unit": analysis.unit if analysis.unit in _UNIT else "registered_unit",
        "min_effect": analysis.min_effect, "equivalence_band": list(analysis.equivalence_band),
    }


# --- Strict structured response ---


class _ResponseError(ValueError):
    """A fixed schema correction safe to include in the next request."""


def _unique_object(pairs: list[tuple[str, Any]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise _ResponseError("Do not repeat JSON field names.")
        result[key] = value
    return result


def _decode(raw: str, prompt: _Prompt) -> JsonObject:
    if not isinstance(raw, str) or len(raw) > 32768:
        raise _ResponseError("Return one bounded JSON object.")
    try:
        response = json.loads(raw, object_pairs_hook=_unique_object)
    except _ResponseError:
        raise
    except (ValueError, RecursionError):
        raise _ResponseError("Return one valid JSON object.") from None
    if not isinstance(response, dict) or response.keys() != {"action", "option", "reason", "predictions"}:
        raise _ResponseError("Return exactly action, option, reason, and predictions.")
    action, option, reason = response["action"], response["option"], response["reason"]
    if (not isinstance(reason, str) or not reason.strip() or len(reason) > 300 or not reason.isprintable()
            or action not in ("select", "keep_parent", "abstain")):
        raise _ResponseError("Use a registered action and a brief printable reason.")
    if action == "select":
        if not isinstance(option, str) or option not in prompt.candidates:
            raise _ResponseError("Select one of the listed anonymous options.")
    elif option is not None:
        raise _ResponseError("Set option to null when retaining the baseline or abstaining.")
    forecasts = response["predictions"]
    if not isinstance(forecasts, list) or len(forecasts) != len(prompt.targets):
        raise _ResponseError("Forecast every listed option and target exactly once.")
    covered = set()
    for row in forecasts:
        if not isinstance(row, dict) or row.keys() != {"option", "target", "direction", "effect_interval"}:
            raise _ResponseError("Each prediction needs option, target, direction, and effect_interval.")
        if not isinstance(row["option"], str) or not isinstance(row["target"], str):
            raise _ResponseError("Use the listed anonymous option and target IDs.")
        key = (row["option"], row["target"])
        direction = row["direction"]
        if (key not in prompt.targets or key in covered or not isinstance(direction, str)
                or direction not in _PRED_STATES or action == "abstain" and direction != "abstain"):
            raise _ResponseError("Use each listed pair once and choose a registered direction.")
        covered.add(key)
        if direction in ("increase", "decrease"):
            if row["effect_interval"] is not None and _interval(row["effect_interval"]) is None:
                raise _ResponseError("A directional effect_interval must contain two finite ordered numbers.")
        elif row["effect_interval"] is not None:
            raise _ResponseError("Set effect_interval to null for abstain or unchanged.")
    return response


def _fallback(prompt: _Prompt, reason: str) -> JsonObject:
    return {
        "action": "abstain", "option": None, "reason": reason,
        "predictions": [
            {"option": option, "target": target, "direction": "abstain", "effect_interval": None}
            for option, target in prompt.targets
        ],
    }


def _offline_default(prompt: _Prompt) -> JsonObject:
    response = _fallback(prompt, "Offline policy has no calibrated effect forecast.")
    if prompt.candidates:
        response["action"] = "select"
        response["option"] = next(iter(prompt.candidates))
        response["reason"] = "Select the first eligible anonymous option under the offline policy."
    return response


# --- Selectors and client construction ---


class LLMSelector:
    """Synchronous Entry callable built from one group of a validated development manifest."""

    def __init__(self, manifest: ExperimentManifest, group: GroupSpec, *, mode: Literal["offline", "real"],
                 client_factory: ClientFactory | None, usage_root: Path | None, retry_limit: int,
                 max_output_tokens: int, count_prompt_tokens: PromptCounter, offline_responder: OfflineResponder | None,
                 system_prompt: str = _SYSTEM) -> None:
        self.manifest = manifest
        self.group = group
        self.mode = mode
        self.offline_responder = offline_responder
        self.client_factory = client_factory
        self.retry_limit = retry_limit
        self.max_output_tokens = max_output_tokens
        self.count_prompt_tokens = count_prompt_tokens
        self._system_prompt = system_prompt
        self._client: ChatClient | None = None
        self._ledger = _UsageLedger(
            group, manifest.exp_id, usage_root if mode == "real" else None,
            retry_limit=retry_limit, output_cap=max_output_tokens, system_prompt=system_prompt,
        )
        self._recipes = {row.candidate_id: row.recipe for row in manifest.candidates}
        self._scenarios = {row.scenario_id: row for row in manifest.scenarios}

    def usage(self) -> dict[str, int]:
        """Return the group attempt count, reported tokens and outstanding reserved tokens."""
        return self._ledger.summary()

    def __call__(self, snapshot: JsonObject, scenario: ScenarioSpec) -> Decision:
        if (snapshot.get("exp_id") != self.manifest.exp_id or snapshot.get("group_id") != self.group.group_id
                or self._scenarios.get(scenario.scenario_id) != scenario):
            raise ValueError("selector requires a registered development group and scenario")
        round_index = snapshot.get("round_index")
        if (not _integer(round_index) or round_index < 1
                or snapshot.get("dev_snapshot_id") != f"round-{round_index}"):
            raise ValueError("selector requires an Entry development snapshot")
        round_index = int(round_index)
        prompt = self._prompt(snapshot, scenario, round_index=round_index)
        if not prompt.candidates:
            response = _fallback(prompt, "No eligible anonymous option remains.")
        elif self.mode == "offline":
            return self._offline(round_index, scenario, prompt)
        else:
            return self._real(round_index, scenario, prompt)
        return self._decision(round_index, scenario, prompt, response)

    def _prompt(self, snapshot: JsonObject, scenario: ScenarioSpec, *, round_index: int) -> _Prompt:
        scene_id = scenario.scenario_id
        candidates = [candidate for candidate in scenario.candidate_ids if candidate != scenario.parent_id]
        aliases = {candidate: f"C{index}" for index, candidate in enumerate(candidates, start=1)}
        analyses = [analysis for analysis in self.manifest.analysis if analysis.scenario_id == scene_id]
        analysis_aliases = {analysis.analysis_id: f"T{index}" for index, analysis in enumerate(analyses, start=1)}
        forecasts = [analysis for analysis in analyses if analysis.acceptance.get("role") == "prediction"]
        if not forecasts:
            forecasts = analyses[:1]
        history = _history(snapshot, self.group.group_id, scene_id, aliases, analysis_aliases)
        exposed = {row["option"] for row in history}
        eligible = {alias: candidate for candidate, alias in aliases.items() if alias not in exposed}
        common = set(self.manifest.prediction_candidate_set[scene_id])
        targets = {}
        for alias, candidate in eligible.items():
            if candidate in common:
                for analysis in forecasts:
                    targets[(alias, analysis_aliases[analysis.analysis_id])] = analysis
        payload = {
            "round": round_index, "baseline": _baseline(snapshot, scene_id),
            "options": [
                {"option": alias, "description": _OPTIONS.get(self._recipes[candidate], "Uses a registered structural change.")}
                for alias, candidate in eligible.items()
            ],
            "targets": [_target(option, alias, analysis) for (option, alias), analysis in targets.items()],
            "feedback": history, "evidence": _evidence(snapshot, self.group, scene_id),
        }
        return _Prompt(payload, eligible, targets)

    def _offline(self, round_index: int, scenario: ScenarioSpec, prompt: _Prompt) -> Decision:
        try:
            if self.offline_responder is None:
                raw = _offline_default(prompt)
            else:
                # The script receives a fresh copy of the exact external projection.
                raw = self.offline_responder(json.loads(json.dumps(prompt.payload, allow_nan=False)))
            response = _decode(raw if isinstance(raw, str) else json.dumps(raw, allow_nan=False), prompt)
            return self._decision(round_index, scenario, prompt, response)
        except Exception:
            failure = _fallback(prompt, "Offline responder did not provide a valid registered proposal.")
            return self._decision(round_index, scenario, prompt, failure)

    def _real(self, round_index: int, scenario: ScenarioSpec, prompt: _Prompt) -> Decision:
        correction = None
        failure = "Provider request failed or returned an invalid registered proposal."
        for _ in range(self.retry_limit + 1):
            payload = dict(prompt.payload)
            if correction is not None:
                payload["validation_feedback"] = correction
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"), allow_nan=False)},
            ]
            try:
                prompt_bound = self.count_prompt_tokens(messages)
            except Exception:
                failure = "Prompt token allowance could not be determined."
                break
            if not _integer(prompt_bound) or prompt_bound < 1:
                failure = "Prompt token allowance must be a positive integer."
                break
            reservation, budget_reason = self._ledger.reserve(int(prompt_bound), self.max_output_tokens)
            if reservation is None:
                failure = budget_reason or "Group inference budget exhausted."
                break
            raw, usage = self._request(messages, reservation.completion_cap)
            if not self._ledger.settle(reservation, usage):
                failure = "Provider usage exceeded the reserved inference allowance."
                break
            if raw is not None:
                try:
                    response = _decode(raw, prompt)
                    return self._decision(round_index, scenario, prompt, response)
                except _ResponseError as error:
                    correction = str(error)
                except (TypeError, ValueError) as error:
                    message = str(error)
                    # Only forward field errors whose contract text contains no record data.
                    correction = (
                        message if message in _SAFE_CONTRACT_ERRORS
                        else "Provide valid values for the registered selection and each structured forecast."
                    )
        if correction is not None:
            failure = f"{failure} Last validation error: {correction}"
        return self._decision(round_index, scenario, prompt, _fallback(prompt, failure))

    def _request(self, messages: list[dict[str, str]], cap: int) -> tuple[str | None, tuple[int, int] | None]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return None, None
        try:
            if self._client is None:
                if self.client_factory is None or self.group.model is None or self.group.model_version is None:
                    return None, None
                self._client = self.client_factory(self.group.model, self.group.model_version)
            before = dict(self._client.total_usage)
        except Exception:
            return None, None
        try:
            response = asyncio.run(self._client.chat(messages, max_tokens=cap))
        except Exception:
            # Backend exceptions may include response bodies and credentials.
            response = None
        try:
            usage = _reported_usage(before, self._client.total_usage)
        except Exception:
            usage = None
        return response, usage

    def _decision(self, round_index: int, scenario: ScenarioSpec, prompt: _Prompt, response: JsonObject) -> Decision:
        args = {
            "exp_id": self.manifest.exp_id, "group_id": self.group.group_id, "scenario_id": scenario.scenario_id,
            "parent_id": scenario.parent_id, "round_index": round_index, "dev_snapshot_id": f"round-{round_index}",
        }
        group_id = quote(self.group.group_id, safe="")
        scene_id = quote(scenario.scenario_id, safe="")
        record_id = f"r{round_index}:{group_id}:{scene_id}"
        selection = SelectionRecord(
            **args, selection_id=record_id, sequence=round_index, menu_id=f"{scenario.scenario_id}-menu",
            action=response["action"], candidate_id=prompt.candidates.get(response["option"]), reason=response["reason"],
        )
        selection.validate()
        forecasts = []
        for row in response["predictions"]:
            option, target = row["option"], row["target"]
            analysis = prompt.targets[(option, target)]
            abstained = row["direction"] == "abstain"
            abstain_reason = None
            if abstained:
                abstain_reason = response["reason"] if selection.action == "abstain" else "No supported ex ante forecast."
            forecast = StructuredPrediction(
                **args, prediction_id=f"{record_id}:{option}:{target}", explanation_id=f"registered:{target}",
                candidate_id=prompt.candidates[option], analysis_id=analysis.analysis_id,
                readout=analysis.readout, unit=analysis.unit, window=analysis.window,
                direction=None if abstained else Direction(row["direction"]),
                effect_interval=_interval(row["effect_interval"]), abstained=abstained,
                abstain_reason=abstain_reason,
                conditions=None if abstained else "Registered scenario, intervention and measurement window.",
                falsifier=None if abstained else "Paired measurement interval is disjoint from the registered forecast region.",
            )
            forecast.validate()
            forecasts.append(forecast)
        return selection, tuple(forecasts)


def build_selectors(manifest: ExperimentManifest, *, mode: Literal["offline", "real"] = "offline",
                    client_factory: ClientFactory | None = None, usage_root: Path | None = None, retry_limit: int = 1,
                    max_output_tokens: int = 512, count_prompt_tokens: PromptCounter = _default_prompt_bound,
                    offline_responder: OfflineResponder | None = None,
                    prompt_templates: Mapping[str, str] | None = None) -> dict[str, LLMSelector]:
    """Build the registered policies for the LLM groups of a development manifest.

    A real client factory must implement one reserved attempt per chat call.
    Supply usage_root to enforce the registered budget across process restarts.
    A provider-specific upper bound can replace the default prompt allowance.
    Prompt ablations in real mode require preregistered instructions for every template ID.
    The shared task and JSON response instructions follow each template's text.
    """
    manifest.validate()
    if mode not in ("offline", "real") or mode == "real" and client_factory is None:
        raise ValueError("a client factory is required for explicit real mode")
    if not _integer(retry_limit) or retry_limit < 0 or not _integer(max_output_tokens) or max_output_tokens < 1:
        raise ValueError("retry limit and completion cap must be valid integers")
    llm_groups = [group for group in manifest.groups if group.kind != "no_llm"]
    template_ids = {group.prompt_template_id for group in llm_groups}
    if mode == "real" and len(template_ids) > 1 and prompt_templates is None:
        raise ValueError("real mode requires system instructions for each registered prompt template")
    selectors: dict[str, LLMSelector] = {}
    for group in llm_groups:
        system_prompt = _SYSTEM
        if prompt_templates is not None:
            registered_prompt = prompt_templates.get(group.prompt_template_id or "")
            if not isinstance(registered_prompt, str) or not registered_prompt.strip():
                raise ValueError(f"missing system instructions for prompt template {group.prompt_template_id!r}")
            system_prompt = f"{registered_prompt}\n\n{_SYSTEM}"
        selectors[group.group_id] = LLMSelector(
            manifest, group, mode=mode, client_factory=client_factory, usage_root=usage_root,
            retry_limit=int(retry_limit), max_output_tokens=int(max_output_tokens), count_prompt_tokens=count_prompt_tokens,
            offline_responder=offline_responder, system_prompt=system_prompt,
        )
    return selectors


def registry_client_factory(models_path: Path, *, env_path: Path | None = None) -> ClientFactory:
    """Reuse Stochopia's registered transports with one HTTP attempt per reserved call."""
    from stochopia.llm import AnthropicClient, OpenAICompatClient, create_client, load_model_registry

    def factory(model: str, version: str) -> ChatClient:
        if env_path is not None:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        registry = load_model_registry(models_path)
        if model not in registry:
            raise ValueError("model is absent from the registered client configuration")
        registry[model] = replace(registry[model], model=version)
        client = create_client(model, registry)
        if isinstance(client, (OpenAICompatClient, AnthropicClient)):
            client.RETRY_DELAYS = ()
            client.TRUNCATION_RETRIES = 0
        return client

    return factory
