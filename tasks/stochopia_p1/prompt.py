"""Frozen P1 context and synthesis instructions."""

import json
import math
from collections.abc import Mapping, Sequence

# These fields are the only experiment readouts allowed into the proposer.
METRIC_CONTEXT = ("val_nll_total", "lob_impact_score")
EVIDENCE_CONTEXT = (
    "parameters", "state_dim", "nll_dt", "nll_action", "nll_rel_ticks", "nll_volume",
    "event_count_delta", "zero_wait_rate_delta", "sign_acf_delta", "hurst_delta",
    "top_depth_delta", "idle_abs_drift", "impact_peak", "blind_ratio", "impact_y",
    "impact_delta", "recovery_ratio",
    "end_over_peak_ratio", "free_abs_drift_ticks", "trade_sign_acf_exp_gap", "hurst_gap",
    "visible_over_blind_ratio", "impact_exponent_gap", "event_count_rel_gap", "l1_depth_mae",
)
EVIDENCE_PRIMARY = (
    "end_over_peak_ratio", "free_abs_drift_ticks", "trade_sign_acf_exp_gap", "hurst_gap",
    "visible_over_blind_ratio", "impact_exponent_gap", "event_count_rel_gap", "l1_depth_mae",
    "lob_impact_score", "impact_peak",
)
HYPOTHESIS_TEXT_FIELDS = (
    "observed_bottleneck",
    "operator_diagnosis", "structural_prescription", "falsifier",
)
HYPOTHESIS_FIELDS = ("hypothesis_id", "parent_hypothesis_id", *HYPOTHESIS_TEXT_FIELDS)
METRIC_DEFINITIONS = {
    "lob_impact_score": "counterfactual impact realism (0–100 points, HIGHER is better)",
    "val_nll_total": "five-tuple NLL (nats/event, LOWER is better)",
    "visible_over_blind_ratio": "visible/blind peak impact (ratio, C3 passes at ≥3; raise when below 3)",
    "end_over_peak_ratio": "stop-end/execution-end impact (ratio, target [0.50,0.67]; lower when above 0.67)",
    "free_abs_drift_ticks": "absolute free-generation drift (ticks, LOWER)",
    "trade_sign_acf_exp_gap": "absolute trade-sign ACF exponent gap (dimensionless, LOWER)",
    "hurst_gap": "absolute Hurst exponent gap (dimensionless, LOWER)",
    "impact_exponent_gap": "absolute impact exponent gap from 0.5 (dimensionless, LOWER)",
    "event_count_rel_gap": "absolute event-count relative error (fraction, LOWER)",
    "l1_depth_mae": "mean absolute best-level depth error (shares, LOWER)",
    "impact_peak": "signed peak impact (ticks, direction depends on the diagnosis)",
}
CONTRACT = """PyTorch LOB contract (stochopia.neural.lob_interface):
StepObservation(book_ticks: Tensor, book_shares: Tensor, previous_event: Tensor,
is_own: Tensor, has_deep: Tensor | None = None). Pre-event book shape [...,2,10],
previous tokens [...,4], own flag has one value per prefix.
LOBBatch(observation: StepObservation, targets: Tensor); targets [B,T,4], bins (16,6,26,12).
StepSampler(model: LOBWorldModel, context: Tensor, observation: StepObservation).
Methods: conditioned(targets: Tensor) -> FiveTupleLogits; sample(generator: Generator | None = None) -> Tensor.
LOBWorldModel methods to implement:
sequence_context(observation: StepObservation) -> Tensor,
step_context(observation: StepObservation, state: State) -> tuple[Tensor, State],
head_logits(context: Tensor, stage: int, prior: tuple[Tensor,...]) -> Tensor.
Keep inherited forward_sequence(batch: LOBBatch) -> FiveTupleLogits and
forward_step(step_obs: StepObservation, hidden_state: State = None) -> tuple[StepSampler, State].
State is None, a Tensor, or a tuple of Tensors. Use at most 28000 trainable params and
128 real state scalars (complex counts twice). No cross-time access to the future.
From stochopia.neural.primitives:
parallel_affine_scan(gates: Tensor, inputs: Tensor, h0: Tensor | None = None, chunk_size: int = 32) -> Tensor,
affine_scan_step(gate_t: Tensor, input_t: Tensor, h_prev: Tensor) -> Tensor,
real_diagonal(decay_logit: Tensor) -> Tensor, complex_diagonal(decay_logit: Tensor, phase: Tensor) -> Tensor.
Scan inputs/output [B,T,C], optional h0 [B,C]; step input/state/output [B,C].
Supply gates matching inputs or a single shared [C] diagonal. Diagonal helpers preserve input shape."""
PLAN_INSTRUCTION = """Return JSON only. The runner supplies hypothesis_id and parent_hypothesis_id.
Design a causal PyTorch revision within 28,000 parameters and 128 real state scalars.
Give exactly two predictions, primary for the bottleneck and guardrail for another readout's non-degradation.
ALL bounds are signed deltas: candidate_metric - active_parent_metric. Never give absolute levels.
increase: delta_interval [low, high], 0 <= low < high, e.g. [1,8].
decrease: delta_interval [low, high], low < high <= 0, e.g. [-0.5,-0.1].
equivalent: provide only equivalence_band [low, high], low < 0 < high.
Propose an actionable revision and explicit falsifier."""
CODE_INSTRUCTION = """Return exactly one fenced python code block. Define one self-contained
class CandidateLOBWorldModel(LOBWorldModel), constructible with no arguments.
Only import standard math libraries, torch, torch.nn, torch.nn.functional,
and stochopia.neural.lob_interface or stochopia.neural.primitives.
Implement the sealed prescription. Keep evaluation and data out of the code.
Keep source compact so the entire file can be included as a later parent.
SeedGRULOBModel can be imported from stochopia.neural.primitives.
Use one canonical computation for sequence and step; respect all shapes."""


def forecast_metrics(group: str, registered: tuple[str, ...] = ()) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Allowed primary and guardrail metrics for the arm's information view."""
    if group == "metrics" or group == "no_llm":
        return ("lob_impact_score",), ("val_nll_total",)
    primary = registered or EVIDENCE_PRIMARY
    primary = tuple(metric for metric in primary if metric in EVIDENCE_PRIMARY)
    return primary, ("val_nll_total", *primary)


def forecast_targets(group: str, registered: tuple[str, ...]) -> str:
    primary, guardrail = forecast_metrics(group, registered)
    definitions = [f"{name}: {METRIC_DEFINITIONS[name]}." for name in dict.fromkeys((*primary, *guardrail))]
    return "\n".join([
        f"Primary metric in {json.dumps(primary)}. Guardrail metric in {json.dumps(guardrail)}.",
        *definitions,
    ])


def forecast_example(group: str, registered: tuple[str, ...]) -> str:
    """Supply a complete valid JSON specimen with only group-visible metric names."""
    allowed, _ = forecast_metrics(group, registered)
    hypothesis = {
        "observed_bottleneck": "Impact realism can improve.",
        "operator_diagnosis": "One memory scale conflates recent and persistent effects.",
        "structural_prescription": "Separate slow and fast context.",
        "falsifier": "Impact score does not increase by 1–8 points.",
    }
    primary = {"role": "primary", "metric": "lob_impact_score", "direction": "increase", "delta_interval": [1, 8]}
    if group == "evidence" and "end_over_peak_ratio" in allowed:
        hypothesis["observed_bottleneck"] = "Recovery is too slow."
        hypothesis["falsifier"] = "Recovery ratio does not fall by 0.10–0.50."
        primary.update(metric="end_over_peak_ratio", direction="decrease", delta_interval=[-0.50, -0.10])
    elif allowed and primary["metric"] not in allowed:
        metric = allowed[0]
        hypothesis["observed_bottleneck"] = "Order-book response requires calibration."
        hypothesis["falsifier"] = f"{metric} falls outside the forecast band."
        primary = {"role": "primary", "metric": metric, "direction": "equivalent", "equivalence_band": [-0.05, 0.05]}
    guardrail = {"role": "guardrail", "metric": "val_nll_total", "direction": "equivalent", "equivalence_band": [-0.05, 0.05]}
    example = {"hypothesis": hypothesis, "predictions": [primary, guardrail]}
    return "Complete JSON example (illustrative only):\n" + dump_prompt(example)


def dump_prompt(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def context_packet(
    group: str, source: str, evidence: Mapping[str, object], recent: Sequence[Mapping[str, object]],
) -> tuple[str, list[dict[str, object]]]:
    """Project observations and recent hypotheses into their registered information arm."""
    if group not in ("evidence", "metrics"):
        raise ValueError(f"unsupported proposal group {group!r}")
    keys = METRIC_CONTEXT
    if group == "evidence":
        keys += EVIDENCE_CONTEXT
    row = {}
    for key in keys:
        value = evidence.get(key)
        row[key] = value if type(value) in (int, float) and math.isfinite(value) else None
    feedback = []
    if group == "evidence":
        for item in recent[-3:]:
            hypothesis = item["hypothesis"]
            if not isinstance(hypothesis, Mapping):
                raise ValueError("recent hypotheses must be mappings")
            feedback.append({
                "hypothesis": {key: hypothesis.get(key) for key in HYPOTHESIS_FIELDS},
                "verdict": item["verdict"],
            })
    return f"Parent Python source:\n{source}\nDevelopment summary:\n{dump_prompt(row)}", feedback


def message_packet(
    instruction: str, context: str, recent: list[dict[str, object]], suffix: str = "", *, include_contract: bool = True,
) -> list[dict[str, str]]:
    content = context
    if recent:
        content += f"\nRecent group history:\n{dump_prompt(recent)}"
    if suffix:
        content += f"\n{suffix}"
    system = f"{CONTRACT}\n\n{instruction}" if include_contract else instruction
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]
