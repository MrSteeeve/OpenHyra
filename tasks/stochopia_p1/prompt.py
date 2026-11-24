"""Frozen P1 context and synthesis instructions."""

import json

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
HYPOTHESIS_FIELDS = (
    "hypothesis_id", "parent_hypothesis_id", "observed_bottleneck",
    "operator_diagnosis", "structural_prescription", "falsifier",
)
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
real_diagonal(decay_logit: Tensor) -> Tensor, complex_diagonal(decay_logit: Tensor, phase: Tensor) -> Tensor."""
PLAN_INSTRUCTION = """Return one JSON object and no code. Keys: hypothesis and predictions.
hypothesis fields: hypothesis_id, parent_hypothesis_id (null only for a root),
observed_bottleneck, operator_diagnosis, structural_prescription, falsifier.
predictions must contain exactly two items in order: role "primary" tied to
the observed bottleneck, then role "guardrail" forecasting non-degradation
on a different readout. Each has role, metric, direction (increase, decrease
or equivalent), interval [low,high], and optional equivalence_band [low,high]
(required for equivalent). Forecast candidate minus the active parent.
Explicitly revise or open a competing hypothesis when feedback refutes it."""
CODE_INSTRUCTION = """Return exactly one fenced python code block. Define one self-contained
class CandidateLOBWorldModel(LOBWorldModel), constructible with no arguments.
Only import standard math libraries, torch, torch.nn, torch.nn.functional,
and stochopia.neural.lob_interface or stochopia.neural.primitives.
Implement the sealed prescription. Keep evaluation and data out of the code.
Keep source compact so the entire file can be included as a later parent.
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
    return f"Primary metric in {json.dumps(primary)}. Guardrail metric in {json.dumps(guardrail)}."
