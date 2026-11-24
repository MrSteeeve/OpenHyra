from __future__ import annotations

import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import affine_scan_step, parallel_affine_scan


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer(
            "trace_decay", torch.tensor([2.0 ** (-1.0 / h) for h in (4, 16, 64, 256)]),
            persistent=False,
        )
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.trace_projection = nn.Linear(4, 48, bias=False)
        nn.init.zeros_(self.trace_projection.weight)

    def _trace_inputs(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        action = observation.previous_event[..., 1]
        gate = self.trace_decay.to(self.input_projection.weight.dtype)
        sign = (action == 4).to(gate.dtype) - (action == 5).to(gate.dtype)
        return gate, sign.unsqueeze(-1) * (1 - gate)

    def _features(self, observation: StepObservation, traces: Tensor) -> Tensor:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        price = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)
        return (self.input_projection(features.to(dtype)) + self.trace_projection(traces)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        gate, inputs = self._trace_inputs(observation)
        traces = parallel_affine_scan(gate, inputs)
        output, _ = self.gru(self._features(observation, traces))
        return output

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        gate, inputs = self._trace_inputs(observation)
        if state is None:
            hidden, previous = None, torch.zeros_like(inputs)
        else:
            hidden, previous = state.split((48, 4), dim=-1)
            hidden = hidden.unsqueeze(0).contiguous()
        traces = affine_scan_step(gate, inputs, previous)
        output, hidden = self.gru(self._features(observation, traces).unsqueeze(1), hidden)
        return output[:, 0], torch.cat((hidden[0], traces), dim=-1)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
