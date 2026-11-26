from __future__ import annotations

import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import parallel_affine_scan


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.depth_residual = nn.Linear(40, 48, bias=False)
        nn.init.zeros_(self.depth_residual.weight)

    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0).flatten(-2)
        depth = (shares / 100).log1p().flatten(-2)
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat(
            (price, depth, torch.stack((spread, imbalance, own), dim=-1),
             previous.flatten(-2)), dim=-1,
        )
        return self.input_projection(features.to(dtype)).tanh(), depth

    def _context(
        self, x: Tensor, depth: Tensor, state: State
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        raw, hidden = self.gru(x, None if state is None else state[0].unsqueeze(0))
        paired = torch.cat((depth, depth), dim=-1)
        weights = depth.new_tensor([0.25, 0.03125]).repeat_interleave(20)
        initial = paired[:, 0] if state is None else state[1]
        # With no history, set the first EMA exactly to the first observed depth.
        gates = (1 - weights).expand_as(paired)
        inputs = weights * paired
        if state is None:
            gates = torch.cat((torch.zeros_like(gates[:, :1]), gates[:, 1:]), dim=1)
            inputs = torch.cat((paired[:, :1], inputs[:, 1:]), dim=1)
        ema = parallel_affine_scan(gates, inputs, h0=initial)
        previous = torch.cat((initial.unsqueeze(1), ema[:, :-1]), dim=1)
        context = raw + self.depth_residual(paired - previous)
        return context, (hidden[0], ema[:, -1])

    def sequence_context(self, observation: StepObservation) -> Tensor:
        x, depth = self._features(observation)
        return self._context(x, depth, None)[0]

    def step_context(
        self, observation: StepObservation, state: State
    ) -> tuple[Tensor, State]:
        x, depth = self._features(observation)
        context, next_state = self._context(x.unsqueeze(1), depth.unsqueeze(1), state)
        return context[:, 0], next_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
