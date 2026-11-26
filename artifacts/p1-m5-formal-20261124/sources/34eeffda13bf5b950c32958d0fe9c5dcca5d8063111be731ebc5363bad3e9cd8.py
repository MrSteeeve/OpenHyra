from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer("touch_weights", torch.tensor([4., 2., 1.]), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _residual(self, ticks: Tensor, depth: Tensor) -> Tensor:
        present = ticks[..., :3] > 0
        near = depth[..., :3].masked_fill(~present, 0)
        weights = self.touch_weights.to(depth.dtype) * present
        means = (near * weights).sum(-1) / weights.sum(-1).clamp_min(1)
        bid, ask = means.unbind(-1)
        c = 0.25 * (bid - ask).tanh()
        return F.pad(torch.stack((c, -c), dim=-1), (0, 46))

    def _encode(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        shares = observation.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(-1) / present.sum(-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(-1).unbind(-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1),
            previous.flatten(-2),
        ), dim=-1)
        projected = self.input_projection(features.to(shares.dtype)).tanh()
        return projected, self._residual(ticks, depth)

    def _run(
        self, features: Tensor, residual: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        initial = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(features, initial)
        return out + residual.to(out.dtype), hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, residual = self._encode(observation)
        return self._run(features, residual)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        features, residual = self._encode(observation)
        out, hidden = self._run(features.unsqueeze(1), residual.unsqueeze(1), state)
        return out[:, 0], hidden

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
