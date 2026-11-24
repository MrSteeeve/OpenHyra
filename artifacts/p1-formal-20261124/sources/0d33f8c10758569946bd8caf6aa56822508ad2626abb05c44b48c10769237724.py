from __future__ import annotations

import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [sum(BUCKET_COUNTS[:i]) for i in range(len(BUCKET_COUNTS))]
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.volume_residual = nn.Sequential(
            nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, BUCKET_COUNTS[3])
        )
        nn.init.zeros_(self.volume_residual[-1].weight)
        nn.init.zeros_(self.volume_residual[-1].bias)

    def _encode(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        shares = observation.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        prices = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depths = (shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape(spread.shape)
        previous = self.token_embed(observation.previous_event.long() + self.token_offsets)
        features = torch.cat((
            prices.flatten(-2), depths.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)
        encoded = self.input_projection(features.to(shares.dtype)).tanh()

        near_present = ticks[..., :2] > 0
        near_depths = depths[..., :2].masked_fill(~near_present, 0)
        queue = torch.cat((
            near_depths.flatten(-2), near_present.flatten(-2).to(shares.dtype),
        ), dim=-1)
        return encoded, queue

    def sequence_context(self, observation: StepObservation) -> Tensor:
        encoded, queue = self._encode(observation)
        out, _ = self.gru(encoded)
        return torch.cat((out, queue.to(out.dtype)), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        encoded, queue = self._encode(observation)
        hidden = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(encoded.unsqueeze(1), hidden)
        return torch.cat((out[:, 0], queue.to(out.dtype)), dim=-1), hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        recurrent, queue = context.split((48, 8), dim=-1)
        if stage == 0:
            return self.heads[stage](recurrent)
        tokens = torch.stack(prior, dim=-1).long() + self.token_offsets[:stage]
        prefix = self.token_embed(tokens).flatten(-2)
        logits = self.heads[stage](torch.cat((recurrent, prefix), dim=-1))
        if stage == 3:
            logits = logits + self.volume_residual(torch.cat((queue, prefix), dim=-1))
        return logits
