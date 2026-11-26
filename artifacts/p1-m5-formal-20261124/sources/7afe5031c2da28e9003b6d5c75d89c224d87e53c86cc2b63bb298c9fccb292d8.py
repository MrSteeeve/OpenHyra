from __future__ import annotations

import torch
from torch import Tensor, nn
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
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.volume_residual = nn.Sequential(
            nn.Linear(78, 16), nn.SiLU(), nn.Linear(16, BUCKET_COUNTS[3])
        )
        nn.init.zeros_(self.volume_residual[-1].weight)
        nn.init.zeros_(self.volume_residual[-1].bias)

    def _encode(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        shares = observation.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
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
        book = torch.cat((
            depth[..., 0], depth[..., 1], present.to(depth.dtype),
        ), dim=-1)
        return self.input_projection(features.to(shares.dtype)).tanh(), book

    def _context(
        self, observation: StepObservation, state: State, single: bool,
    ) -> tuple[Tensor, Tensor]:
        features, book = self._encode(observation)
        if single:
            features = features.unsqueeze(1)
        out, hidden = self.gru(features, None if state is None else state.unsqueeze(0))
        if single:
            out = out[:, 0]
        return torch.cat((out, book.to(out.dtype)), dim=-1), hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._context(observation, None, False)[0]

    def step_context(
        self, observation: StepObservation, state: State,
    ) -> tuple[Tensor, Tensor]:
        return self._context(observation, state, True)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        hidden, book = context.split((48, 6), dim=-1)
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            hidden = torch.cat((hidden, prefix), dim=-1)
        logits = self.heads[stage](hidden)
        if stage == 3:
            logits = logits + self.volume_residual(torch.cat((book, hidden), dim=-1))
        return logits
