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
        self.timing = nn.Sequential(nn.Linear(5, 16), nn.SiLU(),
                                    nn.Linear(16, BUCKET_COUNTS[0]))
        nn.init.zeros_(self.timing[-1].weight)
        nn.init.zeros_(self.timing[-1].bias)

    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        both = present.all(dim=-1)
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        price = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~both, 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)
        timing = torch.cat((
            torch.stack((spread, imbalance), dim=-1),
            depth[..., 0], both.unsqueeze(-1),
        ), dim=-1).to(dtype)
        return self.input_projection(features.to(dtype)).tanh(), timing

    def _context(
        self, observation: StepObservation, state: State, single: bool
    ) -> tuple[Tensor, Tensor]:
        features, timing = self._features(observation)
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(features.unsqueeze(1) if single else features, prior)
        if single:
            out = out[:, 0]
        return torch.cat((out, timing), dim=-1), hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._context(observation, None, False)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        return self._context(observation, state, True)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        recurrent, timing = context.split((48, 5), dim=-1)
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            recurrent = torch.cat((recurrent, prefix), dim=-1)
        logits = self.heads[stage](recurrent)
        return logits + self.timing(timing) if stage == 0 else logits
