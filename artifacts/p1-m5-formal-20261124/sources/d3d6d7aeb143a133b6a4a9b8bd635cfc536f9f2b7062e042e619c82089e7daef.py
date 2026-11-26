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
        self.change_projection = nn.Linear(6, 48, bias=False)
        nn.init.zeros_(self.change_projection.weight)

    def _book_features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        dtype = self.input_projection.weight.dtype
        ticks = observation.book_ticks
        shares = observation.book_shares.to(dtype)
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
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        own = observation.is_own.reshape_as(spread)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1),
            previous.flatten(-2),
        ), dim=-1)
        # Promote the cache to at least float32 to preserve raw tick precision.
        cache = torch.cat((touch.to(torch.float32).unsqueeze(-1), depth[..., :2]), dim=-1)
        return features.to(dtype), cache

    def _gru_input(self, features: Tensor, cache: Tensor, prior: Tensor) -> Tensor:
        delta = cache - prior
        delta = torch.cat((delta[..., :1] / 20, delta[..., 1:]), dim=-1)
        present = (cache[..., :1] > 0) & (prior[..., :1] > 0)
        delta = delta.clamp(-5, 5).masked_fill(~present, 0).flatten(-2)
        return (self.input_projection(features)
                + self.change_projection(delta.to(features.dtype))).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, cache = self._book_features(observation)
        prior = torch.cat((cache[:, :1], cache[:, :-1]), dim=1)
        out, _ = self.gru(self._gru_input(features, cache, prior))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        features, cache = self._book_features(observation)
        hidden = None
        prior = cache
        if state is not None:
            hidden = state[..., :48].to(features.dtype).unsqueeze(0).contiguous()
            prior = state[..., 48:].reshape_as(cache)
        out, hidden = self.gru(self._gru_input(features, cache, prior).unsqueeze(1), hidden)
        return out[:, 0], torch.cat((hidden[0], cache.flatten(-2)), dim=-1)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
