from __future__ import annotations

import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        self.state_dim = 48
        embedding_dim = 8
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, self.state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(self.state_dim, self.state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(self.state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28_000

    def _features(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        price = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)

        fraction = shares[..., 0] / shares[..., :3].sum(dim=-1).clamp_min(1)
        fraction = fraction.masked_fill(~present, 0)
        next_tick = ticks[..., 1]
        gap = (next_tick - touch).abs().to(dtype).log1p()
        gap = gap.masked_fill(~(present & (next_tick > 0)), 0)
        depth = torch.cat(
            ((shares[..., :8] / 100).log1p(), fraction.unsqueeze(-1), gap.unsqueeze(-1)),
            dim=-1,
        )

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat(
            (price.flatten(-2), depth.flatten(-2),
             torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2)),
            dim=-1,
        )
        return self.input_projection(features.to(dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self.gru(self._features(observation))[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        return out[:, 0], hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
