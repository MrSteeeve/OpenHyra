from __future__ import annotations

import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        h = max(
            w for w in range(1, 49)
            if 6 * w * w + 84 * w + sum(
                count * (9 + w + 8 * stage)
                for stage, count in enumerate(BUCKET_COUNTS)
            ) <= 28000
        )
        offsets = [sum(BUCKET_COUNTS[:i]) for i in range(len(BUCKET_COUNTS))]
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(78, h)
        with torch.no_grad():
            self.input_projection.weight[:, IS_OWN_FEATURE_INDEX].zero_()
            self.input_projection.weight[:, -3:].zero_()
        self.gru = nn.GRU(h, h, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(h + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _features(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        prices = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        prices = prices.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)

        third = ticks[..., 2]
        distance = (third - touch).abs().masked_fill(~(present & (third > 0)), 0)
        distance = distance.to(dtype).log1p()
        bid_top, ask_top = shares[..., :3].sum(dim=-1).unbind(dim=-1)
        top_imbalance = (bid_top - ask_top) / (bid_top + ask_top).clamp_min(1)

        previous = self.token_embed(observation.previous_event + self.token_offsets)
        own = observation.is_own.reshape_as(spread)
        features = torch.cat((
            prices.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1),
            previous.flatten(-2), distance, top_imbalance.unsqueeze(-1),
        ), dim=-1)
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
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
