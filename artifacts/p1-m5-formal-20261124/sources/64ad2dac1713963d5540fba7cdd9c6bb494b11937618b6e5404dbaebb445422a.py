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
        self.dt_residual = nn.Sequential(
            nn.Linear(52, 16), nn.SiLU(),
            nn.Linear(16, BUCKET_COUNTS[0], bias=False),
        )
        nn.init.zeros_(self.dt_residual[-1].weight)

    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        prices = relative.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((
            prices.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)
        book = torch.cat((
            torch.stack((spread, imbalance), dim=-1),
            depth[..., 0].masked_fill(~present, 0),
        ), dim=-1)
        return self.input_projection(features.to(dtype)).tanh(), book.to(dtype)

    def _context(
        self, observation: StepObservation, state: State = None, *, step: bool = False,
    ) -> tuple[Tensor, Tensor]:
        features, book = self._features(observation)
        if step:
            features = features.unsqueeze(1)
            book = book.unsqueeze(1)
        out, hidden = self.gru(features, None if state is None else state.unsqueeze(0))
        context = torch.cat((out, book), dim=-1)
        return (context[:, 0] if step else context), hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._context(observation)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        return self._context(observation, state, step=True)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        recurrent = context[..., :48]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            recurrent = torch.cat((recurrent, self.token_embed(tokens).flatten(-2)), dim=-1)
        logits = self.heads[stage](recurrent)
        return logits + self.dt_residual(context) if stage == 0 else logits
