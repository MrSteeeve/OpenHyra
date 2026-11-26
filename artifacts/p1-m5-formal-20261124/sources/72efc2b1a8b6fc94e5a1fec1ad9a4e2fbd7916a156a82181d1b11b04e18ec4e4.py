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
        self.g = nn.Parameter(torch.zeros(48))
        assert sum(p.numel() for p in self.parameters()) <= 28_000

    def _features(self, obs: StepObservation) -> Tensor:
        ticks = obs.book_ticks
        shares = obs.book_shares.to(self.input_projection.weight.dtype)
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
        own = obs.is_own.reshape_as(imbalance)
        previous = self.token_embed(obs.previous_event.long() + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), -1), previous.flatten(-2),
        ), -1)
        return self.input_projection(features.to(shares.dtype)).tanh()

    def _run(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        hidden, initial = (None, None) if state is None else state
        first = x[..., :40]
        gates = x.new_tensor([0.5, 0.95]).repeat_interleave(40)
        inputs = torch.cat((0.5 * first, 0.05 * first), -1)
        averages = parallel_affine_scan(gates, inputs, initial)
        fast, slow = averages.split(40, dim=-1)
        contrast = torch.cat((fast - slow, torch.zeros_like(x[..., 40:])), -1)
        out, last = self.gru(
            x + self.g * contrast, None if hidden is None else hidden.unsqueeze(0),
        )
        return out, (last[0], averages[:, -1].contiguous())

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._run(self._features(observation), None)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        out, state = self._run(self._features(observation).unsqueeze(1), state)
        return out[:, 0], state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, -1).long() + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), -1)
        return self.heads[stage](context)
