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
            nn.Linear(48 + stage * 8, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.depletion_projection = nn.Linear(4, 48, bias=False)
        nn.init.zeros_(self.depletion_projection.weight)

    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
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
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event.long() + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1),
            previous.flatten(-2),
        ), dim=-1)
        return self.input_projection(features.to(dtype)), depth[..., 0]

    def _run(self, base: Tensor, depth: Tensor, state: State) -> tuple[Tensor, Tensor]:
        if state is None:
            hidden = None
            fast = slow = depth[:, 0]
        else:
            hidden, fast, slow = state.split((48, 2, 2), dim=-1)
            hidden = hidden.unsqueeze(0).contiguous()

        signals = []
        for current in depth.unbind(dim=1):
            delta = current - fast
            signals.append(torch.cat((delta, fast - slow), dim=-1))
            fast = fast + 0.25 * delta
            slow = slow + 0.025 * (current - slow)

        inputs = (base + self.depletion_projection(torch.stack(signals, dim=1))).tanh()
        out, hidden = self.gru(inputs, hidden)
        return out, torch.cat((hidden[0], fast, slow), dim=-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        base, depth = self._features(observation)
        if base.size(1) == 0:
            return base
        return self._run(base, depth, None)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        base, depth = self._features(observation)
        out, state = self._run(base.unsqueeze(1), depth.unsqueeze(1), state)
        return out[:, 0], state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1).long() + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
