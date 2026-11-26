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
            nn.Linear(48 + 8 * i, n) for i, n in enumerate(BUCKET_COUNTS)
        )

        # Signed offsets in absolute tick coordinates; the final bucket has
        # no fixed displacement. Executions have no placement price.
        direction = torch.zeros(BUCKET_COUNTS[1], BUCKET_COUNTS[2])
        direction[:4, :25] = torch.arange(-12, 13).sign()
        self.register_buffer("price_direction", direction, persistent=False)
        self.requires_grad_(False)

    def _features(self, obs: StepObservation) -> Tensor:
        ticks = obs.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = obs.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(-1) / present.sum(-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(-1), 0).log1p()
        bid_size, ask_size = shares.sum(-1).unbind(-1)
        imbalance = (bid_size - ask_size) / (bid_size + ask_size).clamp_min(1)
        own = obs.is_own.reshape_as(spread)
        previous = self.token_embed(obs.previous_event + self.token_offsets)
        x = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), -1), previous.flatten(-2),
        ), -1)
        return self.input_projection(x.to(dtype)).tanh()

    def _touch(self, obs: StepObservation) -> Tensor:
        dtype = torch.promote_types(self.input_projection.weight.dtype, torch.float64)
        return torch.cat((obs.book_ticks[..., 0].to(dtype),
                          obs.book_shares[..., 0].to(dtype)), -1)

    @staticmethod
    def _pressure(touch: Tensor, own: Tensor, previous: Tensor | None) -> Tensor:
        if previous is None:
            return torch.zeros_like(touch[..., :1])
        old_tick, old_size, old_pressure = previous.split((2, 2, 1), -1)
        tick, size = touch.split(2, -1)
        valid = (old_tick > 0) & (tick > 0)
        retreat = torch.stack((tick[..., 0] < old_tick[..., 0],
                               tick[..., 1] > old_tick[..., 1]), -1)
        fraction = (old_size - size).relu() / old_size.clamp_min(1)
        withdrawal = torch.where(tick == old_tick, fraction, retreat.to(fraction.dtype))
        withdrawal = withdrawal.masked_fill(~valid, 0)
        bid, ask = withdrawal.unbind(-1)
        return (0.8 * old_pressure + own.reshape_as(old_pressure).to(touch.dtype)
                * (ask - bid).unsqueeze(-1)).clamp(-1, 1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        hidden, _ = self.gru(self._features(observation))
        touch = self._touch(observation)
        own = observation.is_own.reshape(touch.shape[:-1])
        previous = None
        pressure = []
        for t in range(touch.size(1)):
            current = touch[:, t]
            p = self._pressure(current, own[:, t], previous)
            previous = torch.cat((current, p), -1)
            pressure.append(p)
        return torch.cat((hidden, torch.stack(pressure, 1).to(hidden.dtype)), -1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        x = self._features(observation)
        h0 = None if state is None else state[..., :48].to(x.dtype).unsqueeze(0)
        hidden, _ = self.gru(x.unsqueeze(1), h0)
        h = hidden[:, 0]
        touch = self._touch(observation)
        p = self._pressure(touch, observation.is_own,
                           None if state is None else state[..., 48:])
        return torch.cat((h, p.to(h.dtype)), -1), torch.cat((h.to(touch.dtype), touch, p), -1)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        h, p = context.split((48, 1), -1)
        if prior:
            tokens = torch.stack(prior, -1) + self.token_offsets[:stage]
            h = torch.cat((h, self.token_embed(tokens).flatten(-2)), -1)
        logits = self.heads[stage](h)
        if stage == 2:
            logits = logits + 1.25 * p * self.price_direction[prior[1]].to(logits.dtype)
        return logits
