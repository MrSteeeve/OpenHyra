from __future__ import annotations

import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import affine_scan_step, parallel_affine_scan


class CandidateLOBWorldModel(LOBWorldModel):
    ACTION_STAGE = 1
    BUY_EXECUTION = 4
    SELL_EXECUTION = 5

    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer(
            "trade_rates", torch.tensor([1 / 4, 1 / 16, 1 / 64, 1 / 256]),
            persistent=False,
        )
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + stage * 8, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.trade_residual = nn.Linear(4, BUCKET_COUNTS[self.ACTION_STAGE], bias=False)
        nn.init.zeros_(self.trade_residual.weight)

    def _features(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        prices = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        scalars = torch.stack((spread, imbalance, observation.is_own.reshape_as(spread)), -1)
        features = torch.cat(
            (prices.flatten(-2), depth.flatten(-2), scalars, previous.flatten(-2)), -1
        )
        return self.input_projection(features.to(dtype)).tanh()

    def _trade_affine(self, observation: StepObservation, dtype: torch.dtype):
        action = observation.previous_event[..., self.ACTION_STAGE]
        buy = action == self.BUY_EXECUTION
        sell = action == self.SELL_EXECUTION
        rates = self.trade_rates.to(dtype=dtype)
        update = (buy | sell).unsqueeze(-1).to(dtype) * rates
        signed = (buy.to(dtype) - sell.to(dtype)).unsqueeze(-1) * rates
        return 1 - update, signed

    def sequence_context(self, observation: StepObservation) -> Tensor:
        hidden, _ = self.gru(self._features(observation))
        gates, inputs = self._trade_affine(observation, hidden.dtype)
        traces = parallel_affine_scan(gates, inputs)
        return torch.cat((hidden, traces), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        prior = None if state is None else state[0].unsqueeze(0)
        _, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        hidden = hidden[0]
        gates, inputs = self._trade_affine(observation, hidden.dtype)
        previous = torch.zeros_like(inputs) if state is None else state[1]
        traces = affine_scan_step(gates, inputs, previous)
        return torch.cat((hidden, traces), dim=-1), (hidden, traces)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        hidden, traces = context.split((48, 4), dim=-1)
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            hidden = torch.cat((hidden, self.token_embed(tokens).flatten(-2)), dim=-1)
        logits = self.heads[stage](hidden)
        if stage == self.ACTION_STAGE:
            logits = logits + self.trade_residual(traces)
        return logits
