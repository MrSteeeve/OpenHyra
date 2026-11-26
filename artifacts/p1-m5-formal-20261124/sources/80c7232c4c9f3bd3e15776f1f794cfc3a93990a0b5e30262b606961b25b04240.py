from __future__ import annotations

import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import parallel_affine_scan, affine_scan_step


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        self.state_dim = 48
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer("trade_rhos", torch.tensor([.5, .85, .97, .995]),
                             persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.trade_residual = nn.Linear(4, BUCKET_COUNTS[1], bias=False)
        nn.init.zeros_(self.trade_residual.weight)

    def _features(self, observation: StepObservation) -> Tensor:
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
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)
        return self.input_projection(features.to(dtype)).tanh()

    def _trade_affine(self, observation: StepObservation, dtype: torch.dtype):
        action = observation.previous_event[..., 1]
        # The final two action IDs are market buy and market sell, respectively.
        buy, sell = action == 4, action == 5
        rho = self.trade_rhos.to(dtype=dtype)
        gate = torch.where((buy | sell).unsqueeze(-1), rho, torch.ones_like(rho))
        sign = (buy.to(dtype) - sell.to(dtype)).unsqueeze(-1)
        return gate, (1 - rho) * sign

    def sequence_context(self, observation: StepObservation) -> Tensor:
        context, _ = self.gru(self._features(observation))
        gates, inputs = self._trade_affine(observation, context.dtype)
        traces = parallel_affine_scan(gates, inputs)
        return torch.cat((context, traces), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state[..., :48].unsqueeze(0).contiguous()
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        gates, inputs = self._trade_affine(observation, out.dtype)
        previous = torch.zeros_like(inputs) if state is None else state[..., 48:]
        traces = affine_scan_step(gates, inputs, previous)
        context = torch.cat((hidden[0], traces), dim=-1)
        return context, context

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        base, traces = context[..., :48], context[..., 48:]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            base = torch.cat((base, self.token_embed(tokens).flatten(-2)), dim=-1)
        logits = self.heads[stage](base)
        if stage == 1:
            logits = logits + self.trade_residual(traces)
        return logits
