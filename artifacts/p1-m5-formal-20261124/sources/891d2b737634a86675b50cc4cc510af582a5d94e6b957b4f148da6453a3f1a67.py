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
        self.depth_residual = nn.Linear(4, 48, bias=False)
        nn.init.zeros_(self.depth_residual.weight)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000

    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor, Tensor]:
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
            torch.stack((spread, imbalance, own), dim=-1),
            previous.flatten(-2),
        ), dim=-1)
        return features.to(dtype), ticks[..., :2].flatten(-2), depth[..., :2].flatten(-2)

    @staticmethod
    def _depth_change(
        ticks: Tensor, depth: Tensor, old_ticks: Tensor, old_depth: Tensor
    ) -> Tensor:
        same_queue = (ticks > 0) & (ticks == old_ticks)
        return (depth - old_depth).masked_fill(~same_queue, 0)

    def _project(self, features: Tensor, change: Tensor) -> Tensor:
        return (self.input_projection(features) + self.depth_residual(change)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, ticks, depth = self._features(observation)
        change = torch.cat((
            torch.zeros_like(depth[:, :1]),
            self._depth_change(ticks[:, 1:], depth[:, 1:], ticks[:, :-1], depth[:, :-1]),
        ), dim=1)
        out, _ = self.gru(self._project(features, change))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        features, ticks, depth = self._features(observation)
        hidden = None
        if state is None:
            change = torch.zeros_like(depth)
        else:
            hidden = state[..., :48].to(features.dtype).unsqueeze(0)
            change = self._depth_change(
                ticks, depth, state[..., 48:52].to(ticks.dtype),
                state[..., 52:56].to(depth.dtype),
            )
        out, hidden = self.gru(self._project(features, change).unsqueeze(1), hidden)
        # Use a common storage dtype that preserves integer tick prices.
        storage_dtype = torch.float64 if not ticks.is_floating_point() else (
            torch.promote_types(ticks.dtype, depth.dtype)
        )
        next_state = torch.cat((
            hidden[0].to(storage_dtype), ticks.to(storage_dtype), depth.to(storage_dtype),
        ), dim=-1)
        return out[:, 0], next_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
