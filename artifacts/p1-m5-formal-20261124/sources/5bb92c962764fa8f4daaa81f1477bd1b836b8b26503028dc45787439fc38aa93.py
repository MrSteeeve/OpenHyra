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
        self.input_projection = nn.Linear(81, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        nn.init.zeros_(self.input_projection.weight[:, 75:])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28_000

    @torch.no_grad()
    def transplant_parent(self, parent: LOBWorldModel) -> None:
        weights = parent.state_dict()
        key = "input_projection.weight"
        old = weights[key]
        weights[key] = torch.cat((old, old.new_zeros(48, 6)), dim=-1)
        self.load_state_dict(weights)

    def _depth(self, observation: StepObservation) -> tuple[Tensor, Tensor, Tensor]:
        shares = observation.book_shares.to(self.input_projection.weight.dtype)
        depth = (shares / 100).log1p()
        return shares, depth, depth[..., :3].flatten(-2)

    def _features(
        self, observation: StepObservation, shares: Tensor, depth: Tensor,
        q: Tensor, q_prev: Tensor,
    ) -> Tensor:
        ticks = observation.book_ticks
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1),
            previous.flatten(-2), (q - q_prev).clamp(-3, 3),
        ), dim=-1)
        return self.input_projection(features.to(shares.dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        shares, depth, q = self._depth(observation)
        q_prev = torch.cat((q[:, :1], q[:, :-1]), dim=1)
        out, _ = self.gru(self._features(observation, shares, depth, q, q_prev))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        shares, depth, q = self._depth(observation)
        hidden = None if state is None else state[..., :48].unsqueeze(0).contiguous()
        q_prev = q if state is None else state[..., 48:]
        x = self._features(observation, shares, depth, q, q_prev)
        out, hidden = self.gru(x.unsqueeze(1), hidden)
        return out[:, 0], torch.cat((hidden[0], q), dim=-1)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
