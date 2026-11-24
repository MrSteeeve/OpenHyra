from __future__ import annotations

import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        emb = 8
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), emb)

        for width in range(48, 0, -1):
            self.input_projection = nn.Linear(43 + 4 * emb, width)
            self.gru = nn.GRU(width, width, batch_first=True)
            self.heads = nn.ModuleList(
                nn.Linear(width + stage * emb, count)
                for stage, count in enumerate(BUCKET_COUNTS)
            )
            params = sum(p.numel() for p in self.parameters())
            r = max(0, min(4, (28000 - params) // width))
            if r:
                break
        else:
            raise ValueError("No recurrent width fits the parameter budget")

        self.state_dim = width
        self.book_residual = nn.Linear(r, width, bias=False)
        with torch.no_grad():
            self.input_projection.weight[:, IS_OWN_FEATURE_INDEX].zero_()
            self.book_residual.weight.zero_()

    def _encode(self, obs: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = obs.book_ticks
        shares = obs.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid, ask = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid - ask) / (bid + ask).clamp_min(1)
        previous = self.token_embed(obs.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, obs.is_own.reshape_as(spread)), dim=-1),
            previous.flatten(-2),
        ), dim=-1)

        valid = shares[..., :3].masked_fill(ticks[..., :3] <= 0, 0)
        bid1, ask1 = valid[..., 0].unbind(dim=-1)
        bid3, ask3 = valid.sum(dim=-1).unbind(dim=-1)
        total1, total3 = bid1 + ask1, bid3 + ask3
        local = torch.stack((
            (bid1 - ask1) / total1.clamp_min(1),
            (bid3 - ask3) / total3.clamp_min(1),
            (total1 / 100).log1p(),
            (total3 / 100).log1p(),
        ), dim=-1)
        return (
            self.input_projection(features.to(shares.dtype)).tanh(),
            self.book_residual(local[..., :self.book_residual.in_features]),
        )

    def _context(
        self, obs: StepObservation, state: State, step: bool
    ) -> tuple[Tensor, Tensor]:
        encoded, residual = self._encode(obs)
        out, hidden = self.gru(
            encoded.unsqueeze(1) if step else encoded,
            None if state is None else state.unsqueeze(0),
        )
        return (out[:, 0] if step else out) + residual, hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._context(observation, None, False)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        return self._context(observation, state, True)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
