import math

import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import affine_scan_step, parallel_affine_scan


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer(
            "execution_sign", torch.tensor([0., 0., 0., 0., 1., -1.]),
            persistent=False,
        )
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _encode(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        shares = observation.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        previous = observation.previous_event
        own = observation.is_own.reshape(previous.shape[:-1])
        embedded = self.token_embed(previous + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), (shares / 100).log1p().flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), embedded.flatten(-2),
        ), dim=-1)
        x = self.input_projection(features.to(shares.dtype)).tanh()
        impulse = self.execution_sign[previous[..., 1]].to(x.dtype)
        impulse = impulse.masked_fill(~own.bool(), 0).unsqueeze(-1)
        return x, impulse

    def sequence_context(self, observation: StepObservation) -> Tensor:
        x, impulse = self._encode(observation)
        h, _ = self.gru(x)
        gate = impulse.new_full((1,), math.exp(-1 / 16))
        z = parallel_affine_scan(gate, impulse)
        return torch.cat((h, z), dim=-1)

    def step_context(
        self, observation: StepObservation, state: State
    ) -> tuple[Tensor, State]:
        x, impulse = self._encode(observation)
        h_prev, z_prev = (None, torch.zeros_like(impulse)) if state is None else state
        _, hidden = self.gru(x.unsqueeze(1), None if h_prev is None else h_prev.unsqueeze(0))
        h = hidden[0]
        gate = impulse.new_full((1,), math.exp(-1 / 16))
        z = affine_scan_step(gate, impulse, z_prev)
        return torch.cat((h, z), dim=-1), (h, z)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        h, z = context[..., :-1], context[..., -1:]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            h = torch.cat((h, self.token_embed(tokens).flatten(-2)), dim=-1)
        logits = self.heads[stage](h)
        if stage == 1:
            logits = logits + (0.75 * z.tanh()) * self.execution_sign.to(logits.dtype)
        return logits
