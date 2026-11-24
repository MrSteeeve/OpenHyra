from __future__ import annotations

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
        self.register_buffer("decay", torch.tensor([.70, .97]), persistent=False)
        self.register_buffer("injection", torch.tensor([.30, .03]), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + stage * 8, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _inputs(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(-1) / present.sum(-1).clamp_min(1)
        relative = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        prices = relative.masked_fill(ticks <= 0, 0)
        bid, ask = touch.unbind(-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(-1).unbind(-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        tokens = observation.previous_event.long()
        previous = self.token_embed(tokens + self.token_offsets)
        features = torch.cat((
            prices.flatten(-2), (shares / 100).log1p().flatten(-2),
            torch.stack((spread, imbalance, own), -1), previous.flatten(-2),
        ), -1)
        signed = self.execution_sign[tokens[..., 1]].masked_fill(~own.bool(), 0)
        drive = signed.to(dtype).unsqueeze(-1) * self.injection.to(dtype)
        return self.input_projection(features.to(dtype)).tanh(), drive

    def _context(self, hidden: Tensor, traces: Tensor) -> Tensor:
        fast, slow = traces.unbind(-1)
        recovery = (4 * slow).tanh() * (1 - 3 * fast.abs()).clamp(0, 1)
        return torch.cat((hidden, recovery.unsqueeze(-1)), -1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, drive = self._inputs(observation)
        hidden, _ = self.gru(features)
        traces = parallel_affine_scan(self.decay.to(drive), drive)
        return self._context(hidden, traces)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        features, drive = self._inputs(observation)
        if state is None:
            prior_hidden, prior_traces = None, torch.zeros_like(drive)
        else:
            prior_hidden, prior_traces = state
            prior_hidden = prior_hidden.unsqueeze(0)
        hidden, _ = self.gru(features.unsqueeze(1), prior_hidden)
        traces = affine_scan_step(self.decay.to(drive).expand_as(drive), drive, prior_traces)
        hidden = hidden[:, 0]
        return self._context(hidden, traces), (hidden, traces)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        hidden, recovery = context[..., :-1], context[..., -1:]
        if prior:
            tokens = torch.stack(prior, -1).long() + self.token_offsets[:stage]
            hidden = torch.cat((hidden, self.token_embed(tokens).flatten(-2)), -1)
        logits = self.heads[stage](hidden)
        if stage == 1:
            logits = logits - .60 * recovery * self.execution_sign.to(logits)
        return logits
