from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder, GRU, and a leaky slow-sign memory feeding discrete heads."""

    def __init__(
        self,
        state_dim: int = 48,
        embedding_dim: int = 8,
        sign_dim: int = 8,
        sign_decay_logit: float = 2.9,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.sign_dim = sign_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)

        # Slow-sign leaky integrator: small low-rank projection of previous
        # event embedding plus imbalance, integrated with learned per-channel decay.
        self.sign_in = nn.Linear(2 * embedding_dim + 2, sign_dim)
        with torch.no_grad():
            self.sign_in.weight.mul_(0.1)
            self.sign_in.bias.zero_()
        self.sign_decay_logit = nn.Parameter(torch.full((sign_dim,), float(sign_decay_logit)))

        self.heads = nn.ModuleList(
            nn.Linear(state_dim + sign_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _features(self, observation: StepObservation, need_sign_input: bool = False):
        ticks = observation.book_ticks
        book_shares = observation.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        price_feature = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth_feature = (book_shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = book_shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat(
            (
                price_feature.flatten(-2),
                depth_feature.flatten(-2),
                torch.stack((spread, imbalance, observation.is_own), dim=-1),
                previous.flatten(-2),
            ),
            dim=-1,
        )
        out = self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()
        if not need_sign_input:
            return out
        sign_input = torch.cat(
            (previous.flatten(-2), torch.stack((spread, imbalance), dim=-1)), dim=-1
        )
        return out, sign_input

    def _sign_step(self, sign_input: Tensor, sign_state: Tensor | None) -> Tensor:
        drive = self.sign_in(sign_input)
        if sign_state is None:
            return torch.tanh(drive)
        gate = torch.sigmoid(self.sign_decay_logit)
        new_state = gate * sign_state + (1 - gate) * drive
        return torch.tanh(new_state)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        base, sign_input = self._features(observation, need_sign_input=True)
        gru_out, _ = self.gru(base)
        sign_drive = self.sign_in(sign_input)
        gate = torch.sigmoid(self.sign_decay_logit)
        sign_out = torch.empty_like(sign_drive)
        h = torch.zeros_like(sign_drive[:, 0])
        for t in range(sign_drive.shape[1]):
            h = gate * h + (1 - gate) * sign_drive[:, t]
            sign_out[:, t] = h
        sign_out = torch.tanh(sign_out)
        return torch.cat((gru_out, sign_out), dim=-1)

    def step_context(self, observation: StepObservation, state: State):
        base, sign_input = self._features(observation, need_sign_input=True)
        if state is None:
            gru_state = None
            sign_state = None
        else:
            gru_state = state[: self.state_dim].unsqueeze(0)
            sign_state = state[self.state_dim :]
        gru_out, hidden = self.gru(base.unsqueeze(1), gru_state)
        new_sign = self._sign_step(sign_input, sign_state)
        context = torch.cat((gru_out[:, 0], new_sign), dim=-1)
        new_state = torch.cat((hidden[0], new_sign), dim=-1)
        return context, new_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
