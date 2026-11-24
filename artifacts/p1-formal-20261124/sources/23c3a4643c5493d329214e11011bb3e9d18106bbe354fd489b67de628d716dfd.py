from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU backbone with a leaky slow-sign memory branch feeding the heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8,
                 sign_dim: int = 8) -> None:
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
        # Slow-sign leaky integrator: low-rank projection of previous-event embedding + imbalance.
        self.sign_in = nn.Linear(4 * embedding_dim + 1, sign_dim, bias=False)
        self.sign_decay_logit = nn.Parameter(torch.zeros(1))
        self.sign_out = nn.Linear(sign_dim, sign_dim, bias=False)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim + sign_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _features(self, observation: StepObservation):
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
        prev_flat = previous.flatten(-2)
        sign_input = torch.cat((prev_flat, imbalance.unsqueeze(-1)), dim=-1)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, observation.is_own), dim=-1),
                              prev_flat), dim=-1)
        feat = self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()
        return feat, sign_input

    def _sign_decay(self) -> Tensor:
        return torch.sigmoid(self.sign_decay_logit).clamp(0.5, 0.995)

    def _sign_scan(self, sign_inputs: Tensor) -> Tensor:
        # sign_inputs: [B,T,sign_dim_proj] -> gated leaky accumulation [B,T,sign_dim]
        gates = torch.full_like(sign_inputs[..., :1], float(self._sign_decay()))
        gates = gates.expand_as(sign_inputs[..., :1]).squeeze(-1)
        # manual scan (state small: sign_dim)
        b, t, _ = sign_inputs.shape
        decay = self._sign_decay()
        out = sign_inputs.new_zeros(b, t, self.sign_dim)
        h = sign_inputs.new_zeros(b, self.sign_dim)
        for i in range(t):
            h = decay * h + self.sign_in(sign_inputs[:, i])
            out[:, i] = h
        return torch.tanh(self.sign_out(out))

    def _sign_step(self, sign_input: Tensor, sign_state: Tensor) -> tuple[Tensor, Tensor]:
        decay = self._sign_decay()
        h = decay * sign_state + self.sign_in(sign_input)
        return torch.tanh(self.sign_out(h)), h

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feat, sign_input = self._features(observation)
        out, _ = self.gru(feat)
        sign = self._sign_scan(sign_input)
        return torch.cat((out, sign), dim=-1)

    def step_context(self, observation: StepObservation, state: State):
        feat, sign_input = self._features(observation)
        feat = feat.unsqueeze(1)
        sign_input = sign_input.unsqueeze(1)
        if state is None:
            gru_state = None
            sign_state = self._features(observation)[0].new_zeros(
                feat.shape[0], self.sign_dim)
        else:
            if isinstance(state, tuple):
                gru_state, sign_state = state
            else:
                gru_state, sign_state = state, None
                sign_state = feat.new_zeros(feat.shape[0], self.sign_dim)
            if gru_state is not None:
                gru_state = gru_state.unsqueeze(0)
        out, hidden = self.gru(feat, gru_state)
        sign, sign_state = self._sign_step(sign_input[:, 0], sign_state)
        ctx = torch.cat((out[:, 0], sign), dim=-1)
        return ctx, (hidden[0], sign_state)

    def head_logits(self, context: Tensor, stage: int, prior: tuple) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
