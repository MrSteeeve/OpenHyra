from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder + GRU + leaky slow-sign memory feeding discrete heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8,
                 sign_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.sign_dim = sign_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets),
                             persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)

        # Slow-sign memory: low-rank drive from previous-event embedding + imbalance.
        sign_in_dim = 4 * embedding_dim + 1
        self.sign_in = nn.Linear(sign_in_dim, sign_dim)
        nn.init.normal_(self.sign_in.weight, std=0.02)
        nn.init.zeros_(self.sign_in.bias)
        # Learned per-channel decay logit ~ logit(0.95).
        self.sign_decay_logit = nn.Parameter(torch.full((sign_dim,), 2.944))

        self.heads = nn.ModuleList(
            nn.Linear(state_dim + sign_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _features(self, observation: StepObservation) -> Tensor:
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
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, observation.is_own), dim=-1),
                              previous.flatten(-2)), dim=-1)
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    def _sign_inputs(self, observation: StepObservation) -> Tensor:
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        book_shares = observation.book_shares.to(self.sign_in.weight.dtype)
        bid_depth, ask_depth = book_shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        return torch.cat((previous.flatten(-2), imbalance.unsqueeze(-1)), dim=-1)

    def _affine_scan(self, gate: Tensor, inp: Tensor,
                     h0: Tensor | None) -> Tensor:
        # gate, inp: [..., T, C]; state: [..., C]
        t_steps = inp.shape[-2]
        gate = gate.clamp(0.0, 1.0 - 1e-4)
        h = torch.zeros(inp.shape[:-2] + (inp.shape[-1],),
                        dtype=inp.dtype, device=inp.device) if h0 is None else h0
        out = []
        for t in range(t_steps):
            g = gate[..., t, :]
            h = g * h + (1.0 - g) * inp[..., t, :]
            out.append(h)
        return torch.stack(out, dim=-2)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        gru_out, _ = self.gru(feats)
        sign_in = self._sign_inputs(observation)
        drive = self.sign_in(sign_in)
        decay = torch.sigmoid(self.sign_decay_logit)
        sign_state = self._affine_scan(decay, drive, h0=None)
        return torch.cat((gru_out, sign_state), dim=-1)

    def step_context(self, observation: StepObservation,
                     state: State) -> tuple[Tensor, State]:
        feat = self._features(observation)
        if state is None:
            gru_h = None
            sign_h = None
        else:
            size = self.state_dim
            gru_h = state[..., :size]
            sign_h = state[..., size:size + self.sign_dim]
        out, hidden = self.gru(feat.unsqueeze(-2),
                               None if gru_h is None else gru_h.unsqueeze(0))
        gru_out = out[..., 0, :]
        sign_in = self._sign_inputs(observation)
        drive = self.sign_in(sign_in)
        decay = torch.sigmoid(self.sign_decay_logit)
        if sign_h is None:
            sign_h = torch.zeros_like(drive)
        new_sign = decay * sign_h + (1.0 - decay) * drive
        context = torch.cat((gru_out, new_sign), dim=-1)
        new_state = torch.cat((hidden[0], new_sign), dim=-1)
        return context, new_state

    def head_logits(self, context: Tensor, stage: int,
                    prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
