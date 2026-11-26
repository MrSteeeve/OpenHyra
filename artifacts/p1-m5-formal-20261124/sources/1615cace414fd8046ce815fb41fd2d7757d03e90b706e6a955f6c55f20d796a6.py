from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Parent GRU book encoder with a two-timescale EMA side-channel into the heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Two-timescale leaky integrator gates (fast, slow) over projected features.
        self.ema_gate_logits = nn.Parameter(torch.tensor([-1.0, -4.0]))
        head_in = state_dim + 2 * state_dim
        self.heads = nn.ModuleList(
            nn.Linear(head_in + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Zero-init the EMA-to-head columns so the parent function is recovered at step 0.
        for head in self.heads:
            with torch.no_grad():
                head.weight[:, state_dim:head_in].zero_()

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

    def _ema_gates(self) -> Tensor:
        # gate in (0,1): sigmoid of a bounded logit so integrator stays stable.
        return torch.sigmoid(self.ema_gate_logits.clamp(-8.0, 8.0))

    def _augment(self, context: Tensor, ema: Tensor) -> Tensor:
        return torch.cat((context, ema.flatten(-2)), dim=-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feat = self._features(observation)
        context, _ = self.gru(feat)
        gates = self._ema_gates().to(context.dtype)  # [2]
        b, t, c = feat.shape
        ema = feat.new_zeros(b, 2, c)
        outs = []
        for i in range(t):
            g = gates.view(1, 2, 1)
            ema = g * ema + (1.0 - g) * feat[:, i:i + 1]
            outs.append(self._augment(context[:, i], ema))
        return torch.stack(outs, dim=1)

    def step_context(self, observation: StepObservation, state: State):
        feat = self._features(observation)
        if state is None:
            h_prev = None
            ema = feat.new_zeros(feat.shape[0], 2, feat.shape[-1])
        else:
            h_prev = state[:, : self.state_dim].unsqueeze(0)
            ema = state[:, self.state_dim:].view(feat.shape[0], 2, feat.shape[-1])
        out, hidden = self.gru(feat.unsqueeze(1), h_prev)
        gates = self._ema_gates().to(feat.dtype).view(1, 2, 1)
        ema = gates * ema + (1.0 - gates) * feat.unsqueeze(1)
        context = self._augment(out[:, 0], ema)
        new_state = torch.cat((hidden[0], ema.flatten(1)), dim=-1)
        return context, new_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
