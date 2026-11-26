from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import parallel_affine_scan


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder with GRU plus a scalar decayed impact accumulator."""

    def __init__(self, state_dim: int = 40, embedding_dim: int = 6) -> None:
        super().__init__()
        self.state_dim = state_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        # 43 base features + 4*embedding + 1 impact scalar
        self.input_projection = nn.Linear(43 + 4 * embedding_dim + 1, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        # Impact pathway parameters
        self.impact_decay_logit = nn.Parameter(torch.tensor(1.5))
        self.impact_gain = nn.Parameter(torch.tensor(0.5))
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _raw_features(self, observation: StepObservation):
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
        return features.to(self.input_projection.weight.dtype), imbalance, spread

    def _impact_inputs(self, observation: StepObservation):
        """Signed own-order pressure per step and decay gate."""
        _, imbalance, spread = self._raw_features(observation)
        own = observation.is_own.to(imbalance.dtype)
        pressure = own * imbalance.clamp(-1.0, 1.0) * (1.0 + spread)
        decay = torch.sigmoid(self.impact_decay_logit)
        return pressure, decay

    def _run_impact(self, pressure: Tensor, decay: Tensor, h0: Tensor | None) -> Tensor:
        b, t = pressure.shape
        gate = decay * torch.ones_like(pressure)
        # impact_t = decay * impact_{t-1} + pressure
        if h0 is not None:
            h0 = h0.reshape(b, 1)
        vals = parallel_affine_scan(gate.unsqueeze(-1), pressure.unsqueeze(-1),
                                    None if h0 is None else h0)
        return vals.squeeze(-1)

    def _compose(self, observation: StepObservation, impact_seq: Tensor) -> Tensor:
        base, _, _ = self._raw_features(observation)
        impact_feat = (self.impact_gain * torch.tanh(impact_seq)).unsqueeze(-1)
        features = torch.cat((base, impact_feat), dim=-1)
        return self.input_projection(features).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        pressure, decay = self._impact_inputs(observation)
        impact_seq = self._run_impact(pressure, decay, None)
        feats = self._compose(observation, impact_seq)
        out, _ = self.gru(feats)
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, tuple]:
        pressure, decay = self._impact_inputs(observation)  # [B, 1]
        if state is None:
            h0 = None
            impact0 = None
        else:
            h0 = state[0]
            impact0 = state[1]
        impact_now = self._run_impact(pressure, decay, impact0)  # [B,1]
        feats = self._compose(observation, impact_now)  # [B,1,D]
        prior = None if h0 is None else h0.unsqueeze(0)
        out, hidden = self.gru(feats, prior)
        return out[:, 0], (hidden[0], impact_now[:, -1])

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
