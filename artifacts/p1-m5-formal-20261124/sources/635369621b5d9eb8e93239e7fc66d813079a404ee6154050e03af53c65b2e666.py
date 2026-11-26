from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder plus GRU with a learned per-step damping gate."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        base_in = 43 + 4 * embedding_dim
        self.input_projection = nn.Linear(base_in, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        # Scalar gate logit from (spread, imbalance): small extra readout of raw features.
        self.gate_projection = nn.Linear(2, 1)
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _raw(self, observation: StepObservation):
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
        gate_raw = torch.stack((spread, imbalance), dim=-1)
        return features, gate_raw

    def _features(self, observation: StepObservation) -> Tensor:
        features, _ = self._raw(observation)
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    def _gate(self, gate_raw: Tensor) -> Tensor:
        return torch.sigmoid(self.gate_projection(gate_raw.to(self.gate_projection.weight.dtype))).squeeze(-1)

    def _recurrence(self, features: Tensor, gates: Tensor, h0: Tensor | None) -> Tensor:
        out, _ = self.gru(features, None if h0 is None else h0.unsqueeze(0))
        if h0 is None:
            prev = torch.zeros_like(out)
            prev[:, 0] = 0.0
            # For t=0, blend with zero prior hidden.
            prev = out.new_zeros(out.shape)
            blended = gates.unsqueeze(-1) * out + (1 - gates.unsqueeze(-1)) * prev
            return blended
        # Shifted prior: hidden state at t-1 concatenated with initial h0 for t=0.
        shifted = torch.cat((h0.unsqueeze(1).expand(-1, 1, -1), out[:, :-1]), dim=1)
        return gates.unsqueeze(-1) * out + (1 - gates.unsqueeze(-1)) * shifted

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, gate_raw = self._raw(observation)
        feats = self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()
        gates = self._gate(gate_raw)
        return self._recurrence(feats, gates, None)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        features, gate_raw = self._raw(observation)
        feats = self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()
        gate = self._gate(gate_raw)
        h0 = state
        out, _ = self.gru(feats.unsqueeze(1), None if h0 is None else h0.unsqueeze(0))
        new_h = out[:, 0]
        prev = torch.zeros_like(new_h) if h0 is None else h0
        blended = gate * new_h + (1 - gate) * prev
        return blended, blended

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
