from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, gated GRU, and discrete conditional order heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        # +2 extra features: (spread, imbalance) already used, plus 1 gate input.
        self.input_projection = nn.Linear(43 + 4 * embedding_dim + 1, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Scalar gate blending GRU update with prior hidden.
        self.decay_gate = nn.Linear(state_dim, state_dim)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        book_shares = observation.book_shares.to(dtype)
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
        gate_in = (spread - imbalance.abs()).unsqueeze(-1)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, observation.is_own), dim=-1),
                              previous.flatten(-2), gate_in), dim=-1)
        z = self.input_projection(features.to(dtype)).tanh()
        gate = torch.sigmoid(self.decay_gate(z))
        return z, gate

    def _blend(self, update: Tensor, prior: Tensor, gate: Tensor) -> Tensor:
        return gate * update + (1.0 - gate) * prior

    def sequence_context(self, observation: StepObservation) -> Tensor:
        z, gate = self._features(observation)
        raw, _ = self.gru(z)
        # Causal damped recursion over raw GRU outputs.
        outputs = []
        h = torch.zeros_like(raw[:, 0])
        for t in range(raw.shape[1]):
            h = gate[:, t] * raw[:, t] + (1.0 - gate[:, t]) * h
            outputs.append(h)
        return torch.stack(outputs, dim=1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        z, gate = self._features(observation)
        prior_h = torch.zeros_like(z) if state is None else state
        raw, hidden = self.gru(z.unsqueeze(1), prior_h.unsqueeze(0))
        out = self._blend(raw[:, 0], prior_h, gate)
        return out, hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
