from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """A ten-level book encoder and GRU with a learned per-step decay gate."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        # features: 40 (prices) + 20 (depth) + 3 (spread, imbalance, own) + 4*emb (prev)
        feat_dim = 43 + 4 * embedding_dim
        self.input_projection = nn.Linear(feat_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        # Scalar decay gate derived from (spread, imbalance) - 3 extra params.
        self.gate_proj = nn.Linear(2, 1)
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _raw_features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
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
        stats = torch.stack((spread, imbalance, observation.is_own), dim=-1)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              stats, previous.flatten(-2)), dim=-1)
        return features, torch.stack((spread, imbalance), dim=-1)

    def _gate(self, stats: Tensor) -> Tensor:
        return torch.sigmoid(self.gate_proj(stats)).squeeze(-1)

    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        features, stats = self._raw_features(observation)
        projected = self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()
        return projected, self._gate(stats)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        projected, gate = self._features(observation)
        gru_out, _ = self.gru(projected)
        # Gated decay: blend GRU output with its own input via gate.
        return gate.unsqueeze(-1) * gru_out + (1 - gate.unsqueeze(-1)) * projected

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        projected, gate = self._features(observation)
        prior = None if state is None else state.unsqueeze(0)
        gru_out, hidden = self.gru(projected.unsqueeze(1), prior)
        out = gate.unsqueeze(-1) * gru_out[:, 0] + (1 - gate.unsqueeze(-1)) * projected
        return out, hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
