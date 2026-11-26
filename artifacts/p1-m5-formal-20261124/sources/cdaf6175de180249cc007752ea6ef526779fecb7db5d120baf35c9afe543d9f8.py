from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, GRU backbone, and depth-gated residual heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Depth-aware residual gate over the GRU context.
        self.gate = nn.Linear(3, state_dim)
        self.residual = nn.Linear(state_dim, state_dim)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Residual path starts near-identity-free so GRU path dominates initially.
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def _features_and_depth(self, observation: StepObservation):
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
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth.clamp_min(1)
        depth_scale = total_depth.clamp_min(1).log1p() / 6.0
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, observation.is_own), dim=-1),
                              previous.flatten(-2)), dim=-1)
        depth_gate = torch.stack((spread, imbalance, depth_scale), dim=-1)
        return features.to(self.input_projection.weight.dtype), depth_gate

    def _apply_gate(self, context: Tensor, depth_gate: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate(depth_gate.to(self.gate.weight.dtype)))
        return context + g * self.residual(context)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, depth_gate = self._features_and_depth(observation)
        out, _ = self.gru(self.input_projection(features))
        return self._apply_gate(out, depth_gate)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        features, depth_gate = self._features_and_depth(observation)
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self.input_projection(features).unsqueeze(1), prior)
        ctx = self._apply_gate(out[:, 0], depth_gate)
        return ctx, hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
