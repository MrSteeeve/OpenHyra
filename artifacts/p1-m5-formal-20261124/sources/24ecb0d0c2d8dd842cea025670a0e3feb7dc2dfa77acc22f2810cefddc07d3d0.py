from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder, GRU, and impact-scaled discrete conditional order heads."""

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
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Impact gate: bounded scalar multiplier from the state vector.
        self.impact_gate = nn.Linear(state_dim, 1)
        nn.init.zeros_(self.impact_gate.weight)
        nn.init.zeros_(self.impact_gate.bias)
        # Per-stage fixed scale for the impact multiplier.
        self.register_buffer(
            "stage_scale", torch.linspace(0.0, 1.0, len(BUCKET_COUNTS)), persistent=False
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

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        return out[:, 0], hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        impact = torch.tanh(self.impact_gate(context)).squeeze(-1)
        scale = self.stage_scale[stage]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            prefix = prefix * (1.0 + scale * impact).unsqueeze(-1)
            context = torch.cat((context, prefix), dim=-1)
        logits = self.heads[stage](context)
        bias = self.heads[stage].bias
        if bias is not None:
            logits = logits + scale * impact.unsqueeze(-1) * bias
        return logits
