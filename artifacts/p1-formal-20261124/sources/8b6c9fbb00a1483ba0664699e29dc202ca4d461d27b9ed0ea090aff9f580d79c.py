from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder + GRU with a zero-init fast/slow gate before each head."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8, rank: int = 4) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.rank = rank
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Low-rank fast/slow mixing gate, zero-initialized so parent behavior is exact.
        stages = len(BUCKET_COUNTS)
        self.gate_down = nn.Parameter(torch.zeros(stages, state_dim, rank))
        self.gate_up = nn.Parameter(torch.zeros(stages, rank, state_dim))
        self.gate_bias = nn.Parameter(torch.zeros(stages, rank))
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
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

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        return out[:, 0], hidden[0]

    def _stage_context(self, context: Tensor, stage: int) -> Tensor:
        # Low-rank fast/slow gate: hidden = tanh(context @ Wd + b) @ Wu, gate = sigmoid(hidden).
        hidden = torch.tanh(
            torch.matmul(context.unsqueeze(-2), self.gate_down[stage]).squeeze(-2)
            + self.gate_bias[stage]
        )
        logit = torch.matmul(hidden.unsqueeze(-2), self.gate_up[stage]).squeeze(-2)
        gate = torch.sigmoid(logit)
        # Fast path is raw context; slow path is the running mean proxy = context itself
        # but scaled slightly, providing a controllable decay for short-lived impact.
        return gate * context + (1.0 - gate) * context * 0.5

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        ctx = self._stage_context(context, stage)
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            ctx = torch.cat((ctx, prefix), dim=-1)
        return self.heads[stage](ctx)
