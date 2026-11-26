from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, GRU core, and a slow depth-sensitive branch
    that is blended into the context before the discrete conditional heads."""

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

        # Slow branch: per-state learnable multiplicative decay (~0.9) applied to
        # the fast GRU context, plus a zero-initialized linear correction driven
        # by a depth/imbalance sub-vector (same 4 scalars used as features).
        self.slow_decay_logit = nn.Parameter(torch.full((state_dim,), 2.2))
        self.slow_corr = nn.Linear(4, state_dim)
        nn.init.zeros_(self.slow_corr.weight)
        nn.init.zeros_(self.slow_corr.bias)

        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _book_features(self, observation: StepObservation) -> Tensor:
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
        return price_feature, depth_feature, spread, imbalance, previous

    def _fast_features(self, observation: StepObservation) -> Tensor:
        price_feature, depth_feature, spread, imbalance, previous = self._book_features(observation)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, observation.is_own.float()), dim=-1),
                              previous.flatten(-2)), dim=-1)
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    def _slow_features(self, observation: StepObservation) -> Tensor:
        _, _, spread, imbalance, _ = self._book_features(observation)
        depth = observation.book_shares.to(self.slow_corr.weight.dtype).mean(dim=-1).sum(dim=-1)
        extra = torch.stack((spread, imbalance, depth.log1p(), spread * imbalance), dim=-1)
        return extra

    def _blend(self, context: Tensor, slow_obs: Tensor) -> Tensor:
        decay = torch.sigmoid(self.slow_decay_logit)
        corr = self.slow_corr(slow_obs)
        return context * decay + corr

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._fast_features(observation))
        return self._blend(out, self._slow_features(observation))

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._fast_features(observation).unsqueeze(1), prior)
        ctx = self._blend(out[:, 0], self._slow_features(observation))
        return ctx, hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
