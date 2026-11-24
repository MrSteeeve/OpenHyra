from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU encoder plus a parameter-free exponential slow-memory channel."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.slow_decay = 0.95
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Low-rank fusion of fast GRU context with slow EMA memory.
        self.slow_projection = nn.Linear(state_dim, state_dim)
        self.fuse_projection = nn.Linear(state_dim, state_dim)
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
        fast, _ = self.gru(self._features(observation))
        # Slow EMA along time: s_t = d * s_{t-1} + (1-d) * fast_t
        d = self.slow_decay
        slow = torch.empty_like(fast)
        acc = torch.zeros_like(fast[:, 0])
        for t in range(fast.shape[1]):
            acc = d * acc + (1.0 - d) * fast[:, t]
            slow[:, t] = acc
        fused = fast + self.fuse_projection(self.slow_projection(slow)).tanh()
        return fused

    def step_context(self, observation: StepObservation, state: State):
        fast_in = self._features(observation)
        if state is None:
            hidden = None
            slow_prev = torch.zeros_like(fast_in)
        else:
            hidden = state.unsqueeze(0)
            slow_prev = torch.zeros_like(fast_in)
        out, hidden = self.gru(fast_in.unsqueeze(1), hidden)
        fast = out[:, 0]
        d = self.slow_decay
        slow = d * slow_prev + (1.0 - d) * fast
        fused = fast + self.fuse_projection(self.slow_projection(slow)).tanh()
        return fused, hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
