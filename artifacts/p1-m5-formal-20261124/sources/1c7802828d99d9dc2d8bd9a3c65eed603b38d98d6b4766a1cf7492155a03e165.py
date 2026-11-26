from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Dual-timescale GRU (fast transient + slow persistent) book encoder."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.slow_dim = embedding_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        # Fast transient channel: forget bias positive -> quick decay.
        self.fast_gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Slow persistent channel: forget bias negative -> long memory.
        self.slow_gru = nn.GRU(state_dim, self.slow_dim, batch_first=True)
        for name, param in self.fast_gru.named_parameters():
            if name.startswith("bias_hh"):
                n = param.shape[0] // 3
                param.data[n:2 * n] += 1.0  # forget gate bias
        for name, param in self.slow_gru.named_parameters():
            if name.startswith("bias_hh"):
                n = param.shape[0] // 3
                param.data[n:2 * n] -= 1.0
        total_dim = state_dim + self.slow_dim
        self.heads = nn.ModuleList(
            nn.Linear(total_dim + stage * embedding_dim, count)
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
        return self.input_projection(
            features.to(self.input_projection.weight.dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        x = self._features(observation)
        fast, _ = self.fast_gru(x)
        slow, _ = self.slow_gru(x)
        return torch.cat((fast, slow), dim=-1)

    def step_context(self, observation: StepObservation, state: State
                     ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        x = self._features(observation).unsqueeze(1)
        if state is None:
            fast_h = None
            slow_h = None
        else:
            fast_h, slow_h = state
        fast, fast_h = self.fast_gru(x, None if fast_h is None else fast_h.unsqueeze(0))
        slow, slow_h = self.slow_gru(x, None if slow_h is None else slow_h.unsqueeze(0))
        return torch.cat((fast[:, 0], slow[:, 0]), dim=-1), (fast_h[0], slow_h[0])

    def head_logits(self, context: Tensor, stage: int,
                    prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
