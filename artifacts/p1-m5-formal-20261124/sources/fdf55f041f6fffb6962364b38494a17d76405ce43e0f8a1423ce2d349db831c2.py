from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU with fast/slow context split for stage-conditional order heads."""

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
        # Gate producing a scalar mixing weight per unit between fast and slow context.
        self.slow_gate = nn.Linear(state_dim, state_dim)
        nn.init.zeros_(self.slow_gate.weight)
        nn.init.constant_(self.slow_gate.bias, 2.0)  # sigmoid ~0.88 toward slow init
        head_in = 2 * state_dim + (len(BUCKET_COUNTS) - 1) * embedding_dim
        self.heads = nn.ModuleList(
            nn.Linear(2 * state_dim + stage * embedding_dim, count)
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

    def _slow_fast(self, fast: Tensor, slow: Tensor | None) -> tuple[Tensor, Tensor]:
        # fast: [..., C]; slow: [..., C] or None. One EMA refresh plus learned mix.
        if slow is None:
            slow = torch.zeros_like(fast)
        slow_new = 0.5 * slow + 0.5 * fast
        alpha = torch.sigmoid(self.slow_gate(fast))
        mixed = alpha * slow_new + (1.0 - alpha) * fast
        return mixed, slow_new

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        gru_out, _ = self.gru(feats)
        # Parallel EMA over time via cumulative sum trick.
        B, T, C = gru_out.shape
        mix_out, _ = self._slow_fast(gru_out, None)
        # Recompute with sequential EMA to share the canonical step path.
        slow = gru_out.new_zeros(B, C)
        outs = []
        for t in range(T):
            mixed, slow = self._slow_fast(gru_out[:, t], slow)
            outs.append(mixed)
        return torch.stack(outs, dim=1)

    def step_context(self, observation: StepObservation, state: State):
        # state = (h_gru [C], slow [C])
        feats = self._features(observation)
        if state is None:
            h_prev = None
            slow = None
        else:
            h_prev, slow = state
            h_prev = h_prev.unsqueeze(0)
        out, hidden = self.gru(feats.unsqueeze(1), h_prev)
        fast = out[:, 0]
        mixed, slow_new = self._slow_fast(fast, slow)
        return mixed, (hidden[0], slow_new)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
