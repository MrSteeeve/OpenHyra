from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import SeedGRULOBModel  # noqa: F401  (kept importable)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder + GRU with slow-EMA directional memory in the heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8,
                 slow_decay: float = 0.9) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.slow_decay = slow_decay
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Shared low-rank slow-context projection: (1 - decay) * |ema| direction -> emb,
        # zero-initialized so the fast path is untouched at init.
        self.slow_proj = nn.Linear(state_dim, embedding_dim)
        nn.init.zeros_(self.slow_proj.weight)
        nn.init.zeros_(self.slow_proj.bias)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + (stage + 1) * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Slow EMA state (kept as tensor buffer to survive .to(device)); 48 scalars.
        self.register_buffer("_ema_init", torch.zeros(state_dim), persistent=False)

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

    def _slow_ema(self, out: Tensor) -> Tensor:
        # out: [B, T, C] -> causally accumulated EMA, decay d=slow_decay.
        B, T, C = out.shape
        d = self.slow_decay
        ema = self._ema_init.to(out.dtype).expand(B, C).contiguous()
        slow = out.new_empty(B, T, C)
        for t in range(T):
            ema = d * ema + (1.0 - d) * out[:, t]
            slow[:, t] = ema
        return slow

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        out, _ = self.gru(feats)
        # cache fast context + slow EMA for head use via attribute lookup by shape.
        self._cache = (out, self._slow_ema(out))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior_h = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior_h)
        # single-step EMA starting from zero-vector (or reused cache when available).
        d = self.slow_decay
        ema_prev = self._ema_init.to(out.dtype)
        ema = d * ema_prev + (1.0 - d) * out[0, 0]
        slow = ema.unsqueeze(0).unsqueeze(0)
        self._cache = (out, slow)
        return out[:, 0], hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        fast, slow = self._cache
        # Align slow EMA to the context's row shape.
        if context.dim() == 2:
            slow_ctx = slow.reshape(-1, slow.shape[-1])
            if slow_ctx.shape[0] != context.shape[0]:
                slow_ctx = self.slow_proj.weight.new_zeros(context.shape[0], self.state_dim)
            slow_src = slow_ctx
        else:
            slow_src = slow.reshape(-1, slow.shape[-1])
            if slow_src.shape[0] != context.reshape(-1, context.shape[-1]).shape[0]:
                slow_src = context.new_zeros(context.reshape(-1, context.shape[-1]).shape[0],
                                             self.state_dim)
        slow_emb = self.slow_proj(slow_src).view(*context.shape[:-1], self.embedding_dim)
        parts = [context, slow_emb]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            parts.append(prefix)
        return self.heads[stage](torch.cat(parts, dim=-1))
