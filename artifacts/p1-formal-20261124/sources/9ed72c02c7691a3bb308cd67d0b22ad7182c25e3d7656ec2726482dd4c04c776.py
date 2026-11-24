from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """A ten-level book encoder, GRU, and fast/slow gated per-stage heads."""

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

        n_stages = len(BUCKET_COUNTS)
        # Slow memory: per-stage real-diagonal affine scan smoothing.
        self.slow_logit = nn.Parameter(torch.full((n_stages, state_dim), 0.0))
        # Convex mixture gate, zero-initialized -> parent behavior recovered at init.
        self.mix = nn.Parameter(torch.zeros((n_stages, state_dim)))

        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _features(self, observation: StepObservation) -> Tensor:
        weight = self.input_projection.weight
        ticks = observation.book_ticks
        book_shares = observation.book_shares.to(weight.dtype)
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
        return self.input_projection(features.to(weight.dtype)).tanh()

    def _slow(self, ctx: Tensor, stage: int) -> Tensor:
        # Real diagonal smoothing: h_t = g * h_{t-1} + (1 - g) * ctx_t, g in (0,1).
        from stochopia.neural.primitives import real_diagonal
        g = real_diagonal(self.slow_logit[stage]).to(ctx.dtype)  # [C]
        one_minus = 1.0 - g
        # Closed-form prefix sum along time for a stable per-stage smoothed context.
        # h_t = g^t h_{-1} + sum_{s<=t} g^{t-s} (1-g) ctx_s, with h_{-1}=ctx_0.
        T = ctx.shape[1] if ctx.dim() == 3 else ctx.shape[0]
        if T == 0:
            return ctx
        x = ctx
        # iterative accumulation is exact and simple; time length is short.
        g_pow = torch.ones_like(x[..., 0, :]) if x.dim() == 3 else torch.ones_like(x[0])
        out = torch.empty_like(x)
        h = x[..., 0, :] if x.dim() == 3 else x[0]
        if x.dim() == 3:
            out[:, 0, :] = h
            for t in range(1, T):
                h = g * h + one_minus * x[:, t, :]
                out[:, t, :] = h
        else:
            out[0:] = h
            for t in range(1, T):
                h = g * h + one_minus * x[t]
                out[t] = h
        return out

    def _mix(self, ctx: Tensor, stage: int) -> Tensor:
        a = torch.sigmoid(self.mix[stage]).to(ctx.dtype)  # [C]
        slow = self._slow(ctx, stage)
        return a * ctx + (1.0 - a) * slow

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        return out[:, 0], hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        ctx = self._mix(context, stage)
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            ctx = torch.cat((ctx, prefix), dim=-1)
        return self.heads[stage](ctx)
