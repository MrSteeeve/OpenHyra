from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Two-scale read of a single GRU: slow carry plus fast zero-mean channel."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8, fast_decay: float = 0.8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.fast_decay = fast_decay
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Two-scale residual: zero-init so heads start from the pure slow channel.
        self.two_scale = nn.Linear(2 * state_dim, state_dim)
        nn.init.zeros_(self.two_scale.weight)
        nn.init.zeros_(self.two_scale.bias)
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

    def _combine(self, slow: Tensor, fast: Tensor) -> Tensor:
        residual = self.two_scale(torch.cat((slow, fast), dim=-1))
        return slow + residual

    def _ema_scan(self, slow: Tensor) -> Tensor:
        # slow: [..., T, C]. Fast channel = slow - EMA, EMA init to first step.
        T = slow.shape[-2]
        if T == 0:
            return slow
        lam = self.fast_decay
        first = slow[..., 0, :]
        ema = first
        zero = torch.zeros((), dtype=slow.dtype, device=slow.device)
        outs = [zero]
        for t in range(1, T):
            ema = lam * ema + (1.0 - lam) * slow[..., t, :]
            outs.append(ema - first)
        ema_rest = torch.stack(outs, dim=-2)
        return slow - ema_rest

    def sequence_context(self, observation: StepObservation) -> Tensor:
        slow, _ = self.gru(self._features(observation))
        fast = self._ema_scan(slow)
        return self._combine(slow, fast)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        feature = self._features(observation)
        if isinstance(state, tuple):
            h_prev, ema_prev = state
        elif state is None:
            h_prev, ema_prev = None, None
        else:
            h_prev, ema_prev = state, None
        prior = None if h_prev is None else h_prev.unsqueeze(0)
        out, hidden = self.gru(feature.unsqueeze(1), prior)
        slow = out[:, 0]
        if ema_prev is None:
            ema_new = slow
            fast = torch.zeros_like(slow)
        else:
            ema_new = self.fast_decay * ema_prev + (1.0 - self.fast_decay) * slow
            fast = slow - ema_prev
        return self._combine(slow, fast), (hidden[0], ema_new)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
