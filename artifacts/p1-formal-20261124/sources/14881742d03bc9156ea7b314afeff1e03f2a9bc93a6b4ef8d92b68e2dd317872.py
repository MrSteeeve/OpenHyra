from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, GRU, and fast/slow conditional order heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8, slow_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.slow_dim = slow_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Slow-state projection: state -> slow_dim, shared across heads.
        self.slow_projection = nn.Linear(state_dim, slow_dim)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + slow_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    @property
    def _slow_decay(self) -> float:
        return 0.9

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
        if state is None:
            h_prev = None
            slow_prev = None
        elif isinstance(state, tuple):
            h_prev, slow_prev = state
        else:
            h_prev = state
            slow_prev = None
        prior = None if h_prev is None else h_prev.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        h = hidden[0]
        if slow_prev is None:
            slow = self.slow_projection(h)
        else:
            slow = self._slow_decay * slow_prev + (1.0 - self._slow_decay) * self.slow_projection(h)
        return out[:, 0], (h, slow)

    def _slow_sequence(self, context: Tensor, state: State) -> Tensor:
        # context: [B,T,state]; produce per-step slow projection with EMA at decay.
        proj = self.slow_projection(context)  # [B,T,slow_dim]
        if state is None:
            init = proj[:, 0]
        elif isinstance(state, tuple):
            init = state[1]
        else:
            init = proj[:, 0]
        d = self._slow_decay
        # Build EMA via scan-free sequential fold using cumprod: slow_t = d*slow_{t-1}+(1-d)*proj_t
        # We roll manually per-step; T is modest.
        B, T, C = proj.shape
        out = proj.new_empty(B, T, C)
        cur = init
        for t in range(T):
            cur = d * cur + (1.0 - d) * proj[:, t]
            out[:, t] = cur
        return out

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        prefix_parts = []
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix_parts.append(self.token_embed(tokens).flatten(-2))
        ctx = context
        if self._last_slow is not None and self._last_slow.shape[:-1] == context.shape[:-1]:
            ctx = torch.cat((context, self._last_slow), dim=-1)
        if prefix_parts:
            ctx = torch.cat((ctx, *prefix_parts), dim=-1)
        return self.heads[stage](ctx)

    _last_slow: Tensor | None = None

    def sequence_context(self, observation: StepObservation) -> Tensor:  # override to set slow
        out, _ = self.gru(self._features(observation))
        self._last_slow = self._slow_sequence(out, None)
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        if state is None:
            h_prev = None
            slow_prev = None
        elif isinstance(state, tuple):
            h_prev, slow_prev = state
        else:
            h_prev = state
            slow_prev = None
        prior = None if h_prev is None else h_prev.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        h = hidden[0]
        proj = self.slow_projection(h)
        if slow_prev is None:
            slow = proj
        else:
            d = self._slow_decay
            slow = d * slow_prev + (1.0 - d) * proj
        self._last_slow = slow
        return out[:, 0], (h, slow)
