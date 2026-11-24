from __future__ import annotations

import torch
from torch import Tensor, nn

from stochopia.neural.lob_interface import (
    BUCKET_COUNTS,
    IS_OWN_FEATURE_INDEX,
    LOBWorldModel,
    State,
    StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder + GRU with a fast/slow impact decomposition on stage 0."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim

        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer(
            "token_offsets", torch.tensor(token_offsets), persistent=False
        )

        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        # Historical orders have no own labels; start with no untrained flag response.
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])

        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)

        # Fast/slow decomposition: per-channel leak logit and a depthwise gate
        # used only by the first (impact-relevant) head.
        self.leak_logit = nn.Parameter(torch.zeros(state_dim))
        self.fast_gate = nn.Parameter(torch.zeros(state_dim))

        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Widen only the first-stage head input to accept [slow ; fast].
        self.impact_proj = nn.Linear(2 * state_dim, state_dim + 0 * embedding_dim)

    # ------------------------------------------------------------------ features
    def _features(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        book_shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        price_feature = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth_feature = (book_shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (
            (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        )
        bid_depth, ask_depth = book_shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat(
            (
                price_feature.flatten(-2),
                depth_feature.flatten(-2),
                torch.stack((spread, imbalance, observation.is_own), dim=-1),
                previous.flatten(-2),
            ),
            dim=-1,
        )
        return self.input_projection(features.to(dtype)).tanh()

    # ------------------------------------------------------------------ contexts
    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        return out

    def step_context(
        self, observation: StepObservation, state: State
    ) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        return out[:, 0], hidden[0]

    # ------------------------------------------------------------------ impact head
    def _impact_context(self, context: Tensor, slow: Tensor) -> Tensor:
        # slow is the GRU output at t-1 (zeros at the first step); fast = out - leak*slow
        leak = torch.sigmoid(self.leak_logit)
        fast = (context - leak * slow) * torch.sigmoid(self.fast_gate)
        return self.impact_proj(torch.cat((context, fast), dim=-1))

    def head_logits(
        self, context: Tensor, stage: int, prior: tuple[Tensor, ...]
    ) -> Tensor:
        if stage == 0:
            # `_slow_hint` is populated by forward_sequence / forward_step via the
            # attribute set just before calling head_logits; default to zeros.
            slow = getattr(self, "_slow_hint", None)
            if slow is None or slow.shape != context.shape:
                slow = torch.zeros_like(context)
            ctx0 = self._impact_context(context, slow)
            return self.heads[0](ctx0)
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)

    # ------------------------------------------------------------------ overrides
    def sequence_context(self, observation: StepObservation) -> Tensor:  # type: ignore[override]
        out, _ = self.gru(self._features(observation))
        return out

    def forward_sequence(self, batch) -> "object":  # type: ignore[override]
        # Provide slow hint (shifted GRU context) for the impact decomposition.
        ctx = self.sequence_context(batch.observation)
        slow = torch.cat(
            (torch.zeros_like(ctx[:, :1]), ctx[:, :-1]), dim=1
        )
        self._slow_hint = slow
        try:
            return super().forward_sequence(batch)
        finally:
            self._slow_hint = None

    def forward_step(
        self, step_obs: StepObservation, hidden_state: State = None
    ) -> tuple["object", State]:  # type: ignore[override]
        if isinstance(hidden_state, tuple):
            prev = hidden_state[0]
        else:
            prev = hidden_state
        ctx, new_state = self.step_context(step_obs, hidden_state)
        # Slow hint = previous hidden (detached-safe: it is a real state scalar).
        if prev is None:
            slow = torch.zeros_like(ctx)
        else:
            slow = prev.detach() if prev.shape == ctx.shape else torch.zeros_like(ctx)
        self._slow_hint = slow
        try:
            return super().forward_step(step_obs, hidden_state)
        finally:
            self._slow_hint = None
