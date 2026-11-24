from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU book encoder with a fixed-decay slow EMA feeding the conditional heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8, slow_decay: float = 0.9) -> None:
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
        # Shared low-rank projection of the slow EMA into a head-side embedding.
        self.slow_to_embed = nn.Linear(state_dim, embedding_dim)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim + embedding_dim, count)
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

    def _slow_head_input(self, context: Tensor, slow: Tensor) -> Tensor:
        return torch.cat((context, self.slow_to_embed(slow)), dim=-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features = self._features(observation)
        out, _ = self.gru(features)
        # Slow EMA across time: recurrence s_t = (1-d) * s_{t-1} + d * out_t.
        d = self.slow_decay
        slow = torch.empty_like(out)
        acc = torch.zeros_like(out[:, 0])
        for t in range(out.shape[1]):
            acc = (1.0 - d) * acc + d * out[:, t]
            slow[:, t] = acc
        self._last_slow = slow
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        if isinstance(state, tuple):
            hidden, slow = state
            prior = hidden.unsqueeze(0)
        elif state is None:
            prior = None
            slow = None
        else:
            prior = state.unsqueeze(0)
            slow = None
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        ctx = out[:, 0]
        d = self.slow_decay
        if slow is None:
            slow_new = ctx
        else:
            slow_new = (1.0 - d) * slow + d * ctx
        self._last_slow = slow_new.unsqueeze(1)
        return ctx, (hidden[0], slow_new)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        slow = getattr(self, "_last_slow", None)
        if slow is not None:
            slow_vec = slow.reshape(*context.shape[:-1], self.state_dim)
            context = self._slow_head_input(context, slow_vec.to(context.dtype))
        else:
            context = torch.cat(
                (context, context.new_zeros(*context.shape[:-1], self.embedding_dim)), dim=-1
            )
        return self.heads[stage](context)
