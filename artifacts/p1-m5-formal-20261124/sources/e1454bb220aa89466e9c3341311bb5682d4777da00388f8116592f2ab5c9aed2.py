from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Fast/slow gated GRU encoder with per-stage conditional order heads."""

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
        # Slow memory: per-unit gate and EMA decay, learned scalars.
        self.slow_gate = nn.Parameter(torch.zeros(state_dim))
        self.slow_decay = nn.Parameter(torch.zeros(state_dim))
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

    def _combine(self, fast: Tensor, slow: Tensor) -> Tensor:
        gate = torch.sigmoid(self.slow_gate)
        return torch.cat((fast, slow * gate), dim=-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        fast, _ = self.gru(feats)
        # EMA over time, starting from zero (matches step with state=None).
        decay = torch.sigmoid(self.slow_decay)
        B, T, C = fast.shape
        slow_states = []
        h = fast.new_zeros(B, C)
        d = decay.view(1, C)
        for t in range(T):
            h = d * h + (1 - d) * fast[:, t]
            slow_states.append(h)
        slow = torch.stack(slow_states, dim=1)
        return self._combine(fast, slow)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        feats = self._features(observation)
        if state is None:
            gru_state = None
            slow_prev = feats.new_zeros(feats.shape[0], self.state_dim)
        else:
            gru_state = state[:self.state_dim].unsqueeze(0)
            slow_prev = state[self.state_dim:]
        out, hidden = self.gru(feats.unsqueeze(1), gru_state)
        fast = out[:, 0]
        d = torch.sigmoid(self.slow_decay)
        slow_new = d * slow_prev + (1 - d) * fast
        ctx = self._combine(fast, slow_new)
        new_state = torch.cat((hidden[0], slow_new), dim=-1)
        return ctx, new_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
