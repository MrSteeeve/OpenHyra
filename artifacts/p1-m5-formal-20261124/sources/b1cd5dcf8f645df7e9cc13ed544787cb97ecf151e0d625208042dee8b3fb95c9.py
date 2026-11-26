from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """A ten-level book encoder and GRU with discrete conditional order heads,
    augmented with a parallel slow leaky-average context stream."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        slow_dim = state_dim // 2
        self.slow_dim = slow_dim
        self.slow_decay = 0.9
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.slow_projection = nn.Linear(state_dim, slow_dim)
        # Zero-init so parent function preserved at step 0.
        nn.init.zeros_(self.slow_projection.weight)
        nn.init.zeros_(self.slow_projection.bias)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + slow_dim + stage * embedding_dim, count)
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

    def _slow_scan(self, feats: Tensor) -> Tensor:
        # feats: [B, T, state_dim] -> [B, T, slow_dim]
        proj = self.slow_projection(feats)
        decay = self.slow_decay
        out = torch.empty_like(proj)
        acc = torch.zeros_like(proj[:, 0])
        for t in range(proj.shape[1]):
            acc = decay * acc + (1.0 - decay) * proj[:, t]
            out[:, t] = acc
        return out

    def _combine(self, gru_out: Tensor, slow: Tensor) -> Tensor:
        return torch.cat((gru_out, slow), dim=-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        gru_out, _ = self.gru(feats)
        slow = self._slow_scan(feats)
        return self._combine(gru_out, slow)

    def step_context(self, observation: StepObservation, state: State):
        # state: (gru_hidden [B,state_dim], slow_acc [B,slow_dim]) or None
        if state is None:
            gru_prior = None
            slow_acc = None
        else:
            if isinstance(state, tuple):
                gru_prior = state[0].unsqueeze(0)
                slow_acc = state[1]
            else:
                gru_prior = state.unsqueeze(0)
                slow_acc = None
        feats = self._features(observation).unsqueeze(1)
        gru_out, hidden = self.gru(feats, gru_prior)
        gru_out = gru_out[:, 0]
        proj = self.slow_projection(feats[:, 0])
        if slow_acc is None:
            slow_acc = torch.zeros_like(proj)
        slow_acc = self.slow_decay * slow_acc + (1.0 - self.slow_decay) * proj
        ctx = self._combine(gru_out, slow_acc)
        new_state = (hidden[0], slow_acc)
        return ctx, new_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
