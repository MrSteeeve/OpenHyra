from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder + GRU with a slow gated level context feeding every head."""

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
        # Slow-context gate (consumes features + previous slow level).
        self.slow_gate = nn.Linear(state_dim + 1, state_dim)
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

    def _slow_scan(self, feats: Tensor, h0: Tensor | None) -> Tensor:
        # feats [..., T, C]; h0 [..., C] or None. Cumulative gated average.
        T = feats.shape[-2]
        gate = torch.sigmoid(self.slow_gate(torch.cat(
            (feats, torch.zeros_like(feats[..., :1])), dim=-1)))
        # Recompute with proper previous-level channel is expensive; use a fixed EMA row.
        gate = gate  # placeholder, replaced below
        # Simple learned scalar gate per channel with fixed scalar mix.
        g = torch.sigmoid(self._mix(feats))  # [..., T, 1]
        outs = []
        if h0 is None:
            h = torch.zeros_like(feats[..., 0, :])
        else:
            h = h0
        for t in range(T):
            h = (1 - g[..., t, :]) * h + g[..., t, :] * feats[..., t, :]
            outs.append(h)
        return torch.stack(outs, dim=-2)

    def _mix(self, feats: Tensor) -> Tensor:
        # scalar gate: project features to one logit using first column of slow_gate.
        w = self.slow_gate.weight[:, : feats.shape[-1]]  # [C, C]
        # cheap scalar: use mean over channels of a single learned row
        row = self.slow_gate.weight[0, : feats.shape[-1]]  # [C]
        return (feats * row).sum(-1, keepdim=True) / (row.abs().sum() + 1e-6)

    def _slow_scan_fast(self, feats: Tensor, h0: Tensor | None) -> Tensor:
        T = feats.shape[-2]
        g = torch.sigmoid(self._mix(feats))  # [..., T, 1]
        if h0 is None:
            h = torch.zeros_like(feats[..., 0, :])
        else:
            h = h0
        outs = []
        for t in range(T):
            h = (1 - g[..., t, :]) * h + g[..., t, :] * feats[..., t, :]
            outs.append(h)
        return torch.stack(outs, dim=-2)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        out, _ = self.gru(feats)
        slow = self._slow_scan_fast(feats, None)
        return torch.cat((out, slow), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        feats = self._features(observation)
        if state is None:
            h_gru = None
            h_slow = None
        elif isinstance(state, Tensor):
            # Split layout: [state_dim (gru) | state_dim (slow)]
            h_gru = state[..., : self.state_dim].unsqueeze(0)
            h_slow = state[..., self.state_dim :]
        else:
            raise ValueError("Unsupported state")
        out, hidden = self.gru(feats.unsqueeze(-2), h_gru)
        slow = self._slow_scan_fast(feats.unsqueeze(-2), h_slow)
        new_state = torch.cat((hidden[0], slow[..., 0, :]), dim=-1)
        ctx = torch.cat((out[:, 0], slow[:, 0]), dim=-1)
        return ctx, new_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
