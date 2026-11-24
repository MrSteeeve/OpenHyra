from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, leaky impact integrator, and GRU with conditional heads."""

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
        # Lightweight leaky integrator over the same features for slow impact state.
        self.impact_proj = nn.Linear(43 + 4 * embedding_dim, embedding_dim)
        self.impact_scale = nn.Parameter(torch.tensor(0.1))
        self.impact_gate = nn.Parameter(torch.zeros(embedding_dim))
        self.gru = nn.GRU(state_dim + embedding_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + embedding_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _raw_features(self, observation: StepObservation) -> Tensor:
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
        return torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                          torch.stack((spread, imbalance, observation.is_own), dim=-1),
                          previous.flatten(-2)), dim=-1)

    def _drive(self, raw: Tensor) -> Tensor:
        dtype = self.input_projection.weight.dtype
        return torch.tanh(self.impact_proj(raw.to(dtype))) * self.impact_scale

    def _combine(self, raw: Tensor, impact: Tensor) -> Tensor:
        base = self.input_projection(raw.to(self.input_projection.weight.dtype)).tanh()
        return torch.cat((base, impact.to(base.dtype)), dim=-1), base

    def _scan_impact(self, observation: StepObservation, h0: Tensor | None = None):
        raw = self._raw_features(observation)
        drive = self._drive(raw)
        gate = torch.sigmoid(self.impact_gate).to(drive.dtype)
        if drive.dim() == 2:
            h = drive if h0 is None else gate * h0 + drive
            return raw, h, h
        # Sequence: leaky integrate drive over time.
        b, t, _ = drive.shape
        h = torch.zeros(b, drive.shape[-1], dtype=drive.dtype, device=drive.device)
        if h0 is not None:
            h = h0
        outs = []
        for i in range(t):
            h = gate * h + drive[:, i]
            outs.append(h)
        impact_seq = torch.stack(outs, dim=1)
        return raw, impact_seq, h

    def sequence_context(self, observation: StepObservation) -> Tensor:
        raw = self._raw_features(observation)
        drive = self._drive(raw)
        gate = torch.sigmoid(self.impact_gate).to(drive.dtype)
        b, t, d = drive.shape
        h = torch.zeros(b, d, dtype=drive.dtype, device=drive.device)
        outs = []
        for i in range(t):
            h = gate * h + drive[:, i]
            outs.append(h)
        impact_seq = torch.stack(outs, dim=1)
        base = self.input_projection(raw.to(self.input_projection.weight.dtype)).tanh()
        inp = torch.cat((base, impact_seq.to(base.dtype)), dim=-1)
        out, _ = self.gru(inp)
        self._last_impact = impact_seq
        return out

    def step_context(self, observation: StepObservation, state: State):
        raw = self._raw_features(observation)
        drive = self._drive(raw)
        gate = torch.sigmoid(self.impact_gate).to(drive.dtype)
        if state is None:
            h_imp = drive
            h_gru = None
        else:
            h_imp, h_gru = state
            h_imp = gate * h_imp + drive
        base = self.input_projection(raw.to(self.input_projection.weight.dtype)).tanh()
        inp = torch.cat((base, h_imp.to(base.dtype)), dim=-1).unsqueeze(1)
        prior = None if h_gru is None else h_gru.unsqueeze(0)
        out, hidden = self.gru(inp, prior)
        impact = h_imp
        self._last_impact = impact.unsqueeze(1)
        return out[:, 0], (impact, hidden[0])

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        impact = getattr(self, "_last_impact", None)
        base_parts = [context]
        if impact is not None:
            if impact.dim() == 3:
                imp = impact[:, -1, :]
            else:
                imp = impact
            base_parts.append(imp.to(context.dtype))
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            base_parts.append(prefix)
        return self.heads[stage](torch.cat(base_parts, dim=-1))
