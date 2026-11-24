from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder + GRU with a leaky slow-sign memory branch feeding the heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8,
                 sign_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.sign_dim = sign_dim

        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets),
                             persistent=False)

        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)

        # Book/aux features: 2 levels x 10 x (price + depth) = 40, plus
        # spread, imbalance, is_own = 3, plus 4 previous tokens x embedding_dim.
        book_feat_dim = 2 * 10 * 2 + 3
        in_dim = book_feat_dim + 4 * embedding_dim
        self.input_projection = nn.Linear(in_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])

        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)

        # Slow-sign memory: low-rank drive from previous-event embedding + imbalance.
        self.sign_proj = nn.Linear(4 * embedding_dim + 1, sign_dim)
        nn.init.normal_(self.sign_proj.weight, std=0.02)
        nn.init.zeros_(self.sign_proj.bias)
        # Decay logits -> sigmoid gives ~0.95 at init.
        self.sign_decay_logit = nn.Parameter(torch.full((sign_dim,), 2.94))

        head_in = state_dim + sign_dim
        self.heads = nn.ModuleList()
        for stage, count in enumerate(BUCKET_COUNTS):
            self.heads.append(nn.Linear(head_in + stage * embedding_dim, count))

    # ---- feature construction -------------------------------------------
    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
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
        prev_flat = previous.flatten(-2)
        features = torch.cat(
            (price_feature.flatten(-2), depth_feature.flatten(-2),
             torch.stack((spread, imbalance, observation.is_own), dim=-1),
             prev_flat),
            dim=-1,
        )
        # slow-sign drive input (non-recursive): previous embedding + imbalance
        sign_drive_in = torch.cat((prev_flat, imbalance.unsqueeze(-1)), dim=-1)
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh(), sign_drive_in

    # ---- slow-sign integrator -------------------------------------------
    def _sign_scan(self, drives: Tensor, h0: Tensor | None) -> Tensor:
        """drives: [B,T,sign_dim]. Returns [B,T,sign_dim] slow-state outputs."""
        gate = torch.sigmoid(self.sign_decay_logit)  # [sign_dim]
        gate = gate.to(drives.dtype)
        B, T, C = drives.shape
        if h0 is None:
            h = torch.zeros(B, C, device=drives.device, dtype=drives.dtype)
        else:
            h = h0.to(drives.dtype)
        outs = []
        for t in range(T):
            h = gate * h + (1.0 - gate) * drives[:, t]
            outs.append(h)
        return torch.stack(outs, dim=1)

    def _sign_scan_step(self, drive: Tensor, h_prev: Tensor | None) -> Tensor:
        gate = torch.sigmoid(self.sign_decay_logit).to(drive.dtype)
        if h_prev is None:
            h_prev = torch.zeros_like(drive)
        return gate * h_prev + (1.0 - gate) * drive

    # ---- context ---------------------------------------------------------
    def sequence_context(self, observation: StepObservation) -> Tensor:
        fused, sign_drive_in = self._features(observation)
        gru_out, _ = self.gru(fused)
        drive = self.sign_proj(sign_drive_in).tanh()
        sign_out = self._sign_scan(drive, None)
        return torch.cat((gru_out, sign_out), dim=-1)

    def step_context(self, observation: StepObservation, state: State):
        fused, sign_drive_in = self._features(observation)
        if isinstance(fused.dim(), int) and fused.dim() == 2:
            fused_seq = fused.unsqueeze(1)
        else:
            fused_seq = fused
        if state is None:
            gru_h = None
            sign_h = None
        elif isinstance(state, tuple):
            gru_h, sign_h = state
        else:
            gru_h, sign_h = state, None
        prior = None if gru_h is None else gru_h.unsqueeze(0)
        out, hidden = self.gru(fused_seq, prior)
        drive = self.sign_proj(sign_drive_in).tanh()
        if drive.dim() == 2:
            drive_step = drive
        else:
            drive_step = drive[:, 0]
        sign_new = self._sign_scan_step(drive_step, sign_h)
        context = torch.cat((out[:, 0], sign_new), dim=-1)
        return context, (hidden[0], sign_new)

    def head_logits(self, context: Tensor, stage: int,
                    prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
