from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU plus a slow leaky integrator gated into conditional order heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8,
                 slow_dim: int = 32, gate_rank: int = 4) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.slow_dim = slow_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)

        # Slow leaky integrator: decay logit parameter, drive from features + own flag.
        self.slow_decay_logit = nn.Parameter(torch.full((slow_dim,), 2.0))
        self.slow_drive = nn.Linear(state_dim + 1, slow_dim, bias=False)
        # Low-rank gate: heads see fast GRU context plus gated slow state.
        self.gate_down = nn.Linear(state_dim, gate_rank, bias=False)
        self.gate_up = nn.Linear(gate_rank, slow_dim, bias=False)
        nn.init.zeros_(self.gate_up.weight)

        # Project slow into head input via low-rank contribution.
        self.slow_proj = nn.Linear(slow_dim, 8, bias=False)
        nn.init.zeros_(self.slow_proj.weight)

        self.heads = nn.ModuleList(
            nn.Linear(state_dim + 8 + stage * embedding_dim, count)
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

    def _slow_drive(self, feat: Tensor, is_own: Tensor) -> Tensor:
        own = is_own.to(feat.dtype).unsqueeze(-1)
        return self.slow_drive(torch.cat((feat, own), dim=-1)).tanh()

    def _slow_scan(self, drive: Tensor, h0: Tensor | None) -> Tensor:
        # drive [B,T,S]; gate = sigmoid(decay_logit) in (0,1)
        gate = torch.sigmoid(self.slow_decay_logit)
        if h0 is None:
            h = torch.zeros_like(drive[:, 0])
        else:
            h = h0
        outs = []
        for t in range(drive.shape[1]):
            h = gate * h + (1 - gate) * drive[:, t]
            outs.append(h)
        return torch.stack(outs, dim=1)

    def _gated_slow(self, context: Tensor, slow: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate_up(self.gate_down(context).tanh()))
        return self.slow_proj(gate * slow)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feat = self._features(observation)
        out, _ = self.gru(feat)
        drive = self._slow_drive(feat, observation.is_own)
        slow = self._slow_scan(drive, None)
        return torch.cat((out, self._gated_slow(out, slow)), dim=-1)

    def step_context(self, observation: StepObservation, state: State):
        feat = self._features(observation)  # [B, D]
        if state is None:
            h_gru = None
            h_slow = torch.zeros(feat.shape[0], self.slow_dim, device=feat.device, dtype=feat.dtype)
        else:
            h_gru, h_slow = state
            h_gru = h_gru.unsqueeze(0)
        out, hidden = self.gru(feat.unsqueeze(1), h_gru)
        context = out[:, 0]
        drive = self._slow_drive(feat, observation.is_own)
        gate = torch.sigmoid(self.slow_decay_logit)
        h_slow_new = gate * h_slow + (1 - gate) * drive
        ctx = torch.cat((context, self._gated_slow(context, h_slow_new)), dim=-1)
        return ctx, (hidden[0], h_slow_new)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
