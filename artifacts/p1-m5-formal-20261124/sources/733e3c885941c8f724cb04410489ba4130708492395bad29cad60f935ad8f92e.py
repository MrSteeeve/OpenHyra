from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder with a GRU plus a two-timescale EMA side-channel."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8,
                 num_ema: int = 2) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_ema = num_ema
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        head_in = state_dim + num_ema * state_dim
        self.heads = nn.ModuleList(
            nn.Linear(head_in + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Zero the EMA-to-head columns so the parent function is recovered at init.
        for head in self.heads:
            with torch.no_grad():
                head.weight[:, state_dim:head_in].zero_()
        # EMA gates (per-channel) initialized at ~0.5 and ~0.95 (unit-rate EMA).
        eps = 1e-6
        fast_gate = 0.5
        slow_gate = 0.95
        self.ema_logits = nn.Parameter(torch.stack((
            torch.full((state_dim,), float(torch.log(torch.tensor(-torch.log(torch.tensor(fast_gate + eps)))))),
            torch.full((state_dim,), float(torch.log(torch.tensor(-torch.log(torch.tensor(slow_gate + eps)))))),
        )))

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

    def _ema_gates(self) -> Tensor:
        return torch.sigmoid(self.ema_logits)  # [num_ema, state_dim]

    def _ema_scan(self, features: Tensor, h0: Tensor | None) -> Tensor:
        # features: [B, T, C]; returns EMA states [B, T, num_ema*C]
        B, T, C = features.shape
        gates = self._ema_gates()  # [E, C]
        outs = features.new_zeros(B, T, self.num_ema * C)
        for e in range(self.num_ema):
            g = gates[e]  # [C]
            h = features.new_zeros(B, C) if h0 is None else h0[:, e * C:(e + 1) * C]
            seq = []
            for t in range(T):
                h = (1.0 - g) * h + g * features[:, t]
                seq.append(h)
            outs[:, :, e * C:(e + 1) * C] = torch.stack(seq, dim=1)
        return outs

    def _ema_step(self, feature_t: Tensor, h_prev: Tensor | None) -> Tensor:
        # feature_t: [B, C]; h_prev: [B, num_ema*C] or None
        gates = self._ema_gates()
        outs = []
        for e in range(self.num_ema):
            g = gates[e]
            h = feature_t.new_zeros(feature_t.shape[0], feature_t.shape[1]) if h_prev is None \
                else h_prev[:, e * feature_t.shape[-1]:(e + 1) * feature_t.shape[-1]]
            h = (1.0 - g) * h + g * feature_t
            outs.append(h)
        return torch.cat(outs, dim=-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        gru_out, _ = self.gru(feats)
        ema_out = self._ema_scan(feats, None)
        return torch.cat((gru_out, ema_out), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        feats = self._features(observation)  # [B, C]
        if state is None:
            gru_hidden = None
            ema_hidden = None
        else:
            gru_hidden = state[:, :self.state_dim].unsqueeze(0)
            ema_hidden = state[:, self.state_dim:]
        gru_out, hidden = self.gru(feats.unsqueeze(1), gru_hidden)
        ema_out = self._ema_step(feats, ema_hidden)
        new_state = torch.cat((hidden[0], ema_out), dim=-1)
        context = torch.cat((gru_out[:, 0], ema_out), dim=-1)
        return context, new_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
