from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU encoder with a zero-initialized slow leaky-context pathway into heads."""

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
        # Slow leaky context pathway.
        self.slow_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.slow_input = nn.Linear(state_dim, state_dim, bias=False)
        self.heads = nn.ModuleList()
        for stage, count in enumerate(BUCKET_COUNTS):
            head = nn.Linear(state_dim + stage * embedding_dim + state_dim, count)
            # Zero-init the slow-context block so the parent function is preserved.
            nn.init.zeros_(head.weight[:, -(state_dim + state_dim): -stage * embedding_dim if stage else None] if stage else head.weight[:, -state_dim:])
            self.heads.append(head)

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
        # leaky cumulative scan: c_t = (1-g)*c_{t-1} + g * tanh(W x_t)
        g = torch.sigmoid(self.slow_gate_logit)
        u = self.slow_input(feats).tanh()
        B, T, C = u.shape
        if h0 is None:
            h0 = u.new_zeros(B, C)
        # sequential but cheap (state, not weights); fine per step, and for seq use loop
        outs = []
        c = h0
        for t in range(T):
            c = (1.0 - g) * c + g * u[:, t]
            outs.append(c)
        return torch.stack(outs, dim=1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        out, _ = self.gru(feats)
        slow = self._slow_scan(feats, None)
        return torch.cat((out, slow), dim=-1)

    def step_context(self, observation: StepObservation, state: State):
        # state = (gru_hidden [B,state_dim], slow_context [B,state_dim]) or None
        if state is None:
            prior_gru = None
            prior_slow = None
        else:
            prior_gru, prior_slow = state
        feats = self._features(observation).unsqueeze(1)
        out, hidden = self.gru(feats, None if prior_gru is None else prior_gru.unsqueeze(0))
        if prior_slow is None:
            prior_slow = out.new_zeros(out.shape[0], out.shape[-1])
        g = torch.sigmoid(self.slow_gate_logit)
        u = self.slow_input(feats[:, 0]).tanh()
        slow = (1.0 - g) * prior_slow + g * u
        ctx = torch.cat((out[:, 0], slow), dim=-1)
        return ctx, (hidden[0], slow)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        ctx_gru = context[..., : self.state_dim]
        slow = context[..., self.state_dim:]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            ctx_gru = torch.cat((ctx_gru, prefix), dim=-1)
        # head expects [gru(+prefix), stage*emb, slow] -> we pass slow at end
        return self.heads[stage](torch.cat((ctx_gru, slow), dim=-1))
