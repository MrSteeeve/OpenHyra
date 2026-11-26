from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU book encoder plus a two-timescale EMA side-channel feeding the heads."""

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
        # Two learnable EMA decay logits (fast / slow) applied per-feature.
        self.ema_logit_fast = nn.Parameter(torch.full((state_dim,), 1.0))
        self.ema_logit_slow = nn.Parameter(torch.full((state_dim,), 3.0))
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + 2 * state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Zero-init new EMA-to-head columns so the parent function is recovered.
        for head in self.heads:
            with torch.no_grad():
                head.weight[:, state_dim:3 * state_dim].zero_()

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

    def _ema_gates(self) -> tuple[Tensor, Tensor]:
        fast = torch.sigmoid(self.ema_logit_fast)
        slow = torch.sigmoid(self.ema_logit_slow)
        return fast, slow

    def _run_sequence(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (gru_out, ema_fast, ema_slow) for features [B,T,C]."""
        gru_out, _ = self.gru(features)
        fast, slow = self._ema_gates()
        ema_f = torch.empty_like(features)
        ema_s = torch.empty_like(features)
        hf = torch.zeros_like(features[:, 0])
        hs = torch.zeros_like(features[:, 0])
        for t in range(features.shape[1]):
            hf = fast * hf + (1 - fast) * features[:, t]
            hs = slow * hs + (1 - slow) * features[:, t]
            ema_f[:, t] = hf
            ema_s[:, t] = hs
        return gru_out, ema_f, ema_s

    def _combine(self, gru_t: Tensor, ema_f_t: Tensor, ema_s_t: Tensor) -> Tensor:
        return torch.cat((gru_t, ema_f_t, ema_s_t), dim=-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        gru_out, ef, es = self._run_sequence(self._features(observation))
        return self._combine(gru_out, ef, es)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        # state carries (gru_hidden, ema_fast, ema_slow)
        if state is None:
            gru_h, ef, es = None, None, None
        else:
            gru_h, ef, es = state
            gru_h = gru_h.unsqueeze(0) if gru_h is not None else None
        feat = self._features(observation)  # [B,C]
        out, hidden = self.gru(feat.unsqueeze(1), gru_h)
        fast, slow = self._ema_gates()
        ef_prev = torch.zeros_like(feat) if ef is None else ef
        es_prev = torch.zeros_like(feat) if es is None else es
        ef_new = fast * ef_prev + (1 - fast) * feat
        es_new = slow * es_prev + (1 - slow) * feat
        context = self._combine(out[:, 0], ef_new, es_new)
        return context, (hidden[0], ef_new, es_new)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
