from __future__ import annotations

import torch
from torch import Tensor, nn

from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Dual-timescale (fast/slow) GRU encoder with stage-conditioned gated fusion."""

    def __init__(self, fast_dim: int = 24, slow_dim: int = 20, embedding_dim: int = 8) -> None:
        super().__init__()
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.embedding_dim = embedding_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)

        feat_dim = 43 + 4 * embedding_dim
        # Shared shallow projection, then split into fast/slow inputs.
        hidden = fast_dim + slow_dim
        self.input_projection = nn.Linear(feat_dim, hidden)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])

        self.gru_fast = nn.GRU(fast_dim, fast_dim, batch_first=True)
        self.gru_slow = nn.GRU(slow_dim, slow_dim, batch_first=True)

        # Per-branch per-stage input projections (split from shared projection).
        self.hidden = hidden
        self.context_dim = hidden
        self.heads = nn.ModuleList()
        for stage, count in enumerate(BUCKET_COUNTS):
            self.heads.append(
                nn.Linear(self.context_dim + stage * embedding_dim, count)
            )

        # Learnable stage-conditioned gate parameters: fast branch gate ~ sigmoid(w·x + b)
        # where x = [log depth imbalance, |log depth imbalance|]
        self.gate_scale = nn.Parameter(torch.zeros(len(BUCKET_COUNTS)))
        self.gate_bias = nn.Parameter(torch.zeros(len(BUCKET_COUNTS)))

    # ------------------------------------------------------------------ features
    def _book_features(self, observation: StepObservation) -> Tensor:
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
        dsum = (bid_depth + ask_depth).clamp_min(1)
        imbalance = (bid_depth - ask_depth) / dsum
        log_imbalance = ((bid_depth + 1).log() - (ask_depth + 1).log())

        previous = self.token_embed(
            observation.previous_event + self.token_offsets
        )
        return torch.cat(
            (
                price_feature.flatten(-2),
                depth_feature.flatten(-2),
                torch.stack(
                    (spread, imbalance, log_imbalance, observation.is_own), dim=-1
                ),
                previous.flatten(-2),
            ),
            dim=-1,
        ), imbalance

    def _split_gate(self, imbalance: Tensor) -> tuple[Tensor, Tensor]:
        # imbalance: [...,] per prefix. Return fast gate per... returns gate scalar.
        # We use a single scalar per step governing the whole vector.
        x = imbalance * self.gate_scale + self.gate_bias
        fast_gate = torch.sigmoid(x)
        return fast_gate, 1.0 - fast_gate

    def _prepare(self, observation: StepObservation) -> tuple[Tensor, Tensor, Tensor]:
        raw, imbalance = self._book_features(observation)
        projected = self.input_projection(raw).tanh()
        fast_in = projected[..., : self.fast_dim]
        slow_in = projected[..., self.fast_dim :]
        return fast_in, slow_in, imbalance

    # ------------------------------------------------------------------ sequence
    def sequence_context(self, observation: StepObservation) -> Tensor:
        fast_in, slow_in, imbalance = self._prepare(observation)
        fast_out, _ = self.gru_fast(fast_in)
        slow_out, _ = self.gru_slow(slow_in)
        gate_f, gate_s = self._split_gate(imbalance)
        g = gate_f.unsqueeze(-1)
        return g * fast_out + (1.0 - g) * slow_out

    # ------------------------------------------------------------------ step
    def step_context(
        self, observation: StepObservation, state: State
    ) -> tuple[Tensor, State]:
        fast_in, slow_in, imbalance = self._prepare(observation)
        if state is None:
            h_fast = None
            h_slow = None
        else:
            if not isinstance(state, tuple):
                raise ValueError("Expected tuple state for dual-timescale model.")
            h_fast, h_slow = state
            if h_fast is not None:
                h_fast = h_fast.unsqueeze(0)
            if h_slow is not None:
                h_slow = h_slow.unsqueeze(0)
        fast_out, hf = self.gru_fast(fast_in.unsqueeze(1), h_fast)
        slow_out, hs = self.gru_slow(slow_in.unsqueeze(1), h_slow)
        fast_out = fast_out[:, 0]
        slow_out = slow_out[:, 0]
        g = self._split_gate(imbalance)[0].unsqueeze(-1)
        ctx = g * fast_out + (1.0 - g) * slow_out
        new_state = (hf[0], hs[0])
        return ctx, new_state

    # ------------------------------------------------------------------ head
    def head_logits(
        self, context: Tensor, stage: int, prior: tuple[Tensor, ...]
    ) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
