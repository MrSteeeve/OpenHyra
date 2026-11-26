from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder with a gated GRU and discrete conditional heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        # 43 base features + previous-event embedding + 1 scalar gate logit
        self.input_projection = nn.Linear(43 + 4 * embedding_dim + 1, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
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
        gate_logit = (spread.abs() - imbalance.abs()).unsqueeze(-1)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, observation.is_own), dim=-1),
                              previous.flatten(-2), gate_logit), dim=-1)
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    def _gru_step(self, features: Tensor, h_prev: Tensor | None) -> Tensor:
        # Compute the GRU update exactly, then blend with h_prev via a gate derived
        # from the last feature channel. Use the same formula for sequence and step
        # so that outputs match to numerical precision.
        if h_prev is None:
            h_prev = features.new_zeros(features.shape[:-1] + (self.state_dim,))
        # GRU update using the canonical formulation matching nn.GRU equations.
        # We call the module to get the true update, then re-blend.
        raise RuntimeError("unreachable")

    def _gru_run(self, features: Tensor, h0: Tensor | None) -> Tensor:
        """Run gated GRU over time; identical computation for any [...,T,C] input."""
        T = features.shape[-2]
        outs = []
        h = h0
        # The GRU layer without a gate; we blend per-step using a scalar gate.
        # Compute raw recurrence using nn.GRU one step at a time so that the
        # sequence path and step path produce numerically identical results.
        for t in range(T):
            x_t = features[..., t, :]
            if h is None:
                h_in = x_t.new_zeros(x_t.shape[:-1] + (self.state_dim,))
            else:
                h_in = h
            out, h_new = self.gru(x_t.unsqueeze(-2), h_in.unsqueeze(0))
            raw = out[..., 0, :]
            h_new = h_new[0]
            # Gate derived from a scalar computed from the last channel of features.
            # We use a deterministic, input-derived scalar in [0,1].
            gate_scalar = torch.sigmoid(x_t[..., -1:])
            blended = gate_scalar * raw + (1.0 - gate_scalar) * h_in
            outs.append(blended)
            # Feed blended back as hidden so state matches across paths.
            h = blended
        return torch.stack(outs, dim=-2)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._gru_run(self._features(observation), None)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        features = self._features(observation)
        # features shape: [...,state_dim]; unsqueeze for time dim of 1
        out = self._gru_run(features.unsqueeze(-2), state)
        new_state = out[..., 0, :]
        return new_state, new_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
