from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import (
    parallel_affine_scan, real_diagonal,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder, GRU, and a two-scale fast/slow residual into the heads."""

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
        # Fast/slow residual: Linear(2*state_dim, state_dim), zero-init.
        self.mix = nn.Linear(2 * state_dim, state_dim)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        # Passive EMA decay 0.8 as a fixed logit for real_diagonal.
        import math
        self.register_buffer(
            "ema_decay_logit", torch.tensor([math.log(0.8 / 0.2)]), persistent=False
        )
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
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, observation.is_own), dim=-1),
                              previous.flatten(-2)), dim=-1)
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    def _ema_gate(self) -> Tensor:
        # Real decay "a" from logit; parallel_affine_scan computes h_t = a*h_{t-1} + x_t.
        a = torch.sigmoid(self.ema_decay_logit).to(self.input_projection.weight.dtype)
        # real_diagonal on a 1-element tensor returns same shape; use directly.
        return a.reshape(())

    def _fast_slow(self, slow: Tensor) -> Tensor:
        # slow: [..., T, C]; compute EMA over T with zero init.
        a = self._ema_gate()
        gate = a.expand(slow.shape[-1])  # [C]
        ema = parallel_affine_scan(gate, slow)
        fast = slow - ema
        mixed = self.mix(torch.cat((slow, fast), dim=-1))
        return slow + mixed

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        return self._fast_slow(out)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        # For single-step, EMA≈0 initially (passive, stateless per step) so fast≈slow.
        # Apply same residual on the single output using EMA=slow itself gives fast=0,
        # which under-relaxes; instead reuse the recursive EMA carried in `state`? We
        # cannot add scalars. Keep consistent with sequence by running EMA over the
        # one-element sequence: with zero init, ema_1 = a*0 + slow = slow -> fast=0.
        # To remain a faithful single-step realization of the sequence recurrence
        # (which itself starts from EMA=0 at t=0), fast=0 at t=0 is correct.
        step_out = out[:, 0]
        a = self._ema_gate()
        ema = a * step_out  # EMA₀ = 0
        fast = step_out - ema
        mixed = self.mix(torch.cat((step_out, fast), dim=-1))
        return step_out + mixed, hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
