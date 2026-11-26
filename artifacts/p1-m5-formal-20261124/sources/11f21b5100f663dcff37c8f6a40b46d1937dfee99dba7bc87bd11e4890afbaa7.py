from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Parent GRU with a two-timescale EMA side-channel feeding the heads."""

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

        # Two-timescale EMA decay logits. Choose gates so decays are ~0.7 and ~0.95 step.
        fast_gate = 0.7
        slow_gate = 0.95
        fast_logit = math.log(-math.log(fast_gate))
        slow_logit = math.log(-math.log(slow_gate))
        self.ema_decay = nn.Parameter(torch.tensor([fast_logit, slow_logit]))

        head_in = state_dim + 2 * state_dim
        self.heads = nn.ModuleList()
        for stage, count in enumerate(BUCKET_COUNTS):
            linear = nn.Linear(head_in + stage * embedding_dim, count)
            # Zero-init EMA-to-head columns so parent function is recovered at step 0.
            with torch.no_grad():
                linear.weight[:, state_dim:head_in].zero_()
            self.heads.append(linear)

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
        return torch.sigmoid(self.ema_decay)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        out, _ = self.gru(feats)
        # Parallel EMA scan: a_t = 1 - gate (retention). Use affine_scan on feats.
        gates = self._ema_gates()  # [2]
        retention = (1.0 - gates).view(1, 1, 2, 1)  # broadcast
        # Shape feats to [B,T,2,C]
        b, t, c = feats.shape
        x = feats.unsqueeze(2)  # [B,T,1,C]
        a = retention.expand(b, t, 2, c)
        inp = (gates.view(1, 1, 2, 1)) * x
        from stochopia.neural.primitives import parallel_affine_scan
        e = parallel_affine_scan(a.reshape(b, t, 2 * c), inp.reshape(b, t, 2 * c))
        e = e.reshape(b, t, 2 * c)
        return torch.cat((out, e), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        feats = self._features(observation)
        if state is None:
            h_prev = None
            ema_prev = feats.new_zeros(feats.shape[:-1] + (2 * self.state_dim,))
        else:
            h_prev, ema_prev = state
            h_prev = h_prev.unsqueeze(0)
        out, hidden = self.gru(feats.unsqueeze(1), h_prev)
        out = out[:, 0]
        gates = self._ema_gates()
        retention = 1.0 - gates  # [2]
        ema_prev_v = ema_prev.view(*ema_prev.shape[:-1], 2, self.state_dim)
        inp = gates.view(1, 2, 1) * feats.unsqueeze(-2)
        ema_new = retention.view(1, 2, 1) * ema_prev_v + inp
        ema_new = ema_new.reshape(*ema_prev.shape)
        ctx = torch.cat((out, ema_new), dim=-1)
        return ctx, (hidden[0], ema_new)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
