from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder + GRU with a two-timescale EMA head side-channel."""

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

        # Leaky-integrator side-channel: two learnable timescales over projected features.
        # log_decay init so that gates ≈ {exp(-1/(fast_period)) : fast, exp(-1/(slow_period)) : slow}.
        fast_logit = torch.log(torch.tensor(-torch.log(torch.tensor(0.5))))  # gate ~0.5
        slow_logit = torch.log(torch.tensor(-torch.log(torch.tensor(0.95))))  # gate ~0.95
        self.ema_decay_logit = nn.Parameter(torch.stack([fast_logit, slow_logit]))

        side = 2 * state_dim
        self.side_dim = side
        self.heads = nn.ModuleList()
        for stage, count in enumerate(BUCKET_COUNTS):
            layer = nn.Linear(state_dim + stage * embedding_dim + side, count)
            # Zero-init the EMA side-channel columns so function equals parent at step 0.
            with torch.no_grad():
                layer.weight[:, -side:].zero_()
            self.heads.append(layer)

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

    def _ema_step(self, x: Tensor, side: Tensor) -> Tensor:
        # x: [..., C], side: [..., 2C]; single leaky-integrator step per timescale.
        gates = torch.sigmoid(self.ema_decay_logit)  # [2]
        fast, slow = side[..., :self.state_dim], side[..., self.state_dim:]
        fast_new = gates[0] * fast + (1 - gates[0]) * x
        slow_new = gates[1] * slow + (1 - gates[1]) * x
        return torch.cat((fast_new, slow_new), dim=-1)

    def _sequence_side(self, features: Tensor) -> Tensor:
        # features: [B, T, C] -> side [B, T, 2C] via a scan.
        B, T, C = features.shape
        gates = torch.sigmoid(self.ema_decay_logit)  # [2]
        side = features.new_zeros(B, T, 2 * C)
        fast = features.new_zeros(B, C)
        slow = features.new_zeros(B, C)
        for t in range(T):
            xt = features[:, t]
            fast = gates[0] * fast + (1 - gates[0]) * xt
            slow = gates[1] * slow + (1 - gates[1]) * xt
            side[:, t, :C] = fast
            side[:, t, C:] = slow
        return side

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        out, _ = self.gru(feats)
        side = self._sequence_side(feats)
        return torch.cat((out, side), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        feats = self._features(observation)  # [B, C]
        if state is None:
            prior_h = None
            prior_side = None
        else:
            prior_h, prior_side = state
            prior_h = prior_h.unsqueeze(0)
        out, hidden = self.gru(feats.unsqueeze(1), prior_h)
        h = hidden[0]
        side = feats.new_zeros(feats.shape[0], 2 * self.state_dim) if prior_side is None else prior_side
        side = self._ema_step(feats, side)
        ctx = torch.cat((out[:, 0], side), dim=-1)
        return ctx, (h, side)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if context.size(-1) == self.state_dim + self.side_dim:
            base, side = context[..., :self.state_dim], context[..., self.state_dim:]
        else:
            base, side = context, None
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            base = torch.cat((base, prefix), dim=-1)
        if side is not None:
            base = torch.cat((base, side), dim=-1)
        return self.heads[stage](base)
