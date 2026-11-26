from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU encoder with a fast/slow split carrying depth-sensitive slow state."""

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
        # Slow-branch decay: logits -> per-channel decay near 0.9.
        self.decay_logit = nn.Parameter(torch.full((state_dim,), 2.1972))
        # Slow-branch correction on last 4 channels (spread, imbalance, own, spare).
        self.slow_slice = slice(state_dim - 4, state_dim)
        self.slow_corr = nn.Linear(4, state_dim)
        nn.init.zeros_(self.slow_corr.weight)
        nn.init.zeros_(self.slow_corr.bias)
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
        own = observation.is_own.to(self.input_projection.weight.dtype)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, own), dim=-1),
                              previous.flatten(-2)), dim=-1)
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    def _slow_correction(self, sub: Tensor) -> Tensor:
        return self.slow_corr(sub.to(self.slow_corr.weight.dtype))

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        out, _ = self.gru(feats)
        # Fast/slow split: slow = decay-multiplied prefix sum over context.
        decay = torch.sigmoid(self.decay_logit).to(out.dtype)
        # Prefix scan with per-channel multiplicative decay.
        T = out.shape[-2]
        slow = torch.zeros_like(out)
        acc = torch.zeros_like(out[..., 0, :])
        for t in range(T):
            acc = decay * acc + (1 - decay) * out[..., t, :]
            slow[..., t, :] = acc
        corr = self._slow_correction(observation.book_shares.to(out.dtype).sum(dim=-2).sum(dim=-1) * 0)
        # correction uses depth/imbalance features recomputed from feats slice
        sub = self._depth_subvector(observation, out)
        corr = self._slow_correction(sub)
        return out + slow + corr.unsqueeze(-2) if corr.dim() == 2 else out + slow

    def _depth_subvector(self, observation: StepObservation, out: Tensor) -> Tensor:
        book_shares = observation.book_shares.to(out.dtype)
        ticks = observation.book_ticks
        touch = ticks[..., 0]
        present = touch > 0
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0)
        bid_depth, ask_depth = book_shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1) / 100
        own = observation.is_own.to(out.dtype)
        spare = torch.zeros_like(own)
        sub = torch.stack((spread.log1p(), imbalance, own, spare), dim=-1)
        return sub

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        feats = self._features(observation).unsqueeze(1)
        out, hidden = self.gru(feats, prior)
        h = out[:, 0]
        decay = torch.sigmoid(self.decay_logit).to(h.dtype)
        slow_prev = torch.zeros_like(h)
        slow = decay * slow_prev + (1 - decay) * h
        sub = self._depth_subvector(observation.unsqueeze(1) if observation.book_ticks.dim() == 3 else observation, h)
        if sub.dim() == 3:
            sub = sub[:, 0]
        corr = self._slow_correction(sub)
        return h + slow + corr, hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
