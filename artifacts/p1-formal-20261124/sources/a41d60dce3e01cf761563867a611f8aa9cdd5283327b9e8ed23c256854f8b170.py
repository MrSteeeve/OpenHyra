from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, GRU plus fast EMA path, discrete conditional heads."""

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
        # Fast leaky EMA path.
        self.fast_decay_logit = nn.Parameter(torch.tensor(1.3862943611))  # sigmoid ~ 0.8
        self.fast_scale = nn.Parameter(torch.zeros(state_dim))
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

    def _fast_scan(self, features: Tensor, h0: Tensor | None) -> Tensor:
        decay = torch.sigmoid(self.fast_decay_logit)
        gate = torch.full_like(features[..., :1], decay.item())
        if h0 is None:
            h0 = torch.zeros_like(features[..., 0, :])
        outs = []
        h = h0
        for t in range(features.shape[-2]):
            h = gate[..., t, :] * h + (1.0 - gate[..., t, :]) * features[..., t, :]
            outs.append(h)
        return torch.stack(outs, dim=-2)

    def _augment(self, gru_out: Tensor, fast: Tensor) -> Tensor:
        return gru_out + fast * self.fast_scale

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        gru_out, _ = self.gru(feats)
        fast = self._fast_scan(feats, None)
        return self._augment(gru_out, fast)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        feats = self._features(observation).unsqueeze(-2)
        if state is None:
            gru_prior = None
            fast_prior = None
        else:
            gru_h, fast_h = state
            gru_prior = gru_h.unsqueeze(0)
            fast_prior = fast_h
        gru_out, hidden = self.gru(feats, gru_prior)
        fast = self._fast_scan(feats, fast_prior)[..., 0, :]
        ctx = self._augment(gru_out[..., 0, :], fast)
        return ctx, (hidden[0], fast)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
