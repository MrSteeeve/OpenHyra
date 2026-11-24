import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)
from stochopia.neural.primitives import affine_scan_step, parallel_affine_scan


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        # Training-split scales were not supplied; use unit scales as a fallback.
        self.register_buffer("feature_scales", torch.ones(8))
        self.register_buffer("trace_decay", torch.full((8,), 0.97), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000

    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        both = present.all(dim=-1)
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        prices = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        prices = prices.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~both, 0).to(dtype).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        bid_touch, ask_touch = shares[..., 0].unbind(dim=-1)
        touch_imbalance = (
            (bid_touch - ask_touch) / (bid_touch + ask_touch).clamp_min(1)
        )
        second = ticks[..., 1]
        gaps = (second - touch).abs().masked_fill(~(present & (second > 0)), 0)
        bid_gap, ask_gap = gaps.to(dtype).log1p().unbind(dim=-1)
        bid_size, ask_size = depth[..., 0].unbind(dim=-1)
        current = torch.stack(
            (spread, bid_size, ask_size, touch_imbalance, imbalance,
             bid_gap, ask_gap, both.to(dtype)), dim=-1,
        )

        own = observation.is_own.reshape_as(spread).to(dtype)
        previous = self.token_embed(observation.previous_event.long() + self.token_offsets)
        features = torch.cat(
            (prices.flatten(-2).to(dtype), depth.flatten(-2),
             torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2)),
            dim=-1,
        )
        return self.input_projection(features).tanh(), current

    def _context(self, hidden: Tensor, current: Tensor, prior: Tensor) -> Tensor:
        delta = ((current - prior) / self.feature_scales).clamp(-4, 4)
        return torch.cat((hidden[..., :40], delta.to(hidden.dtype)), dim=-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, current = self._features(observation)
        hidden, _ = self.gru(features)
        first = current[:, :1]
        if current.size(1) > 2:
            history = parallel_affine_scan(
                self.trace_decay, 0.03 * current[:, 1:-1], h0=first[:, 0],
            )
            prior = torch.cat((first, first, history), dim=1)
        else:
            prior = first.expand_as(current)
        return self._context(hidden, current, prior)

    def step_context(
        self, observation: StepObservation, state: State,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        features, current = self._features(observation)
        if state is None:
            gru_prior, trace_prior, trace = None, current, current
        else:
            gru_prior = state[0].unsqueeze(0)
            trace_prior = state[1]
            trace = affine_scan_step(self.trace_decay, 0.03 * current, trace_prior)
        _, hidden = self.gru(features.unsqueeze(1), gru_prior)
        hidden = hidden[0]
        return self._context(hidden, current, trace_prior), (hidden, trace)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1).long() + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
