import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Expand after initialization to preserve the parent's seed-dependent weights.
        projection = self.input_projection
        weight = projection.weight.detach()
        projection.weight = nn.Parameter(
            torch.cat((weight, weight.new_zeros(48, 4)), dim=1)
        )
        projection.in_features = 79
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000

    def _book(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        shares = observation.book_shares.to(self.input_projection.weight.dtype)
        totals = shares.sum(dim=-1)
        summary = (torch.cat((shares[..., 0], totals), dim=-1) / 100).log1p()
        return shares, summary

    def _features(
        self, observation: StepObservation, shares: Tensor, delta: Tensor
    ) -> Tensor:
        ticks = observation.book_ticks
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        price = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1),
            previous.flatten(-2), delta,
        ), dim=-1)
        return self.input_projection(features.to(shares.dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        shares, summary = self._book(observation)
        delta = torch.cat((
            torch.zeros_like(summary[:, :1]),
            summary[:, 1:] - summary[:, :-1],
        ), dim=1)
        out, _ = self.gru(self._features(observation, shares, delta))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        shares, summary = self._book(observation)
        if state is None:
            hidden = None
            delta = torch.zeros_like(summary)
        else:
            hidden = state[..., :48].unsqueeze(0).contiguous()
            delta = summary - state[..., 48:]
        out, hidden = self.gru(self._features(observation, shares, delta).unsqueeze(1), hidden)
        return out[:, 0], torch.cat((hidden[0], summary), dim=-1)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
