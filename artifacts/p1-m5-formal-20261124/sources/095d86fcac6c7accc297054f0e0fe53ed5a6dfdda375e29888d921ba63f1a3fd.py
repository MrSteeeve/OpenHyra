import torch
from torch import nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self):
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        # Action buckets are paired by side. Price buckets use a side-relative
        # coordinate, so reflection preserves their indices, including sentinels.
        permutations = (
            torch.arange(BUCKET_COUNTS[0]),
            torch.tensor([1, 0, 3, 2, 5, 4]),
            torch.arange(BUCKET_COUNTS[2]),
            torch.arange(BUCKET_COUNTS[3]),
        )
        for stage, permutation in enumerate(permutations):
            self.register_buffer(f"reflection_{stage}", permutation, persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _permutation(self, stage):
        return getattr(self, f"reflection_{stage}")

    def _reflect_token(self, token, stage):
        permutation = self._permutation(stage)
        valid = (token >= 0) & (token < permutation.numel())
        return torch.where(valid, permutation[token.clamp(0, permutation.numel() - 1)], token)

    def _features(self, observation):
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(-1) / present.sum(-1).clamp_min(1)
        prices = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        prices = prices.masked_fill(ticks <= 0, 0).to(dtype)
        depths = (shares / 100).log1p()
        bid, ask = touch.unbind(-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(-1).unbind(-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = observation.previous_event.long()
        reflected = torch.stack(
            [self._reflect_token(previous[..., i], i) for i in range(4)], dim=-1
        )

        def encode(price, depth, skew, tokens):
            return torch.cat((
                price.flatten(-2),
                depth.flatten(-2),
                torch.stack((spread, skew, own), dim=-1).to(dtype),
                self.token_embed(tokens + self.offsets).flatten(-2),
            ), dim=-1)

        paired = torch.stack((
            encode(prices, depths, imbalance, previous),
            encode(-prices.flip(-2), depths.flip(-2), -imbalance, reflected),
        ), dim=1)
        return self.input_projection(paired).tanh()

    def _run(self, features, state):
        batch, _, steps, _ = features.shape
        h0 = None if state is None else state.reshape(1, batch * 2, 48)
        output, hidden = self.gru(features.reshape(batch * 2, steps, 48), h0)
        context = output.reshape(batch, 2, steps, 48).transpose(1, 2)
        return context, hidden.reshape(batch, 2, 48)

    def sequence_context(self, observation: StepObservation) -> torch.Tensor:
        return self._run(self._features(observation), None)[0]

    def step_context(
        self, observation: StepObservation, state: State
    ) -> tuple[torch.Tensor, State]:
        context, hidden = self._run(self._features(observation).unsqueeze(2), state)
        return context[:, 0], hidden

    def head_logits(
        self, context: torch.Tensor, stage: int, prior: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if stage:
            tokens = torch.stack(prior, dim=-1).long()
            reflected = torch.stack(
                [self._reflect_token(tokens[..., i], i) for i in range(stage)], dim=-1
            )
            paired = torch.stack((tokens, reflected), dim=-2)
            prefix = self.token_embed(paired + self.offsets[:stage]).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        direct, reflected = self.heads[stage](context).unbind(dim=-2)
        return (direct + reflected.index_select(-1, self._permutation(stage))) * 0.5
