import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        total = sum(BUCKET_COUNTS)
        weighted = sum(i * count for i, count in enumerate(BUCKET_COUNTS))
        width = min(8, (28000 - 18144 - 49 * total) // (192 + total + weighted))
        assert width > 0
        offsets = [sum(BUCKET_COUNTS[:i]) for i in range(len(BUCKET_COUNTS))]
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(total, width)
        self.input_projection = nn.Linear(43 + 4 * width, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + i * width, count)
            for i, count in enumerate(BUCKET_COUNTS)
        )
        self.adapter = nn.Linear(40, 48, bias=False)
        nn.init.zeros_(self.adapter.weight)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000

    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        prices = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        x = torch.cat((prices.flatten(-2), depth.flatten(-2)), dim=-1).to(dtype)

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        scalar = torch.stack((spread, imbalance, own), dim=-1).to(dtype)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((x, scalar, previous.flatten(-2)), dim=-1)
        return self.input_projection(features).tanh(), x

    def _context(self, hidden: Tensor, x: Tensor, previous_x: Tensor) -> Tensor:
        return hidden + self.adapter((x - previous_x).clamp(-2, 2))

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, x = self._features(observation)
        if x.size(1) == 0:
            return features
        hidden, _ = self.gru(features)
        previous_x = torch.cat((x[:, :1], x[:, :-1]), dim=1)
        return self._context(hidden, x, previous_x)

    def step_context(
        self, observation: StepObservation, state: State
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        features, x = self._features(observation)
        if state is None:
            initial, previous_x = None, x
        else:
            hidden, previous_x = state
            initial = hidden.unsqueeze(0)
        _, final = self.gru(features.unsqueeze(1), initial)
        hidden = final[0]
        return self._context(hidden, x, previous_x), (hidden, x)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
