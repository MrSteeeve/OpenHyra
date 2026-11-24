import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        counts = tuple(BUCKET_COUNTS)
        embed = 8
        features = 43 + 4 * embed
        offsets = [sum(counts[:i]) for i in range(len(counts))]
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)

        def parameter_count(width: int) -> int:
            return (
                sum(counts) * embed
                + width * (features + 1 + 4)
                + 6 * width * (width + 1)
                + sum(count * (width + i * embed + 1)
                      for i, count in enumerate(counts))
            )

        width = next((w for w in range(48, 0, -1)
                      if parameter_count(w) <= 28000), 0)
        assert width > 0
        self.state_dim = width
        self.token_embed = nn.Embedding(sum(counts), embed)
        self.input_projection = nn.Linear(features, width)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(width, width, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(width + i * embed, count) for i, count in enumerate(counts)
        )
        self.depth_projection = nn.Linear(4, width, bias=False)
        nn.init.zeros_(self.depth_projection.weight)
        assert sum(p.numel() for p in self.parameters()) <= 28000

    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        summaries = (
            torch.stack((shares[..., 0], shares[..., 1:].sum(dim=-1)), dim=-1)
            / 100
        ).log1p().flatten(-2)

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        own = observation.is_own.reshape_as(spread)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)
        return features.to(dtype), summaries

    def _run(
        self, features: Tensor, depth: Tensor, previous_depth: Tensor,
        hidden: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        delta = (depth - previous_depth).clamp(-3, 3)
        inputs = (self.input_projection(features) + self.depth_projection(delta)).tanh()
        return self.gru(inputs, hidden)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, depth = self._features(observation)
        if depth.size(1) == 0:
            return features.new_empty((features.size(0), 0, self.state_dim))
        previous_depth = torch.cat((depth[:, :1], depth[:, :-1]), dim=1)
        out, _ = self._run(features, depth, previous_depth, None)
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        features, depth = self._features(observation)
        if state is None:
            hidden, previous_depth = None, depth
        else:
            hidden = state[:, :self.state_dim].unsqueeze(0).contiguous()
            previous_depth = state[:, self.state_dim:]
        out, hidden = self._run(
            features.unsqueeze(1), depth.unsqueeze(1), previous_depth.unsqueeze(1), hidden,
        )
        return out[:, 0], torch.cat((hidden[0], depth), dim=-1)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
