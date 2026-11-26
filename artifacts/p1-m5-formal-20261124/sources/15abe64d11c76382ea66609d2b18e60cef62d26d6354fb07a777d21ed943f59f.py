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
        self.input_projection = nn.Linear(43 + 32, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + stage * 8, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28_000

    def _features(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        shares = observation.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        price = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        near = shares[..., :3].masked_fill(ticks[..., :3] <= 0, 0)
        bid_depth, ask_depth = near.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat(
            (price.flatten(-2), depth.flatten(-2),
             torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2)),
            dim=-1,
        )
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    def _run(self, features: Tensor, state: State = None) -> tuple[Tensor, Tensor]:
        initial = None if state is None else state.unsqueeze(0)
        output, hidden = self.gru(features, initial)
        return output + 0.25 * features, hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        context, _ = self._run(self._features(observation))
        return context

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        context, hidden = self._run(self._features(observation).unsqueeze(1), state)
        return context[:, 0], hidden

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
