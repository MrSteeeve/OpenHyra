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
            nn.Linear(48 + stage * 8, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.volume_residual = nn.Sequential(
            nn.Linear(76, 16), nn.SiLU(), nn.Linear(16, BUCKET_COUNTS[3])
        )
        nn.init.zeros_(self.volume_residual[-1].weight)
        nn.init.zeros_(self.volume_residual[-1].bias)

    def _encode(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)
        queues = depth[..., :2].masked_fill(ticks[..., :2] <= 0, 0).flatten(-2)
        return self.input_projection(features.to(dtype)).tanh(), queues

    def _run(self, features: Tensor, queues: Tensor, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(features, prior)
        return torch.cat((out, queues.to(out.dtype)), dim=-1), hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, queues = self._encode(observation)
        return self._run(features, queues, None)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        features, queues = self._encode(observation)
        context, hidden = self._run(features.unsqueeze(1), queues.unsqueeze(1), state)
        return context[:, 0], hidden

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        base = context[..., :48]
        if not prior:
            return self.heads[stage](base)
        tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
        prefix = self.token_embed(tokens).flatten(-2)
        logits = self.heads[stage](torch.cat((base, prefix), dim=-1))
        if stage == 3:
            logits = logits + self.volume_residual(torch.cat((context, prefix), dim=-1))
        return logits
