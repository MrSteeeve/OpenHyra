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
        self.register_buffer("wait_decay", torch.tensor([0.5] * 8 + [0.95] * 8),
                             persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + stage * 8, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        self.wait_residual = nn.Linear(24, BUCKET_COUNTS[0], bias=False)
        nn.init.zeros_(self.wait_residual.weight)

    def _encode(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        shares = observation.book_shares.to(self.input_projection.weight.dtype)
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
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        own = observation.is_own.reshape_as(spread)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1),
            previous.flatten(-2),
        ), dim=-1)
        return self.input_projection(features.to(shares.dtype)).tanh(), previous[..., 0, :]

    def _wait_terms(self, wait: Tensor) -> tuple[Tensor, Tensor]:
        decay = self.wait_decay.to(wait)
        return decay, (1 - decay) * torch.cat((wait, wait), dim=-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, wait = self._encode(observation)
        hidden, _ = self.gru(features)
        decay, inputs = self._wait_terms(wait)
        averages = parallel_affine_scan(decay, inputs)
        return torch.cat((hidden, wait, averages), dim=-1)

    def step_context(
        self, observation: StepObservation, state: State
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        features, wait = self._encode(observation)
        decay, inputs = self._wait_terms(wait)
        if state is None:
            initial_hidden = None
            previous_averages = torch.zeros_like(inputs)
        else:
            hidden, previous_averages = state
            initial_hidden = hidden.unsqueeze(0)
        _, next_hidden = self.gru(features.unsqueeze(1), initial_hidden)
        hidden = next_hidden[0]
        averages = affine_scan_step(decay, inputs, previous_averages)
        return torch.cat((hidden, wait, averages), dim=-1), (hidden, averages)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        hidden = context[..., :48]
        if stage == 0:
            return self.heads[0](hidden) + self.wait_residual(context[..., 48:])
        tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
        prefix = self.token_embed(tokens).flatten(-2)
        return self.heads[stage](torch.cat((hidden, prefix), dim=-1))
