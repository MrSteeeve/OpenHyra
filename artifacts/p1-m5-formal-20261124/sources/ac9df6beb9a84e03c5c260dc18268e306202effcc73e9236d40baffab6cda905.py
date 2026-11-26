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
        self.input_projection = nn.Linear(43 + 32, 41)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(41, 41, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + stage * 8, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000

    def _features(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = relative.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0)
        spread = spread.to(dtype).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread).to(dtype)
        previous = self.token_embed(observation.previous_event.long() + self.token_offsets)
        features = torch.cat((
            price.flatten(-2).to(dtype), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)

        valid_shares = shares.masked_fill(ticks <= 0, 0)
        touch_depth = (valid_shares[..., 0] / 100).log1p()
        bid_cum, ask_cum = valid_shares.cumsum(dim=-1).unbind(dim=-2)
        cumulative = (bid_cum - ask_cum) / (bid_cum + ask_cum).clamp_min(1)
        direct = torch.cat(
            (spread.unsqueeze(-1), touch_depth, cumulative[..., [0, 2, 4, 9]]),
            dim=-1,
        )
        return self.input_projection(features).tanh(), direct

    def _context(
        self, observation: StepObservation, state: State, *, step: bool
    ) -> tuple[Tensor, Tensor]:
        features, direct = self._features(observation)
        if step:
            features = features.unsqueeze(1)
        initial = None if state is None else state.unsqueeze(0)
        output, hidden = self.gru(features, initial)
        if step:
            output = output[:, 0]
        return torch.cat((output, direct.to(output.dtype)), dim=-1), hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._context(observation, None, step=False)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        return self._context(observation, state, step=True)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1).long() + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
