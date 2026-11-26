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
        self.input_projection = nn.Linear(43 + 4 * 8, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.near_touch_weight = nn.Parameter(torch.zeros(4, 48))
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + stage * 8, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28_000

    def _features(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
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
        own = observation.is_own.reshape(spread.shape)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat(
            (price.flatten(-2), depth.flatten(-2),
             torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2)),
            dim=-1,
        )

        valid_shares = observation.book_shares.masked_fill(ticks <= 0, 0).clamp_min(0)
        near_touch = []
        for k in (1, 3):
            b, a = valid_shares[..., :k].sum(dim=-1).unbind(dim=-1)
            total = b + a
            near_touch.extend(((b - a) / total.clamp_min(1), (total / 100).log1p()))
        extra = torch.stack(near_touch, dim=-1).to(dtype)
        return (self.input_projection(features.to(dtype)) + extra @ self.near_touch_weight).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        return out[:, 0], hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
