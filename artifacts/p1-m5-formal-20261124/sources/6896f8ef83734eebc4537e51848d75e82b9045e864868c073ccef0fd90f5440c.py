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
        self.delta_projection = nn.Linear(4, 48, bias=False)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        nn.init.zeros_(self.delta_projection.weight)
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000

    def _encode(self, obs: StepObservation) -> tuple[Tensor, Tensor]:
        ticks = obs.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = obs.book_shares.to(dtype)
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
        own = obs.is_own.reshape_as(imbalance)
        previous = self.token_embed(obs.previous_event + self.token_offsets)
        features = torch.cat((
            price.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1),
            previous.flatten(-2),
        ), dim=-1)
        near = shares[..., :5].masked_fill(ticks[..., :5] <= 0, 0)
        stats = torch.stack((near[..., 0], near[..., 1:].sum(dim=-1)), dim=-1)
        return features.to(dtype), (stats.flatten(-2) / 100).log1p()

    def _run(
        self, features: Tensor, stats: Tensor, preceding: Tensor, hidden: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        delta = (stats - preceding).clamp(-3, 3)
        x = (self.input_projection(features) + self.delta_projection(delta)).tanh()
        return self.gru(x, hidden)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features, stats = self._encode(observation)
        preceding = torch.cat((stats[:, :1], stats[:, :-1]), dim=1)
        return self._run(features, stats, preceding, None)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        features, stats = self._encode(observation)
        if state is None:
            preceding, hidden = stats, None
        else:
            h, preceding = state.split((48, 4), dim=-1)
            hidden = h.unsqueeze(0).contiguous()
        out, hidden = self._run(
            features.unsqueeze(1), stats.unsqueeze(1), preceding.unsqueeze(1), hidden
        )
        return out[:, 0], torch.cat((hidden[0], stats), dim=-1)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
