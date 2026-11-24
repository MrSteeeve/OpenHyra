import torch
from torch import nn
from torch.nn import functional as F
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self):
        super().__init__()
        offsets = [sum(BUCKET_COUNTS[:i]) for i in range(len(BUCKET_COUNTS))]
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000

    def _encode(self, observation: StepObservation):
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
        x = torch.cat(
            (price.flatten(-2), depth.flatten(-2),
             torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2)),
            dim=-1,
        ).to(dtype)
        book = F.linear(x[..., :42], self.input_projection.weight[:, :42]).tanh()
        return self.input_projection(x).tanh(), book

    def sequence_context(self, observation: StepObservation) -> torch.Tensor:
        encoded, book = self._encode(observation)
        out, _ = self.gru(encoded)
        return out + 0.25 * book

    def step_context(
        self, observation: StepObservation, state: State
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, book = self._encode(observation)
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(encoded.unsqueeze(1), prior)
        return out[:, 0] + 0.25 * book, hidden[0]

    def head_logits(
        self, context: torch.Tensor, stage: int, prior: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
