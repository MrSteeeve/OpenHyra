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
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000
        self._check_context_parity()

    def _features(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks[..., :8] - mid[..., None, None]) / 20
        prices = relative.clamp(-10, 10).masked_fill(ticks[..., :8] <= 0, 0)

        near = ticks[..., 1:3]
        bid_gap = touch[..., 0, None] - near[..., 0, :]
        ask_gap = near[..., 1, :] - touch[..., 1, None]
        gaps = torch.stack((bid_gap, ask_gap), dim=-2)
        gaps = gaps.clamp(0, 200).to(prices.dtype).log1p()
        gaps = gaps.masked_fill(~(present[..., None] & (near > 0)), 0)
        prices = torch.cat((prices, gaps), dim=-1)

        total = shares.sum(dim=-1)
        fractions = torch.stack((shares[..., 0], shares[..., :3].sum(dim=-1)), dim=-1)
        fractions = fractions / total.clamp_min(1)[..., None]
        depth = torch.cat(((shares[..., :8] / 100).log1p(), fractions), dim=-1)

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = total.unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(spread)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat(
            (prices.flatten(-2), depth.flatten(-2),
             torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2)),
            dim=-1,
        )
        return self.input_projection(features.to(dtype)).tanh()

    def _context(self, features: Tensor, state: State = None) -> tuple[Tensor, Tensor]:
        out, hidden = self.gru(features, None if state is None else state.unsqueeze(0))
        return out, hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._context(self._features(observation))[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        out, hidden = self._context(self._features(observation).unsqueeze(1), state)
        return out[:, 0], hidden

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)

    @torch.no_grad()
    def _check_context_parity(self) -> None:
        device = self.token_offsets.device
        levels = torch.arange(10, device=device)
        ticks = torch.stack((100 - levels, 102 + levels)).expand(2, 3, 2, 10)
        shares = (1 + torch.arange(60, device=device)).reshape(1, 3, 2, 10).expand_as(ticks)
        previous = torch.arange(3, device=device).view(1, 3, 1).expand(2, 3, 4)
        own = torch.zeros((2, 3), device=device)
        expected = self.sequence_context(StepObservation(ticks, shares, previous, own))
        state = None
        contexts = []
        for t in range(3):
            obs = StepObservation(ticks[:, t], shares[:, t], previous[:, t], own[:, t])
            context, state = self.step_context(obs, state)
            contexts.append(context)
        torch.testing.assert_close(torch.stack(contexts, dim=1), expected)
