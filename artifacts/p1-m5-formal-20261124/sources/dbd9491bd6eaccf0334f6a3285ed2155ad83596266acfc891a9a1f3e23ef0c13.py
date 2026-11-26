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
        self.register_buffer("warmup_calls", torch.zeros((), dtype=torch.long))
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, 48)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(48 + 8 * stage, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000

    def _coefficient(self) -> Tensor:
        # Count training forwards, not timesteps; evaluation freezes the schedule.
        coefficient = self.warmup_calls.clamp(max=1000).to(torch.float64) * 0.0001
        if self.training:
            self.warmup_calls.add_(1)
        return coefficient

    def _run(
        self, observation: StepObservation, state: State, single: bool
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        ticks, shares = observation.book_ticks, observation.book_shares
        previous, own = observation.previous_event, observation.is_own
        own = own.reshape(ticks.shape[:-2])
        if single:
            ticks, shares, previous, own = (
                x.unsqueeze(1) for x in (ticks, shares, previous, own)
            )
        dtype = self.input_projection.weight.dtype
        shares = shares.to(dtype)
        touch = ticks[..., 0].to(torch.float64)
        present = touch > 0
        valid = present.all(dim=-1)
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        prices = ((ticks.to(torch.float64) - mid[..., None, None]) / 20)
        prices = prices.clamp(-10, 10).masked_fill(ticks <= 0, 0).to(dtype)
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~valid, 0).log1p().to(dtype)
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)

        if state is None:
            hidden = None
            memory = mid.new_zeros((mid.size(0), 4))
        else:
            hidden, memory = state
            hidden = hidden.unsqueeze(0)
        refs, seen = memory.split(2, dim=-1)
        coefficient = self._coefficient()
        residuals = []
        for current, good in zip(mid.unbind(1), valid.unbind(1)):
            current, good = current[:, None], good[:, None]
            delta = current - refs
            residuals.append(
                (delta / 20).clamp(-2, 2).masked_fill(~(good & seen.bool()), 0)
            )
            updated = torch.cat((refs[:, :1] + delta[:, :1] / 16, current), dim=-1)
            updated = torch.where(seen.bool(), updated, current)
            refs = torch.where(good, updated, refs)
            seen = torch.where(good, torch.ones_like(seen), seen)

        slow, preceding = (coefficient * torch.stack(residuals, dim=1)).to(dtype).unbind(-1)
        scalars = torch.stack((spread + slow, imbalance + preceding, own.to(dtype)), dim=-1)
        tokens = self.token_embed(previous + self.token_offsets).flatten(-2)
        features = torch.cat((
            prices.flatten(-2), (shares / 100).log1p().flatten(-2), scalars, tokens
        ), dim=-1)
        out, hidden = self.gru(self.input_projection(features).tanh(), hidden)
        return out, (hidden[0], torch.cat((refs, seen), dim=-1))

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._run(observation, None, False)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        out, state = self._run(observation, state, True)
        return out[:, 0], state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
