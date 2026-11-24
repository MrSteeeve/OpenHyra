import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        counts = BUCKET_COUNTS
        c, e = sum(counts), 8
        for width in range(48, 0, -1):
            p = c * e + (44 + 4 * e) * width + 6 * width * (width + 1)
            p += sum((width + s * e + 1) * n for s, n in enumerate(counts))
            a = sum(width + 21 + s * e + n for s, n in enumerate(counts))
            r = min(8, (28000 - p - c) // a)
            if r >= 1:
                break
        self.state_dim = width
        self.register_buffer(
            "token_offsets",
            torch.tensor([sum(counts[:s]) for s in range(len(counts))]),
            persistent=False,
        )
        self.token_embed = nn.Embedding(c, e)
        self.input_projection = nn.Linear(43 + 4 * e, width)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(width, width, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(width + s * e, n) for s, n in enumerate(counts)
        )
        self.residuals = nn.ModuleList(
            nn.Sequential(nn.Linear(width + 20 + s * e, r), nn.SiLU(), nn.Linear(r, n))
            for s, n in enumerate(counts)
        )
        for residual in self.residuals:
            nn.init.zeros_(residual[-1].weight)
            nn.init.zeros_(residual[-1].bias)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000

    def _run(self, observation: StepObservation, state: State, step: bool):
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        prices = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = observation.is_own.reshape_as(imbalance)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((
            prices.flatten(-2), depth.flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)
        x = self.input_projection(features.to(dtype)).tanh()
        cumulative = (shares.cumsum(dim=-1) / 100).log1p().flatten(-2)
        out, hidden = self.gru(
            x.unsqueeze(1) if step else x,
            None if state is None else state.unsqueeze(0),
        )
        if step:
            out = out[:, 0]
        return torch.cat((out, cumulative.to(out.dtype)), dim=-1), hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._run(observation, None, False)[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        return self._run(observation, state, True)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        base = context[..., :self.state_dim]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            base = torch.cat((base, prefix), dim=-1)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](base) + self.residuals[stage](context)
