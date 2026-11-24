import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        fixed = sum((9 + 8 * j) * c for j, c in enumerate(BUCKET_COUNTS))
        h = max(k for k in range(1, 49) if 6 * k * k + 82 * k + fixed <= 28000)
        self.state_dim = h
        offsets = [sum(BUCKET_COUNTS[:j]) for j in range(len(BUCKET_COUNTS))]
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), 8)
        self.input_projection = nn.Linear(75, h)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(h, h, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(h + 8 * j, c) for j, c in enumerate(BUCKET_COUNTS)
        )

    def _features(self, obs: StepObservation, touch: Tensor, previous_touch: Tensor) -> Tensor:
        dtype = self.input_projection.weight.dtype
        ticks = obs.book_ticks
        shares = obs.book_shares.to(dtype)
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        deep_ticks = ticks[..., 1:]
        relative = ((deep_ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        relative = relative.masked_fill(deep_ticks <= 0, 0)
        innovation = ((touch - previous_touch) / 20).clamp(-10, 10)
        innovation = innovation.masked_fill(~(present & (previous_touch > 0)), 0)
        prices = torch.cat((innovation.unsqueeze(-1), relative), dim=-1)
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = obs.is_own.reshape_as(spread)
        previous = self.token_embed(obs.previous_event + self.token_offsets)
        features = torch.cat((
            prices.flatten(-2), (shares / 100).log1p().flatten(-2),
            torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2),
        ), dim=-1)
        return self.input_projection(features.to(dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        touch = observation.book_ticks[..., 0].float()
        previous = torch.cat((torch.zeros_like(touch[:, :1]), touch[:, :-1]), dim=1)
        out, _ = self.gru(self._features(observation, touch, previous))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        touch = observation.book_ticks[..., 0].float()
        previous = torch.zeros_like(touch) if state is None else state[..., -2:].float()
        features = self._features(observation, touch, previous)
        prior = None if state is None else state[..., :-2].to(features.dtype).unsqueeze(0).contiguous()
        out, hidden = self.gru(features.unsqueeze(1), prior)
        return out[:, 0], torch.cat((hidden[0], touch), dim=-1)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
