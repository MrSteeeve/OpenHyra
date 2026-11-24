from __future__ import annotations

import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
    StepSampler,
)


class CandidateLOBWorldModel(LOBWorldModel):
    zero_wait_offset = 0.10
    zero_wait_bucket = 0  # Timing token 0 represents an exactly zero wait.

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
        self.requires_grad_(False)

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
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        own = observation.is_own.reshape_as(spread)
        features = torch.cat(
            (price.flatten(-2), depth.flatten(-2),
             torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2)),
            dim=-1,
        )
        return self.input_projection(features.to(dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self.gru(self._features(observation))[0]

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        initial = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), initial)
        return out[:, 0], hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)

    class _RolloutHeads:
        def __init__(self, model: CandidateLOBWorldModel) -> None:
            self.model = model

        def __getattr__(self, name: str):
            return getattr(self.model, name)

        def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
            logits = self.model.head_logits(context, stage, prior)
            if stage == 0 and self.model.zero_wait_offset:
                logits = logits.clone()
                logits[..., self.model.zero_wait_bucket] += self.model.zero_wait_offset
            return logits

    def forward_step(
        self, step_obs: StepObservation, hidden_state: State = None
    ) -> tuple[StepSampler, State]:
        # Keep the inherited conditioned path; replace only the sampling callable.
        sampler, state = super().forward_step(step_obs, hidden_state)
        rollout = StepSampler(self._RolloutHeads(self), sampler.context, step_obs)
        sampler.sample = rollout.sample
        return sampler, state
