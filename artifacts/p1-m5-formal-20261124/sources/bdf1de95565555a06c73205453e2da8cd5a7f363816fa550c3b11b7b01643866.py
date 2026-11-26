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
        self.transition_projection = nn.Linear(24, 48, bias=False)
        nn.init.zeros_(self.transition_projection.weight)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) <= 28000

    def _transition(
        self, ticks: Tensor, depth: Tensor, old_ticks: Tensor, old_depth: Tensor
    ) -> Tensor:
        present = ticks > 0
        matches = (ticks.unsqueeze(-1) == old_ticks.unsqueeze(-2)) & present.unsqueeze(-1)
        matched = matches.any(dim=-1)
        index = matches.long().argmax(dim=-1)
        prior_depth = old_depth.gather(-1, index)
        delta = (depth - prior_depth).clamp(-4, 4).masked_fill(~matched, 0)

        touch, old_touch = ticks[..., 0], old_ticks[..., 0]
        touch_delta = ((touch - old_touch) / 20).clamp(-10, 10)
        touch_delta = touch_delta.masked_fill((touch <= 0) | (old_touch <= 0), 0)
        unmatched = (present & ~matched).sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        return torch.cat((delta.flatten(-2), touch_delta, unmatched), dim=-1).to(depth.dtype)

    def _features(
        self, obs: StepObservation, depth: Tensor, transition: Tensor
    ) -> Tensor:
        ticks = obs.book_ticks
        shares = obs.book_shares.to(depth.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        price = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        price = price.masked_fill(ticks <= 0, 0)
        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        own = obs.is_own.reshape_as(spread)
        previous = self.token_embed(obs.previous_event.long() + self.token_offsets)
        features = torch.cat(
            (price.flatten(-2), depth.flatten(-2),
             torch.stack((spread, imbalance, own), dim=-1), previous.flatten(-2)),
            dim=-1,
        )
        return (self.input_projection(features.to(depth.dtype))
                + self.transition_projection(transition)).tanh()

    def _depth(self, observation: StepObservation) -> Tensor:
        return (observation.book_shares.to(self.input_projection.weight.dtype) / 100).log1p()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        depth = self._depth(observation)
        if ticks.size(1) == 0:
            return depth.new_empty(ticks.size(0), 0, 48)
        rest = self._transition(ticks[:, 1:], depth[:, 1:], ticks[:, :-1], depth[:, :-1])
        transition = torch.cat((depth.new_zeros(ticks.size(0), 1, 24), rest), dim=1)
        out, _ = self.gru(self._features(observation, depth, transition))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        ticks = observation.book_ticks
        depth = self._depth(observation)
        if state is None:
            hidden = None
            transition = depth.new_zeros(ticks.size(0), 24)
        else:
            h, old_ticks, old_depth = state
            hidden = h.unsqueeze(0)
            transition = self._transition(ticks, depth, old_ticks, old_depth)
        out, hidden = self.gru(self._features(observation, depth, transition).unsqueeze(1), hidden)
        return out[:, 0], (hidden[0], ticks.clone(), depth)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1).long() + self.token_offsets[:stage]
            context = torch.cat((context, self.token_embed(tokens).flatten(-2)), dim=-1)
        return self.heads[stage](context)
