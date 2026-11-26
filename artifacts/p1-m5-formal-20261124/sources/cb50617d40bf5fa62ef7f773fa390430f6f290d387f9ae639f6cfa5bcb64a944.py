from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder + GRU with a gated slow own-impact memory path."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Slow own-memory gate: gate on hidden state, memory = carried own scalar.
        self.memory_gate = nn.Linear(state_dim + 1, 1)
        self.memory_mix = nn.Linear(state_dim + 1, state_dim)
        nn.init.zeros_(self.memory_mix.weight)
        nn.init.zeros_(self.memory_mix.bias)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim + 1, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Bias correction on stage-0 logits to favour impact-relevant bins.
        self.register_buffer("impact_bias", torch.zeros(BUCKET_COUNTS[0]), persistent=True)

    def _features(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        book_shares = observation.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        price_feature = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth_feature = (book_shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = book_shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, observation.is_own), dim=-1),
                              previous.flatten(-2)), dim=-1)
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    def _slow_memory(self, hidden: Tensor, own_prev: Tensor) -> Tensor:
        augmented = torch.cat((hidden, own_prev.unsqueeze(-1)), dim=-1)
        gate = torch.sigmoid(self.memory_gate(augmented))
        return gate * self.memory_mix(augmented)

    def _head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...],
                     memory: Tensor) -> Tensor:
        ctx = torch.cat((context, memory), dim=-1)
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            ctx = torch.cat((ctx, prefix), dim=-1)
        logits = self.heads[stage](ctx)
        if stage == 0:
            logits = logits + self.impact_bias
        return logits

    def sequence_context(self, observation: StepObservation) -> Tensor:
        features = self._features(observation)
        out, _ = self.gru(features)
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        feats = self._features(observation).unsqueeze(1)
        out, hidden = self.gru(feats, prior)
        return out[:, 0], hidden[0]

    def _sequence_impl(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        features = self._features(observation)
        out, _ = self.gru(features)
        # slow memory: previous own flag, shifted along time
        own = observation.is_own.to(out.dtype)
        prev_own = torch.zeros_like(own)
        prev_own[..., 1:] = own[..., :-1]
        memory = self._slow_memory(out, prev_own)
        return out, memory

    def _step_impl(self, observation: StepObservation, state: State):
        if state is None:
            hidden_prev = None
        else:
            hidden_prev = state[0].unsqueeze(0)
        feats = self._features(observation).unsqueeze(1)
        out, hidden = self.gru(feats, hidden_prev)
        context = out[:, 0]
        own = observation.is_own.to(context.dtype)
        memory = self._slow_memory(context, own)
        return context, memory, hidden[0]

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._sequence_impl(observation)[0]

    def step_context(self, observation: StepObservation, state: State):
        context, memory, hidden = self._step_impl(observation, state)
        return context, (hidden, memory)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        memory = torch.zeros_like(context)
        return self._head_logits(context, stage, prior, memory)
