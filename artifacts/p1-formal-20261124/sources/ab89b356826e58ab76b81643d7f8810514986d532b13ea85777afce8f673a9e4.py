from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """GRU encoder with a bounded stateless slow-sign readout bias in each head."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8, sign_rank: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.sign_rank = sign_rank
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Bounded slow-sign readout: maps (prev sign cue, imbalance) -> rank -> head bias.
        self.sign_in = nn.Linear(2, sign_rank, bias=False)
        self.sign_out = nn.ModuleList(nn.Linear(sign_rank, count) for count in BUCKET_COUNTS)
        nn.init.normal_(self.sign_in.weight, std=0.01)
        for layer in self.sign_out:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

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

    def _sign_cue(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        book_shares = observation.book_shares.to(self.input_projection.weight.dtype)
        bid_depth, ask_depth = book_shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        # Directional cue from previous event token: sign of its action bin.
        prev_action = observation.previous_event[..., 1]
        action_sign = torch.zeros_like(prev_action, dtype=imbalance.dtype)
        # Action buckets: 0..5 → build around mid; use centered index as signed cue.
        action_sign = (prev_action.to(imbalance.dtype) - 2.5) / 2.5
        cue = torch.stack((action_sign, imbalance), dim=-1)
        return torch.tanh(cue)

    def _sign_bias(self, observation: StepObservation, stage: int) -> Tensor:
        cue = self._sign_cue(observation)
        hidden = torch.tanh(self.sign_in(cue))
        return self.sign_out[stage](hidden)

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
            logits = self.heads[stage](torch.cat((context, prefix), dim=-1))
        else:
            logits = self.heads[stage](context)
        return logits

    def forward_sequence(self, batch) -> "FiveTupleLogits":  # type: ignore[name-defined]
        return super().forward_sequence(batch)

    def forward_step(self, step_obs: StepObservation, hidden_state: State = None):
        return super().forward_step(step_obs, hidden_state)
