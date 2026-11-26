from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder + GRU, with a separate impact-memory scalar path."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        # Magnitude bucket embedding for the impact-memory module.
        self.mag_embed = nn.Embedding(4, 4)
        # Learnable scalar gain on is_own (small but nonzero init).
        self.own_gain = nn.Parameter(torch.tensor(0.05))
        impact_width = 4
        self.input_projection = nn.Linear(
            43 + 4 * embedding_dim + impact_width, state_dim
        )
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Impact integrator: decay and input gains (per magnitude bucket).
        self.impact_decay = nn.Parameter(torch.tensor(0.9))
        self.impact_in = nn.Parameter(torch.zeros(4))
        # Heads see GRU context + impact scalar + prefix tokens.
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + 1 + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _book_features(self, observation: StepObservation) -> Tensor:
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
        own = observation.is_own * self.own_gain
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        return torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                          torch.stack((spread, imbalance, own), dim=-1),
                          previous.flatten(-2)), dim=-1)

    def _impact_bucket(self, observation: StepObservation) -> Tensor:
        # Magnitude = total executed size at the event; map to 0..3 buckets.
        own = observation.is_own
        size = observation.previous_event[..., 0].to(torch.float32)
        mag = (size.abs() + 1).log2()
        return mag.clamp(0, 3).to(torch.long) * own.to(torch.long)

    def _impact_input(self, observation: StepObservation) -> Tensor:
        own = observation.is_own.to(self.input_projection.weight.dtype)
        bucket = self._impact_bucket(observation)
        size = observation.previous_event[..., 0].to(self.input_projection.weight.dtype)
        sign = torch.sign(size)
        gain = self.impact_in[bucket]
        return gain * sign * own

    def _combined_features(self, observation: StepObservation, impact: Tensor) -> Tensor:
        base = self._book_features(observation)
        mag_emb = self.mag_embed(self._impact_bucket(observation))
        return torch.cat((base, mag_emb, impact.unsqueeze(-1)), dim=-1)

    def _project(self, observation: StepObservation, impact: Tensor) -> Tensor:
        return self.input_projection(
            self._combined_features(observation, impact)
        ).tanh()

    def _step_impact(self, observation: StepObservation, impact: Tensor) -> Tensor:
        decay = self.impact_decay.sigmoid()
        return decay * impact + self._impact_input(observation)

    # --- canonical computations ---

    def sequence_context(self, observation: StepObservation) -> Tensor:
        impact_in = self._impact_input(observation)
        T = impact_in.shape[-1]
        decay = self.impact_decay.sigmoid()
        impacts = []
        h = torch.zeros_like(impact_in[..., 0])
        for t in range(T):
            h = decay * h + impact_in[..., t]
            impacts.append(h)
        impact_seq = torch.stack(impacts, dim=-1)
        feats = self._project(observation, impact_seq)
        out, _ = self.gru(feats)
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        if state is None:
            state = (torch.zeros(observation.is_own.shape[:-1] + (self.state_dim,),
                                 dtype=self.input_projection.weight.dtype,
                                 device=observation.is_own.device),
                     torch.zeros(observation.is_own.shape[:-1],
                                 dtype=self.input_projection.weight.dtype,
                                 device=observation.is_own.device))
        hidden, impact = state
        impact = self._step_impact(observation, impact)
        feats = self._project(observation, impact)
        out, new_hidden = self.gru(feats.unsqueeze(1), hidden.unsqueeze(0))
        return out[:, 0], (new_hidden[0], impact)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        return self._head_logits(context, stage, prior)

    def _head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
