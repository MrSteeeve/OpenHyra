from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, leaky impact integrator, and conditional GRU heads."""

    def __init__(self, state_dim: int = 44, embedding_dim: int = 8, leak: float = 0.85) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.leak = leak
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.impact_dim = 8
        self.input_projection = nn.Linear(
            43 + 4 * embedding_dim + self.impact_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        # Leaky impact integrator: per-step impact features -> impact state.
        self.impact_proj = nn.Linear(43, self.impact_dim)
        self.impact_scale = nn.Parameter(torch.full((self.impact_dim,), 0.5))
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + self.impact_dim + stage * embedding_dim, count)
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
        base = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                          torch.stack((spread, imbalance), dim=-1)), dim=-1)
        return base.to(self.input_projection.weight.dtype)

    def _step_inputs(self, observation: StepObservation, impact_state: Tensor) -> Tensor:
        book = self._book_features(observation)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        is_own = observation.is_own.to(book.dtype).unsqueeze(-1)
        features = torch.cat((book, is_own, previous.flatten(-2), impact_state), dim=-1)
        return self.input_projection(features).tanh()

    def _impact_step(self, observation: StepObservation, impact_state: Tensor) -> Tensor:
        book = self._book_features(observation)
        drive = torch.tanh(self.impact_proj(book)) * self.impact_scale
        return self.leak * impact_state + (1.0 - self.leak) * drive

    def _scan(self, observation: StepObservation) -> tuple[Tensor, Tensor]:
        book = self._book_features(observation)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        is_own = observation.is_own.to(book.dtype).unsqueeze(-1)
        drive = torch.tanh(self.impact_proj(book)) * self.impact_scale
        T = book.shape[-2]
        impact = book.new_zeros(book.shape[:-1] + (self.impact_dim,))
        impacts = []
        for t in range(T):
            impact = self.leak * impact + (1.0 - self.leak) * drive[..., t, :]
            impacts.append(impact)
        impact_states = torch.stack(impacts, dim=-2)
        features = torch.cat((book, is_own, previous.flatten(-2), impact_states), dim=-1)
        inputs = self.input_projection(features).tanh()
        out, _ = self.gru(inputs)
        return out, impact_states

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, impacts = self._scan(observation)
        self._last_impacts = impacts
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        if state is None:
            hidden = None
            impact = self._book_features(observation).new_zeros(
                self._book_features(observation).shape[:-1] + (self.impact_dim,))
        else:
            if isinstance(state, tuple):
                hidden, impact = state
            else:
                hidden, impact = None, state
        impact = self._impact_step(observation, impact)
        x = self._step_inputs(observation, impact).unsqueeze(-2)
        prior = None if hidden is None else hidden.unsqueeze(0)
        out, new_hidden = self.gru(x, prior)
        return out[..., 0, :], (new_hidden[0], impact)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        impact = getattr(self, "_last_impacts", None)
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        if impact is not None and impact.shape[-2] == 1:
            impact = impact[..., 0, :]
            context = torch.cat((context, impact), dim=-1)
        elif impact is not None and impact.shape[:-1] == context.shape[:-1]:
            context = torch.cat((context, impact), dim=-1)
        else:
            context = torch.cat((context, context.new_zeros(
                context.shape[:-1] + (self.impact_dim,))), dim=-1)
        return self.heads[stage](context)
