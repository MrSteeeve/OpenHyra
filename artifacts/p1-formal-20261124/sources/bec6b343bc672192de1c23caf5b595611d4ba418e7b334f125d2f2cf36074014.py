from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder + GRU + zero-init fast/slow memory blend heads."""

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
        # Fast/slow memory blend: single shared smoothing decay and mix gate.
        self.fast_logit = nn.Parameter(torch.zeros(state_dim))
        self.mix_logit = nn.Parameter(torch.zeros(state_dim))
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

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

    def _blend(self, raw: Tensor, fast: Tensor) -> Tensor:
        # raw slow path; fast is per-step smoothed copy; mix zero-init -> parent behavior.
        mix = torch.sigmoid(self.mix_logit)
        return torch.lerp(raw, fast, mix)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        raw, _ = self.gru(self._features(observation))
        keep = torch.sigmoid(self.fast_logit)
        init = raw.new_zeros(raw.shape[:-2] + raw.shape[-1:])
        smooth = init
        outs = []
        for t in range(raw.shape[-2]):
            smooth = keep * smooth + (1 - keep) * raw[..., t, :]
            outs.append(self._blend(raw[..., t, :], smooth))
        return torch.stack(outs, dim=-2)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        # State carries (gru_hidden[state_dim], fast_memory[state_dim]).
        if state is None:
            gru_prior = None
            fast_prior = None
        elif isinstance(state, tuple):
            gru_prior, fast_prior = state
        else:
            gru_prior, fast_prior = state, None
        prior = None if gru_prior is None else gru_prior.unsqueeze(0)
        raw, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        raw = raw[:, 0]
        keep = torch.sigmoid(self.fast_logit)
        if fast_prior is None:
            smooth = raw * (1 - keep)
        else:
            smooth = keep * fast_prior + (1 - keep) * raw
        ctx = self._blend(raw, smooth)
        return ctx, (hidden[0], smooth)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
