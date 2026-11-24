from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder + GRU + leaky-integrator impact state feeding discrete heads."""

    def __init__(self, state_dim: int = 44, embedding_dim: int = 8, impact_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.impact_dim = impact_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        raw_feat_dim = 43 + 4 * embedding_dim
        self.input_projection = nn.Linear(raw_feat_dim + impact_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        # Impact integrator: projects raw features to impact_dim, leaky-accumulates.
        self.impact_proj = nn.Linear(raw_feat_dim, impact_dim)
        self.impact_gate = nn.Parameter(torch.zeros(impact_dim))
        self.register_buffer("impact_init", torch.zeros(impact_dim), persistent=False)
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + impact_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _raw_features(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        dtype = self.input_projection.weight.dtype
        book_shares = observation.book_shares.to(dtype)
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
        own = observation.is_own.to(dtype)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        return torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                          torch.stack((spread, imbalance, own), dim=-1),
                          previous.flatten(-2)), dim=-1).to(dtype)

    def _features(self, observation: StepObservation, impact_state: Tensor) -> Tensor:
        raw = self._raw_features(observation)
        feats = torch.cat((raw, impact_state), dim=-1)
        return self.input_projection(feats.to(self.input_projection.weight.dtype)).tanh()

    def _impact_step(self, raw: Tensor, impact_prev: Tensor) -> Tensor:
        gate = torch.sigmoid(self.impact_gate)
        drive = self.impact_proj(raw)
        return gate * impact_prev + (1.0 - gate) * torch.tanh(drive)

    def _impact_scan(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        # raw: [..., T, F]. Returns impact states [..., T, I] and final [..., I].
        drive = torch.tanh(self.impact_proj(raw))
        gate = torch.sigmoid(self.impact_gate)
        T = raw.shape[-2]
        init = self.impact_init.expand(drive.shape[:-1]).to(drive.dtype)
        h = init
        outs = []
        for t in range(T):
            h = gate * h + (1.0 - gate) * drive[..., t, :]
            outs.append(h)
        return torch.stack(outs, dim=-2), h

    def sequence_context(self, observation: StepObservation) -> Tensor:
        raw = self._raw_features(observation)
        impacts, _ = self._impact_scan(raw)
        feats = self._features(observation, impacts)
        out, _ = self.gru(feats)
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        impact_prev, hidden_prev = state
        raw = self._raw_features(observation)
        impact = self._impact_step(raw, impact_prev)
        feats = self._features(observation, impact).unsqueeze(1)
        out, hidden = self.gru(feats, hidden_prev.unsqueeze(0))
        return out[:, 0], (impact, hidden[0])

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        impact = context[..., self.state_dim:self.state_dim + self.impact_dim]
        gru_ctx = context[..., :self.state_dim]
        parts = [gru_ctx, impact]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            parts.append(prefix)
        return self.heads[stage](torch.cat(parts, dim=-1))

    def forward_sequence(self, batch) -> object:
        context = self.sequence_context(batch.observation)
        from stochopia.neural.lob_interface import StepSampler
        return StepSampler(self, context, batch.observation).conditioned(batch.targets)

    def forward_step(self, step_obs: StepObservation, hidden_state: State = None):
        from stochopia.neural.lob_interface import StepSampler
        if hidden_state is None:
            B = step_obs.book_ticks.shape[0]
            impact = self.impact_init.unsqueeze(0).expand(B, -1).to(self.input_projection.weight.dtype)
            hidden = torch.zeros(1, B, self.state_dim, dtype=self.input_projection.weight.dtype,
                                 device=step_obs.book_ticks.device)
            state = (impact, hidden[0])
        else:
            state = hidden_state
        context, new_state = self.step_context(step_obs, state)
        return StepSampler(self, context, step_obs), new_state
