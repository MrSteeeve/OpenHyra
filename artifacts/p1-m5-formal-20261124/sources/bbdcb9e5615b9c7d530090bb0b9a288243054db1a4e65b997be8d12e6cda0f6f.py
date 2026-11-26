from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """A ten-level book encoder, GRU, and a dual timescale EMA side-channel."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Two-timescale EMA gates (fast, slow) over projected features.
        self.ema_gate_logit = nn.Parameter(torch.tensor([-2.0, -6.0]))
        head_in = state_dim + 2 * state_dim + 4 * embedding_dim
        self.stage_dims = [state_dim + 2 * state_dim + stage * embedding_dim
                           for stage in range(len(BUCKET_COUNTS))]
        self.heads = nn.ModuleList(
            nn.Linear(dim, count) for dim, count in zip(self.stage_dims, BUCKET_COUNTS)
        )
        # Zero-init new EMA-to-head columns on stage 0 so parent behaviour recovers.
        for head in self.heads:
            with torch.no_grad():
                head.weight[:, state_dim:state_dim + 2 * state_dim].zero_()

    def _raw_features(self, observation: StepObservation) -> Tensor:
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
        return features.to(self.input_projection.weight.dtype)

    def _project(self, raw: Tensor) -> Tensor:
        return self.input_projection(raw).tanh()

    def _ema_gates(self) -> Tensor:
        return torch.sigmoid(self.ema_gate_logit)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        x = self._project(self._raw_features(observation))
        gru_out, _ = self.gru(x)
        gates = self._ema_gates()
        # [B,T,2,C] -> EMA along T for two timescales.
        ema = torch.zeros_like(x[:, 0:1]).expand(-1, 2, -1).clone()
        emissions = []
        for t in range(x.shape[1]):
            ema = gates[None, :, None] * ema + (1 - gates[None, :, None]) * x[:, t:t + 1]
            emissions.append(ema)
        ema_out = torch.stack(emissions, dim=1)  # [B,T,2,C]
        return torch.cat((gru_out, ema_out.flatten(-2)), dim=-1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        if state is None:
            gru_state = None
            ema = torch.zeros(observation.previous_event.shape[0], 2, self.state_dim,
                              dtype=self.input_projection.weight.dtype,
                              device=observation.previous_event.device)
        else:
            gru_state = state["gru"].unsqueeze(0)
            ema = state["ema"]
        x = self._project(self._raw_features(observation)).unsqueeze(1)
        out, hidden = self.gru(x, gru_state)
        gates = self._ema_gates()
        ema = gates[None, :, None] * ema + (1 - gates[None, :, None]) * x[:, 0]
        context = torch.cat((out[:, 0], ema.flatten(-2)), dim=-1)
        new_state = {"gru": hidden[0], "ema": ema}
        return context, new_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        state_dim = self.state_dim
        gru_part = context[..., :state_dim]
        ema_part = context[..., state_dim:state_dim + 2 * state_dim]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            parts = [gru_part, ema_part, prefix]
        else:
            parts = [gru_part]
        return self.heads[stage](torch.cat(parts, dim=-1))
