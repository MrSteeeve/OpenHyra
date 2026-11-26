from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Book encoder + GRU with stage-conditioned signed-transient decay."""

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
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim + 2, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Transient signed-flow projection: state_dim -> (sign, magnitude).
        self.transient_proj = nn.Linear(state_dim, 2)
        nn.init.zeros_(self.transient_proj.weight)
        nn.init.zeros_(self.transient_proj.bias)
        # Decay gate: state_dim -> 2 scalars in (0,1) via sigmoid.
        self.decay_gate = nn.Linear(state_dim, 2)
        nn.init.zeros_(self.decay_gate.weight)
        nn.init.constant_(self.decay_gate.bias, -0.85)  # sigmoid ~0.3
        # Stage-conditioned discount table, sigmoid-initialized from ~0.3 to ~0.05.
        num_stages = len(BUCKET_COUNTS)
        init = torch.linspace(0.3, 0.05, num_stages)
        init = torch.log(init / (1 - init))
        self.register_buffer("stage_gate_logit", init, persistent=True)
        self.stage_gate_logit = nn.Parameter(init.clone())
        # Transient buffer recurrence scalars (2 state scalars: sign, magnitude).
        self.transient_mix = nn.Linear(state_dim + 2, 2)
        nn.init.zeros_(self.transient_mix.weight)
        nn.init.zeros_(self.transient_mix.bias)

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

    def _run_gru(self, features: Tensor, h0: Tensor | None = None) -> Tensor:
        out, _ = self.gru(features, h0)
        return out

    def _transient_step(self, h: Tensor, buf: Tensor) -> Tensor:
        raw = self.transient_proj(h)
        sign = torch.tanh(raw[..., 0:1])
        mag = torch.sigmoid(raw[..., 1:2])
        gate = torch.sigmoid(self.decay_gate(h))
        prev_sign, prev_mag = buf[..., 0:1], buf[..., 1:2]
        new_sign = gate[..., 0:1] * prev_sign + (1 - gate[..., 0:1]) * sign
        new_mag = gate[..., 1:2] * prev_mag + (1 - gate[..., 1:2]) * mag
        mixed = torch.tanh(self.transient_mix(torch.cat((h, new_sign, new_mag), dim=-1)))
        return torch.cat((new_sign * mixed[..., 0:1], new_mag * mixed[..., 1:2]), dim=-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        out = self._run_gru(feats)
        B, T, C = out.shape
        buf = out.new_zeros(B, 2)
        pieces = []
        for t in range(T):
            buf = self._transient_step(out[:, t], buf)
            pieces.append(torch.cat((out[:, t], buf), dim=-1))
        return torch.stack(pieces, dim=1)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        if state is None:
            h_prev = None
            buf = self.input_projection.weight.new_zeros(
                observation.is_own.shape[0], 2)
        else:
            h_prev = state[..., :self.state_dim]
            buf = state[..., self.state_dim:self.state_dim + 2]
        feats = self._features(observation).unsqueeze(1)
        out = self._run_gru(feats, None if h_prev is None else h_prev.unsqueeze(0))
        h = out[:, 0]
        buf = self._transient_step(h, buf)
        new_state = torch.cat((h, buf), dim=-1)
        return torch.cat((h, buf), dim=-1), new_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        h = context[..., :self.state_dim]
        buf = context[..., self.state_dim:self.state_dim + 2]
        pieces = [h]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            pieces.append(prefix)
        stage_scale = torch.sigmoid(self.stage_gate_logit[stage])
        pieces.append(buf * stage_scale)
        return self.heads[stage](torch.cat(pieces, dim=-1))
