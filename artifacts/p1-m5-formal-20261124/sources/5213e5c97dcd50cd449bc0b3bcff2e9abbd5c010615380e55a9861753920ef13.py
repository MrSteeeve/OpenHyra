from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Dual-timescale GRU encoder with stage-conditioned gated heads."""

    def __init__(self, fast_dim: int = 24, slow_dim: int = 24,
                 embedding_dim: int = 3) -> None:
        super().__init__()
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.state_dim = fast_dim + slow_dim
        self.embedding_dim = embedding_dim
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        # book 2*10*2 prices + 2*10 shares + spread + imbalance + is_own = 43
        in_dim = 43 + 4 * embedding_dim
        self.input_projection = nn.Linear(in_dim, fast_dim + slow_dim)
        # Historical orders have no own labels: zero the is_own column.
        with torch.no_grad():
            self.input_projection.weight[:, IS_OWN_FEATURE_INDEX].zero_()
        self.gru_fast = nn.GRU(fast_dim + slow_dim, fast_dim, batch_first=True)
        self.gru_slow = nn.GRU(fast_dim + slow_dim, slow_dim, batch_first=True)
        # Scalar gate from imbalance: slow path weight.
        self.slow_gate = nn.Linear(1, 1)
        nn.init.zeros_(self.slow_gate.weight)
        nn.init.zeros_(self.slow_gate.bias)
        self.heads = nn.ModuleList(
            nn.Linear(self.state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _prepare(self, observation: StepObservation):
        ticks = observation.book_ticks
        present = ticks > 0
        raw_dtype = self.input_projection.weight.dtype
        shares = observation.book_shares.to(raw_dtype)
        touch = ticks[..., 0]
        has_touch = touch > 0
        mid = touch.sum(dim=-1) / has_touch.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        price_feature = relative.clamp(-10, 10).masked_fill(~present, 0)
        depth_feature = (shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~has_touch.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)

        prev = self.token_embed(observation.previous_event + self.token_offsets)
        raw = torch.cat((
            price_feature.flatten(-2),
            depth_feature.flatten(-2),
            torch.stack((spread, imbalance, observation.is_own.to(raw_dtype)), dim=-1),
            prev.flatten(-2),
        ), dim=-1)
        projected = self.input_projection(raw.to(raw_dtype)).tanh()
        fast_in = projected[..., :self.fast_dim]
        slow_in = projected[..., self.fast_dim:]
        return fast_in, slow_in, imbalance.to(raw_dtype)

    def _run_sequence(self, observation: StepObservation) -> Tensor:
        fwd = torch.cat(self._prepare(observation)[:2], dim=-1)
        fast_out, _ = self.gru_fast(fwd)
        slow_out, _ = self.gru_slow(fwd)
        _, _, imbalance = self._prepare(observation)
        gate = torch.sigmoid(self.slow_gate(imbalance.unsqueeze(-1))).squeeze(-1)
        return fast_out * (1 - gate.unsqueeze(-1)) + slow_out * gate.unsqueeze(-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return self._run_sequence(observation)

    def _step_run(self, observation: StepObservation, h_fast, h_slow):
        fast_in, slow_in, imbalance = self._prepare(observation)
        fwd = torch.cat((fast_in, slow_in), dim=-1).unsqueeze(1)
        fast_out, hf = self.gru_fast(fwd, h_fast)
        slow_out, hs = self.gru_slow(fwd, h_slow)
        gate = torch.sigmoid(self.slow_gate(imbalance.unsqueeze(-1)))
        ctx = fast_out[:, 0] * (1 - gate) + slow_out[:, 0] * gate
        return ctx, hf, hs

    def step_context(self, observation: StepObservation, state: State):
        state_dim = self.fast_dim + self.slow_dim
        if state is None:
            h_fast = torch.zeros(1, observation.is_own.shape[0], self.fast_dim,
                                 dtype=self.input_projection.weight.dtype)
            h_slow = torch.zeros(1, observation.is_own.shape[0], self.slow_dim,
                                 dtype=self.input_projection.weight.dtype)
        else:
            if isinstance(state, (tuple, list)):
                h_fast, h_slow = state[0], state[1]
                if h_fast.dim() == 2:
                    h_fast = h_fast.unsqueeze(0)
                if h_slow.dim() == 2:
                    h_slow = h_slow.unsqueeze(0)
            else:
                flat = state if state.dim() == 2 else state.unsqueeze(0)
                h_fast = flat[..., :self.fast_dim].contiguous()
                h_slow = flat[..., self.fast_dim:].contiguous()
                if h_fast.dim() == 2:
                    h_fast = h_fast.unsqueeze(0)
                if h_slow.dim() == 2:
                    h_slow = h_slow.unsqueeze(0)
        ctx, hf, hs = self._step_run(observation, h_fast, h_slow)
        flat = torch.cat((hf[0], hs[0]), dim=-1)
        return ctx, flat

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
