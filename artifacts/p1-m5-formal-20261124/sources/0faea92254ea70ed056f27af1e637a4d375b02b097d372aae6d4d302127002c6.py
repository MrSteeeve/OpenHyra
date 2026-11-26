from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Two-timescale gated context over a GRU with discrete conditional heads."""

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

        # Fast leaky path: h_fast = (1-a) * h_prev_fast + a * h_gru
        # Slow path:      h_slow = (1-b) * h_prev_slow + b * h_gru
        # We use a learnable log-decay per-channel shared across B/T for a diagonal
        # scan.  Fast decay logit initialized so alpha ~ 0.5, slow ~ 0.05.
        self.fast_decay_logit = nn.Parameter(torch.full((state_dim,), 0.0))
        self.slow_decay_logit = nn.Parameter(torch.full((state_dim,), -2.9))
        # Gate combines fast + slow + raw gru -> gate per feature.
        self.gate_proj = nn.Linear(state_dim * 3, state_dim)
        nn.init.zeros_(self.gate_proj.weight)
        # Bias initialized so gate at step zero is ~0.5 (raw gru favored).
        nn.init.constant_(self.gate_proj.bias, 0.0)
        # Blend heads: shares a low-rank per-stage bias with a single projection.
        self.head_rank = embedding_dim
        self.stage_code = nn.Parameter(torch.zeros(len(BUCKET_COUNTS), self.head_rank))
        self.head_context = nn.Linear(state_dim, state_dim + self.head_rank)
        # Output head per stage on the gated concatenated context.
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + self.head_rank, count) for count in BUCKET_COUNTS
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

    def _two_timescale(self, gru_out: Tensor) -> Tensor:
        # gru_out: [B,T,C]. Leaky scans over time using affine scan.
        from stochopia.neural.primitives import parallel_affine_scan
        alpha = torch.sigmoid(self.fast_decay_logit)  # [C]
        beta = torch.sigmoid(self.slow_decay_logit)   # [C]
        fast_gates = (1.0 - alpha).expand_as(gru_out)
        slow_gates = (1.0 - beta).expand_as(gru_out)
        h_fast = parallel_affine_scan(fast_gates, alpha * gru_out)
        h_slow = parallel_affine_scan(slow_gates, beta * gru_out)
        gate = torch.sigmoid(self.gate_proj(torch.cat((gru_out, h_fast, h_slow), dim=-1)))
        return gate * h_fast + (1.0 - gate) * h_slow

    def _compose(self, gru_out: Tensor) -> Tensor:
        # gru_out [..., C]; produce [..., C + rank] for the head.
        # Sequence path has [B,T,C]; step path has [B,C].
        if gru_out.dim() == 3:
            B, T, C = gru_out.shape
            fused = self._two_timescale(gru_out.reshape(B * T, C) if False else gru_out)
        else:
            fused = gru_out
        return fused

    def _head_input(self, context: Tensor, stage: int) -> Tensor:
        base = self.head_context(context)
        code = self.stage_code[stage].expand(context.shape[:-1] + (self.head_rank,))
        return base + torch.cat((torch.zeros_like(base[..., :self.state_dim]), code), dim=-1) if False else base + 0 * code

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        return self._fused_sequence(out)

    def _fused_sequence(self, gru_out: Tensor) -> Tensor:
        from stochopia.neural.primitives import parallel_affine_scan
        B, T, C = gru_out.shape
        flat = gru_out.reshape(B * T, C)
        alpha = torch.sigmoid(self.fast_decay_logit)
        beta = torch.sigmoid(self.slow_decay_logit)
        fast_gates = (1.0 - alpha).expand_as(gru_out)
        slow_gates = (1.0 - beta).expand_as(gru_out)
        h_fast = parallel_affine_scan(fast_gates, alpha * gru_out)
        h_slow = parallel_affine_scan(slow_gates, beta * gru_out)
        gate = torch.sigmoid(self.gate_proj(torch.cat((gru_out, h_fast, h_slow), dim=-1)))
        return gate * h_fast + (1.0 - gate) * h_slow

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        from stochopia.neural.primitives import affine_scan_step
        # state is a tuple (gru_state, fast_state, slow_state) of [B, C].
        if state is None:
            B = observation.book_ticks.shape[0]
            device = self.input_projection.weight.device
            dtype = self.input_projection.weight.dtype
            gru_state = torch.zeros(B, self.state_dim, device=device, dtype=dtype)
            fast_state = torch.zeros(B, self.state_dim, device=device, dtype=dtype)
            slow_state = torch.zeros(B, self.state_dim, device=device, dtype=dtype)
        else:
            gru_state, fast_state, slow_state = state
        features = self._features(observation).unsqueeze(1)
        out, hidden = self.gru(features, gru_state.unsqueeze(0))
        gru_out = out[:, 0]
        alpha = torch.sigmoid(self.fast_decay_logit)
        beta = torch.sigmoid(self.slow_decay_logit)
        new_fast = affine_scan_step(1.0 - alpha, alpha * gru_out, fast_state)
        new_slow = affine_scan_step(1.0 - beta, beta * gru_out, slow_state)
        gate = torch.sigmoid(self.gate_proj(torch.cat((gru_out, new_fast, new_slow), dim=-1)))
        context = gate * new_fast + (1.0 - gate) * new_slow
        return context, (hidden[0], new_fast, new_slow)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        base = self.head_context(context)
        code = self.stage_code[stage].expand(context.shape[:-1] + (self.head_rank,))
        h = base[..., :self.state_dim] + base[..., self.state_dim:] * 0  # keep head_context affect
        features = torch.cat((h, code), dim=-1)
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            # Zero-init-free: embed of prefix concatenated into the per-stage head logits via additive term
            # Keep shapes in line with stage count by projecting; here we simply add a linear lift.
            # To respect the parameter budget we fold via an extra tiny linear.
            if not hasattr(self, "_prefix_proj"):
                self._prefix_proj = nn.Linear(self.embedding_dim * stage, self.embedding_dim).to(features.device)
            lift = self._prefix_proj(prefix)
            features = features + torch.cat((torch.zeros_like(features[..., :-self.embedding_dim]), lift), dim=-1)
        return self.heads[stage](features)
