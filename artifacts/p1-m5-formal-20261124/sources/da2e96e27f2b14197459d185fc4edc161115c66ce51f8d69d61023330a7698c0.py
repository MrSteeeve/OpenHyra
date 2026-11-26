from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, GRU, dual-timescale gated context, low-rank tied heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8, rank: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.rank = rank
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Two-timescale gated context.
        self.fast_decay = nn.Parameter(torch.full((state_dim,), -1.0))  # sigmoid -> ~0.27
        self.slow_decay = nn.Parameter(torch.full((state_dim,), 2.0))   # sigmoid -> ~0.88
        self.gate = nn.Linear(2 * state_dim, state_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)  # favor fast path at init
        # Shared low-rank stage embedding + tied emit.
        self.stage_embed = nn.Parameter(torch.zeros(len(BUCKET_COUNTS), rank))
        nn.init.normal_(self.stage_embed, std=0.02)
        self.emit_proj = nn.Linear(state_dim + rank, rank)
        self.emit_scale = nn.Parameter(torch.ones(rank))
        self.emit_bias = nn.Parameter(torch.zeros(rank))
        self.stage_bias = nn.Parameter(torch.zeros(len(BUCKET_COUNTS), rank))
        # Per-stage output projection from shared rank space.
        total = sum(BUCKET_COUNTS)
        self.out_proj = nn.Parameter(torch.zeros(rank, total))
        offsets = [0]
        for c in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + c)
        self.register_buffer("out_offsets", torch.tensor(offsets), persistent=False)

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

    def _dual_timescale(self, gru_out: Tensor) -> Tensor:
        g_fast = torch.sigmoid(self.fast_decay)
        g_slow = torch.sigmoid(self.slow_decay)
        if gru_out.dim() == 2:
            f_seq = gru_out.unsqueeze(1)
            b, t, c = f_seq.shape
            hf = gru_out.new_zeros(b, c)
            hs = gru_out.new_zeros(b, c)
            hf = g_fast * hf + (1 - g_fast) * f_seq[:, 0]
            hs = g_slow * hs + (1 - g_slow) * f_seq[:, 0]
            gate = torch.sigmoid(self.gate(torch.cat((hf, hs), dim=-1)))
            return gate * hf + (1 - gate) * hs
        b, t, c = gru_out.shape
        hf = gru_out.new_zeros(b, c)
        hs = gru_out.new_zeros(b, c)
        fast_list, slow_list = [], []
        for i in range(t):
            hf = g_fast * hf + (1 - g_fast) * gru_out[:, i]
            hs = g_slow * hs + (1 - g_slow) * gru_out[:, i]
            fast_list.append(hf)
            slow_list.append(hs)
        fast_stack = torch.stack(fast_list, dim=1)
        slow_stack = torch.stack(slow_list, dim=1)
        gate = torch.sigmoid(self.gate(torch.cat((fast_stack, slow_stack), dim=-1)))
        return gate * fast_stack + (1 - gate) * slow_stack

    def sequence_context(self, observation: StepObservation) -> Tensor:
        gru_out, _ = self.gru(self._features(observation))
        return self._dual_timescale(gru_out)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        gru_out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        ctx = self._dual_timescale(gru_out[:, 0])
        return ctx, hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        base = context
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens)
            base_summary = torch.cat(
                (base, prefix.flatten(-2)[..., :self.rank].pad(
                    (0, max(0, self.rank - prefix.flatten(-2).shape[-1])), value=0)),
                dim=-1) if prefix.flatten(-2).shape[-1] < self.rank else \
                torch.cat((base, prefix.flatten(-2)[..., :self.rank]), dim=-1)
        else:
            base_summary = torch.cat(
                (base, torch.zeros_like(base[..., :self.rank])), dim=-1)
        z = self.emit_proj(base_summary) * self.emit_scale + self.emit_bias
        z = z + self.stage_bias[stage]
        count = BUCKET_COUNTS[stage]
        offset = int(self.out_offsets[stage].item())
        return torch.einsum("...r,ro->...o", z, self.out_proj[:, offset:offset + count])
