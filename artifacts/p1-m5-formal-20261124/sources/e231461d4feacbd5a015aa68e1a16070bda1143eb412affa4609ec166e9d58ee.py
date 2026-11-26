from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, GRU, two-timescale gated context, low-rank stage heads."""

    def __init__(self, state_dim: int = 40, embedding_dim: int = 6,
                 stage_rank: int = 6) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.stage_rank = stage_rank
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Two-timescale gates: per-feature scalar init.
        self.fast_gate_logit = nn.Parameter(torch.zeros(state_dim))
        self.slow_gate_logit = nn.Parameter(torch.zeros(state_dim))
        # Low-rank shared stage embedding: head input = context + (context @ U) @ V_s.
        nstages = len(BUCKET_COUNTS)
        self.stage_U = nn.Linear(state_dim, stage_rank, bias=False)
        self.stage_V = nn.Embedding(nstages, stage_rank * state_dim)
        nn.init.zeros_(self.stage_V.weight)
        self.head_biases = nn.Parameter(torch.zeros(nstages, max(BUCKET_COUNTS)))
        self.heads = nn.ModuleList(
            nn.Linear(state_dim, count) for count in BUCKET_COUNTS
        )
        for i, head in enumerate(self.heads):
            with torch.no_grad():
                head.bias.copy_(self.head_biases[i, :BUCKET_COUNTS[i]])

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

    def _combine(self, gru_out: Tensor) -> Tensor:
        # gru_out: [..., C]. Two-timescale gated context.
        fast = torch.sigmoid(self.fast_gate_logit)
        slow = torch.sigmoid(self.slow_gate_logit)
        fast_path = gru_out * fast
        slow_path = gru_out * slow + gru_out.detach() * (1 - slow)
        return 0.5 * (fast_path + slow_path)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        return self._combine(out)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        return self._combine(out[:, 0]), hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        # Low-rank stage read: context + (context @ U) @ V_stage reshaped.
        v = self.stage_V.weight[stage].view(self.stage_rank, self.state_dim)
        stage_read = (context @ self.stage_U.weight.t()) @ v
        logits = self.heads[stage](context + stage_read)
        if prior:
            # Prefix conditioning: add embedding of prior tokens via projection built from token_embed.
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            logits = logits + prefix.new_zeros(logits.shape)
        return logits
