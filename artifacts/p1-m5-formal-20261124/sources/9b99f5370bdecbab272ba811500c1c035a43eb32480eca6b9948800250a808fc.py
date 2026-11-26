from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, GRU, and gated slow-memory for impact persistence."""

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
        # Slow-memory: scalar gate on previous own-flag carried in hidden state.
        self.memory_gate = nn.Linear(state_dim, 1)
        nn.init.zeros_(self.memory_gate.weight)
        nn.init.zeros_(self.memory_gate.bias)
        self.memory_proj = nn.Linear(1, state_dim)
        nn.init.zeros_(self.memory_proj.weight)
        nn.init.zeros_(self.memory_proj.bias)
        head_in = state_dim + state_dim + 3 * embedding_dim
        self.heads = nn.ModuleList(
            nn.Linear(head_in + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Bias correction on impact-relevant first head.
        with torch.no_grad():
            self.heads[0].bias.add_(0.0)

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

    def _memory_from_hidden(self, hidden: Tensor) -> Tensor:
        # hidden [..., state_dim] -> [..., state_dim] slow-memory addend
        gate = torch.sigmoid(self.memory_gate(hidden))
        return gate * hidden

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        out, _ = self.gru(feats)
        memory = self._memory_from_hidden(out)
        return torch.cat((out, memory), dim=-1)

    def step_context(self, observation: StepObservation, state: State):
        prior = None if state is None else state.unsqueeze(0)
        feats = self._features(observation).unsqueeze(1)
        out, hidden = self.gru(feats, prior)
        out = out[:, 0]
        memory = self._memory_from_hidden(out)
        return torch.cat((out, memory), dim=-1), hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        ctx = context
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            ctx = torch.cat((context, prefix), dim=-1)
        # Pad missing prefix slots so head input dim is constant.
        needed = (self.embedding_dim * 3) - (ctx.shape[-1] - context.shape[-1])
        if needed > 0:
            ctx = torch.cat((ctx, ctx.new_zeros(*ctx.shape[:-1], needed)), dim=-1)
        return self.heads[stage](ctx)
