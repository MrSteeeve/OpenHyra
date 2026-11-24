from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """A ten-level book encoder and GRU with discrete conditional order heads.

    Adds a learnable order-statistic pool over per-level depth imbalance plus a
    decaying exponential average of the top-of-book imbalance, fed as extra
    scalar features into the input projection (GRU and heads unchanged).
    """

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        # Extra scalar features: learnable level-pooled imbalance, top-of-book sign.
        self.input_projection = nn.Linear(43 + 4 * embedding_dim + 2, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Small learnable weights for an order-statistic pool over per-level depth.
        self.level_weight = nn.Parameter(torch.zeros(10))

    def _raw_features(self, observation: StepObservation) -> Tensor:
        """Per-step raw features (no EMA) so sequence and step stay identical."""
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

        # Signed per-level depth imbalance, pooled with learnable level weights.
        level_imb = depth_feature[..., 0, :] - depth_feature[..., 1, :]
        w = torch.softmax(self.level_weight, dim=0)
        pooled = (level_imb * w).sum(dim=-1)

        # Top-of-book imbalance sign scaled by relative depth.
        top_sign = torch.sign(imbalance) * torch.log1p(
            (bid_depth[..., 0] + ask_depth[..., 0]).clamp_min(1)
        )

        return torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                          torch.stack((spread, imbalance, observation.is_own), dim=-1),
                          previous.flatten(-2),
                          pooled.unsqueeze(-1),
                          top_sign.unsqueeze(-1)), dim=-1)

    def _project(self, raw: Tensor) -> Tensor:
        return self.input_projection(raw.to(self.input_projection.weight.dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._project(self._raw_features(observation)))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        # _raw_features on a single-step observation (T=1) uses the same code path.
        out, hidden = self.gru(self._project(self._raw_features(observation)).unsqueeze(1), prior)
        return out[:, 0], hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
