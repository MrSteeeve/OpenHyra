from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder plus GRU, with a cheap cross-time difference head input."""

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
        # Zero-init scalar scaling the fast (difference) branch -> parent at step 0.
        self.fast_scale = nn.Parameter(torch.zeros(()))
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim + state_dim, count)
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

    def _ema(self, context: Tensor) -> Tensor:
        """Causal EMA over time via vectorized cumsum; [...,T,C] -> [...,T,C]."""
        *lead, T, C = context.shape
        flat = context.reshape(-1, T, C)
        decay = 0.9
        # weights decay^(T-1-i); normalized so most recent weight = 1
        steps = torch.arange(T, device=context.device, dtype=context.dtype)
        w = decay ** steps  # [T], increasing
        num = torch.cumsum(flat * w.view(1, T, 1), dim=1)
        den = torch.cumsum(w, dim=0).view(1, T, 1)
        ema = num / den
        return ema.reshape(*lead, T, C)

    def _head_input(self, context: Tensor, ema: Tensor) -> Tensor:
        fast = (context - ema) * self.fast_scale
        return torch.cat((context, fast), dim=-1)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        ema = self._ema(out)
        return self._head_input(out, ema)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        ctx = out[:, 0]
        # Step-level EMA equals context itself: fast branch is zero.
        return self._head_input(ctx, ctx), hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        ctx_dim = self.state_dim
        slow = context[..., :ctx_dim]
        extra = context[..., ctx_dim:]
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            ctx = torch.cat((slow, prefix, extra), dim=-1)
        else:
            ctx = context
        return self.heads[stage](ctx)
