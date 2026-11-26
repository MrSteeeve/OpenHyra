from __future__ import annotations

import torch
from torch import Tensor, nn

from stochopia.neural.lob_interface import (
    BUCKET_COUNTS,
    IS_OWN_FEATURE_INDEX,
    LOBWorldModel,
    State,
    StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, GRU, and discrete conditional order heads
    augmented with a leaky slow-context accumulator feeding stage-0 logits."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer(
            "token_offsets", torch.tensor(token_offsets), persistent=False
        )
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        # 40 book features + 3 scalars + 4 * embedding_dim previous-event features
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Slow-context accumulator: leaky EMA over the GRU output, injected
        # as a small zero-init residual on the stage-0 head input.
        self.slow_decay_logit = nn.Parameter(torch.tensor(-2.0))
        self.slow_residual = nn.Linear(state_dim, state_dim)
        nn.init.zeros_(self.slow_residual.weight)
        nn.init.zeros_(self.slow_residual.bias)

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
        features = torch.cat(
            (
                price_feature.flatten(-2),
                depth_feature.flatten(-2),
                torch.stack((spread, imbalance, observation.is_own), dim=-1),
                previous.flatten(-2),
            ),
            dim=-1,
        )
        return self.input_projection(
            features.to(self.input_projection.weight.dtype)
        ).tanh()

    def _slow_gate(self) -> Tensor:
        # sigmoid(-2.0) ~= 0.119 ; gate is 1 - decay in [0, 1]
        return torch.sigmoid(self.slow_decay_logit)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        feats = self._features(observation)
        out, _ = self.gru(feats)
        # Leaky EMA over the time axis (causal).
        decay = 1.0 - self._slow_gate()
        original_shape = out.shape
        flat = out.reshape(-1, original_shape[-2], original_shape[-1])
        # Simple causal cumulative EMA via scan-over-seq loop is too slow; use
        # a chunk-free sequential scan on the flattened batch (memory-light).
        B, T, C = flat.shape
        slow = torch.empty_like(flat)
        running = torch.zeros(B, C, dtype=flat.dtype, device=flat.device)
        for t in range(T):
            running = decay * running + (1.0 - decay) * flat[:, t]
            slow[:, t] = running
        # Inject zero-init residual signature into context; heads stage-0
        # receive this via head_logits by adding the residual once here so
        # the canonical computation is shared with step mode.
        slow = self.slow_residual(slow)
        return out + slow.view(original_shape)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        # State packs (gru_hidden[state_dim], slow[state_dim]) into one Tensor.
        if state is None:
            gru_h = None
            slow_prev = None
        else:
            gru_h = state[..., : self.state_dim].unsqueeze(0)
            slow_prev = state[..., self.state_dim :]
        feats = self._features(observation).unsqueeze(1)
        out, hidden = self.gru(feats, gru_h)
        ctx = out[:, 0]
        decay = 1.0 - self._slow_gate()
        if slow_prev is None:
            slow = (1.0 - decay) * ctx
        else:
            slow = decay * slow_prev + (1.0 - decay) * ctx
        ctx_aug = ctx + self.slow_residual(slow)
        new_state = torch.cat((hidden[0], slow), dim=-1)
        return ctx_aug, new_state

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
