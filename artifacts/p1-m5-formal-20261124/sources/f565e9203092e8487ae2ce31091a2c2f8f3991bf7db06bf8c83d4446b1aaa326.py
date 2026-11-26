from __future__ import annotations
import torch
from torch import Tensor, nn
from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,
)


class CandidateLOBWorldModel(LOBWorldModel):
    """Ten-level book encoder, GRU, and slow-context buffer for persistent impact."""

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
        # Slow-context leak, per state dim, initialized near-identity (slow decay).
        leak_logit = torch.full((state_dim,), 3.0)
        self.leak_logit = nn.Parameter(leak_logit)
        # Zero-initialized slow-context projection: begins as the parent model.
        self.slow_projection = nn.Linear(state_dim, state_dim)
        nn.init.zeros_(self.slow_projection.weight)
        nn.init.zeros_(self.slow_projection.bias)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )
        # Impact-relevant stages reuse slow context via the projected sum.
        self.register_buffer("slow_stages", torch.tensor([1, 2, 3]), persistent=False)

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

    def _leak(self) -> Tensor:
        return torch.sigmoid(self.leak_logit)

    def _slow_update(self, slow: Tensor, h: Tensor) -> Tensor:
        leak = self._leak()
        return (1.0 - leak) * slow + leak * h

    def _slow_scan(self, hidden_seq: Tensor, h0: Tensor | None) -> Tensor:
        batch, time, _ = hidden_seq.shape
        slow = hidden_seq.new_zeros((batch, self.state_dim)) if h0 is None else h0
        leak = self._leak()
        outputs = []
        for t in range(time):
            slow = (1.0 - leak) * slow + leak * hidden_seq[:, t]
            outputs.append(slow)
        return torch.stack(outputs, dim=1)

    def _blend(self, context: Tensor, slow: Tensor) -> Tensor:
        return context + self.slow_projection(slow)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        slow = self._slow_scan(out, None)
        return self._blend(out, slow)

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if state is None:
            h_prev = None
            slow_prev = None
        elif isinstance(state, tuple):
            h_prev = state[0].unsqueeze(0)
            slow_prev = state[1]
        else:
            h_prev = state.unsqueeze(0)
            slow_prev = None
        feat = self._features(observation).unsqueeze(1)
        out, hidden = self.gru(feat, h_prev)
        h_t = out[:, 0]
        slow_t = self._slow_update(
            h_t.new_zeros((h_t.shape[0], self.state_dim)) if slow_prev is None else slow_prev,
            h_t,
        )
        blended = self._blend(h_t, slow_t)
        return blended, (hidden[0], slow_t)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)
